"""Disk request cache (memoisation hook for evaluation)."""

import sys
import threading
import types

from evaluation.cache import DiskRequestCache, install_request_cache


def test_get_set_and_persistence(tmp_path):
    path = tmp_path / "c" / "llm_cache.sqlite"
    cache = DiskRequestCache(path)
    assert cache.get("k") is None
    cache.set("k", '{"a": 1}')
    cache.set("k", '{"a": 2}')  # last write wins
    assert cache.get("k") == '{"a": 2}'
    assert len(cache) == 1 and "k" in cache and "x" not in cache and 3 not in cache
    assert cache.stats()["hits"] == 1 and cache.stats()["misses"] == 1
    cache.close()
    reopened = DiskRequestCache(path)
    assert reopened.get("k") == '{"a": 2}'
    reopened.close()


def test_thread_safety(tmp_path):
    cache = DiskRequestCache(tmp_path / "c.sqlite")

    def worker(n: int) -> None:
        for i in range(50):
            cache.set(f"{n}-{i}", str(i))
            assert cache.get(f"{n}-{i}") == str(i)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(cache) == 400
    cache.close()


def test_two_instances_share_file(tmp_path):
    a = DiskRequestCache(tmp_path / "shared.sqlite")
    b = DiskRequestCache(tmp_path / "shared.sqlite")
    a.set("x", "1")
    assert b.get("x") == "1"
    a.close()
    b.close()


def test_install_registers_with_wrapper(tmp_path, monkeypatch):
    installed = []
    fake = types.ModuleType("medbridge.ai.llm_wrapper")
    fake.set_request_cache = installed.append
    monkeypatch.setitem(sys.modules, "medbridge.ai.llm_wrapper", fake)
    cache = install_request_cache(tmp_path / "c.sqlite")
    assert installed == [cache]
    assert install_request_cache(None) is None
    assert installed[-1] is None
    cache.close()


# ----------------------------------------------------------------------
# Model-digest namespacing (finding ablation-study-validity-5)
# ----------------------------------------------------------------------

import asyncio

import pytest

from evaluation.cache import (
    ALL_CALLS,
    CacheNamespaceError,
    bypass_cache_reads,
    instrument_llm,
    is_complete_json,
    model_digests,
    namespaced_path,
)


def test_namespace_changes_when_a_model_is_repulled(tmp_path):
    base = tmp_path / "llm_cache.sqlite"
    old = namespaced_path(base, {"qwen3:8b": "sha256:aaa"})
    new = namespaced_path(base, {"qwen3:8b": "sha256:bbb"})
    assert old != new and old.parent == base.parent and old.suffix == ".sqlite"
    assert namespaced_path(base, {"a": "1", "b": "2"}) == namespaced_path(
        base, {"b": "2", "a": "1"}
    )
    cache = DiskRequestCache(old, model_digests={"qwen3:8b": "sha256:aaa"})
    cache.set("k", '{"text": "old weights"}')
    cache.close()
    repulled = DiskRequestCache(new, model_digests={"qwen3:8b": "sha256:bbb"})
    assert repulled.get("k") is None  # outputs of the old weights are never replayed
    repulled.close()
    with pytest.raises(CacheNamespaceError):
        DiskRequestCache(old, model_digests={"qwen3:8b": "sha256:bbb"})


def test_model_digests_from_ollama_listing():
    info = {"qwen3:8b": {"digest": "d1"}, "llama3:latest": {"digest": "d2"}}
    assert model_digests(info, ["qwen3:8b", "llama3"]) == {"qwen3:8b": "d1", "llama3": "d2"}
    with pytest.raises(LookupError):
        model_digests(info, ["gemma3:12b"])
    with pytest.raises(LookupError):
        model_digests({"error": "ConnectError"}, ["qwen3:8b"])


def test_install_namespaces_by_digest_and_instruments_wrapper(tmp_path, monkeypatch):
    installed = []

    class Wrapper:
        async def call(self, call_name, *args, **kwargs):
            return call_name

    wrapper = Wrapper()
    fake = types.ModuleType("medbridge.ai.llm_wrapper")
    fake.set_request_cache = installed.append
    fake.get_llm_wrapper = lambda: wrapper
    monkeypatch.setitem(sys.modules, "medbridge.ai.llm_wrapper", fake)
    cache = install_request_cache(tmp_path / "c.sqlite", model_digests={"m": "sha256:1"})
    assert cache.path == namespaced_path(tmp_path / "c.sqlite", {"m": "sha256:1"})
    assert cache.stats()["key_includes_digest"] and cache.stats()["model_digests"] == {
        "m": "sha256:1"
    }
    assert cache.metadata()["model_digests"] == '{"m": "sha256:1"}'
    assert getattr(wrapper.call, "_evaluation_instrumented", False)
    instrument_llm(wrapper)  # idempotent
    assert asyncio.run(wrapper.call("judge_label")) == "judge_label"
    cache.close()


# ----------------------------------------------------------------------
# Incomplete (truncated) outputs are never memoised (finding statistics-judges-5)
# ----------------------------------------------------------------------


def test_truncated_json_is_neither_stored_nor_served(tmp_path):
    cache = DiskRequestCache(tmp_path / "c.sqlite")
    truncated = '{"claims": [{"claim": "Lisinopril lowers BP", "source_sentence": "Lis'
    assert not is_complete_json(truncated) and is_complete_json('{"claims": []}')
    cache.set("k", truncated)
    assert "k" not in cache and cache.stats()["rejected_incomplete"] == 1
    cache._conn.execute("INSERT INTO llm_cache VALUES ('old', ?, 0)", (truncated,))  # legacy row
    assert cache.get("old") is None
    cache.close()


# ----------------------------------------------------------------------
# Call-level read bypass for re-runs of degraded turns
# ----------------------------------------------------------------------


class _CachedLLM:
    """Minimal LLM whose ``call`` consults the cache like the wrapper does."""

    def __init__(self, cache):
        self.cache = cache

    async def call(self, call_name, *args, **kwargs):
        await asyncio.sleep(0)
        hit = self.cache.get(f"key-{call_name}")
        if hit is None:
            self.cache.set(f"key-{call_name}", '{"fresh": true}')
            return "computed"
        return "cached"


def test_bypass_only_affects_the_named_calls(tmp_path):
    cache = DiskRequestCache(tmp_path / "c.sqlite")
    llm = _CachedLLM(cache)
    instrument_llm(llm)
    for name in ("context_extractor", "response_generator"):
        cache.set(f"key-{name}", '{"x": 1}')

    async def turn():
        return await asyncio.gather(
            llm.call("context_extractor"), llm.call(call_name="response_generator")
        )

    async def scenario():
        plain = await turn()
        with bypass_cache_reads({"response_generator"}):
            targeted = await turn()
        with bypass_cache_reads({ALL_CALLS}):
            everything = await turn()
        after = await turn()
        return plain, targeted, everything, after

    plain, targeted, everything, after = asyncio.run(scenario())
    assert plain == ["cached", "cached"]
    assert targeted == ["cached", "computed"]
    assert everything == ["computed", "computed"]
    assert after == ["cached", "cached"]
    assert cache.stats()["bypassed_reads"] == 3
    cache.close()


def test_unknown_call_inside_bypass_scope_misses(tmp_path):
    cache = DiskRequestCache(tmp_path / "c.sqlite")
    cache.set("k", '{"a": 1}')
    with bypass_cache_reads({"response_generator"}):
        assert cache.get("k") is None  # no instrumentation → the call is unknown → bypassed
    assert cache.get("k") == '{"a": 1}'
    cache.close()
