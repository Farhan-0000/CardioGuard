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
