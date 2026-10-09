"""
Unit tests for the Resilient LLM Wrapper (TASK-17, ADL-028).

All HTTP traffic goes through an ``httpx.MockTransport`` — no real Ollama or
remote API is contacted. Covers: native Ollama payload (structured-output
``format`` schema, options, keep_alive, think gating), per-call model/seed,
LOCAL_ONLY_MODE, optional remote providers and failover, retry/backoff,
JSON extraction and repair, call statistics, the request cache, warm-up and
health.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from medbridge.ai import llm_wrapper
from medbridge.ai.llm_wrapper import (
    LLMCallStat,
    ResilientLLMWrapper,
    get_llm_wrapper,
    llm_call_log,
    llm_format_schema,
    request_cache_key,
    set_request_cache,
)
from medbridge.ai.schemas.extractor import ExtractorOutput
from medbridge.ai.schemas.response_generator import ResponseGeneratorOutput
from medbridge.config import Settings


class SampleOutput(BaseModel):
    action: str
    rationale: str


VALID_JSON = json.dumps({"action": "PROCEED", "rationale": "Context sufficient"})


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "LLM_PROVIDER": "ollama",
        "LOCAL_ONLY_MODE": True,
        "OLLAMA_BASE_URL": "http://127.0.0.1:11434/v1",
        "LLM_MODEL": "test-model",
        "LLM_MAX_RETRIES": 3,
        "LLM_RETRY_BASE_DELAY": 0.0,
        "LLM_SEED": 42,
        "LLM_KEEP_ALIVE": "30m",
        "LLM_NUM_CTX": 8192,
        "LLM_THINK": False,
        "GROQ_API_KEY": "",
        "OPENAI_API_KEY": "",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


class FakeServer:
    """Routes requests to per-path handlers and records everything it receives."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.chat_contents: list[str] = [VALID_JSON]
        self.chat_handler: Callable[[httpx.Request], httpx.Response] | None = None
        self.capabilities: dict[str, list[str]] = {}
        self.tags: list[str] = ["test-model"]
        self.loaded: list[str] = []

    def bodies(self, path: str) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.url.path == path]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/api/show":
            model = json.loads(request.content)["model"]
            return httpx.Response(200, json={"capabilities": self.capabilities.get(model, ["completion"])})
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": n} for n in self.tags]})
        if path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": n} for n in self.loaded]})
        if path == "/api/generate":
            return httpx.Response(200, json={"done": True})
        if self.chat_handler is not None:
            return self.chat_handler(request)
        content = self.chat_contents.pop(0) if len(self.chat_contents) > 1 else self.chat_contents[0]
        if path == "/api/chat":
            return httpx.Response(200, json={
                "message": {"role": "assistant", "content": content},
                "prompt_eval_count": 120, "eval_count": 30,
                "total_duration": 2_500_000_000, "load_duration": 1_000_000_000,
                "done_reason": "stop",
            })
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
        })


@pytest.fixture
def server() -> FakeServer:
    return FakeServer()


def make_wrapper(server: FakeServer, **overrides: Any) -> ResilientLLMWrapper:
    return ResilientLLMWrapper(make_settings(**overrides), transport=httpx.MockTransport(server.handler))


async def call(wrapper: ResilientLLMWrapper, schema: type[BaseModel] = SampleOutput, **kwargs: Any):
    return await wrapper.call("test", "system", "user", schema, **kwargs)


# ---------------------------------------------------------------------------
# Native Ollama request
# ---------------------------------------------------------------------------


class TestNativeOllamaPayload:
    async def test_posts_native_chat_endpoint_with_v1_stripped(self, server: FakeServer) -> None:
        result = await call(make_wrapper(server))
        assert isinstance(result, SampleOutput) and result.action == "PROCEED"
        chat = [r for r in server.requests if r.url.path == "/api/chat"]
        assert len(chat) == 1
        assert str(chat[0].url) == "http://127.0.0.1:11434/api/chat"

    async def test_payload_shape(self, server: FakeServer) -> None:
        await call(make_wrapper(server), temperature=0.0, max_tokens=256)
        body = server.bodies("/api/chat")[0]
        assert body["model"] == "test-model"
        assert body["stream"] is False
        assert body["messages"] == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]
        assert body["format"] == llm_format_schema(SampleOutput)
        assert body["options"] == {
            "temperature": 0.0, "seed": 42, "num_ctx": 8192, "num_predict": 256, "top_p": 1.0,
        }
        assert body["keep_alive"] == "30m"

    async def test_generator_sampling_keeps_seed_and_uses_top_p_095(self, server: FakeServer) -> None:
        await call(make_wrapper(server), temperature=0.3)
        options = server.bodies("/api/chat")[0]["options"]
        assert options["temperature"] == 0.3
        assert options["top_p"] == 0.95
        assert options["seed"] == 42

    async def test_model_and_seed_override(self, server: FakeServer) -> None:
        await call(make_wrapper(server), model="gemma3:4b", seed=7)
        body = server.bodies("/api/chat")[0]
        assert body["model"] == "gemma3:4b"
        assert body["options"]["seed"] == 7

    async def test_think_sent_only_to_thinking_models(self, server: FakeServer) -> None:
        server.capabilities = {"qwen3:8b": ["completion", "thinking"], "gemma3:4b": ["completion"]}
        wrapper = make_wrapper(server)
        await call(wrapper, model="qwen3:8b")
        await call(wrapper, model="gemma3:4b")
        first, second = server.bodies("/api/chat")
        assert first["think"] is False
        assert "think" not in second

    async def test_capabilities_cached_per_model(self, server: FakeServer) -> None:
        server.capabilities = {"qwen3:8b": ["thinking"]}
        wrapper = make_wrapper(server)
        for _ in range(3):
            await call(wrapper, model="qwen3:8b")
        assert len(server.bodies("/api/show")) == 1

    async def test_show_failure_omits_think(self, server: FakeServer) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/show":
                return httpx.Response(500)
            return server.handler(request)

        wrapper = ResilientLLMWrapper(make_settings(), transport=httpx.MockTransport(handler))
        assert await call(wrapper) is not None
        assert "think" not in server.bodies("/api/chat")[0]


class TestFormatSchema:
    def test_refs_inlined_and_all_properties_required(self) -> None:
        schema = llm_format_schema(ExtractorOutput)
        assert "$defs" not in json.dumps(schema)
        assert schema["required"] == ["delta_events", "search_query", "raw_intent"]
        item = schema["properties"]["delta_events"]["items"]
        assert item["required"] == ["event_type", "payload"]
        assert item["properties"]["event_type"]["enum"][0] == "BP_READING"

    def test_llm_omit_properties_removed(self) -> None:
        schema = llm_format_schema(ResponseGeneratorOutput)
        assert list(schema["properties"]) == ["response_text"]
        assert schema["required"] == ["response_text"]


# ---------------------------------------------------------------------------
# Local-only mode and remote providers
# ---------------------------------------------------------------------------


class TestProviders:
    async def test_local_only_blocks_remote_provider(self, server: FakeServer) -> None:
        wrapper = make_wrapper(
            server, LLM_PROVIDER="groq", GROQ_API_KEY="k1", OPENAI_API_KEY="k2", LOCAL_ONLY_MODE=True
        )
        assert wrapper.provider_names == ["ollama"]
        server.chat_handler = lambda request: httpx.Response(503)
        assert await call(wrapper) is None
        assert all(r.url.host == "127.0.0.1" for r in server.requests)

    async def test_remote_provider_when_local_only_disabled(self, server: FakeServer) -> None:
        wrapper = make_wrapper(
            server, LLM_PROVIDER="groq", GROQ_API_KEY="secret", LOCAL_ONLY_MODE=False,
            GROQ_MODEL="llama-3.1-8b-instant",
        )
        assert wrapper.provider_names == ["groq"]
        with llm_call_log() as stats:
            result = await call(wrapper)
        assert result is not None
        request = server.requests[-1]
        assert str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer secret"
        body = json.loads(request.content)
        assert body["model"] == "llama-3.1-8b-instant"
        assert body["response_format"] == {"type": "json_object"}
        assert body["seed"] == 42
        assert stats[0].provider == "groq" and stats[0].prompt_tokens == 50

    async def test_remote_without_key_falls_back_to_ollama(self, server: FakeServer) -> None:
        wrapper = make_wrapper(server, LLM_PROVIDER="openai", LOCAL_ONLY_MODE=False)
        assert wrapper.provider_names == ["ollama"]

    async def test_failover_to_remote_when_allowed(self, server: FakeServer) -> None:
        wrapper = make_wrapper(
            server, LLM_PROVIDER="ollama", LOCAL_ONLY_MODE=False, GROQ_API_KEY="k", LLM_MAX_RETRIES=2,
        )
        assert wrapper.provider_names == ["ollama", "groq"]

        def chat(request: httpx.Request) -> httpx.Response:
            if request.url.host == "127.0.0.1":
                raise httpx.ReadTimeout("slow", request=request)
            return httpx.Response(200, json={"choices": [{"message": {"content": VALID_JSON}}]})

        server.chat_handler = chat
        result = await call(wrapper)
        assert result is not None
        hosts = [r.url.host for r in server.requests if r.url.path.endswith(("chat", "completions"))]
        assert hosts == ["127.0.0.1", "127.0.0.1", "api.groq.com"]

    async def test_remote_blocked_at_send_time_in_local_only(self, server: FakeServer) -> None:
        wrapper = make_wrapper(server)
        remote = llm_wrapper._Provider("groq", "https://api.groq.com/openai/v1", "k", "m")
        wrapper._providers = [remote]
        assert await call(wrapper) is None
        assert not any(r.url.host == "api.groq.com" for r in server.requests)


# ---------------------------------------------------------------------------
# Retry, extraction and repair
# ---------------------------------------------------------------------------


class TestRetries:
    async def test_retry_on_timeout_then_succeed(self, server: FakeServer) -> None:
        attempts: list[int] = []

        def chat(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            if len(attempts) == 1:
                raise httpx.ReadTimeout("timeout", request=request)
            return httpx.Response(200, json={"message": {"content": VALID_JSON}})

        server.chat_handler = chat
        with llm_call_log() as stats:
            assert await call(make_wrapper(server)) is not None
        assert len(attempts) == 2 and stats[0].attempts == 2

    async def test_retry_on_500_then_succeed(self, server: FakeServer) -> None:
        responses = [httpx.Response(500), httpx.Response(200, json={"message": {"content": VALID_JSON}})]
        server.chat_handler = lambda request: responses.pop(0)
        assert await call(make_wrapper(server)) is not None

    async def test_connect_error_retried_then_none(self, server: FakeServer) -> None:
        def chat(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        server.chat_handler = chat
        with llm_call_log() as stats:
            assert await call(make_wrapper(server)) is None
        assert stats[0].attempts == 3 and stats[0].success is False

    async def test_client_error_not_retried(self, server: FakeServer) -> None:
        server.chat_handler = lambda request: httpx.Response(404, json={"error": "model not found"})
        with llm_call_log() as stats:
            assert await call(make_wrapper(server)) is None
        assert stats[0].attempts == 1

    async def test_quota_429_not_retried_but_transient_429_is(self, server: FakeServer) -> None:
        server.chat_handler = lambda request: httpx.Response(429, text="quota exceeded")
        with llm_call_log() as stats:
            assert await call(make_wrapper(server)) is None
        assert stats[0].attempts == 1

        server.chat_handler = lambda request: httpx.Response(429, text="slow down")
        with llm_call_log() as stats:
            assert await call(make_wrapper(server)) is None
        assert stats[0].attempts == 3

    async def test_repairable_json_parsed_without_new_request(self, server: FakeServer) -> None:
        server.chat_contents = ['{"action": "PROCEED", "rationale": "ok",}']
        assert await call(make_wrapper(server)) is not None
        assert len(server.bodies("/api/chat")) == 1

    async def test_invalid_output_retried_then_none(self, server: FakeServer) -> None:
        server.chat_contents = ["this is not json at all"]
        with llm_call_log() as stats:
            assert await call(make_wrapper(server)) is None
        assert len(server.bodies("/api/chat")) == 3 and stats[0].attempts == 3

    async def test_schema_violation_returns_none(self, server: FakeServer) -> None:
        server.chat_contents = [json.dumps({"wrong_field": "value"})]
        assert await call(make_wrapper(server)) is None

    async def test_malformed_response_body_retried(self, server: FakeServer) -> None:
        responses = [
            httpx.Response(200, json={"unexpected": True}),
            httpx.Response(200, json={"message": {"content": VALID_JSON}}),
        ]
        server.chat_handler = lambda request: responses.pop(0)
        assert await call(make_wrapper(server)) is not None


class TestJsonExtraction:
    @pytest.mark.parametrize("raw", [
        '```json\n{"action": "PROCEED", "rationale": "ok"}\n```',
        '```\n{"action": "PROCEED", "rationale": "ok"}\n```',
        '{"action": "PROCEED", "rationale": "ok"}',
        'Here is the result:\n```json\n{"action": "PROCEED", "rationale": "ok"}\n```\nDone.',
        '<think>reasoning {x}</think>\n{"action": "PROCEED", "rationale": "ok"}',
        'Sure! {"action": "PROCEED", "rationale": "ok"} Hope that helps.',
    ])
    def test_extract(self, raw: str) -> None:
        assert json.loads(ResilientLLMWrapper._extract_json(raw))["action"] == "PROCEED"

    @pytest.mark.parametrize(("raw", "expected"), [
        ('{"response_text": "Fine [1]."}\u201d}', '{"response_text": "Fine [1]."}'),
        ('{"action": "PROCEED", "rationale": "a } inside"} trailing }',
         '{"action": "PROCEED", "rationale": "a } inside"}'),
        ('note {not json} then {"action": "PROCEED", "rationale": "x"}',
         '{"action": "PROCEED", "rationale": "x"}'),
    ])
    def test_first_complete_object_extracted_exactly(self, raw: str, expected: str) -> None:
        """Regression (llm-prompts-03): the end is never taken from a later or quoted '}'."""
        assert ResilientLLMWrapper._extract_json(raw) == expected


class TestJsonRepair:
    @pytest.mark.parametrize(("malformed", "key", "expected"), [
        ('{"action": "PROCEED", "rationale": "ok",}', "action", "PROCEED"),
        ('{"items": ["a", "b",]}', "items", ["a", "b"]),
        ('{"action": "PROCEED", "rationale": "ok"', "action", "PROCEED"),
        ('{"items": ["a", "b"', "items", ["a", "b"]),
        ("{'action': 'PROCEED', 'rationale': 'ok'}", "action", "PROCEED"),
        ('{"action": "PROCEED", "rationale": "ok",', "action", "PROCEED"),
        ('{"a": [{"b": 1', "a", [{"b": 1}]),
        ('{"response_text": "Hello wor', "response_text", "Hello wor"),
    ])
    def test_repair(self, malformed: str, key: str, expected: Any) -> None:
        assert json.loads(ResilientLLMWrapper._repair_json(malformed))[key] == expected

    def test_apostrophes_preserved(self) -> None:
        repaired = ResilientLLMWrapper._repair_json("""{'text': "patient's reading isn't normal"}""")
        assert json.loads(repaired)["text"] == "patient's reading isn't normal"


# ---------------------------------------------------------------------------
# Telemetry and request cache
# ---------------------------------------------------------------------------


class TestCallLog:
    async def test_stats_collected(self, server: FakeServer) -> None:
        with llm_call_log() as stats:
            await call(make_wrapper(server), model="m1")
        assert len(stats) == 1
        stat = stats[0]
        assert isinstance(stat, LLMCallStat)
        assert (stat.call_name, stat.model, stat.provider) == ("test", "m1", "ollama")
        assert (stat.prompt_tokens, stat.completion_tokens) == (120, 30)
        assert stat.total_duration_ms == 2500.0 and stat.load_duration_ms == 1000.0
        assert stat.success and not stat.cached and stat.attempts == 1
        assert stat.latency_ms >= 0.0

    async def test_logs_are_scoped(self, server: FakeServer) -> None:
        wrapper = make_wrapper(server)
        await call(wrapper)  # outside any log: not recorded anywhere
        with llm_call_log() as outer:
            await call(wrapper)
            with llm_call_log() as inner:
                await call(wrapper)
            await call(wrapper)
        assert len(outer) == 2 and len(inner) == 1


class DictCache:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def set(self, key: str, value: str) -> None:
        self.store[key] = value


class TestRequestCache:
    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        yield
        set_request_cache(None)

    async def test_miss_then_hit(self, server: FakeServer) -> None:
        cache = DictCache()
        set_request_cache(cache)
        wrapper = make_wrapper(server)
        with llm_call_log() as stats:
            first = await call(wrapper)
            second = await call(wrapper)
        assert first == second
        assert len(server.bodies("/api/chat")) == 1
        assert [s.cached for s in stats] == [False, True]
        assert list(cache.store.values()) == [VALID_JSON]

    async def test_failed_responses_not_stored(self, server: FakeServer) -> None:
        cache = DictCache()
        set_request_cache(cache)
        server.chat_contents = ["not json"]
        assert await call(make_wrapper(server)) is None
        assert cache.store == {}

    async def test_key_ignores_keep_alive_but_not_options(self, server: FakeServer) -> None:
        cache = DictCache()
        set_request_cache(cache)
        await call(make_wrapper(server, LLM_KEEP_ALIVE="5m"))
        await call(make_wrapper(server, LLM_KEEP_ALIVE="1h"))
        assert len(server.bodies("/api/chat")) == 1
        await call(make_wrapper(server), temperature=0.3)
        assert len(server.bodies("/api/chat")) == 2

    def test_key_is_sha256_of_canonical_payload(self) -> None:
        a = request_cache_key({"model": "m", "keep_alive": "5m", "options": {"seed": 1, "top_p": 1}})
        b = request_cache_key({"options": {"top_p": 1, "seed": 1}, "model": "m"})
        assert a == b and len(a) == 64


# ---------------------------------------------------------------------------
# Warm-up, health, lifecycle
# ---------------------------------------------------------------------------


class TestWarmupHealth:
    async def test_warmup_loads_unique_models(self, server: FakeServer) -> None:
        wrapper = make_wrapper(server, LLM_MODEL="a", LLM_MODEL_GATES="", LLM_MODEL_GENERATOR="b")
        assert await wrapper.warmup() == {"a": True, "b": True}
        bodies = server.bodies("/api/generate")
        assert [b["model"] for b in bodies] == ["a", "b"]
        assert all(b["prompt"] == "" and b["keep_alive"] == "30m" for b in bodies)

    async def test_warmup_failure_reported(self, server: FakeServer) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/generate":
                return httpx.Response(404)
            return server.handler(request)

        wrapper = ResilientLLMWrapper(make_settings(), transport=httpx.MockTransport(handler))
        assert await wrapper.warmup(["missing"]) == {"missing": False}

    async def test_health_reachable(self, server: FakeServer) -> None:
        server.tags = ["qwen3:8b", "gemma3:latest"]
        server.loaded = ["qwen3:8b"]
        wrapper = make_wrapper(server, LLM_MODEL="qwen3:8b", LLM_MODEL_GENERATOR="gemma3")
        health = await wrapper.health()
        assert health["provider"] == "ollama" and health["reachable"] is True
        assert health["models"] == {"qwen3:8b": True, "gemma3": True}
        assert health["loaded"] == ["qwen3:8b"]
        assert health["local_only"] is True

    async def test_health_unreachable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        wrapper = ResilientLLMWrapper(make_settings(), transport=httpx.MockTransport(handler))
        health = await wrapper.health()
        assert health["reachable"] is False and health["models"] == {"test-model": False}

    async def test_close_and_reuse(self, server: FakeServer) -> None:
        wrapper = make_wrapper(server)
        await call(wrapper)
        await wrapper.close()
        assert await call(wrapper) is not None

    def test_singleton(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_wrapper, "_llm_wrapper_instance", None)
        assert get_llm_wrapper() is get_llm_wrapper()


# ---------------------------------------------------------------------------
# Regression (llm-prompts-03): truncated output is a failure, never repaired or cached
# ---------------------------------------------------------------------------

TRUNCATED_ANSWER = (
    '{"response_text": "Yes, it is generally fine to stop taking your amlodipine [1]. '
    "However, never stop it"
)
FULL_ANSWER = json.dumps({"response_text": "Do not stop amlodipine suddenly [1]."})


def _ollama_reply(content: str, done_reason: str) -> httpx.Response:
    return httpx.Response(200, json={
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": 10, "eval_count": 900, "done_reason": done_reason,
    })


class TestTruncation:
    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        yield
        set_request_cache(None)

    async def test_length_stop_retried_once_with_double_budget(self, server: FakeServer) -> None:
        replies = [_ollama_reply(TRUNCATED_ANSWER, "length"), _ollama_reply(FULL_ANSWER, "stop")]
        server.chat_handler = lambda request: replies.pop(0)
        cache = DictCache()
        set_request_cache(cache)
        with llm_call_log() as stats:
            result = await call(make_wrapper(server), ResponseGeneratorOutput, max_tokens=900)
        assert result is not None and result.response_text == "Do not stop amlodipine suddenly [1]."
        budgets = [b["options"]["num_predict"] for b in server.bodies("/api/chat")]
        assert budgets == [900, 1800]
        assert stats[0].truncated is True and stats[0].success is True
        # Only the complete answer is cached (under the original request).
        assert list(cache.store.values()) == [FULL_ANSWER]

    async def test_truncated_twice_fails_without_repair_or_cache(self, server: FakeServer) -> None:
        server.chat_handler = lambda request: _ollama_reply(TRUNCATED_ANSWER, "length")
        cache = DictCache()
        set_request_cache(cache)
        with llm_call_log() as stats:
            result = await call(make_wrapper(server), ResponseGeneratorOutput, max_tokens=900)
        assert result is None  # the orchestrator then uses its pre-vetted fallback
        assert len(server.bodies("/api/chat")) == 2  # no third identical (deterministic) request
        assert stats[0].truncated is True and stats[0].success is False
        assert cache.store == {}

    async def test_truncated_gate_output_is_not_accepted(self, server: FakeServer) -> None:
        cut = '{"rationale": "Known.", "action": "PROCEED", "missing_fields": ["age"'
        server.chat_handler = lambda request: _ollama_reply(cut, "length")
        assert await call(make_wrapper(server)) is None

    async def test_remote_finish_reason_length_is_truncation(self, server: FakeServer) -> None:
        wrapper = make_wrapper(server, LLM_PROVIDER="groq", GROQ_API_KEY="k", LOCAL_ONLY_MODE=False)
        replies = [
            httpx.Response(200, json={"choices": [{"message": {"content": TRUNCATED_ANSWER},
                                                   "finish_reason": "length"}]}),
            httpx.Response(200, json={"choices": [{"message": {"content": FULL_ANSWER},
                                                   "finish_reason": "stop"}]}),
        ]
        server.chat_handler = lambda request: replies.pop(0)
        result = await call(wrapper, ResponseGeneratorOutput, max_tokens=300)
        assert result is not None
        assert [json.loads(r.content)["max_tokens"] for r in server.requests] == [300, 600]

    async def test_incomplete_json_without_done_reason_is_not_repaired(
        self, server: FakeServer
    ) -> None:
        server.chat_contents = [TRUNCATED_ANSWER]
        with llm_call_log() as stats:
            assert await call(make_wrapper(server), ResponseGeneratorOutput) is None
        assert stats[0].attempts == 3

    async def test_truncated_entry_in_an_old_cache_is_ignored(self, server: FakeServer) -> None:
        cache = DictCache()
        set_request_cache(cache)
        server.chat_contents = [FULL_ANSWER]
        wrapper = make_wrapper(server)
        await call(wrapper, ResponseGeneratorOutput)
        key = next(iter(cache.store))
        cache.store[key] = TRUNCATED_ANSWER  # written by an older version of the wrapper
        with llm_call_log() as stats:
            result = await call(wrapper, ResponseGeneratorOutput)
        assert result is not None and result.response_text.endswith("suddenly [1].")
        assert stats[0].cached is False and cache.store[key] == FULL_ANSWER

    def test_incomplete_detection(self) -> None:
        assert ResilientLLMWrapper._is_incomplete('{"a": "unterminated')
        assert ResilientLLMWrapper._is_incomplete('{"a": [1, 2')
        assert not ResilientLLMWrapper._is_incomplete('{"a": "x}"}')
        assert not ResilientLLMWrapper._is_incomplete('{"a": "ok",}')
        assert not ResilientLLMWrapper._is_incomplete("{'a': 'single quotes'}")


# ---------------------------------------------------------------------------
# Regression (llm-prompts-08): LOCAL_ONLY_MODE fails closed on a non-loopback Ollama
# ---------------------------------------------------------------------------


class TestLocalOnlyEnforcement:
    @pytest.mark.parametrize("url", [
        "http://192.168.1.20:11434", "https://ollama.example.com/v1", "http://10.0.0.5:11434",
        "http://host.docker.internal:11434", "http://localhost.evil.com:11434",
    ])
    async def test_non_loopback_host_refused_and_never_contacted(self, url: str) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"message": {"content": VALID_JSON}})

        wrapper = ResilientLLMWrapper(
            make_settings(OLLAMA_BASE_URL=url, LOCAL_ONLY_MODE=True),
            transport=httpx.MockTransport(handler),
        )
        assert wrapper.blocked_reason
        with llm_call_log() as stats:
            assert await call(wrapper) is None
        assert stats[0].success is False
        assert await wrapper.warmup() == dict.fromkeys(
            dict.fromkeys([wrapper._settings.gates_model, wrapper._settings.generator_model]),
            False,
        )
        health = await wrapper.health()
        assert health["reachable"] is False and health["blocked"]
        assert requests == []

    @pytest.mark.parametrize("url", [
        "http://127.0.0.1:11434", "http://localhost:11434/v1", "http://[::1]:11434",
        "http://127.0.1.1:11434", "http://LOCALHOST.:11434",
    ])
    async def test_loopback_hosts_allowed(self, server: FakeServer, url: str) -> None:
        wrapper = make_wrapper(server, OLLAMA_BASE_URL=url)
        assert wrapper.blocked_reason == ""
        assert await call(wrapper) is not None

    async def test_non_loopback_allowed_when_local_only_disabled(self, server: FakeServer) -> None:
        wrapper = make_wrapper(server, OLLAMA_BASE_URL="http://192.168.1.20:11434",
                               LOCAL_ONLY_MODE=False)
        assert wrapper.blocked_reason == ""
        assert await call(wrapper) is not None
        assert server.requests[-1].url.host == "192.168.1.20"

    @pytest.mark.parametrize("model", ["gpt-oss:120b-cloud", "qwen3-coder:480b-cloud", "x:cloud"])
    async def test_cloud_models_refused_in_local_only(self, server: FakeServer, model: str) -> None:
        wrapper = make_wrapper(server)
        assert await call(wrapper, model=model) is None
        assert server.bodies("/api/chat") == [] and server.bodies("/api/show") == []
        assert await wrapper.warmup([model]) == {model: False}
        assert server.bodies("/api/generate") == []

    def test_loopback_helper(self) -> None:
        assert llm_wrapper.is_loopback_host("127.0.0.1")
        assert llm_wrapper.is_loopback_host("[::1]")
        assert not llm_wrapper.is_loopback_host("")
        assert not llm_wrapper.is_loopback_host("0.0.0.0")
        assert not llm_wrapper.is_loopback_host("host.docker.internal")
