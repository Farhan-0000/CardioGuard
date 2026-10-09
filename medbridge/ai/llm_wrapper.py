"""
Resilient LLM Wrapper (ADL-004, ADL-028).

Single gateway for every LLM call in MedBridge:

* **Ollama native API** (default, local): ``POST {OLLAMA_BASE_URL}/api/chat`` with
  structured outputs (``format`` = the output model's JSON schema, enforced by
  grammar-constrained decoding), ``options`` (temperature, seed, num_ctx,
  num_predict, top_p), ``keep_alive`` and ``think`` (sent only to models whose
  ``/api/show`` capabilities include ``thinking``).
* **Optional remote providers** (Groq / OpenAI, OpenAI-compatible
  ``/chat/completions`` with ``response_format=json_object``). With
  ``LOCAL_ONLY_MODE=true`` (default) they are never contacted, whatever
  ``LLM_PROVIDER`` says — no patient text leaves the machine.
* Exponential-backoff retry (``base · 2^(n-1)``), markdown/think-block JSON
  extraction and malformed-JSON repair, Pydantic validation → model or ``None``.
* Per-turn call statistics via :func:`llm_call_log` (context-variable scoped)
  and an optional content-addressed request cache (:func:`set_request_cache`)
  used by the evaluation harness to memoise identical requests.

Technical Specification Part III §3.9.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar
from urllib.parse import urlparse

import httpx
import structlog
from pydantic import BaseModel, ValidationError

from medbridge.config import Settings, get_settings

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

ProviderName = Literal["ollama", "groq", "openai"]

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_OPENAI_BASE_URL = "https://api.openai.com/v1"
_CONNECT_TIMEOUT_S = 5.0
_PROBE_TIMEOUT_S = 5.0
_TOP_P_DETERMINISTIC = 1.0
_TOP_P_SAMPLING = 0.95
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "host.docker.internal"})


# ---------------------------------------------------------------------------
# Call statistics & request cache
# ---------------------------------------------------------------------------


@dataclass
class LLMCallStat:
    """Telemetry for one ``ResilientLLMWrapper.call`` (no prompt or response text)."""

    call_name: str
    model: str
    provider: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    attempts: int = 0
    success: bool = False
    cached: bool = False
    # Server-side timings reported by Ollama (0 when unavailable / cached).
    total_duration_ms: float = 0.0
    load_duration_ms: float = 0.0


class RequestCache(Protocol):
    """Content-addressed store for raw LLM responses (evaluation memoisation)."""

    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str) -> None: ...


_call_log: ContextVar[list[LLMCallStat] | None] = ContextVar("medbridge_llm_call_log", default=None)
_request_cache: RequestCache | None = None


@contextmanager
def llm_call_log() -> Iterator[list[LLMCallStat]]:
    """Collect an :class:`LLMCallStat` for every LLM call made inside the ``with`` block.

    The list is bound to a context variable, so calls made from tasks spawned
    inside the block (e.g. ``asyncio.gather``) are collected too, while
    concurrent requests keep separate logs.
    """
    stats: list[LLMCallStat] = []
    token = _call_log.set(stats)
    try:
        yield stats
    finally:
        _call_log.reset(token)


def set_request_cache(cache: RequestCache | None) -> None:
    """Install (or remove with ``None``) the process-wide request cache."""
    global _request_cache
    _request_cache = cache


def request_cache_key(payload: dict[str, Any]) -> str:
    """sha256 of the canonical JSON request payload, excluding ``keep_alive``."""
    canonical = {k: v for k, v in payload.items() if k != "keep_alive"}
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def llm_format_schema(output_schema: type[BaseModel]) -> dict[str, Any]:
    """JSON schema sent as Ollama ``format`` (structured outputs).

    ``$ref``s are inlined, every remaining property is marked required (so the
    grammar emits all fields in declaration order), and properties flagged with
    ``json_schema_extra={"llm_omit": True}`` are removed (they are filled by the
    server, not the model).
    """
    schema = output_schema.model_json_schema()
    defs: dict[str, Any] = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, list):
            return [resolve(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            target = defs[str(node["$ref"]).rsplit("/", 1)[-1]]
            return resolve({**target, **{k: v for k, v in node.items() if k != "$ref"}})
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                kept = {
                    name: resolve(prop)
                    for name, prop in value.items()
                    if not (isinstance(prop, dict) and prop.get("llm_omit"))
                }
                out["properties"] = kept
                out["required"] = list(kept)
            elif key != "required":
                out[key] = resolve(value)
        return out

    resolved: dict[str, Any] = resolve(schema)
    return resolved


def _native_ollama_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    return url[: -len("/v1")] if url.endswith("/v1") else url


def _model_present(name: str, available: list[str]) -> bool:
    return name in available or (":" not in name and f"{name}:latest" in available)


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


@dataclass(frozen=True)
class _Provider:
    name: ProviderName
    base_url: str
    api_key: str = ""
    model: str | None = None  # fixed model for remote providers


class _RemoteBlockedError(RuntimeError):
    """Raised if a remote provider is reached while LOCAL_ONLY_MODE is on (defence in depth)."""


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class ResilientLLMWrapper:
    """Wraps all LLM calls with retry, JSON repair, telemetry and deterministic fallback.

    On permanent failure (all retries exhausted across all providers) ``call``
    returns ``None``; the orchestrator then applies the spec fallbacks (§3.9).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        s = self._settings
        self.max_retries: int = max(1, s.LLM_MAX_RETRIES)
        self.retry_base_delay: float = s.LLM_RETRY_BASE_DELAY
        self.seed: int = s.LLM_SEED
        self.timeout: float = s.LLM_TIMEOUT
        self.local_only: bool = s.LOCAL_ONLY_MODE
        self.ollama_base_url: str = _native_ollama_url(s.OLLAMA_BASE_URL)
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        self._thinking_support: dict[str, bool] = {}
        self._providers: list[_Provider] = self._build_providers()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @property
    def provider_mode(self) -> str:
        """Name of the primary provider actually used."""
        return self._providers[0].name

    @property
    def provider_names(self) -> list[str]:
        """Providers in failover order."""
        return [p.name for p in self._providers]

    def _build_providers(self) -> list[_Provider]:
        s = self._settings
        ollama = _Provider("ollama", self.ollama_base_url)
        if self.local_only:
            if s.LLM_PROVIDER != "ollama":
                logger.error(
                    "llm_remote_provider_blocked",
                    requested=s.LLM_PROVIDER,
                    reason="LOCAL_ONLY_MODE is enabled; using local Ollama only",
                )
            host = urlparse(self.ollama_base_url).hostname or ""
            if host not in _LOCAL_HOSTS:
                logger.warning("ollama_host_not_loopback", host=host)
            return [ollama]

        remotes: list[_Provider] = []
        if s.GROQ_API_KEY:
            remotes.append(_Provider("groq", _GROQ_BASE_URL, s.GROQ_API_KEY, s.GROQ_MODEL))
        if s.OPENAI_API_KEY:
            remotes.append(_Provider("openai", _OPENAI_BASE_URL, s.OPENAI_API_KEY, s.OPENAI_MODEL))
        if s.LLM_PROVIDER == "ollama":
            return [ollama, *remotes]
        primary = next((p for p in remotes if p.name == s.LLM_PROVIDER), None)
        if primary is None:
            logger.error("llm_provider_missing_key", requested=s.LLM_PROVIDER, using="ollama")
            return [ollama, *remotes]
        return [primary, *(p for p in remotes if p is not primary)]

    # ------------------------------------------------------------------
    # HTTP client lifecycle
    # ------------------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        """Shared persistent client, recreated if the event loop changed (scripts/tests)."""
        loop = asyncio.get_running_loop()
        if self._client is None or self._client.is_closed or self._client_loop is not loop:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=_CONNECT_TIMEOUT_S),
                transport=self._transport,
            )
            self._client_loop = loop
        return self._client

    async def close(self) -> None:
        """Close the persistent HTTP client (called during app shutdown)."""
        client, self._client = self._client, None
        if client is not None and not client.is_closed:
            if self._client_loop is asyncio.get_running_loop():
                await client.aclose()
        self._client_loop = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def call(
        self,
        call_name: str,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[T],
        temperature: float = 0.0,
        max_tokens: int = 512,
        *,
        model: str | None = None,
        seed: int | None = None,
    ) -> T | None:
        """Execute an LLM call with retry, JSON repair and (optional) provider failover.

        Args:
            call_name: Label for logs/telemetry (e.g. ``"context_gate"``).
            system_prompt: System message.
            user_prompt: User message (XML-partitioned by the caller).
            output_schema: Pydantic model the JSON answer must validate against.
            temperature: Sampling temperature (0.0 for extractor and gates).
            max_tokens: Generation cap (Ollama ``num_predict``).
            model: Ollama model tag; defaults to ``settings.LLM_MODEL``. Remote
                providers always use their configured model.
            seed: Sampling seed; defaults to ``settings.LLM_SEED`` (42).

        Returns:
            A validated model instance, or ``None`` on permanent failure.
        """
        started = time.perf_counter()
        stat = LLMCallStat(
            call_name=call_name,
            model=model or self._settings.LLM_MODEL,
            provider=self._providers[0].name,
        )
        result: T | None = None
        for provider in self._providers:
            result = await self._call_provider(
                provider, stat, call_name, system_prompt, user_prompt, output_schema,
                temperature, max_tokens, model, seed,
            )
            if result is not None:
                break
            logger.warning("llm_provider_exhausted", call=call_name, provider=provider.name)
        stat.latency_ms = round((time.perf_counter() - started) * 1000.0, 1)
        stat.success = result is not None
        self._record(stat)
        if result is None:
            logger.error("llm_permanent_failure", call=call_name, attempts=stat.attempts)
        return result

    async def warmup(self, models: list[str] | None = None) -> dict[str, bool]:
        """Load models into (GPU) memory with ``keep_alive`` so the first turn is fast.

        Returns ``{model: loaded_ok}``; empty when the primary provider is remote.
        """
        if self._providers[0].name != "ollama":
            return {}
        targets = _unique(models or [self._settings.gates_model, self._settings.generator_model])
        client = self._get_client()
        results: dict[str, bool] = {}
        for name in targets:
            try:
                response = await client.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": name,
                        "prompt": "",
                        "stream": False,
                        "keep_alive": self._settings.LLM_KEEP_ALIVE,
                    },
                )
                response.raise_for_status()
                results[name] = True
                await self._supports_thinking(name)
            except httpx.HTTPError as exc:
                logger.warning("llm_warmup_failed", model=name, error=type(exc).__name__)
                results[name] = False
        return results

    async def health(self) -> dict[str, Any]:
        """Provider readiness.

        Returns ``{"provider", "reachable", "models": {name: present}, "loaded", "local_only"}``.
        """
        primary = self._providers[0]
        if primary.name != "ollama":
            return {
                "provider": primary.name,
                "reachable": bool(primary.api_key),
                "models": {primary.model or "": True},
                "loaded": [],
                "local_only": self.local_only,
            }
        configured = _unique([self._settings.gates_model, self._settings.generator_model])
        client = self._get_client()
        try:
            tags = await client.get(f"{self.ollama_base_url}/api/tags", timeout=_PROBE_TIMEOUT_S)
            tags.raise_for_status()
            available = [str(m.get("name", "")) for m in tags.json().get("models", [])]
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            logger.warning("ollama_unreachable", error=type(exc).__name__)
            return {
                "provider": "ollama",
                "reachable": False,
                "models": dict.fromkeys(configured, False),
                "loaded": [],
                "local_only": self.local_only,
            }
        loaded: list[str] = []
        try:
            ps = await client.get(f"{self.ollama_base_url}/api/ps", timeout=_PROBE_TIMEOUT_S)
            ps.raise_for_status()
            loaded = [str(m.get("name", "")) for m in ps.json().get("models", [])]
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            logger.warning("ollama_ps_failed", error=type(exc).__name__)
        return {
            "provider": "ollama",
            "reachable": True,
            "models": {name: _model_present(name, available) for name in configured},
            "loaded": loaded,
            "local_only": self.local_only,
        }

    # ------------------------------------------------------------------
    # Provider-level retry loop
    # ------------------------------------------------------------------

    async def _call_provider(
        self,
        provider: _Provider,
        stat: LLMCallStat,
        call_name: str,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[T],
        temperature: float,
        max_tokens: int,
        model: str | None,
        seed: int | None,
    ) -> T | None:
        """Up to ``max_retries`` attempts against one provider (cache first)."""
        stat.provider = provider.name
        payload = await self._build_payload(
            provider, system_prompt, user_prompt, output_schema, temperature, max_tokens,
            model, seed,
        )
        stat.model = str(payload["model"])
        cache = _request_cache
        cache_key = request_cache_key(payload) if cache is not None else ""
        if cache is not None:
            cached_raw = cache.get(cache_key)
            if cached_raw is not None:
                parsed = self._parse(cached_raw, output_schema, call_name)
                if parsed is not None:
                    stat.cached = True
                    return parsed

        for attempt in range(1, self.max_retries + 1):
            stat.attempts += 1
            delay = self.retry_base_delay * (2 ** (attempt - 1))
            try:
                raw = await self._send(provider, payload, stat)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                logger.error("llm_http_error", call=call_name, provider=provider.name,
                             attempt=attempt, status=status)
                if not self._is_retryable(exc):
                    break
                retry_after = exc.response.headers.get("Retry-After")
                if retry_after and retry_after.replace(".", "", 1).isdigit():
                    delay = float(retry_after)
            except httpx.TimeoutException:
                logger.error("llm_timeout", call=call_name, provider=provider.name, attempt=attempt)
            except httpx.TransportError as exc:
                logger.error("llm_transport_error", call=call_name, provider=provider.name,
                             attempt=attempt, error=type(exc).__name__)
            except _RemoteBlockedError:
                logger.error("llm_remote_provider_blocked", call=call_name, provider=provider.name)
                return None
            except (KeyError, TypeError, ValueError) as exc:
                logger.error("llm_malformed_response", call=call_name, provider=provider.name,
                             attempt=attempt, error=type(exc).__name__)
            else:
                parsed = self._parse(raw, output_schema, call_name)
                if parsed is not None:
                    if cache is not None:
                        cache.set(cache_key, raw)
                    return parsed
            if attempt < self.max_retries:
                await asyncio.sleep(delay)
        return None

    @staticmethod
    def _is_retryable(exc: httpx.HTTPStatusError) -> bool:
        """5xx and transient 429 are retried; quota exhaustion and other 4xx are not."""
        status = exc.response.status_code
        if status >= 500:
            return True
        if status == 429:
            text = exc.response.text.lower()
            return not any(word in text for word in ("quota", "credit", "billing"))
        return False

    # ------------------------------------------------------------------
    # Request building & sending
    # ------------------------------------------------------------------

    async def _build_payload(
        self,
        provider: _Provider,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        temperature: float,
        max_tokens: int,
        model: str | None,
        seed: int | None,
    ) -> dict[str, Any]:
        s = self._settings
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        top_p = _TOP_P_DETERMINISTIC if temperature == 0.0 else _TOP_P_SAMPLING
        effective_seed = self.seed if seed is None else seed
        if provider.name != "ollama":
            return {
                "model": provider.model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "seed": effective_seed,
                "response_format": {"type": "json_object"},
            }
        model_name = model or s.LLM_MODEL
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "format": llm_format_schema(output_schema),
            "options": {
                "temperature": temperature,
                "seed": effective_seed,
                "num_ctx": s.LLM_NUM_CTX,
                "num_predict": max_tokens,
                "top_p": top_p,
            },
            "keep_alive": s.LLM_KEEP_ALIVE,
        }
        if await self._supports_thinking(model_name):
            payload["think"] = s.LLM_THINK
        return payload

    async def _supports_thinking(self, model: str) -> bool:
        """Whether ``model`` advertises the ``thinking`` capability (cached per model)."""
        if model in self._thinking_support:
            return self._thinking_support[model]
        try:
            response = await self._get_client().post(
                f"{self.ollama_base_url}/api/show", json={"model": model}, timeout=_PROBE_TIMEOUT_S
            )
            response.raise_for_status()
            capabilities = response.json().get("capabilities") or []
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            logger.warning("ollama_show_failed", model=model, error=type(exc).__name__)
            return False
        supported = "thinking" in capabilities
        self._thinking_support[model] = supported
        return supported

    async def _send(self, provider: _Provider, payload: dict[str, Any], stat: LLMCallStat) -> str:
        """POST the request and return the raw message content."""
        client = self._get_client()
        if provider.name == "ollama":
            response = await client.post(f"{provider.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            stat.prompt_tokens = int(data.get("prompt_eval_count") or 0)
            stat.completion_tokens = int(data.get("eval_count") or 0)
            stat.total_duration_ms = round(int(data.get("total_duration") or 0) / 1e6, 1)
            stat.load_duration_ms = round(int(data.get("load_duration") or 0) / 1e6, 1)
            if data.get("done_reason") == "length":
                logger.warning("llm_output_truncated", model=payload["model"])
            return str(data["message"]["content"])

        if self.local_only:
            raise _RemoteBlockedError(provider.name)
        response = await client.post(
            f"{provider.base_url}/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}
        stat.prompt_tokens = int(usage.get("prompt_tokens") or 0)
        stat.completion_tokens = int(usage.get("completion_tokens") or 0)
        return str(data["choices"][0]["message"]["content"])

    def _record(self, stat: LLMCallStat) -> None:
        log = _call_log.get()
        if log is not None:
            log.append(stat)
        logger.info(
            "llm_call",
            call=stat.call_name,
            model=stat.model,
            provider=stat.provider,
            latency_ms=stat.latency_ms,
            prompt_tokens=stat.prompt_tokens,
            completion_tokens=stat.completion_tokens,
            attempts=stat.attempts,
            success=stat.success,
            cached=stat.cached,
        )

    # ------------------------------------------------------------------
    # JSON extraction, repair & validation
    # ------------------------------------------------------------------

    def _parse(self, raw: str, output_schema: type[T], call_name: str) -> T | None:
        """Validate the raw content; on failure retry once with the repaired JSON."""
        candidate = self._extract_json(raw)
        try:
            return output_schema.model_validate_json(candidate)
        except ValidationError:
            pass
        try:
            return output_schema.model_validate_json(self._repair_json(candidate))
        except ValidationError as exc:
            logger.warning("llm_parse_failure", call=call_name, errors=exc.error_count())
            return None

    @staticmethod
    def _extract_json(raw: str) -> str:
        """Extract the JSON object from a model reply.

        Handles ``<think>…</think>`` traces, markdown code fences (```json … ```)
        and surrounding prose (keeps the outermost ``{…}`` span).
        """
        text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
        fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        start = text.find("{")
        if start == -1:
            return text
        end = text.rfind("}")
        return text[start:end + 1] if end > start else text[start:]

    @staticmethod
    def _repair_json(json_str: str) -> str:
        """Repair common malformed-JSON patterns.

        Fixes single-quoted keys/values (apostrophes inside words are kept),
        trailing commas, an unterminated final string and unclosed
        brackets/braces (closed in the correct nesting order).
        """
        repaired = re.sub(r"(?<!\w)'|'(?!\w)", '"', json_str)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        repaired = re.sub(r",\s*$", "", repaired)

        stack: list[str] = []
        in_string = False
        escaped = False
        for char in repaired:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in "{[":
                stack.append(char)
            elif char in "}]" and stack:
                stack.pop()
        suffix = '"' if in_string else ""
        suffix += "".join("}" if opener == "{" else "]" for opener in reversed(stack))
        return repaired + suffix


# ---------------------------------------------------------------------------
# Module-level singleton — shared across all AI components
# ---------------------------------------------------------------------------

_llm_wrapper_instance: ResilientLLMWrapper | None = None


def get_llm_wrapper() -> ResilientLLMWrapper:
    """Return the shared ResilientLLMWrapper singleton (one connection pool per process)."""
    global _llm_wrapper_instance
    if _llm_wrapper_instance is None:
        _llm_wrapper_instance = ResilientLLMWrapper()
    return _llm_wrapper_instance
