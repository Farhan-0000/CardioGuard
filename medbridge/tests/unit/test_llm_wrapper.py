"""
Unit tests for TASK-17: Resilient LLM Wrapper.

All tests mock HTTP calls — no real API requests are made.
Tests cover: valid parsing, JSON extraction from code blocks, JSON repair,
retry on timeout/5xx, graceful None on permanent failure, and provider
failover.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel

from medbridge.ai.llm_wrapper import ResilientLLMWrapper


# ---------------------------------------------------------------------------
# Test schema
# ---------------------------------------------------------------------------

class SampleOutput(BaseModel):
    """Minimal schema for testing."""
    action: str
    rationale: str


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wrapper():
    """Create a wrapper with fast retries for testing."""
    with patch("medbridge.ai.llm_wrapper.get_settings") as mock_settings:
        settings = MagicMock()
        settings.LLM_MODEL = "test-model"
        settings.LLM_MAX_RETRIES = 3
        settings.LLM_RETRY_BASE_DELAY = 0.01  # Fast retries for tests
        settings.LLM_SEED = 42
        settings.GROQ_API_KEY = "test-groq-key"
        settings.OPENAI_API_KEY = ""  # No fallback
        mock_settings.return_value = settings
        return ResilientLLMWrapper()


@pytest.fixture
def wrapper_with_failover():
    """Create a wrapper with both Groq and OpenAI providers."""
    with patch("medbridge.ai.llm_wrapper.get_settings") as mock_settings:
        settings = MagicMock()
        settings.LLM_MODEL = "test-model"
        settings.LLM_MAX_RETRIES = 2
        settings.LLM_RETRY_BASE_DELAY = 0.01
        settings.LLM_SEED = 42
        settings.GROQ_API_KEY = "test-groq-key"
        settings.OPENAI_API_KEY = "test-openai-key"
        mock_settings.return_value = settings
        return ResilientLLMWrapper()


VALID_JSON = json.dumps({"action": "PROCEED", "rationale": "Context sufficient"})


def _make_httpx_response(content: str, status_code: int = 200) -> httpx.Response:
    """Build a mock httpx.Response with chat-completions JSON body."""
    body = json.dumps({
        "choices": [{"message": {"content": content}}]
    })
    return httpx.Response(
        status_code=status_code,
        json=json.loads(body),
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )


# ---------------------------------------------------------------------------
# Valid JSON response parsing
# ---------------------------------------------------------------------------

class TestValidParsing:
    """Mocked valid JSON response → validated Pydantic model."""

    @pytest.mark.asyncio
    async def test_parses_clean_json(self, wrapper):
        mock_resp = _make_httpx_response(VALID_JSON)

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await wrapper.call(
                call_name="test",
                system_prompt="You are a test.",
                user_prompt="Test input",
                output_schema=SampleOutput,
            )

        assert result is not None
        assert isinstance(result, SampleOutput)
        assert result.action == "PROCEED"
        assert result.rationale == "Context sufficient"


# ---------------------------------------------------------------------------
# JSON extraction from code blocks
# ---------------------------------------------------------------------------

class TestJsonExtraction:
    """Extract JSON from markdown code blocks."""

    def test_extract_from_json_code_block(self, wrapper):
        raw = '```json\n{"action": "PROCEED", "rationale": "ok"}\n```'
        result = wrapper._extract_json(raw)
        parsed = json.loads(result)
        assert parsed["action"] == "PROCEED"

    def test_extract_from_plain_code_block(self, wrapper):
        raw = '```\n{"action": "ANSWER", "rationale": "found"}\n```'
        result = wrapper._extract_json(raw)
        parsed = json.loads(result)
        assert parsed["action"] == "ANSWER"

    def test_extract_raw_json(self, wrapper):
        raw = '{"action": "ABSTAIN", "rationale": "no evidence"}'
        result = wrapper._extract_json(raw)
        parsed = json.loads(result)
        assert parsed["action"] == "ABSTAIN"

    def test_extract_with_surrounding_text(self, wrapper):
        raw = 'Here is the result:\n```json\n{"action": "PROCEED", "rationale": "ok"}\n```\nDone.'
        result = wrapper._extract_json(raw)
        parsed = json.loads(result)
        assert parsed["action"] == "PROCEED"

    @pytest.mark.asyncio
    async def test_code_block_response_parsed(self, wrapper):
        """Full end-to-end: code-block wrapped JSON → Pydantic model."""
        content = '```json\n{"action": "PROCEED", "rationale": "ok"}\n```'
        mock_resp = _make_httpx_response(content)

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await wrapper.call(
                call_name="test",
                system_prompt="sys",
                user_prompt="usr",
                output_schema=SampleOutput,
            )

        assert result is not None
        assert result.action == "PROCEED"


# ---------------------------------------------------------------------------
# JSON repair on malformed strings
# ---------------------------------------------------------------------------

class TestJsonRepair:
    """Common LLM JSON mistakes are repaired."""

    def test_trailing_comma_before_brace(self, wrapper):
        malformed = '{"action": "PROCEED", "rationale": "ok",}'
        repaired = wrapper._repair_json(malformed)
        parsed = json.loads(repaired)
        assert parsed["action"] == "PROCEED"

    def test_trailing_comma_before_bracket(self, wrapper):
        malformed = '{"items": ["a", "b",]}'
        repaired = wrapper._repair_json(malformed)
        parsed = json.loads(repaired)
        assert parsed["items"] == ["a", "b"]

    def test_unclosed_brace(self, wrapper):
        malformed = '{"action": "PROCEED", "rationale": "ok"'
        repaired = wrapper._repair_json(malformed)
        parsed = json.loads(repaired)
        assert parsed["action"] == "PROCEED"

    def test_unclosed_bracket(self, wrapper):
        malformed = '{"items": ["a", "b"]}'
        # Missing outer brace close — but this is already valid
        # Test a truly unclosed bracket:
        malformed2 = '{"items": ["a", "b"'
        repaired = wrapper._repair_json(malformed2)
        # After repair: adds ] then } → valid
        parsed = json.loads(repaired)
        assert parsed["items"] == ["a", "b"]

    def test_single_quotes_to_double(self, wrapper):
        malformed = "{'action': 'PROCEED', 'rationale': 'ok'}"
        repaired = wrapper._repair_json(malformed)
        parsed = json.loads(repaired)
        assert parsed["action"] == "PROCEED"

    def test_apostrophes_preserved_in_values(self, wrapper):
        """Mid-word apostrophes (e.g. patient's) must NOT be replaced."""
        malformed = """{'rationale': "The patient's BP is high"}"""
        repaired = wrapper._repair_json(malformed)
        parsed = json.loads(repaired)
        assert "patient's" in parsed["rationale"]

    def test_mixed_quotes_with_apostrophe(self, wrapper):
        """Single-quoted keys with double-quoted values containing apostrophes."""
        malformed = """{'text': "patient's reading isn't normal", 'action': 'PROCEED'}"""
        repaired = wrapper._repair_json(malformed)
        parsed = json.loads(repaired)
        assert "patient's" in parsed["text"]
        assert parsed["action"] == "PROCEED"

    def test_combined_errors(self, wrapper):
        """Trailing comma + unclosed brace."""
        malformed = '{"action": "PROCEED", "rationale": "ok",'
        repaired = wrapper._repair_json(malformed)
        parsed = json.loads(repaired)
        assert parsed["action"] == "PROCEED"


# ---------------------------------------------------------------------------
# Retry on timeout / 5xx errors
# ---------------------------------------------------------------------------

class TestRetryBehavior:
    """Retry with exponential backoff on transient errors."""

    @pytest.mark.asyncio
    async def test_retry_on_timeout_then_succeed(self, wrapper):
        """First call times out, second succeeds."""
        mock_resp = _make_httpx_response(VALID_JSON)

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.TimeoutException("Connection timed out")
            return mock_resp

        with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
            result = await wrapper.call(
                call_name="test",
                system_prompt="sys",
                user_prompt="usr",
                output_schema=SampleOutput,
            )

        assert result is not None
        assert result.action == "PROCEED"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_500_then_succeed(self, wrapper):
        """First call returns 500, second succeeds."""
        error_resp = _make_httpx_response("Server Error", status_code=500)
        ok_resp = _make_httpx_response(VALID_JSON)

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.HTTPStatusError(
                    "Server Error",
                    request=httpx.Request("POST", "https://test"),
                    response=error_resp,
                )
            return ok_resp

        with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
            result = await wrapper.call(
                call_name="test",
                system_prompt="sys",
                user_prompt="usr",
                output_schema=SampleOutput,
            )

        assert result is not None
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_malformed_json_with_repair(self, wrapper):
        """First call returns bad JSON, repair kicks in on attempt 2."""
        # First response: malformed JSON (trailing comma)
        bad_json = '{"action": "PROCEED", "rationale": "ok",}'
        bad_resp = _make_httpx_response(bad_json)
        # After repair on attempt 2, same malformed JSON should parse
        good_resp = _make_httpx_response(bad_json)

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return bad_resp  # Will fail parse (no repair on attempt 1)
            return good_resp  # Will succeed (repair on attempt 2)

        with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
            result = await wrapper.call(
                call_name="test",
                system_prompt="sys",
                user_prompt="usr",
                output_schema=SampleOutput,
            )

        assert result is not None
        assert result.action == "PROCEED"
        assert call_count == 2


# ---------------------------------------------------------------------------
# Permanent failure → None
# ---------------------------------------------------------------------------

class TestPermanentFailure:
    """Return None after all retries exhausted."""

    @pytest.mark.asyncio
    async def test_returns_none_after_all_retries(self, wrapper):
        """3 consecutive timeouts → None."""
        async def mock_post(*args, **kwargs):
            raise httpx.TimeoutException("Timed out")

        with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
            result = await wrapper.call(
                call_name="test",
                system_prompt="sys",
                user_prompt="usr",
                output_schema=SampleOutput,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_permanent_parse_failure(self, wrapper):
        """3 consecutive unparseable responses → None."""
        bad_resp = _make_httpx_response("this is not json at all")

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=bad_resp):
            result = await wrapper.call(
                call_name="test",
                system_prompt="sys",
                user_prompt="usr",
                output_schema=SampleOutput,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_validation_failure(self, wrapper):
        """Valid JSON but wrong schema → None after retries."""
        wrong_schema_json = json.dumps({"wrong_field": "value"})
        bad_resp = _make_httpx_response(wrong_schema_json)

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=bad_resp):
            result = await wrapper.call(
                call_name="test",
                system_prompt="sys",
                user_prompt="usr",
                output_schema=SampleOutput,
            )

        assert result is None


# ---------------------------------------------------------------------------
# Provider failover
# ---------------------------------------------------------------------------

class TestProviderFailover:
    """Fall back to OpenAI when Groq exhausted."""

    @pytest.mark.asyncio
    async def test_failover_to_openai(self, wrapper_with_failover):
        """Groq times out → OpenAI succeeds."""
        ok_resp = _make_httpx_response(VALID_JSON)

        call_count = 0

        async def mock_post(url, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if "groq" in str(url):
                raise httpx.TimeoutException("Groq timed out")
            return ok_resp

        with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
            result = await wrapper_with_failover.call(
                call_name="test",
                system_prompt="sys",
                user_prompt="usr",
                output_schema=SampleOutput,
            )

        assert result is not None
        assert result.action == "PROCEED"
        # 2 retries on Groq + at least 1 on OpenAI
        assert call_count >= 3

    @pytest.mark.asyncio
    async def test_both_providers_fail(self, wrapper_with_failover):
        """Both Groq and OpenAI fail → None."""
        async def mock_post(*args, **kwargs):
            raise httpx.TimeoutException("All timed out")

        with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
            result = await wrapper_with_failover.call(
                call_name="test",
                system_prompt="sys",
                user_prompt="usr",
                output_schema=SampleOutput,
            )

        assert result is None
