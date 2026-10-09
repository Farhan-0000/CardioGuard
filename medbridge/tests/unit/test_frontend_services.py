"""
Unit Tests for Frontend Services (TASK-25, Module M-20).

Tests:
1. APIClient communication with backend (create_session, send_message, get_history, health_check).
2. APIClient network timeout and connection error handling.
3. APIClient HTTP error parsing and safe fallback mapping.
4. SessionManager state management and query parameter synchronization.
"""
import uuid
from datetime import UTC, datetime, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest
import streamlit as st

from medbridge.api.schemas.enums import ActionEnum
from medbridge.api.schemas.responses import (
    HealthResponse,
    HistoryResponse,
    MessageResponse,
)
from medbridge.frontend.services.api_client import (
    DEFAULT_SAFE_FALLBACK,
    APIClient,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)
from medbridge.frontend.services.session import SessionManager


@pytest.fixture(autouse=True)
def clean_streamlit_context():
    """Ensure clean st.session_state and st.query_params before and after each test."""
    st.session_state.clear()
    st.query_params.clear()
    yield
    st.session_state.clear()
    st.query_params.clear()


# ===========================================================================
# 1. APIClient Unit Tests
# ===========================================================================

class TestAPIClient:
    """Verify APIClient synchronous and asynchronous HTTP methods."""

    def test_create_session_success(self) -> None:
        """create_session() calls POST /api/sessions and returns UUID."""
        client = APIClient(base_url="http://testserver")
        expected_uuid = uuid.uuid4()

        mock_resp = MagicMock()
        mock_resp.is_error = False
        mock_resp.status_code = 201
        mock_resp.json.return_value = {
            "session_id": str(expected_uuid),
            "created_at": "2026-08-15T02:00:00Z",
            "session_token": "tok-123",
        }

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            res_uuid = client.create_session()
            assert res_uuid == expected_uuid
            mock_post.assert_called_once_with("http://testserver/api/sessions")
        assert client.get_session_token(expected_uuid) == "tok-123"

    def test_create_session_without_token_keeps_working(self) -> None:
        """Backends with AUTH_REQUIRED=false return no token; no header is sent then."""
        client = APIClient(base_url="http://testserver")
        sid = uuid.uuid4()
        mock_resp = MagicMock(is_error=False, status_code=201)
        mock_resp.json.return_value = {"session_id": str(sid)}
        with patch("httpx.Client.post", return_value=mock_resp):
            assert client.create_session() == sid
        assert client.get_session_token(sid) is None
        assert client._auth_headers(sid) == {}

    def test_session_token_set_and_clear(self) -> None:
        """set_session_token() stores per-session tokens; a falsy token removes it."""
        client = APIClient(base_url="http://testserver")
        sid = uuid.uuid4()
        client.set_session_token(sid, "abc")
        assert client._auth_headers(str(sid)) == {"Authorization": "Bearer abc"}
        client.set_session_token(sid, None)
        assert client.get_session_token(sid) is None

    def test_send_message_success(self) -> None:
        """send_message() calls POST /api/sessions/{id}/messages and returns MessageResponse."""
        client = APIClient(base_url="http://testserver")
        sid = uuid.uuid4()
        now = datetime.now(UTC)

        mock_resp = MagicMock()
        mock_resp.is_error = False
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "session_id": str(sid),
            "response_text": "Your target BP is <130/80 mmHg [1].",
            "action": "ANSWER",
            "citations": [
                {
                    "marker": "[1]",
                    "chunk_id": "aha_c1",
                    "source": "AHA 2025",
                    "section": "S8",
                    "excerpt": "Target BP <130/80.",
                }
            ],
            "soft_ask_count": 0,
            "timestamp": now.isoformat(),
        }

        client.set_session_token(sid, "tok-xyz")
        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            resp = client.send_message(sid, "What is my target BP?")
            assert isinstance(resp, MessageResponse)
            assert resp.session_id == sid
            assert resp.action == ActionEnum.ANSWER
            assert len(resp.citations) == 1
            mock_post.assert_called_once_with(
                f"http://testserver/api/sessions/{sid}/messages",
                json={"message": "What is my target BP?"},
                headers={"Authorization": "Bearer tok-xyz"},
            )

    def test_get_history_success(self) -> None:
        """get_history() calls GET /api/sessions/{id}/history and returns HistoryResponse."""
        client = APIClient(base_url="http://testserver")
        sid = uuid.uuid4()
        now = datetime.now(UTC)

        mock_resp = MagicMock()
        mock_resp.is_error = False
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "session_id": str(sid),
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                    "action": None,
                    "citations": [],
                    "timestamp": now.isoformat(),
                }
            ],
        }

        client.set_session_token(sid, "tok-h")
        with patch("httpx.Client.get", return_value=mock_resp) as mock_get:
            resp = client.get_history(sid)
            assert isinstance(resp, HistoryResponse)
            assert resp.session_id == sid
            assert len(resp.messages) == 1
            assert resp.messages[0].role == "user"
            mock_get.assert_called_once_with(
                f"http://testserver/api/sessions/{sid}/history",
                headers={"Authorization": "Bearer tok-h"},
            )

    def test_health_check_success(self) -> None:
        """health_check() calls GET /health and returns HealthResponse."""
        client = APIClient(base_url="http://testserver")
        now = datetime.now(UTC)

        mock_resp = MagicMock()
        mock_resp.is_error = False
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "healthy",
            "postgres_connected": True,
            "qdrant_connected": True,
            "timestamp": now.isoformat(),
        }

        with patch("httpx.Client.get", return_value=mock_resp) as mock_get:
            resp = client.health_check()
            assert isinstance(resp, HealthResponse)
            assert resp.status == "healthy"
            assert resp.postgres_connected is True
            mock_get.assert_called_once_with("http://testserver/health")

    def test_connection_error_raises_api_connection_error(self) -> None:
        """ConnectError should be converted to APIConnectionError with safe fallback."""
        client = APIClient(base_url="http://testserver")

        with patch("httpx.Client.post", side_effect=httpx.ConnectError("Connection refused")):
            with pytest.raises(APIConnectionError) as exc_info:
                client.create_session()
            assert "Unable to connect" in exc_info.value.safe_fallback
            assert exc_info.value.error_code == "CONNECTION_ERROR"

    def test_timeout_error_raises_api_timeout_error(self) -> None:
        """TimeoutException should be converted to APITimeoutError with retry guidance."""
        client = APIClient(base_url="http://testserver")

        with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Read timed out")):
            with pytest.raises(APITimeoutError) as exc_info:
                client.create_session()
            assert "took too long" in exc_info.value.safe_fallback
            assert exc_info.value.error_code == "TIMEOUT_ERROR"

    def test_http_error_parses_server_safe_fallback(self) -> None:
        """4xx/5xx responses parse structured ErrorResponse and extract safe_fallback."""
        client = APIClient(base_url="http://testserver")
        mock_resp = MagicMock()
        mock_resp.is_error = True
        mock_resp.status_code = 500
        mock_resp.json.return_value = {
            "error_code": "INTERNAL_ERROR",
            "message": "Internal failure.",
            "safe_fallback": "Server provided clinical fallback instruction.",
        }

        with patch("httpx.Client.post", return_value=mock_resp):
            with pytest.raises(APIStatusError) as exc_info:
                client.create_session()
            assert exc_info.value.status_code == 500
            assert exc_info.value.error_code == "INTERNAL_ERROR"
            assert exc_info.value.safe_fallback == "Server provided clinical fallback instruction."

    def test_http_error_without_json_uses_default_safe_fallback(self) -> None:
        """Non-JSON 502/503 errors use default safe fallback."""
        client = APIClient(base_url="http://testserver")
        mock_resp = MagicMock()
        mock_resp.is_error = True
        mock_resp.status_code = 502
        mock_resp.text = "Bad Gateway"
        mock_resp.json.side_effect = ValueError("Not JSON")

        with patch("httpx.Client.post", return_value=mock_resp):
            with pytest.raises(APIStatusError) as exc_info:
                client.create_session()
            assert exc_info.value.status_code == 502
            assert exc_info.value.safe_fallback == DEFAULT_SAFE_FALLBACK

    @pytest.mark.asyncio
    async def test_async_methods(self) -> None:
        """Async variants function equivalently to sync counterparts."""
        client = APIClient(base_url="http://testserver")
        expected_uuid = uuid.uuid4()

        mock_resp = MagicMock()
        mock_resp.is_error = False
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"session_id": str(expected_uuid), "session_token": "atok"}

        with patch("httpx.AsyncClient.post", return_value=mock_resp):
            res_uuid = await client.acreate_session()
            assert res_uuid == expected_uuid
        assert client.get_session_token(expected_uuid) == "atok"

    @pytest.mark.asyncio
    async def test_async_session_calls_send_bearer_token(self) -> None:
        """asend_message() and aget_history() send the captured bearer token."""
        client = APIClient(base_url="http://testserver")
        sid = uuid.uuid4()
        client.set_session_token(sid, "atok")
        now = datetime.now(UTC).isoformat()
        msg_resp = MagicMock(is_error=False, status_code=200)
        msg_resp.json.return_value = {
            "session_id": str(sid), "response_text": "ok", "action": "GENERALIZE", "timestamp": now,
        }
        hist_resp = MagicMock(is_error=False, status_code=200)
        hist_resp.json.return_value = {"session_id": str(sid), "messages": []}
        with patch("httpx.AsyncClient.post", return_value=msg_resp) as mock_post, \
             patch("httpx.AsyncClient.get", return_value=hist_resp) as mock_get:
            await client.asend_message(sid, "hi")
            await client.aget_history(sid)
        assert mock_post.call_args.kwargs["headers"] == {"Authorization": "Bearer atok"}
        assert mock_get.call_args.kwargs["headers"] == {"Authorization": "Bearer atok"}

    def test_http_error_truncation_with_ellipsis(self) -> None:
        """Non-JSON errors exceeding 200 characters should be truncated with ellipsis."""
        client = APIClient(base_url="http://testserver")

        # Long text (> 200 chars)
        long_text = "A" * 250
        mock_resp = MagicMock()
        mock_resp.is_error = True
        mock_resp.status_code = 500
        mock_resp.text = long_text
        mock_resp.json.side_effect = ValueError("Not JSON")

        with patch("httpx.Client.post", return_value=mock_resp):
            with pytest.raises(APIStatusError) as exc_info:
                client.create_session()
            assert exc_info.value.message == f"HTTP 500: {'A' * 200}..."

        # Short text (<= 200 chars)
        short_text = "B" * 50
        mock_resp.text = short_text
        with patch("httpx.Client.post", return_value=mock_resp):
            with pytest.raises(APIStatusError) as exc_info:
                client.create_session()
            assert exc_info.value.message == f"HTTP 500: {short_text}"

    def test_client_close_and_context_manager(self) -> None:
        """APIClient close() closes persistent client, and context manager handles lifecycle."""
        client = APIClient(base_url="http://testserver")
        assert not client._client.is_closed
        client.close()
        assert client._client.is_closed

        # Re-accessing via _get_client creates fresh active client
        reopened = client._get_client()
        assert not reopened.is_closed
        client.close()

        # Context manager
        with APIClient(base_url="http://testserver") as cm_client:
            assert not cm_client._client.is_closed
        assert cm_client._client.is_closed



# ===========================================================================
# 2. SessionManager Unit Tests
# ===========================================================================

class TestSessionManager:
    """Verify SessionManager state and query parameter synchronization."""

    def test_get_session_id_from_state(self) -> None:
        """get_session_id() returns UUID from st.session_state."""
        test_uuid = uuid.uuid4()
        st.session_state["session_id"] = test_uuid

        retrieved = SessionManager.get_session_id()
        assert retrieved == test_uuid

    def test_get_session_id_from_query_params_and_syncs_state(self) -> None:
        """get_session_id() retrieves UUID from st.query_params and updates state."""
        test_uuid = uuid.uuid4()
        st.query_params["session_id"] = str(test_uuid)

        retrieved = SessionManager.get_session_id()
        assert retrieved == test_uuid
        assert st.session_state["session_id"] == test_uuid

    def test_invalid_query_param_is_cleaned_up(self) -> None:
        """Invalid non-UUID query parameter is dropped and returns None."""
        st.query_params["session_id"] = "invalid-uuid-format"

        retrieved = SessionManager.get_session_id()
        assert retrieved is None
        assert "session_id" not in st.query_params

    def test_set_session_id_syncs_state_and_query_params(self) -> None:
        """set_session_id() updates both session_state and query_params."""
        test_uuid = uuid.uuid4()
        SessionManager.set_session_id(test_uuid)

        assert st.session_state["session_id"] == test_uuid
        assert st.query_params["session_id"] == str(test_uuid)

    def test_set_session_id_accepts_string(self) -> None:
        """set_session_id() parses valid UUID string."""
        test_uuid = uuid.uuid4()
        SessionManager.set_session_id(str(test_uuid))

        assert st.session_state["session_id"] == test_uuid
        assert st.query_params["session_id"] == str(test_uuid)

    def test_clear_session(self) -> None:
        """clear_session() removes session_id, token and messages from state and URL."""
        test_uuid = uuid.uuid4()
        st.session_state["session_id"] = test_uuid
        st.session_state["session_token"] = "secret"
        st.session_state["messages"] = [{"content": "hello"}]
        st.query_params["session_id"] = str(test_uuid)

        SessionManager.clear_session()

        assert "session_id" not in st.session_state
        assert "session_token" not in st.session_state
        assert "messages" not in st.session_state
        assert "session_id" not in st.query_params

    def test_session_token_is_kept_out_of_the_url(self) -> None:
        """set_session_token() stores the token in session state only."""
        SessionManager.set_session_token("tok")
        assert SessionManager.get_session_token() == "tok"
        assert "session_token" not in st.query_params
        SessionManager.set_session_token(None)
        assert SessionManager.get_session_token() is None

    def test_init_session_with_existing(self) -> None:
        """init_session() returns existing session if already established."""
        test_uuid = uuid.uuid4()
        st.session_state["session_id"] = test_uuid

        mock_api = MagicMock(spec=APIClient)
        active_id = SessionManager.init_session(mock_api)

        assert active_id == test_uuid
        mock_api.create_session.assert_not_called()

    def test_init_session_creates_new_via_api(self) -> None:
        """init_session() requests new session via APIClient when none exists."""
        expected_uuid = uuid.uuid4()
        mock_api = MagicMock(spec=APIClient)
        mock_api.create_session.return_value = expected_uuid
        mock_api.get_session_token.return_value = "new-token"

        active_id = SessionManager.init_session(mock_api)

        assert active_id == expected_uuid
        assert st.session_state["session_id"] == expected_uuid
        assert st.query_params["session_id"] == str(expected_uuid)
        assert SessionManager.get_session_token() == "new-token"
        mock_api.create_session.assert_called_once()

    def test_messages_management(self) -> None:
        """add_message() and get_messages() manage the message list."""
        assert SessionManager.get_messages() == []

        SessionManager.add_message("user", "Hello MedBridge")
        SessionManager.add_message("assistant", "Hello! How can I help?", action="ANSWER")

        msgs = SessionManager.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello MedBridge"
        assert msgs[1]["action"] == "ANSWER"
