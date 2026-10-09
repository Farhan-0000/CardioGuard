"""
Unit Tests for Streamlit Chat Interface & Main Application (TASK-27, Module M-20).

Tests:
- Conversation history restoration on initial page load.
- User message processing and successful API response handling.
- Fail-safe fallback shielding for connection errors, timeouts, and HTTP errors.
- Main page component assembly and input box validation.
"""
import uuid
from datetime import UTC, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from medbridge.api.schemas.enums import ActionEnum
from medbridge.api.schemas.responses import (
    CitationResponse,
    HistoryMessage,
    HistoryResponse,
    MessageResponse,
)
from medbridge.frontend.app import (
    WELCOME_MESSAGE,
    main,
    process_user_query,
    restore_history,
    send_with_session_recovery,
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
def clean_session_state() -> None:
    """Ensure clean session state for each test."""
    SessionManager.clear_session()
    yield
    SessionManager.clear_session()


# ===========================================================================
# 1. History Restoration Tests
# ===========================================================================

def test_restore_history_skips_if_messages_already_present() -> None:
    """If session state already has messages, backend history is not queried."""
    SessionManager.add_message(role="user", content="Existing message")
    client = MagicMock(spec=APIClient)

    restore_history(client, uuid.uuid4())
    client.get_history.assert_not_called()


def test_restore_history_populates_empty_state() -> None:
    """If state is empty, backend history messages are restored into state."""
    sid = uuid.uuid4()
    now = datetime.now(UTC)
    mock_history = HistoryResponse(
        session_id=sid,
        messages=[
            HistoryMessage(
                role="user",
                content="What is normal BP?",
                action=None,
                citations=[],
                timestamp=now,
            ),
            HistoryMessage(
                role="assistant",
                content="Normal BP is <120/<80 mmHg [1].",
                action=ActionEnum.ANSWER,
                citations=[
                    CitationResponse(
                        marker="[1]",
                        chunk_id="aha_01",
                        source="AHA 2025",
                        section="Table 1",
                        excerpt="Normal BP is <120 and <80 mmHg.",
                    )
                ],
                timestamp=now,
            ),
        ],
    )
    client = MagicMock(spec=APIClient)
    client.get_history.return_value = mock_history

    restore_history(client, sid)
    client.get_history.assert_called_once_with(sid)

    messages = SessionManager.get_messages()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is normal BP?"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["action"] == "ANSWER"
    assert len(messages[1]["citations"]) == 1


def test_restore_history_swallows_backend_error() -> None:
    """Backend connection failure during restore logs a warning without crashing."""
    client = MagicMock(spec=APIClient)
    client.get_history.side_effect = APIConnectionError("Backend unavailable")

    sid = uuid.uuid4()
    # Should not raise
    restore_history(client, sid)
    assert len(SessionManager.get_messages()) == 0


# ===========================================================================
# 2. Query Processing & Fallback Shielding Tests
# ===========================================================================

def test_process_user_query_empty_string_ignored() -> None:
    """Empty or whitespace-only prompt performs no action."""
    client = MagicMock(spec=APIClient)
    process_user_query(client, uuid.uuid4(), "   \n\t  ")
    assert len(SessionManager.get_messages()) == 0
    client.send_message.assert_not_called()


def test_process_user_query_successful_response() -> None:
    """Valid prompt dispatches to APIClient and appends assistant response."""
    sid = uuid.uuid4()
    client = MagicMock(spec=APIClient)
    client.send_message.return_value = MessageResponse(
        session_id=sid,
        action=ActionEnum.ANSWER,
        response_text="Target BP is <130/80 mmHg [1].",
        citations=[
            CitationResponse(
                marker="[1]",
                chunk_id="aha_c1",
                source="AHA 2025",
                section="S8",
                excerpt="Target is <130/80.",
            )
        ],
        timestamp=datetime.now(UTC),
    )

    with patch("streamlit.spinner"):
        process_user_query(client, sid, "What is target BP for diabetes?")

    messages = SessionManager.get_messages()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is target BP for diabetes?"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Target BP is <130/80 mmHg [1]."
    assert messages[1]["action"] == "ANSWER"
    assert len(messages[1]["citations"]) == 1


def test_process_user_query_handles_connection_error() -> None:
    """Network connection failure renders safe clinical fallback with ABSTAIN action."""
    sid = uuid.uuid4()
    client = MagicMock(spec=APIClient)
    client.send_message.side_effect = APIConnectionError("Connection refused")

    with patch("streamlit.spinner"):
        process_user_query(client, sid, "Check my blood pressure")

    messages = SessionManager.get_messages()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["action"] == "ABSTAIN"
    assert "Unable to connect to the MedBridge service" in messages[1]["content"]


def test_process_user_query_handles_timeout_error() -> None:
    """Request timeout renders safe clinical fallback with ABSTAIN action."""
    sid = uuid.uuid4()
    client = MagicMock(spec=APIClient)
    client.send_message.side_effect = APITimeoutError("Request timed out after 15s")

    with patch("streamlit.spinner"):
        process_user_query(client, sid, "Complex multi-morbidity query")

    messages = SessionManager.get_messages()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["action"] == "ABSTAIN"
    assert "took too long to process" in messages[1]["content"]


def test_process_user_query_handles_status_error() -> None:
    """HTTP error code renders safe clinical fallback with ABSTAIN action."""
    sid = uuid.uuid4()
    client = MagicMock(spec=APIClient)
    client.send_message.side_effect = APIStatusError(
        status_code=500,
        error_code="INTERNAL_ERROR",
        message="500 Internal Server Error",
        safe_fallback="An unexpected clinical verification issue occurred. Please try asking again.",
    )

    with patch("streamlit.spinner"):
        process_user_query(client, sid, "My blood pressure is high")

    messages = SessionManager.get_messages()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["action"] == "ABSTAIN"
    assert "unexpected clinical verification issue" in messages[1]["content"]


def test_process_user_query_handles_unhandled_exception() -> None:
    """Generic unhandled exceptions fall back to DEFAULT_SAFE_FALLBACK."""
    sid = uuid.uuid4()
    client = MagicMock(spec=APIClient)
    client.send_message.side_effect = RuntimeError("Unexpected internal crash")

    with patch("streamlit.spinner"):
        process_user_query(client, sid, "Any query")

    messages = SessionManager.get_messages()
    assert len(messages) == 2
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == DEFAULT_SAFE_FALLBACK
    assert messages[1]["action"] == "ABSTAIN"


def test_rejected_session_token_starts_new_session_and_retries() -> None:
    """A 401 for the stored session creates a new session (with token) and resends once."""
    old_sid, new_sid = uuid.uuid4(), uuid.uuid4()
    ok = MessageResponse(
        session_id=new_sid,
        action=ActionEnum.GENERALIZE,
        response_text="General guidance.",
        timestamp=datetime.now(UTC),
    )
    client = MagicMock(spec=APIClient)
    client.send_message.side_effect = [
        APIStatusError(status_code=401, error_code="UNAUTHORIZED", message="bad token", safe_fallback="x"),
        ok,
    ]
    client.create_session.return_value = new_sid
    client.get_session_token.return_value = "fresh-token"

    result = send_with_session_recovery(client, old_sid, "hello")

    assert result is ok
    assert client.send_message.call_args_list[1].kwargs == {"session_id": new_sid, "message": "hello"}
    assert SessionManager.get_session_id() == new_sid
    assert SessionManager.get_session_token() == "fresh-token"


def test_non_session_errors_are_not_retried() -> None:
    """Errors other than 401/403/404 propagate unchanged (no new session)."""
    client = MagicMock(spec=APIClient)
    client.send_message.side_effect = APIStatusError(
        status_code=429, error_code="RATE_LIMITED", message="slow down", safe_fallback="Please wait."
    )
    with pytest.raises(APIStatusError):
        send_with_session_recovery(client, uuid.uuid4(), "hello")
    client.create_session.assert_not_called()


def test_welcome_message_matches_indexed_corpus() -> None:
    """Welcome text names the curated corpus generically and no unindexed sources."""
    assert "curated cardiology and hypertension guidelines" in WELCOME_MESSAGE
    assert "AHA/ACC, ESC, NICE, WHO" in WELCOME_MESSAGE
    for absent in ("ADA 2025", "KDIGO"):
        assert absent not in WELCOME_MESSAGE


# ===========================================================================
# 3. Main Application Assembly Tests
# ===========================================================================

def test_main_initial_render_flow() -> None:
    """Verify full main() flow on initial empty visit."""
    sid = uuid.uuid4()

    mock_chat_ctx = MagicMock()
    with patch("streamlit.set_page_config") as mock_cfg, \
         patch("medbridge.frontend.app.APIClient") as mock_client_cls, \
         patch("medbridge.frontend.app.SessionManager.init_session", return_value=sid) as mock_init_session, \
         patch("medbridge.frontend.app.restore_history") as mock_restore, \
         patch("medbridge.frontend.app.render_sidebar") as mock_sidebar, \
         patch("medbridge.frontend.app.render_disclaimer") as mock_disclaimer, \
         patch("streamlit.title") as mock_title, \
         patch("streamlit.caption") as mock_caption, \
         patch("streamlit.divider") as mock_divider, \
         patch("streamlit.chat_message", return_value=mock_chat_ctx) as mock_chat_msg, \
         patch("streamlit.markdown") as mock_markdown, \
         patch("streamlit.chat_input", return_value=None) as mock_input:

        main()

        mock_cfg.assert_called_once()
        mock_init_session.assert_called_once()
        mock_client_cls.return_value.set_session_token.assert_called_once()
        mock_restore.assert_called_once()
        mock_sidebar.assert_called_once_with(session_id=sid)
        mock_disclaimer.assert_called_once()
        mock_title.assert_called_once()
        mock_caption.assert_called_once()
        mock_divider.assert_called_once()
        # Initial empty state displays welcome message
        mock_chat_msg.assert_called_once_with("assistant")
        mock_markdown.assert_any_call(WELCOME_MESSAGE)
        mock_input.assert_called_once_with(
            placeholder="Type your clinical question here (e.g., 'My BP was 142/90 today, what should I do?')...",
            max_chars=2000,
        )


def test_main_renders_existing_messages_and_submits_input() -> None:
    """When messages exist and user types input, query is processed and page reruns."""
    sid = uuid.uuid4()
    SessionManager.add_message(role="user", content="Hello")
    SessionManager.add_message(role="assistant", content="Hi there", action="ANSWER")

    with patch("streamlit.set_page_config"), \
         patch("medbridge.frontend.app.APIClient"), \
         patch("medbridge.frontend.app.SessionManager.init_session", return_value=sid), \
         patch("medbridge.frontend.app.restore_history"), \
         patch("medbridge.frontend.app.render_sidebar"), \
         patch("medbridge.frontend.app.render_disclaimer"), \
         patch("streamlit.title"), \
         patch("streamlit.caption"), \
         patch("streamlit.divider"), \
         patch("medbridge.frontend.app.render_chat_message") as mock_render_msg, \
         patch("streamlit.chat_input", return_value="What is Stage 1 hypertension?"), \
         patch("medbridge.frontend.app.process_user_query") as mock_process, \
         patch("streamlit.rerun") as mock_rerun:

        main()

        assert mock_render_msg.call_count == 2
        mock_process.assert_called_once()
        assert mock_process.call_args[0][2] == "What is Stage 1 hypertension?"
        mock_rerun.assert_called_once()
