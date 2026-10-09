"""
MedBridge Streamlit Chat Interface & Main Application (TASK-27, Module M-20).

Assembles:
1. Streamlit page configuration and responsive layout.
2. Persistent medical disclaimer banner (R-FE-06).
3. Sidebar with active session UUID and 'New Conversation' trigger.
4. Multi-turn conversation message history (R-FE-01).
5. Triage action badge rendering (R-FE-03) and collapsible citations accordion (R-FE-04).
6. Chat input box with 2,000-character constraint (Constitution §8.5).
7. Fail-safe error handling and network fallback shielding (Non-Negotiable #7).

Technical Specification Part I §3.2, §3.3.
"""
from pathlib import Path
import sys
from typing import Optional
import uuid

# Ensure repository root is on sys.path when launched via standalone streamlit CLI
repo_root = str(Path(__file__).resolve().parent.parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import structlog
import streamlit as st

from medbridge.frontend.components import (
    render_chat_message,
    render_disclaimer,
    render_sidebar,
)
from medbridge.frontend.services.api_client import (
    DEFAULT_SAFE_FALLBACK,
    APIClient,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)
from medbridge.frontend.services.session import SessionManager

logger = structlog.get_logger(__name__)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Custom disclaimer styling */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: 1px solid #FDE68A !important;
    background: linear-gradient(135deg, #FFFDF5 0%, #FEF9C3 100%) !important;
    box-shadow: 0 2px 6px rgba(245, 158, 11, 0.07) !important;
    color: #92400E !important;
    margin-bottom: 1.25rem !important;
}

/* Sidebar styling - Sleek High-Contrast Dark Theme */
section[data-testid="stSidebar"] {
    background-color: #111827 !important;
    border-right: 1px solid #1F2937 !important;
}

/* Sidebar Headings & Titles */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] strong {
    color: #F8FAFC !important;
    font-weight: 700 !important;
}

/* Sidebar Body text, spans, labels, and list items */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] div {
    color: #E2E8F0 !important;
}

/* Sidebar Captions & secondary notes */
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] .stCaption {
    color: #94A3B8 !important;
}

/* Sidebar Dividers */
section[data-testid="stSidebar"] hr {
    border-color: #1F2937 !important;
}

/* Session ID code block in sidebar: clean, bright, high-contrast */
section[data-testid="stSidebar"] [data-testid="stCodeBlock"],
section[data-testid="stSidebar"] [data-testid="stCode"],
section[data-testid="stSidebar"] pre {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.3) !important;
    overflow: hidden !important;
}
section[data-testid="stSidebar"] [data-testid="stCodeBlock"] *,
section[data-testid="stSidebar"] [data-testid="stCode"] *,
section[data-testid="stSidebar"] pre code,
section[data-testid="stSidebar"] code {
    white-space: pre-wrap !important;
    word-break: break-all !important;
    overflow-wrap: anywhere !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #38BDF8 !important;
    font-family: 'Fira Code', 'Cascadia Code', Consolas, monospace !important;
}

/* Primary buttons (e.g. New Conversation) */
button[kind="primary"], .stButton > button[type="primary"] {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
}
button[kind="primary"]:hover, .stButton > button[type="primary"]:hover {
    background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.45) !important;
    transform: translateY(-1px);
}

/* Chat Messages */
div[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    margin-bottom: 1rem !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    transition: box-shadow 0.2s ease !important;
}
div[data-testid="stChatMessage"]:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
}

/* Citations Expander */
div[data-testid="stExpander"] {
    background-color: #1E293B !important;
    border-radius: 8px !important;
    border: 1px solid #334155 !important;
    margin-top: 0.75rem !important;
}
div[data-testid="stExpander"] * {
    color: #E2E8F0 !important;
}

/* Chat Input Bar */
div[data-testid="stChatInput"] {
    border-radius: 12px !important;
    border: 1.5px solid #334155 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stChatInput"]:focus-within {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2) !important;
}
</style>
"""


def inject_custom_css() -> None:
    """Inject polished clinical CSS stylesheet into Streamlit DOM."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


WELCOME_MESSAGE = (
    "### 👋 Welcome to MedBridge\n"
    "**Evidence-Grounded Clinical Guidance for Hypertension Management**\n\n"
    "I assist patients and clinicians by referencing validated clinical guidelines "
    "(**AHA/ACC 2025**, **ESC/ESH 2024**, **ADA 2025**, and **KDIGO 2024**) "
    "with transparent evidence citations and safety guardrails.\n\n"
    "#### 💡 What you can ask:\n"
    "- 🎯 **Blood Pressure Targets**: *\"My BP was 145/92 this morning, is this considered high?\"*\n"
    "- 📚 **Clinical Guidelines**: *\"What does AHA/ACC 2025 recommend for Stage 1 hypertension?\"*\n"
    "- 💊 **Medication Classes**: *\"What are first-line medications for high blood pressure with diabetes?\"*\n"
    "- 🥗 **Lifestyle & Self-Care**: *\"What dietary sodium limits and exercise habits help lower BP?\"*\n\n"
    "To begin, type your question below. All guidance is strictly grounded in peer-reviewed clinical guidelines."
)


def restore_history(client: APIClient, session_id: uuid.UUID) -> None:
    """
    Restore conversation message history from the backend upon initial session load.
    Only queries the backend if the local session message list is empty.
    """
    messages = SessionManager.get_messages()
    if messages:
        return

    try:
        history_resp = client.get_history(session_id)
        for msg in history_resp.messages:
            action_val = msg.action.value if msg.action else None
            SessionManager.add_message(
                role=msg.role,
                content=msg.content,
                action=action_val,
                citations=msg.citations,
            )
    except Exception as exc:
        logger.warning(
            "history_restore_skipped",
            session_id=str(session_id),
            error=str(exc),
        )


def process_user_query(
    client: APIClient,
    session_id: uuid.UUID,
    prompt: str,
) -> None:
    """
    Submit user query to the backend API and store the response in session state.
    Handles network errors, timeouts, and API status codes gracefully.
    """
    clean_prompt = prompt.strip()
    if not clean_prompt:
        return

    # 1. Record user message in state
    SessionManager.add_message(role="user", content=clean_prompt)

    # 2. Invoke MedBridge 8-stage pipeline via backend API
    try:
        with st.spinner("Consulting clinical guidelines (AHA/ACC 2025, ESC/ESH 2024, ADA 2025)..."):
            response = client.send_message(session_id=session_id, message=clean_prompt)

        action_str = response.action.value if hasattr(response.action, "value") else str(response.action)
        SessionManager.add_message(
            role="assistant",
            content=response.response_text,
            action=action_str,
            citations=response.citations,
        )

    except APIConnectionError as exc:
        logger.error("api_connection_error", error=str(exc))
        SessionManager.add_message(
            role="assistant",
            content=exc.safe_fallback,
            action="ABSTAIN",
        )
    except APITimeoutError as exc:
        logger.warning("api_timeout_error", error=str(exc))
        SessionManager.add_message(
            role="assistant",
            content=exc.safe_fallback,
            action="ABSTAIN",
        )
    except APIStatusError as exc:
        logger.error("api_status_error", status_code=exc.status_code, error=str(exc))
        SessionManager.add_message(
            role="assistant",
            content=exc.safe_fallback,
            action="ABSTAIN",
        )
    except Exception as exc:
        logger.error("unhandled_frontend_exception", error=str(exc), exc_info=True)
        SessionManager.add_message(
            role="assistant",
            content=DEFAULT_SAFE_FALLBACK,
            action="ABSTAIN",
        )


def main() -> None:
    """Main Streamlit application entry point."""
    # 1. Page Configuration
    st.set_page_config(
        page_title="MedBridge — Clinical Guidance Assistant",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    # 2. Service Initialization
    client = APIClient()

    # 3. Session Initialization (syncs with ?session_id=... URL query param)
    try:
        session_id = SessionManager.init_session(client)
    except Exception as exc:
        logger.warning("session_init_backend_offline_fallback", error=str(exc))
        session_id = SessionManager.init_session(None)

    # 4. Restore history if page was refreshed
    restore_history(client, session_id)

    # 5. Render Sidebar Navigation
    render_sidebar(session_id=session_id)

    # 6. Render Persistent Medical Disclaimer Banner (R-FE-06)
    render_disclaimer()

    # 7. Main Application Header
    st.title("🩺 MedBridge Clinical Guidance")
    st.caption("Evidence-Grounded AI Guidance for Hypertension Management")
    st.divider()

    # 8. Render Message History
    messages = SessionManager.get_messages()
    if not messages:
        # Display welcoming instructions if no messages yet
        with st.chat_message("assistant"):
            st.markdown(WELCOME_MESSAGE)
    else:
        for msg in messages:
            render_chat_message(
                role=msg["role"],
                content=msg["content"],
                action=msg.get("action"),
                citations=msg.get("citations"),
            )

    # 9. Conversational Input Box (2,000 max character constraint)
    prompt = st.chat_input(
        placeholder="Type your clinical question here (e.g., 'My BP was 142/90 today, what should I do?')...",
        max_chars=2000,
    )

    if prompt:
        process_user_query(client, session_id, prompt)
        st.rerun()


if __name__ == "__main__":
    main()
