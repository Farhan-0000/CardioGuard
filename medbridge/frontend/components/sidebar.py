"""
Sidebar Presentation Component (TASK-26, Module M-20).

Renders the Streamlit sidebar displaying active session UUID,
guideline knowledge base sources, and the 'New Chat' reset trigger.

Technical Specification Part I §3.2, §3.3.
"""
from typing import Callable, Optional
import uuid

import streamlit as st

from medbridge.frontend.services.session import SessionManager


def render_sidebar(
    session_id: Optional[uuid.UUID] = None,
    on_new_chat: Optional[Callable[[], None]] = None,
) -> None:
    """
    Render sidebar navigation and session management controls.
    
    Args:
        session_id: Active conversation session UUID.
        on_new_chat: Optional callback executed when user clicks 'New Conversation'.
    """
    with st.sidebar:
        st.title("🩺 MedBridge")
        st.caption("Hypertension Clinical Guidance Assistant")
        st.divider()

        # Session Status
        st.subheader("💬 Active Session")
        if session_id:
            st.code(str(session_id), language="text")
            st.caption("Session UUID (shareable URL)")
        else:
            st.info("No active session initialized.")

        # New Chat Control
        if st.button("➕ New Conversation", use_container_width=True, type="primary"):
            SessionManager.clear_session()
            if on_new_chat is not None:
                on_new_chat()
            st.rerun()

        st.divider()

        # Clinical Knowledge Base Sources
        st.subheader("📚 Guideline Index")
        st.markdown(
            "• **AHA/ACC 2025** Hypertension  \n"
            "• **ESC 2024** Guidelines  \n"
            "• **NICE NG136** Management"
        )

        st.divider()
        st.caption("MedBridge v3.0 • Non-Negotiable Safety Architecture")
