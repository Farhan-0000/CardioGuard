"""
Streamlit Session State & Query Parameter Management (TASK-25, Module M-20).

Manages conversation session state and synchronizes with Streamlit's URL
query parameters (?session_id=...). The session's bearer token is kept only in
``st.session_state`` (never in the URL), so after a full page reload the old
session cannot be reused and the app starts a fresh one when the backend
rejects the request.

Technical Specification Part I §3.2, §3.3.
"""
import uuid
from typing import Any

import streamlit as st

from medbridge.frontend.services.api_client import APIClient


class SessionManager:
    """Manages session lifecycle and state synchronization in Streamlit."""

    @staticmethod
    def get_session_id() -> uuid.UUID | None:
        """
        Retrieve active session UUID from st.session_state or st.query_params.

        If found in query params but not in state, synchronizes into session state.
        Returns None if no valid session is found.
        """
        # 1. Check Streamlit session state
        if "session_id" in st.session_state:
            val = st.session_state["session_id"]
            if isinstance(val, uuid.UUID):
                return val
            try:
                parsed_uuid = uuid.UUID(str(val))
                st.session_state["session_id"] = parsed_uuid
                return parsed_uuid
            except ValueError:
                pass

        # 2. Check URL query parameters (?session_id=...)
        query_param_val = st.query_params.get("session_id")
        if query_param_val:
            try:
                parsed_uuid = uuid.UUID(str(query_param_val))
                st.session_state["session_id"] = parsed_uuid
                st.query_params["session_id"] = str(parsed_uuid)
                return parsed_uuid
            except ValueError:
                # Remove invalid query parameter
                if "session_id" in st.query_params:
                    del st.query_params["session_id"]

        return None

    @staticmethod
    def set_session_id(session_id: uuid.UUID | str) -> uuid.UUID:
        """
        Set active session UUID in both st.session_state and st.query_params.

        Args:
            session_id: UUID instance or valid UUID string.

        Returns:
            Normalized UUID object.
        """
        if isinstance(session_id, str):
            session_uuid = uuid.UUID(session_id)
        else:
            session_uuid = session_id

        st.session_state["session_id"] = session_uuid
        st.query_params["session_id"] = str(session_uuid)
        return session_uuid

    @staticmethod
    def get_session_token() -> str | None:
        """Return the bearer token of the active session (kept in session state only)."""
        token = st.session_state.get("session_token")
        return token if isinstance(token, str) and token else None

    @staticmethod
    def set_session_token(token: str | None) -> None:
        """Store (or clear) the bearer token of the active session."""
        if token:
            st.session_state["session_token"] = token
        elif "session_token" in st.session_state:
            del st.session_state["session_token"]

    @staticmethod
    def clear_session() -> None:
        """Clear the active session, its token and messages from state and the URL."""
        for key in ("session_id", "session_token", "messages"):
            if key in st.session_state:
                del st.session_state[key]
        if "session_id" in st.query_params:
            del st.query_params["session_id"]

    @classmethod
    def init_session(cls, api_client: APIClient | None = None) -> uuid.UUID:
        """
        Ensure an active session exists.

        If a session ID is already stored in state or query params, returns it.
        Otherwise, requests a new session from the backend via api_client.create_session()
        (or generates a local UUID fallback) and synchronizes state, including the
        session's bearer token.
        """
        existing = cls.get_session_id()
        if existing is not None:
            return existing

        if api_client is not None:
            new_id = api_client.create_session()
            cls.set_session_token(api_client.get_session_token(new_id))
        else:
            new_id = uuid.uuid4()

        cls.set_session_id(new_id)
        return new_id

    @staticmethod
    def get_messages() -> list[dict[str, Any]]:
        """Retrieve conversation messages from Streamlit session state."""
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        return st.session_state["messages"]

    @classmethod
    def add_message(
        cls,
        role: str,
        content: str,
        action: str | None = None,
        citations: list[Any] | None = None,
    ) -> None:
        """Append a message to the current conversation state."""
        messages = cls.get_messages()
        messages.append({
            "role": role,
            "content": content,
            "action": action,
            "citations": citations or [],
        })
