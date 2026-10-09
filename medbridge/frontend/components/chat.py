"""
Chat Presentation Components (TASK-26, Module M-20).

Renders chat messages, clinical triage routing badges (ANSWER, SOFT-ASK, etc.),
and integrates inline citation accordions in accordance with Technical Specification Part I §3.2, §3.3.
"""
from typing import Any

import streamlit as st

from medbridge.api.schemas.enums import ActionEnum
from medbridge.frontend.components.citations import render_citations_panel

ACTION_BADGE_CONFIG = {
    "ANSWER": {
        "label": "ANSWER",
        "emoji": "🟢",
        "color": "#065F46",
        "bg": "#D1FAE5",
        "border": "#A7F3D0",
        "desc": "Answered from guideline evidence that matches your situation",
    },
    "SOFT-ASK": {
        "label": "SOFT-ASK",
        "emoji": "🟡",
        "color": "#92400E",
        "bg": "#FEF3C7",
        "border": "#FDE68A",
        "desc": "I need a bit more information to answer safely",
    },
    "GENERALIZE": {
        "label": "GENERALIZE",
        "emoji": "🔵",
        "color": "#1E40AF",
        "bg": "#DBEAFE",
        "border": "#BFDBFE",
        "desc": "General guideline information — not personalised",
    },
    "ABSTAIN": {
        "label": "ABSTAIN",
        "emoji": "⚪",
        "color": "#374151",
        "bg": "#F3F4F6",
        "border": "#E5E7EB",
        "desc": "Outside what my guidelines can safely answer",
    },
    "ESCALATE": {
        "label": "ESCALATE",
        "emoji": "🔴",
        "color": "#991B1B",
        "bg": "#FEE2E2",
        "border": "#FECACA",
        "desc": "Possible emergency — seek care now",
    },
}


def get_action_badge_html(action: ActionEnum | str) -> str:
    """
    Generate styled HTML markup for a clinical routing action badge.

    Args:
        action: ActionEnum or string ('ANSWER', 'SOFT-ASK', 'GENERALIZE', 'ABSTAIN', 'ESCALATE').

    Returns:
        Safe inline HTML span element with custom styling.
    """
    if isinstance(action, ActionEnum):
        key = action.value.upper()
    else:
        key = str(action).strip().upper()

    cfg = ACTION_BADGE_CONFIG.get(
        key,
        {
            "label": key,
            "emoji": "ℹ️",
            "color": "#374151",
            "bg": "#F3F4F6",
            "border": "#E5E7EB",
            "desc": "Triage Action",
        },
    )

    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;'
        f'color:{cfg["color"]};background-color:{cfg["bg"]};border:1px solid {cfg["border"]};'
        f'margin-bottom:8px;" title="{cfg["desc"]}">'
        f'{cfg["emoji"]} {cfg["label"]}'
        f'</span>'
    )


def render_action_badge(action: ActionEnum | str) -> None:
    """Render a styled clinical routing action badge in Streamlit."""
    badge_html = get_action_badge_html(action)
    st.markdown(badge_html, unsafe_allow_html=True)


def render_chat_message(
    role: str,
    content: str,
    action: ActionEnum | str | None = None,
    citations: list[Any] | None = None,
) -> None:
    """
    Render a single conversational turn in the chat interface.

    Args:
        role: "user" | "patient" or "assistant" | "medbridge".
        content: Markdown message text with inline citation markers.
        action: Optional final routing action for assistant turns.
        citations: Optional list of citation objects.
    """
    # Map patient -> user for standard Streamlit chat avatar
    display_role = "user" if role in ("user", "patient") else "assistant"

    with st.chat_message(display_role):
        if display_role == "assistant" and action:
            render_action_badge(action)

        st.markdown(content)

        if citations:
            render_citations_panel(citations)
