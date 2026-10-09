"""Frontend Presentation Components Package (TASK-26, Module M-20)."""
from medbridge.frontend.components.chat import (
    ACTION_BADGE_CONFIG,
    get_action_badge_html,
    render_action_badge,
    render_chat_message,
)
from medbridge.frontend.components.citations import render_citations_panel
from medbridge.frontend.components.disclaimer import (
    DISCLAIMER_TEXT,
    render_disclaimer,
)
from medbridge.frontend.components.sidebar import render_sidebar

__all__ = [
    "ACTION_BADGE_CONFIG",
    "DISCLAIMER_TEXT",
    "get_action_badge_html",
    "render_action_badge",
    "render_chat_message",
    "render_citations_panel",
    "render_disclaimer",
    "render_sidebar",
]
