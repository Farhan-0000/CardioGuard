"""
Unit Tests for Frontend Presentation Components (TASK-26, Module M-20).

Verifies disclaimer banner, action badges, citation accordion, and sidebar rendering.
"""
import uuid
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st

from medbridge.api.schemas.enums import ActionEnum
from medbridge.api.schemas.responses import CitationResponse
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

# ===========================================================================
# 1. Disclaimer Component Tests
# ===========================================================================

def test_render_disclaimer() -> None:
    """render_disclaimer() renders persistent warning banner with exact text."""
    with patch("streamlit.warning") as mock_warning:
        render_disclaimer()
        mock_warning.assert_called_once_with(DISCLAIMER_TEXT, icon="⚠️")


# ===========================================================================
# 2. Chat & Action Badge Component Tests
# ===========================================================================

class TestActionBadges:
    """Verify routing action badge HTML generation and Streamlit rendering."""

    @pytest.mark.parametrize(
        "action,expected_emoji,expected_label",
        [
            ("ANSWER", "🟢", "ANSWER"),
            ("SOFT-ASK", "🟡", "SOFT-ASK"),
            ("GENERALIZE", "🔵", "GENERALIZE"),
            ("ABSTAIN", "⚪", "ABSTAIN"),
            ("ESCALATE", "🔴", "ESCALATE"),
        ],
    )
    def test_action_badge_html_all_actions(self, action: str, expected_emoji: str, expected_label: str) -> None:
        """All 5 clinical actions produce styled badge with correct emoji and text."""
        html = get_action_badge_html(action)
        assert expected_emoji in html
        assert expected_label in html
        assert ACTION_BADGE_CONFIG[action]["color"] in html
        assert ACTION_BADGE_CONFIG[action]["bg"] in html

    def test_action_badge_html_with_enum(self) -> None:
        """get_action_badge_html() accepts ActionEnum instance."""
        html = get_action_badge_html(ActionEnum.ANSWER)
        assert "🟢" in html
        assert "ANSWER" in html

    def test_action_badge_html_unknown_action(self) -> None:
        """Unknown action strings fall back to neutral informational badge."""
        html = get_action_badge_html("CUSTOM_ACTION")
        assert "CUSTOM_ACTION" in html
        assert "ℹ️" in html

    def test_render_action_badge(self) -> None:
        """render_action_badge() calls st.markdown with unsafe_allow_html=True."""
        with patch("streamlit.markdown") as mock_markdown:
            render_action_badge(ActionEnum.ESCALATE)
            mock_markdown.assert_called_once()
            args, kwargs = mock_markdown.call_args
            assert "🔴" in args[0]
            assert "ESCALATE" in args[0]
            assert kwargs.get("unsafe_allow_html") is True

    def test_render_chat_message_assistant_abstain_shows_badge(self) -> None:
        """ABSTAIN answers (templates, no citations) still show their action badge."""
        mock_chat_ctx = MagicMock()
        with patch("streamlit.chat_message", return_value=mock_chat_ctx), \
             patch("streamlit.markdown") as mock_md:
            render_chat_message(role="assistant", content="Outside scope.", action="ABSTAIN")
            badge_html = mock_md.call_args_list[0][0][0]
            assert "ABSTAIN" in badge_html
            assert ACTION_BADGE_CONFIG["ABSTAIN"]["desc"] in badge_html

    def test_render_chat_message_user(self) -> None:
        """User messages render without action badge inside user chat bubble."""
        mock_chat_ctx = MagicMock()
        with patch("streamlit.chat_message", return_value=mock_chat_ctx) as mock_cm, \
             patch("streamlit.markdown") as mock_md:
            render_chat_message(role="user", content="My BP is 150/90.")
            mock_cm.assert_called_once_with("user")
            mock_md.assert_called_once_with("My BP is 150/90.")

    def test_render_chat_message_assistant_with_action_and_citations(self) -> None:
        """Assistant messages render action badge, text, and citations panel."""
        mock_chat_ctx = MagicMock()
        citations = [
            CitationResponse(
                marker="[1]",
                chunk_id="aha_c1",
                source="AHA 2025",
                section="S8.1",
                excerpt="Target is <130/80.",
            )
        ]

        with patch("streamlit.chat_message", return_value=mock_chat_ctx) as mock_cm, \
             patch("streamlit.markdown") as mock_md, \
             patch("medbridge.frontend.components.chat.render_citations_panel") as mock_cite:
            render_chat_message(
                role="assistant",
                content="Based on guidelines [1]...",
                action=ActionEnum.ANSWER,
                citations=citations,
            )
            mock_cm.assert_called_once_with("assistant")
            assert mock_md.call_count >= 2  # badge + content
            mock_cite.assert_called_once_with(citations)


# ===========================================================================
# 3. Citations Panel Tests
# ===========================================================================

class TestCitationsPanel:
    """Verify collapsible citations panel rendering."""

    def test_render_citations_panel_empty(self) -> None:
        """Empty or None citation list renders nothing."""
        with patch("streamlit.expander") as mock_expander:
            render_citations_panel([])
            mock_expander.assert_not_called()

            render_citations_panel(None)  # type: ignore[arg-type]
            mock_expander.assert_not_called()

    def test_render_citations_panel_with_pydantic_citations(self) -> None:
        """Pydantic CitationResponse models format marker, source, section, excerpt."""
        citations = [
            CitationResponse(
                marker="[1]",
                chunk_id="aha_c1",
                source="AHA/ACC 2025",
                section="Section 8.2",
                excerpt="For adults with diabetes, target is <130/80 mmHg.",
            ),
            CitationResponse(
                marker="[2]",
                chunk_id="esc_c3",
                source="ESC/ESH 2024",
                section="Section 6.1",
                excerpt="Initiate treatment with dual therapy.",
            ),
        ]

        mock_exp_ctx = MagicMock()
        with patch("streamlit.expander", return_value=mock_exp_ctx) as mock_exp, \
             patch("streamlit.markdown") as mock_md, \
             patch("streamlit.divider") as mock_div:
            render_citations_panel(citations)
            mock_exp.assert_called_once_with("📚 Evidence Citations (2)", expanded=False)
            assert mock_md.call_count == 4  # 2 headers + 2 excerpts
            mock_div.assert_called_once()  # Divider between citation 1 and 2

    def test_render_citations_panel_with_dicts(self) -> None:
        """Dict-based citations format correctly."""
        citations = [
            {
                "marker": "[1]",
                "source": "KDIGO 2024",
                "section": "Chapter 3",
                "excerpt": "Target SBP <120 mmHg in CKD.",
            }
        ]

        mock_exp_ctx = MagicMock()
        with patch("streamlit.expander", return_value=mock_exp_ctx), \
             patch("streamlit.markdown") as mock_md:
            render_citations_panel(citations)
            assert mock_md.call_count == 2
            rendered_header = mock_md.call_args_list[0][0][0]
            rendered_excerpt = mock_md.call_args_list[1][0][0]
            assert "[1] KDIGO 2024 — Chapter 3" in rendered_header
            assert "Target SBP <120 mmHg in CKD." in rendered_excerpt


# ===========================================================================
# 4. Sidebar Component Tests
# ===========================================================================

class TestSidebar:
    """Verify sidebar session status and reset triggers."""

    def test_render_sidebar_displays_session_id(self) -> None:
        """Sidebar displays code block with active session UUID."""
        sid = uuid.uuid4()
        mock_sidebar_ctx = MagicMock()

        with patch("streamlit.sidebar", mock_sidebar_ctx), \
             patch("streamlit.title") as mock_title, \
             patch("streamlit.code") as mock_code, \
             patch("streamlit.button", return_value=False):
            render_sidebar(session_id=sid)
            mock_title.assert_called_once_with("🩺 MedBridge")
            mock_code.assert_called_once_with(str(sid), language="text")

    def test_render_sidebar_no_session_shows_info(self) -> None:
        """Sidebar indicates no active session when None is provided."""
        mock_sidebar_ctx = MagicMock()

        with patch("streamlit.sidebar", mock_sidebar_ctx), \
             patch("streamlit.info") as mock_info, \
             patch("streamlit.button", return_value=False):
            render_sidebar(session_id=None)
            mock_info.assert_called_once()

    def test_render_sidebar_new_chat_clicked(self) -> None:
        """Clicking 'New Conversation' invokes callback, clears session, and reruns."""
        mock_sidebar_ctx = MagicMock()
        mock_callback = MagicMock()

        with patch("streamlit.sidebar", mock_sidebar_ctx), \
             patch("streamlit.button", return_value=True), \
             patch("medbridge.frontend.services.session.SessionManager.clear_session") as mock_clear, \
             patch("streamlit.rerun") as mock_rerun:
            render_sidebar(session_id=uuid.uuid4(), on_new_chat=mock_callback)
            mock_clear.assert_called_once()
            mock_callback.assert_called_once()
            mock_rerun.assert_called_once()
