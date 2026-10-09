"""
Medical Disclaimer Presentation Component (TASK-26, Module M-20).

Renders a persistent, non-dismissable medical safety warning banner
in accordance with Technical Specification Part I §2 (R-FE-06).
"""
import streamlit as st

DISCLAIMER_TEXT = (
    "**Medical Disclaimer**: This tool provides informational guidance only. "
    "It does not replace professional medical advice, diagnosis, or treatment. "
    "If you are experiencing a medical emergency (e.g., severe chest pain, "
    "shortness of breath, sudden numbness), call your local emergency number immediately "
    "(112 in India, 911 in the US)."
)


def render_disclaimer() -> None:
    """Render the persistent medical safety disclaimer banner at the top of the UI."""
    st.warning(DISCLAIMER_TEXT, icon="⚠️")
