"""
Citations Presentation Component (TASK-26, Module M-20).

Renders a collapsible citations accordion panel linking inline citation markers
to guideline sources, sections, and excerpts in accordance with Technical Specification Part I §3.2, §3.3.
"""
from typing import Any

import streamlit as st

from medbridge.api.schemas.responses import CitationResponse


def render_citations_panel(citations: list[CitationResponse | dict[str, Any] | Any]) -> None:
    """
    Render a collapsible citations panel for guideline evidence.

    Args:
        citations: List of CitationResponse models, dicts, or dataclasses.
    """
    if not citations:
        return

    count = len(citations)
    label = f"📚 Evidence Citations ({count})"

    with st.expander(label, expanded=False):
        for idx, item in enumerate(citations):
            if isinstance(item, CitationResponse):
                marker = item.marker
                source = item.source
                section = item.section
                excerpt = item.excerpt
            elif isinstance(item, dict):
                marker = item.get("marker", f"[{idx+1}]")
                source = item.get("source", "Clinical Guideline")
                section = item.get("section", "Guideline Section")
                excerpt = item.get("excerpt", "")
            else:
                marker = getattr(item, "marker", f"[{idx+1}]")
                source = getattr(item, "source", "Clinical Guideline")
                section = getattr(item, "section", "Guideline Section")
                excerpt = getattr(item, "excerpt", "")

            st.markdown(f"**{marker} {source} — {section}**")
            if excerpt:
                st.markdown(f"> *\"{excerpt.strip()}\"*")

            if idx < count - 1:
                st.divider()
