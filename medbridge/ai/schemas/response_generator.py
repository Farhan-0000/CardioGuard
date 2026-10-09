"""Response Generator (LLM Call 4) output schema."""
from pydantic import BaseModel, Field


class GeneratedCitation(BaseModel):
    """Citation reference corresponding to an inline marker [1], [2]."""

    marker: str = Field(..., description="Inline citation marker, e.g. '[1]'")
    chunk_id: str = Field(default="", description="Identifier of the cited guideline chunk")
    source: str = Field(
        default="", description="Source guideline document, e.g. 'AHA_ACC_HTN_2025'"
    )
    section: str = Field(default="", description="Section or chapter title")
    excerpt: str = Field(default="", description="Verbatim or summarized evidence excerpt")


class ResponseGeneratorOutput(BaseModel):
    """Output schema for Response Generator (LLM Call 4, ADL-025).

    The model only writes ``response_text`` with inline ``[n]`` markers (n = the
    evidence chunk index); the server rebuilds citations from the markers, so
    ``citations`` is optional and is not requested from the model
    (``llm_omit`` removes it from the structured-output schema).
    """

    response_text: str = Field(
        ...,
        min_length=1,
        description="Patient-facing markdown answer with inline [n] citation markers",
    )
    citations: list[GeneratedCitation] = Field(
        default_factory=list,
        description="Optional citation references matching inline markers in response_text",
        json_schema_extra={"llm_omit": True},
    )
