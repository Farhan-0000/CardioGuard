"""Context Extractor (LLM Call 1) output schema — typed clinical delta events."""
from typing import Annotated, Any

from pydantic import BaseModel, Field, WithJsonSchema

from medbridge.api.schemas.enums import EventTypeEnum

# The LLM-facing JSON schema restricts event types to the six known values so
# Ollama's grammar-constrained decoding can never invent a new type. Python-side
# validation stays lenient (plain strings are accepted and filtered later).
_EVENT_TYPE_SCHEMA: dict[str, Any] = {
    "type": "string",
    "enum": [e.value for e in EventTypeEnum],
}


class DeltaEvent(BaseModel):
    """One new clinical fact (BP_READING, MEDICATION_ADDED, …) with its payload."""

    event_type: Annotated[EventTypeEnum | str, WithJsonSchema(_EVENT_TYPE_SCHEMA)]
    payload: dict[str, Any] = {}


class ExtractorOutput(BaseModel):
    """Delta events, a standalone clinical search query and the patient's intent."""

    delta_events: list[DeltaEvent] = Field(default_factory=list)
    search_query: str
    raw_intent: str
