"""
API Request Pydantic Models.

Defines validated request schemas for inbound API endpoints.
Constitution §8.5: messages are limited to 2,000 characters (HTTP 422 beyond).
"""

from pydantic import BaseModel, Field, field_validator


class MessageRequest(BaseModel):
    """Request body for ``POST /api/sessions/{session_id}/messages`` (and ``/stream``)."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Patient's clinical question",
    )

    @field_validator("message")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        """A message made only of whitespace is not a question (HTTP 422)."""
        if not value.strip():
            raise ValueError("message must contain non-whitespace characters")
        return value
