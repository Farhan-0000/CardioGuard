"""Evidence Gate (LLM Call 3, Gate 2) output schema."""
from typing import Literal

from pydantic import BaseModel, Field


class EvidenceGateOutput(BaseModel):
    """Evidence Gate classification output (LLM Call 3, ADL-024).

    ``rationale`` is the first field so that, under grammar-constrained decoding,
    the model states its reasoning before committing to an action.
    """

    rationale: str = Field(
        default="",
        description="Clinical rationale explaining the routing action.",
    )
    action: Literal["ANSWER", "GENERALIZE", "ABSTAIN", "ESCALATE"] = Field(
        ...,
        description="Routing action decision based on evidence sufficiency and clinical safety.",
    )
    evidence_sufficient: bool = Field(
        ...,
        description="Whether the retrieved evidence is clinically sufficient to answer the query.",
    )
    evidence_used: list[int] = Field(
        default_factory=list,
        description="1-based indices of the evidence chunks that support the decision.",
    )
