from typing import Literal
from pydantic import BaseModel, Field

class ContextGateOutput(BaseModel):
    action: Literal["SOFT-ASK", "PROCEED"]
    missing_fields: list[str] = Field(default_factory=list)
    rationale: str
