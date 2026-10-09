from pydantic import BaseModel
from typing import Any, Union
from medbridge.api.schemas.enums import EventTypeEnum

class DeltaEvent(BaseModel):
    event_type: Union[EventTypeEnum, str]
    payload: dict[str, Any] = {}

class ExtractorOutput(BaseModel):
    delta_events: list[DeltaEvent] = []
    search_query: str
    raw_intent: str
