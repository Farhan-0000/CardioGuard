"""
structlog processor that removes PII from log events (ADL-035).

Every string value of the event dict — including the event message, nested
dicts/lists and formatted exception text — is passed through
:func:`medbridge.privacy.redaction.redact`. Work is bounded: strings longer
than ``MAX_SCRUB_CHARS`` are truncated and nesting deeper than
``MAX_DEPTH`` is replaced by a marker. UUIDs (session/message ids) and
the structural keys added by structlog are left untouched.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping
from typing import Any

from medbridge.privacy.redaction import redact

MAX_SCRUB_CHARS = 4000
MAX_DEPTH = 4
MAX_ITEMS = 200
_SKIP_KEYS = frozenset({"timestamp", "level", "logger", "logger_name"})
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _scrub_str(value: str) -> str:
    if _UUID_RE.match(value):
        return value
    if len(value) > MAX_SCRUB_CHARS:
        value = value[:MAX_SCRUB_CHARS] + "…[truncated]"
    return redact(value).text


def scrub_value(value: Any, depth: int = 0) -> Any:
    """Return ``value`` with PII removed from every string it contains."""
    if isinstance(value, str):
        return _scrub_str(value)
    if depth >= MAX_DEPTH and isinstance(value, Mapping | list | tuple | set):
        return "[nested]"
    if isinstance(value, Mapping):
        return {k: scrub_value(v, depth + 1) for k, v in list(value.items())[:MAX_ITEMS]}
    if isinstance(value, list | tuple | set):
        return [scrub_value(v, depth + 1) for v in list(value)[:MAX_ITEMS]]
    return value


def scrub_pii_processor(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """structlog processor: redact PII in all string values of ``event_dict``."""
    for key in list(event_dict.keys()):
        if key not in _SKIP_KEYS:
            event_dict[key] = scrub_value(event_dict[key])
    return event_dict
