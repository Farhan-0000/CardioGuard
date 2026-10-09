"""
Structured logging configuration (structlog → stdlib → JSON/console).

Two privacy processors run on every event (Constitution §8.4, ADL-035):

* ``mask_sensitive_data`` — replaces values of secret-bearing keys
  (API keys, passwords, tokens, authorization headers) with ``***MASKED***``;
* ``scrub_pii_processor`` — redacts PII (names, phones, e-mails, ids, …) from
  every remaining string value, including formatted exception text.
"""
import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog
from structlog.contextvars import merge_contextvars
from structlog.typing import Processor

from medbridge.privacy.log_scrub import scrub_pii_processor

# Keys whose value is masked when the key *contains* one of these substrings.
SENSITIVE_KEYS = {
    "api_key", "authorization", "password", "groq_api_key", "openai_api_key",
    "postgres_password", "secret", "master_key", "session_token", "wrapped_key", "cookie",
}
# Keys masked only on an exact (case-insensitive) match — avoids masking e.g. "prompt_tokens".
SENSITIVE_EXACT_KEYS = {"token", "access_token", "admin_token", "bearer", "x-session-token"}
MASK = "***MASKED***"


def mask_sensitive_data(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Processor to mask sensitive fields in the log event dictionary."""
    for key in list(event_dict.keys()):
        lowered = key.lower()
        if lowered in SENSITIVE_EXACT_KEYS or any(s in lowered for s in SENSITIVE_KEYS):
            event_dict[key] = MASK
    return event_dict


def configure_logging(json_format: bool = True, log_level: int = logging.INFO) -> None:
    """
    Configure structlog and standard logging.
    If json_format is True, outputs structured JSON.
    Otherwise outputs colored console logging.
    """
    shared_processors: list[Processor] = [
        merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        mask_sensitive_data,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        # Last, so formatted exceptions and positional args are scrubbed too.
        scrub_pii_processor,
    ]

    renderer: Processor = (
        structlog.processors.JSONRenderer() if json_format else structlog.dev.ConsoleRenderer()
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    processors: list[Processor] = [
        structlog.stdlib.filter_by_level,
        *shared_processors,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]
    # Loggers are cached on first use and keep a reference to the processor
    # list that was configured at that moment. Re-configuring (create_app() is
    # called more than once in one process, e.g. by tests or a reloader) must
    # therefore update the existing list in place; a new list would leave the
    # already-cached module loggers on a stale chain that later configuration
    # (or structlog.testing.capture_logs) never reaches.
    current = structlog.get_config().get("processors")
    if structlog.is_configured() and isinstance(current, list):
        current[:] = processors
        processors = current
    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
