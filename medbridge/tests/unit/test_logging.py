import io
import json
import logging

import pytest
import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

from medbridge.logging_config import configure_logging


@pytest.fixture(autouse=True)
def reset_logging():
    yield
    clear_contextvars()
    logging.getLogger().handlers.clear()
    structlog.reset_defaults()

def test_json_log_emission_and_masking():
    # Capture stdout
    log_output = io.StringIO()

    # We manually set up a handler to capture output for the test
    configure_logging(json_format=True, log_level=logging.INFO)

    # Replace stdout handler with our StringIO handler
    handler = logging.StreamHandler(log_output)
    formatter = logging.getLogger().handlers[0].formatter
    handler.setFormatter(formatter)
    logging.getLogger().handlers = [handler]

    # Also bind some context to test context variable binding
    bind_contextvars(session_id="12345-abcde", call_name="orchestrator")

    logger = structlog.get_logger("test_logger")

    # Emit a log event with a mix of safe and sensitive fields
    logger.info(
        "request_received",
        message_length=50,
        groq_api_key="secret_key_123",
        authorization="Bearer xyz",
        normal_field="value"
    )

    log_content = log_output.getvalue()
    log_data = json.loads(log_content)

    # Assert base requirements
    assert log_data["event"] == "request_received"
    assert "timestamp" in log_data
    assert log_data["level"] == "info"

    # Assert context variables are bound
    assert log_data["session_id"] == "12345-abcde"
    assert log_data["call_name"] == "orchestrator"

    # Assert arbitrary kwargs are added
    assert log_data["message_length"] == 50
    assert log_data["normal_field"] == "value"

    # Assert sensitive data is masked
    assert log_data["groq_api_key"] == "***MASKED***"
    assert log_data["authorization"] == "***MASKED***"
    assert "secret_key_123" not in log_content
    assert "Bearer xyz" not in log_content



def test_pii_is_scrubbed_from_log_events():
    log_output = io.StringIO()
    configure_logging(json_format=True, log_level=logging.INFO)
    handler = logging.StreamHandler(log_output)
    handler.setFormatter(logging.getLogger().handlers[0].formatter)
    logging.getLogger().handlers = [handler]

    logger = structlog.get_logger("test_logger")
    logger.info("user said my name is Rahul Sharma, email rahul@example.com",
                note="BP 150/95 on Telma 40", prompt_tokens=120)

    log_data = json.loads(log_output.getvalue())
    assert log_data["event"] == "user said my name is [NAME], email [EMAIL]"
    assert log_data["note"] == "BP 150/95 on Telma 40"  # clinical values untouched
    assert log_data["prompt_tokens"] == 120  # token *counts* are not masked
    assert "Rahul" not in log_output.getvalue()


def test_reconfiguring_keeps_cached_loggers_on_the_live_chain():
    """create_app() may run twice; loggers cached in between must follow the new chain."""
    from structlog.testing import capture_logs

    configure_logging(json_format=True, log_level=logging.INFO)
    first = structlog.get_config()["processors"]
    cached = structlog.get_logger("medbridge.test.cached")
    cached.info("first_use_caches_the_logger")
    configure_logging(json_format=False, log_level=logging.INFO)
    assert structlog.get_config()["processors"] is first  # updated in place
    with capture_logs() as logs:
        cached.warning("seen_by_capture")
    assert [e["event"] for e in logs] == ["seen_by_capture"]
