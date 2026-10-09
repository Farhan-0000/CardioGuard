"""
Unit Tests for Global Error Handling Middleware (TASK-28, Module M-03).

Constitutional Invariant Non-Negotiable #7:
Guarantees zero raw 500 errors, structured ErrorResponse schemas, safe clinical
fallbacks, and strict server-side stack trace containment.
"""
import uuid

import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from medbridge.api.middleware.error_handler import (
    SAFE_FALLBACK,
    database_error_handler,
    global_exception_handler,
    http_exception_handler,
    register_error_handlers,
    session_not_found_handler,
    validation_error_handler,
)
from medbridge.api.schemas.responses import ErrorResponse
from medbridge.state.session_manager import (
    MessageNotFoundError,
    SessionErasedError,
    SessionNotFoundError,
)


class DummyPayload(BaseModel):
    message: str = Field(..., min_length=2, max_length=100)


@pytest.fixture
def test_app() -> FastAPI:
    """Create a FastAPI application with registered error handlers and test routes."""
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/test/unhandled")
    async def route_unhandled():
        raise RuntimeError("Sensitive internal database connection failed: secret_token_123")

    @app.get("/test/session-not-found/{session_id}")
    async def route_session_not_found(session_id: uuid.UUID):
        raise SessionNotFoundError(session_id)

    @app.post("/test/validation")
    async def route_validation(payload: DummyPayload):
        return {"status": "ok", "message": payload.message}

    @app.get("/test/db-error")
    async def route_db_error():
        raise OperationalError(
            statement="SELECT * FROM sessions",
            params={},
            orig=Exception("Connection refused at 127.0.0.1:5432"),
        )

    @app.get("/test/http-403")
    async def route_http_403():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden for this clinical resource.",
        )

    @app.get("/test/http-429")
    async def route_http_429():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded.",
            headers={"Retry-After": "30"},
        )

    @app.get("/test/http-401")
    async def route_http_401():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.get("/test/http-413")
    async def route_http_413():
        raise HTTPException(status_code=413, detail="Request body too large.")

    @app.get("/test/session-erased/{session_id}")
    async def route_session_erased(session_id: uuid.UUID):
        raise SessionErasedError(session_id)

    @app.get("/test/message-not-found/{message_id}")
    async def route_message_not_found(message_id: uuid.UUID):
        raise MessageNotFoundError(message_id)

    return app


@pytest.fixture
def client(test_app: FastAPI) -> TestClient:
    """FastAPI TestClient configured with error handled app."""
    return TestClient(test_app, raise_server_exceptions=False)


class TestGlobalErrorHandler:
    """Verify behavior of global error handlers and constitutional invariants."""

    def test_unhandled_exception_returns_500_with_safe_fallback(self, client: TestClient) -> None:
        """Verify unhandled exceptions return 500 with safe clinical fallback message."""
        response = client.get("/test/unhandled")
        assert response.status_code == 500

        data = response.json()
        assert data["error_code"] == "INTERNAL_ERROR"
        assert data["message"] == "An internal error occurred."
        assert data["safe_fallback"] == SAFE_FALLBACK

        # Ensure model validation passes against ErrorResponse
        error_resp = ErrorResponse.model_validate(data)
        assert error_resp.error_code == "INTERNAL_ERROR"

    def test_unhandled_exception_never_leaks_stack_trace_or_sensitive_data(self, client: TestClient) -> None:
        """Verify stack traces, python traces, and sensitive strings are masked from response."""
        response = client.get("/test/unhandled")
        body_text = response.text

        assert "secret_token_123" not in body_text
        assert "Traceback" not in body_text
        assert "RuntimeError" not in body_text
        assert ".py" not in body_text
        assert "line " not in body_text

    def test_session_not_found_returns_404_with_structured_error_response(self, client: TestClient) -> None:
        """Verify SessionNotFoundError returns 404 with SESSION_NOT_FOUND code."""
        test_id = uuid.uuid4()
        response = client.get(f"/test/session-not-found/{test_id}")
        assert response.status_code == 404

        data = response.json()
        assert data["error_code"] == "SESSION_NOT_FOUND"
        assert str(test_id) in data["message"]
        assert data["safe_fallback"] == SAFE_FALLBACK

        error_resp = ErrorResponse.model_validate(data)
        assert error_resp.error_code == "SESSION_NOT_FOUND"

    def test_validation_error_returns_422_with_field_details(self, client: TestClient) -> None:
        """Verify Pydantic RequestValidationError returns 422 with field-level details."""
        # Empty body
        response = client.post("/test/validation", json={})
        assert response.status_code == 422

        data = response.json()
        assert data["error_code"] == "VALIDATION_ERROR"
        assert data["message"] == "Request validation failed."
        assert data["safe_fallback"] == SAFE_FALLBACK
        assert "details" in data
        assert isinstance(data["details"], list)
        assert len(data["details"]) > 0

        # String constraint violation (min_length=2)
        response_short = client.post("/test/validation", json={"message": "a"})
        assert response_short.status_code == 422
        short_data = response_short.json()
        assert short_data["error_code"] == "VALIDATION_ERROR"
        assert any("message" in str(err.get("loc", [])) for err in short_data["details"])

    def test_database_error_returns_503_database_unavailable(self, client: TestClient) -> None:
        """Verify database connection errors return 503 DATABASE_UNAVAILABLE."""
        response = client.get("/test/db-error")
        assert response.status_code == 503

        data = response.json()
        assert data["error_code"] == "DATABASE_UNAVAILABLE"
        assert data["message"] == "Database service is currently unavailable."
        assert data["safe_fallback"] == SAFE_FALLBACK

        # Ensure no internal DB details leaked
        assert "Connection refused" not in response.text
        assert "SELECT * FROM sessions" not in response.text

    def test_http_exceptions_preserve_status_and_format_error_response(self, client: TestClient) -> None:
        """Verify HTTPExceptions return mapped error_code and standard ErrorResponse."""
        resp_403 = client.get("/test/http-403")
        assert resp_403.status_code == 403
        data_403 = resp_403.json()
        assert data_403["error_code"] == "FORBIDDEN"
        assert "forbidden" in data_403["message"]
        assert data_403["safe_fallback"] == SAFE_FALLBACK

        resp_429 = client.get("/test/http-429")
        assert resp_429.status_code == 429
        data_429 = resp_429.json()
        assert data_429["error_code"] == "RATE_LIMITED"
        assert "Rate limit" in data_429["message"]
        assert data_429["safe_fallback"] == SAFE_FALLBACK

    def test_unmatched_route_returns_404_error_response(self, client: TestClient) -> None:
        """Verify unmatched route (Starlette 404) returns ErrorResponse."""
        response = client.get("/nonexistent/endpoint/path")
        assert response.status_code == 404

        data = response.json()
        assert data["error_code"] == "NOT_FOUND"
        assert data["safe_fallback"] == SAFE_FALLBACK


class TestPrivacyAndAuthErrors:
    """401/403/410/413/429 mapping and PHI-free error logging (finalization)."""

    def test_auth_and_limit_codes_keep_headers(self, client: TestClient) -> None:
        resp_401 = client.get("/test/http-401")
        assert resp_401.status_code == 401
        assert resp_401.json()["error_code"] == "UNAUTHORIZED"
        assert resp_401.headers["www-authenticate"] == "Bearer"

        resp_413 = client.get("/test/http-413")
        assert resp_413.status_code == 413
        assert resp_413.json()["error_code"] == "PAYLOAD_TOO_LARGE"

        resp_429 = client.get("/test/http-429")
        assert resp_429.headers["retry-after"] == "30"

    def test_erased_session_returns_410(self, client: TestClient) -> None:
        sid = uuid.uuid4()
        response = client.get(f"/test/session-erased/{sid}")
        assert response.status_code == 410
        data = response.json()
        assert data["error_code"] == "SESSION_ERASED"
        assert data["safe_fallback"] == SAFE_FALLBACK

    def test_message_not_found_returns_404(self, client: TestClient) -> None:
        response = client.get(f"/test/message-not-found/{uuid.uuid4()}")
        assert response.status_code == 404
        assert response.json()["error_code"] == "MESSAGE_NOT_FOUND"

    def test_validation_errors_do_not_echo_or_log_the_input(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import structlog
        from structlog.testing import capture_logs

        from medbridge.api.middleware import error_handler

        with capture_logs() as logs:  # "x" is too short → the error carries the input
            # The module logger is cached on first use (cache_logger_on_first_use) and
            # keeps the processor chain of that moment; an earlier test that reset
            # structlog (structlog.reset_defaults) would hide it from capture_logs.
            # A fresh proxy bound inside capture_logs is captured whatever ran before.
            monkeypatch.setattr(
                error_handler, "logger", structlog.get_logger(error_handler.__name__)
            )
            response = client.post("/test/validation", json={"message": "x"})
        assert response.status_code == 422
        for err in response.json()["details"]:
            assert "input" not in err and "ctx" not in err
        logged = [e for e in logs if e["event"] == "request_validation_error"]
        assert logged and all("input" not in err for err in logged[0]["errors"])
