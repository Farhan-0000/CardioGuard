"""
Global Error Handling Middleware and Exception Handlers (TASK-28, Module M-03).

Constitutional Invariant Non-Negotiable #7:
"The backend must never return raw 500 errors to the patient. Unhandled exceptions
must be caught, logged server-side with stack traces, and converted into structured
ErrorResponse payloads containing pre-vetted clinically safe fallback messages."

Privacy: handlers never log request bodies — validation errors are logged
without the offending ``input`` value, and every log event additionally
passes through the PII-scrubbing structlog processor.

Technical Specification Part II §9.
"""
from typing import Any

import structlog
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from medbridge.api.schemas.responses import ErrorResponse
from medbridge.state.session_manager import (
    MessageNotFoundError,
    SessionErasedError,
    SessionNotFoundError,
)

logger = structlog.get_logger(__name__)

SAFE_FALLBACK = (
    "We're unable to process your question right now. "
    "For immediate health concerns, please contact your healthcare provider "
    "or call emergency services."
)

HTTP_ERROR_CODES: dict[int, str] = {
    status.HTTP_400_BAD_REQUEST: "BAD_REQUEST",
    status.HTTP_401_UNAUTHORIZED: "UNAUTHORIZED",
    status.HTTP_403_FORBIDDEN: "FORBIDDEN",
    status.HTTP_404_NOT_FOUND: "NOT_FOUND",
    status.HTTP_405_METHOD_NOT_ALLOWED: "METHOD_NOT_ALLOWED",
    status.HTTP_410_GONE: "GONE",
    status.HTTP_413_CONTENT_TOO_LARGE: "PAYLOAD_TOO_LARGE",
    status.HTTP_429_TOO_MANY_REQUESTS: "RATE_LIMITED",
    status.HTTP_503_SERVICE_UNAVAILABLE: "SERVICE_UNAVAILABLE",
}

# Keys of a pydantic error that may echo user input (never logged or returned).
_ERROR_ECHO_KEYS = ("input", "ctx", "url")


def _error_json(status_code: int, error_code: str, message: str,
                details: Any = None, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error_code=error_code,
            message=message,
            details=details,
            safe_fallback=SAFE_FALLBACK,
        ).model_dump(mode="json"),
        headers=headers,
    )


def _sanitize_validation_errors(errors: Any) -> list[dict[str, Any]]:
    """Drop the fields of pydantic errors that echo the (possibly PHI-bearing) input."""
    return [
        {k: v for k, v in err.items() if k not in _ERROR_ECHO_KEYS}
        for err in errors
        if isinstance(err, dict)
    ]


async def session_not_found_handler(
    request: Request,
    exc: SessionNotFoundError,
) -> JSONResponse:
    """Handle SessionNotFoundError, returning HTTP 404 with structured ErrorResponse."""
    logger.warning(
        "session_not_found",
        session_id=str(exc.session_id),
        path=request.url.path,
    )
    return _error_json(
        status.HTTP_404_NOT_FOUND, "SESSION_NOT_FOUND", f"Session {exc.session_id} not found."
    )


async def session_erased_handler(
    request: Request,
    exc: SessionErasedError,
) -> JSONResponse:
    """Handle SessionErasedError (crypto-shredded session), returning HTTP 410."""
    logger.info("session_erased_access", session_id=str(exc.session_id), path=request.url.path)
    return _error_json(
        status.HTTP_410_GONE, "SESSION_ERASED",
        f"Session {exc.session_id} was erased at the user's request.",
    )


async def message_not_found_handler(
    request: Request,
    exc: MessageNotFoundError,
) -> JSONResponse:
    """Handle MessageNotFoundError, returning HTTP 404."""
    logger.warning("message_not_found", message_id=str(exc.message_id), path=request.url.path)
    return _error_json(
        status.HTTP_404_NOT_FOUND, "MESSAGE_NOT_FOUND", f"Message {exc.message_id} not found."
    )


async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle RequestValidationError, returning HTTP 422 with field-level details."""
    errors = _sanitize_validation_errors(exc.errors())
    logger.warning(
        "request_validation_error",
        path=request.url.path,
        errors=errors,
    )
    return _error_json(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "VALIDATION_ERROR",
        "Request validation failed.",
        details=errors,
    )


async def database_error_handler(
    request: Request,
    exc: SQLAlchemyError,
) -> JSONResponse:
    """Handle SQLAlchemy errors, returning HTTP 503 DATABASE_UNAVAILABLE."""
    logger.error(
        "database_unavailable",
        path=request.url.path,
        error_type=type(exc).__name__,
        error=str(exc),
        exc_info=True,
    )
    return _error_json(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "DATABASE_UNAVAILABLE",
        "Database service is currently unavailable.",
    )


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """Handle standard HTTP exceptions (incl. 401/403/413/429) as ErrorResponse."""
    error_code = HTTP_ERROR_CODES.get(exc.status_code, f"HTTP_{exc.status_code}")

    logger.warning(
        "http_exception",
        status_code=exc.status_code,
        error_code=error_code,
        detail=str(exc.detail),
        path=request.url.path,
    )
    return _error_json(
        exc.status_code, error_code, str(exc.detail), headers=getattr(exc, "headers", None)
    )


async def global_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Catch-all global exception handler (Non-Negotiable #7).

    Guarantees no raw 500 error pages or unhandled stack traces are returned to clients.
    Logs the full exception trace server-side and returns a structured ErrorResponse.
    """
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        error_type=type(exc).__name__,
        error=str(exc),
        exc_info=True,
    )
    return _error_json(
        status.HTTP_500_INTERNAL_SERVER_ERROR, "INTERNAL_ERROR", "An internal error occurred."
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all global exception handlers onto a FastAPI application instance."""
    app.add_exception_handler(SessionErasedError, session_erased_handler)  # type: ignore[arg-type]
    app.add_exception_handler(SessionNotFoundError, session_not_found_handler)  # type: ignore[arg-type]
    app.add_exception_handler(MessageNotFoundError, message_not_found_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(SQLAlchemyError, database_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, global_exception_handler)
