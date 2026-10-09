"""
Frontend API Client Service (TASK-25, Module M-20).

Provides synchronous and asynchronous communication with the MedBridge FastAPI backend,
handling network timeouts, connection errors, and structured safe fallback mapping.

Session routes are protected by signed session tokens: ``create_session`` captures the
``session_token`` returned by ``POST /api/sessions`` and every later call for that session
sends ``Authorization: Bearer <token>``.

Technical Specification Part I §4, §9, §10.
"""
import os
import uuid
from typing import Any

import httpx

from medbridge.api.schemas.responses import (
    HealthResponse,
    HistoryResponse,
    MessageResponse,
)

DEFAULT_API_URL = os.getenv("MEDBRIDGE_API_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = float(os.getenv("MEDBRIDGE_API_TIMEOUT", "30.0"))  # Technical Specification Part I §9

DEFAULT_SAFE_FALLBACK = (
    "We're unable to process your question right now. "
    "Please try again, or consult your healthcare provider."
)


class APIError(Exception):
    """Base exception for API client operations."""

    def __init__(
        self,
        message: str,
        safe_fallback: str = DEFAULT_SAFE_FALLBACK,
        error_code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.safe_fallback = safe_fallback
        self.error_code = error_code
        self.status_code = status_code


class APIConnectionError(APIError):
    """Raised when the backend server is unreachable."""

    def __init__(self, message: str = "Unable to connect to the MedBridge service.") -> None:
        super().__init__(
            message=message,
            safe_fallback=(
                "Unable to connect to the MedBridge service. "
                "Please verify the backend server is running and try again."
            ),
            error_code="CONNECTION_ERROR",
        )


class APITimeoutError(APIError):
    """Raised when a request exceeds the 15-second timeout limit."""

    def __init__(self, message: str = "Request to MedBridge timed out.") -> None:
        super().__init__(
            message=message,
            safe_fallback="Your question took too long to process. Please try asking again.",
            error_code="TIMEOUT_ERROR",
        )


class APIStatusError(APIError):
    """Raised when the API returns an HTTP 4xx or 5xx response."""

    def __init__(
        self,
        status_code: int,
        error_code: str | None,
        message: str,
        safe_fallback: str,
    ) -> None:
        super().__init__(
            message=message,
            safe_fallback=safe_fallback,
            error_code=error_code,
            status_code=status_code,
        )


def _parse_error_response(response: httpx.Response) -> APIStatusError:
    """Extract structured ErrorResponse fields or construct safe fallback."""
    status_code = response.status_code
    try:
        data = response.json()
        error_code = data.get("error_code")
        message = data.get("message", f"HTTP {status_code} error from server.")
        safe_fallback = data.get("safe_fallback", DEFAULT_SAFE_FALLBACK)
    except ValueError:
        error_code = f"HTTP_{status_code}"
        raw = response.text
        message = f"HTTP {status_code}: {raw[:200]}{'...' if len(raw) > 200 else ''}"
        safe_fallback = DEFAULT_SAFE_FALLBACK

    return APIStatusError(
        status_code=status_code,
        error_code=error_code,
        message=message,
        safe_fallback=safe_fallback,
    )


class APIClient:
    """Client for synchronous and asynchronous interaction with the MedBridge REST API."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = (base_url or DEFAULT_API_URL).rstrip("/")
        self.timeout = timeout
        self._client: httpx.Client = httpx.Client(timeout=self.timeout)
        self._session_tokens: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Session tokens
    # ------------------------------------------------------------------

    def set_session_token(self, session_id: uuid.UUID | str, token: str | None) -> None:
        """Remember (or forget, when ``token`` is falsy) the bearer token of a session."""
        if token:
            self._session_tokens[str(session_id)] = token
        else:
            self._session_tokens.pop(str(session_id), None)

    def get_session_token(self, session_id: uuid.UUID | str) -> str | None:
        """Return the bearer token captured for ``session_id``, if any."""
        return self._session_tokens.get(str(session_id))

    def _auth_headers(self, session_id: uuid.UUID | str) -> dict[str, str]:
        """``Authorization`` header for session-scoped routes (empty if no token)."""
        token = self.get_session_token(session_id)
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _remember_session(self, data: dict[str, Any]) -> uuid.UUID:
        """Parse a SessionResponse body and store its token."""
        session_id = uuid.UUID(data["session_id"])
        self.set_session_token(session_id, data.get("session_token"))
        return session_id

    def _get_client(self) -> httpx.Client:
        """Return the active HTTP client, recreating it if closed."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self) -> None:
        """Close the underlying persistent HTTP client."""
        if self._client is not None and not self._client.is_closed:
            self._client.close()

    def __enter__(self) -> "APIClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Synchronous Methods (Streamlit UI execution)
    # ------------------------------------------------------------------

    def create_session(self) -> uuid.UUID:
        """
        Create a new conversation session on the backend and capture its bearer token.

        Returns:
            Newly initialized session UUID.
        Raises:
            APIConnectionError, APITimeoutError, APIStatusError
        """
        url = f"{self.base_url}/api/sessions"
        try:
            client = self._get_client()
            resp = client.post(url)
        except httpx.ConnectError as exc:
            raise APIConnectionError(str(exc)) from exc
        except httpx.TimeoutException as exc:
            raise APITimeoutError(str(exc)) from exc
        except httpx.RequestError as exc:
            raise APIConnectionError(str(exc)) from exc

        if resp.is_error:
            raise _parse_error_response(resp)

        return self._remember_session(resp.json())

    def send_message(
        self,
        session_id: uuid.UUID | str,
        message: str,
    ) -> MessageResponse:
        """
        Submit a clinical query to the MedBridge pipeline for a given session.

        Returns:
            Validated MessageResponse containing action, response_text, citations.
        Raises:
            APIConnectionError, APITimeoutError, APIStatusError
        """
        url = f"{self.base_url}/api/sessions/{session_id}/messages"
        payload = {"message": message}
        try:
            client = self._get_client()
            resp = client.post(url, json=payload, headers=self._auth_headers(session_id))
        except httpx.ConnectError as exc:
            raise APIConnectionError(str(exc)) from exc
        except httpx.TimeoutException as exc:
            raise APITimeoutError(str(exc)) from exc
        except httpx.RequestError as exc:
            raise APIConnectionError(str(exc)) from exc

        if resp.is_error:
            raise _parse_error_response(resp)

        return MessageResponse.model_validate(resp.json())

    def get_history(
        self,
        session_id: uuid.UUID | str,
    ) -> HistoryResponse:
        """
        Retrieve conversation message history for a given session.

        Returns:
            Validated HistoryResponse containing chronological messages.
        Raises:
            APIConnectionError, APITimeoutError, APIStatusError
        """
        url = f"{self.base_url}/api/sessions/{session_id}/history"
        try:
            client = self._get_client()
            resp = client.get(url, headers=self._auth_headers(session_id))
        except httpx.ConnectError as exc:
            raise APIConnectionError(str(exc)) from exc
        except httpx.TimeoutException as exc:
            raise APITimeoutError(str(exc)) from exc
        except httpx.RequestError as exc:
            raise APIConnectionError(str(exc)) from exc

        if resp.is_error:
            raise _parse_error_response(resp)

        return HistoryResponse.model_validate(resp.json())

    def health_check(self) -> HealthResponse:
        """
        Probe system health status (PostgreSQL + Qdrant).

        Returns:
            Validated HealthResponse.
        Raises:
            APIConnectionError, APITimeoutError, APIStatusError
        """
        url = f"{self.base_url}/health"
        try:
            client = self._get_client()
            resp = client.get(url)
        except httpx.ConnectError as exc:
            raise APIConnectionError(str(exc)) from exc
        except httpx.TimeoutException as exc:
            raise APITimeoutError(str(exc)) from exc
        except httpx.RequestError as exc:
            raise APIConnectionError(str(exc)) from exc

        # 503 is a valid health response (unhealthy status)
        if resp.status_code == 503:
            return HealthResponse.model_validate(resp.json())

        if resp.is_error:
            raise _parse_error_response(resp)

        return HealthResponse.model_validate(resp.json())

    # ------------------------------------------------------------------
    # Asynchronous Variants (for async callers and scripts)
    # ------------------------------------------------------------------

    async def acreate_session(self) -> uuid.UUID:
        """Asynchronously create a new conversation session."""
        url = f"{self.base_url}/api/sessions"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url)
        except httpx.ConnectError as exc:
            raise APIConnectionError(str(exc)) from exc
        except httpx.TimeoutException as exc:
            raise APITimeoutError(str(exc)) from exc
        except httpx.RequestError as exc:
            raise APIConnectionError(str(exc)) from exc

        if resp.is_error:
            raise _parse_error_response(resp)

        return self._remember_session(resp.json())

    async def asend_message(
        self,
        session_id: uuid.UUID | str,
        message: str,
    ) -> MessageResponse:
        """Asynchronously submit a patient message."""
        url = f"{self.base_url}/api/sessions/{session_id}/messages"
        payload = {"message": message}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload, headers=self._auth_headers(session_id))
        except httpx.ConnectError as exc:
            raise APIConnectionError(str(exc)) from exc
        except httpx.TimeoutException as exc:
            raise APITimeoutError(str(exc)) from exc
        except httpx.RequestError as exc:
            raise APIConnectionError(str(exc)) from exc

        if resp.is_error:
            raise _parse_error_response(resp)

        return MessageResponse.model_validate(resp.json())

    async def aget_history(
        self,
        session_id: uuid.UUID | str,
    ) -> HistoryResponse:
        """Asynchronously retrieve conversation history."""
        url = f"{self.base_url}/api/sessions/{session_id}/history"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url, headers=self._auth_headers(session_id))
        except httpx.ConnectError as exc:
            raise APIConnectionError(str(exc)) from exc
        except httpx.TimeoutException as exc:
            raise APITimeoutError(str(exc)) from exc
        except httpx.RequestError as exc:
            raise APIConnectionError(str(exc)) from exc

        if resp.is_error:
            raise _parse_error_response(resp)

        return HistoryResponse.model_validate(resp.json())

    async def ahealth_check(self) -> HealthResponse:
        """Asynchronously probe health status."""
        url = f"{self.base_url}/health"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url)
        except httpx.ConnectError as exc:
            raise APIConnectionError(str(exc)) from exc
        except httpx.TimeoutException as exc:
            raise APITimeoutError(str(exc)) from exc
        except httpx.RequestError as exc:
            raise APIConnectionError(str(exc)) from exc

        if resp.status_code == 503:
            return HealthResponse.model_validate(resp.json())

        if resp.is_error:
            raise _parse_error_response(resp)

        return HealthResponse.model_validate(resp.json())
