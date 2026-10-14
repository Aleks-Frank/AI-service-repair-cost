import secrets
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import APIKeyHeader

from app.core.errors import ServiceError

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


def check_secret(provided: str | None, expected: str, name: str) -> None:
    if not expected:
        raise ServiceError(503, "authentication_not_configured", f"{name} is not configured")
    if provided is None or not secrets.compare_digest(provided.encode(), expected.encode()):
        raise ServiceError(401, "invalid_credentials", "Invalid credentials")


def require_api_key(
    request: Request, api_key: Annotated[str | None, Depends(api_key_header)]
) -> None:
    check_secret(api_key, request.app.state.settings.api_key.get_secret_value(), "API_KEY")
