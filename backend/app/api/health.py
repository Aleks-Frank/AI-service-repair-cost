import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_repository
from app.core.errors import ServiceError
from app.repository import RequestRepository

router = APIRouter(tags=["Проверка сервера"])


@router.get(
    "/health",
    summary="Проверить работу сервера и базы данных",
    response_description="Состояние сервера и базы данных",
)
def health(repository: Annotated[RequestRepository, Depends(get_repository)]) -> dict[str, str]:
    try:
        repository.check_health()
    except sqlite3.Error:
        raise ServiceError(503, "database_unavailable", "Database is unavailable") from None
    return {"status": "ok", "database": "ok"}
