import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, Request
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_repository
from app.core.errors import ServiceError
from app.core.security import require_api_key
from app.integrations.agent import agent_input
from app.models import (
    CreateRequest,
    Receipt,
    RequestPage,
    RequestRecord,
    RequestStatus,
)
from app.repository import RequestRepository

router = APIRouter(prefix="/requests", tags=["Заявки"], dependencies=[Depends(require_api_key)])
Repository = Annotated[RequestRepository, Depends(get_repository)]
logger = logging.getLogger(__name__)


@router.post(
    "",
    status_code=202,
    response_model=Receipt,
    summary="Создать заявку вручную или через API",
    response_description="Идентификатор, состояние заявки и признак повторного запроса",
)
def create_request(
    body: CreateRequest,
    repository: Repository,
    request: Request,
    idempotency_key: Annotated[
        str | None,
        Header(
            min_length=1,
            max_length=200,
            alias="Idempotency-Key",
        ),
    ] = None,
) -> Receipt:
    settings = request.app.state.settings
    if settings.ai_enabled:
        agent_input(body.data, "api", settings)
    record, duplicate = repository.create(
        "api",
        body.external_id,
        body.data,
        f"api:{idempotency_key}" if idempotency_key else None,
        queue_limit=settings.ai_queue_limit if settings.ai_enabled else None,
    )
    return Receipt(request_id=record.id, status=record.status, duplicate=duplicate)


@router.get(
    "",
    response_model=RequestPage,
    summary="Получить список сохранённых заявок",
    response_description="Список заявок с параметрами пагинации",
)
def list_requests(
    repository: Repository,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    status: RequestStatus | None = None,
) -> RequestPage:
    return RequestPage(items=repository.list(limit, offset, status), limit=limit, offset=offset)


@router.get(
    "/{request_id}",
    response_model=RequestRecord,
    summary="Получить заявку, её состояние и результат",
    response_description="Данные заявки, текущее состояние и сохранённый результат",
)
def get_request(request_id: UUID, repository: Repository) -> RequestRecord:
    return repository.get(str(request_id))


@router.post(
    "/{request_id}/retry",
    status_code=202,
    response_model=Receipt,
    summary="Повторить обработку заявки после ошибки",
    response_description="Идентификатор и состояние заявки",
)
def retry_processing(request_id: UUID, request: Request, repository: Repository) -> Receipt:
    settings = request.app.state.settings
    if not settings.ai_enabled:
        raise ServiceError(503, "agent_disabled", "AI processing is disabled")
    record = repository.get(str(request_id))
    agent_input(record.payload, record.source, settings)
    record = repository.retry_processing(str(request_id), settings.ai_queue_limit)
    return Receipt(request_id=record.id, status=record.status, duplicate=False)


@router.post(
    "/{request_id}/send-to-bitrix",
    response_model=RequestRecord,
    summary="Отправить сохранённый результат в Битрикс24",
    response_description="Заявка с обновлённым состоянием доставки",
)
async def send_to_bitrix(
    request_id: UUID, request: Request, repository: Repository
) -> RequestRecord:
    request_id_str = str(request_id)
    current = await run_in_threadpool(repository.get, request_id_str)
    if current.status == RequestStatus.DELIVERED:
        return current
    bitrix = request.app.state.bitrix
    bitrix.ensure_configured()
    record, claimed = await run_in_threadpool(repository.claim_delivery, request_id_str)
    if not claimed:
        return record
    try:
        await bitrix.send_result(record)
    except ServiceError as error:
        await run_in_threadpool(repository.finish_delivery, request_id_str, error.code)
        logger.warning("Bitrix delivery failed request_id=%s code=%s", request_id, error.code)
        raise
    except Exception:
        await run_in_threadpool(
            repository.finish_delivery, request_id_str, "delivery_internal_error"
        )
        logger.error("Unexpected delivery error request_id=%s", request_id)
        raise ServiceError(502, "delivery_internal_error", "Delivery failed") from None
    return await run_in_threadpool(repository.finish_delivery, request_id_str)
