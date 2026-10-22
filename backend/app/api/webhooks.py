from typing import Annotated

from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_repository
from app.core.errors import ServiceError
from app.core.security import check_secret
from app.integrations.agent import agent_input
from app.integrations.webhook import get_path, read_webhook
from app.models import Receipt
from app.repository import RequestRepository, fingerprint

router = APIRouter(prefix="/webhooks", tags=["Обмен с Битрикс24"])


@router.post(
    "/bitrix",
    status_code=202,
    response_model=Receipt,
    summary="Принять событие от Битрикс24",
    response_description="Идентификатор сохранённой заявки, состояние и признак повтора",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {"schema": {"type": "object"}},
                "application/x-www-form-urlencoded": {"schema": {"type": "object"}},
            },
        }
    },
)
async def receive_bitrix(
    request: Request, repository: Annotated[RequestRepository, Depends(get_repository)]
) -> Receipt:
    payload = await read_webhook(request)
    settings = request.app.state.settings
    auth = payload.get("auth")
    token = auth.get("application_token") if isinstance(auth, dict) else None
    check_secret(
        token if isinstance(token, str) else None,
        settings.bitrix_outbound_token.get_secret_value(),
        "BITRIX_OUTBOUND_TOKEN",
    )
    event = payload.get("event")
    if not isinstance(event, str) or not event.strip() or len(event) > 200:
        raise ServiceError(422, "invalid_event", "A non-empty event name is required")
    if not isinstance(payload.get("data"), dict):
        raise ServiceError(422, "invalid_event_data", "Event data must be an object")
    external_id = get_path(payload, settings.bitrix_entity_id_path)
    payload.pop("auth", None)
    if settings.ai_enabled:
        await run_in_threadpool(agent_input, payload, "bitrix", settings)
    dedup_key = None
    event_time = payload.get("ts")
    if isinstance(event_time, (str, int)) and not isinstance(event_time, bool) and str(event_time):
        dedup_key = "bitrix:" + fingerprint(payload)
    record, duplicate = await run_in_threadpool(
        repository.create,
        "bitrix",
        external_id,
        payload,
        dedup_key,
        queue_limit=settings.ai_queue_limit if settings.ai_enabled else None,
    )
    return Receipt(request_id=record.id, status=record.status, duplicate=duplicate)
