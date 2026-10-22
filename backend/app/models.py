from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class RequestStatus(StrEnum):
    RECEIVED = "received"
    QUEUED = "queued"
    PROCESSING = "processing"
    PROCESSING_FAILED = "processing_failed"
    RESULT_READY = "result_ready"
    SENDING = "sending"
    DELIVERED = "delivered"
    DELIVERY_FAILED = "delivery_failed"


class CreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    external_id: str | None = Field(default=None, min_length=1, max_length=200)
    data: dict[str, JsonValue] = Field(
        min_length=1,
        examples=[{"description": "Заменить 1 неисправную розетку.", "images": []}],
    )


class RequestRecord(BaseModel):
    id: UUID
    source: Literal["api", "bitrix"]
    external_id: str | None
    status: RequestStatus
    payload: dict[str, JsonValue]
    result: dict[str, JsonValue] | None
    delivery_attempts: int
    last_error: str | None
    created_at: datetime
    updated_at: datetime


class Receipt(BaseModel):
    request_id: UUID
    status: RequestStatus
    duplicate: bool


class RequestPage(BaseModel):
    items: list[RequestRecord]
    limit: int
    offset: int
