import json
from typing import Any

import httpx

from app.core.config import Settings
from app.core.errors import ServiceError
from app.models import RequestRecord


class BitrixClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.client = client

    def ensure_configured(self) -> None:
        if not all(
            (
                self.settings.bitrix_webhook_url.get_secret_value(),
                self.settings.bitrix_result_method,
                self.settings.bitrix_result_field,
            )
        ):
            raise ServiceError(
                503, "bitrix_not_configured", "Configure the inbound webhook and result mapping"
            )

    def result_parameters(self, record: RequestRecord) -> dict[str, Any]:
        return {
            **self.settings.bitrix_result_static_params,
            self.settings.bitrix_result_id_param: record.external_id,
            "fields": {
                self.settings.bitrix_result_field: json.dumps(record.result, ensure_ascii=False)
            },
        }

    async def send_result(self, record: RequestRecord) -> None:
        self.ensure_configured()
        base = self.settings.bitrix_webhook_url.get_secret_value().rstrip("/")
        url = f"{base}/{self.settings.bitrix_result_method}.json"
        try:
            response = await self.client.post(url, json=self.result_parameters(record))
        except httpx.TimeoutException:
            raise ServiceError(
                504, "bitrix_timeout", "Bitrix timed out; check the object before retrying"
            ) from None
        except httpx.RequestError:
            raise ServiceError(502, "bitrix_unavailable", "Could not reach Bitrix") from None
        if not response.is_success:
            raise ServiceError(502, "bitrix_http_error", "Bitrix returned an HTTP error")
        try:
            body = response.json()
        except ValueError:
            raise ServiceError(
                502, "bitrix_invalid_response", "Bitrix returned invalid JSON"
            ) from None
        if not isinstance(body, dict) or "error" in body:
            raise ServiceError(502, "bitrix_api_error", "Bitrix rejected the REST request")
        if "result" not in body or body["result"] is None or body["result"] is False:
            raise ServiceError(502, "bitrix_invalid_response", "Bitrix did not confirm delivery")
