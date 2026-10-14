import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.errors import ServiceError
from app.main import create_app
from tests.test_api import create_request


def ready_request(client, headers, external_id="42"):
    request_id = create_request(client, headers, external_id=external_id)
    client.app.state.repository.save_result(
        request_id,
        {"works": [], "requires_review": True, "note": "Проверить"},
    )
    return request_id


def test_successful_delivery_exact_mapping_and_no_duplicate(
    client_factory, bitrix_config, headers, caplog
):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"result": {"item": {"id": 42}}})

    client = client_factory(handler, **bitrix_config)
    request_id = ready_request(client, headers)
    endpoint = f"/api/v1/requests/{request_id}/send-to-bitrix"
    response = client.post(endpoint, headers=headers)
    assert response.status_code == 200
    assert response.json()["status"] == "delivered"
    assert response.json()["delivery_attempts"] == 1
    assert len(calls) == 1
    assert (
        str(calls[0].url) == "https://example.bitrix24.ru/rest/1/secret-token/crm.item.update.json"
    )
    payload = json.loads(calls[0].content)
    assert payload["entityTypeId"] == 2
    assert payload["id"] == "42"
    assert json.loads(payload["fields"]["ufCrmExampleResult"])["note"] == "Проверить"
    assert client.post(endpoint, headers=headers).status_code == 200
    assert len(calls) == 1
    assert "secret-token" not in caplog.text
    with pytest.raises(ServiceError) as locked:
        client.app.state.repository.save_result(request_id, {"changed": True})
    assert locked.value.status_code == 409


@pytest.mark.parametrize(
    "mode,status,code",
    [
        ("timeout", 504, "bitrix_timeout"),
        ("connection", 502, "bitrix_unavailable"),
        ("api_error", 502, "bitrix_api_error"),
        ("http_error", 502, "bitrix_http_error"),
        ("redirect", 502, "bitrix_http_error"),
        ("invalid_json", 502, "bitrix_invalid_response"),
        ("missing_result", 502, "bitrix_invalid_response"),
        ("false_result", 502, "bitrix_invalid_response"),
    ],
)
def test_delivery_failures_and_explicit_retry(
    client_factory, bitrix_config, headers, caplog, mode, status, code
):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) > 1:
            return httpx.Response(200, json={"result": True})
        if mode == "timeout":
            raise httpx.ReadTimeout("secret-token", request=request)
        if mode == "connection":
            raise httpx.ConnectError("secret-token", request=request)
        responses = {
            "api_error": httpx.Response(200, json={"error": "secret-token"}),
            "http_error": httpx.Response(500, text="secret-token"),
            "redirect": httpx.Response(302, headers={"Location": "https://untrusted.example"}),
            "invalid_json": httpx.Response(200, text="not json secret-token"),
            "missing_result": httpx.Response(200, json={"time": {}}),
            "false_result": httpx.Response(200, json={"result": False}),
        }
        return responses[mode]

    client = client_factory(handler, **bitrix_config)
    request_id = ready_request(client, headers)
    endpoint = f"/api/v1/requests/{request_id}/send-to-bitrix"
    response = client.post(endpoint, headers=headers)
    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
    assert "secret-token" not in response.text + caplog.text
    failed = client.get(f"/api/v1/requests/{request_id}", headers=headers).json()
    assert failed["status"] == "delivery_failed"
    assert failed["last_error"] == code
    assert len(calls) == 1
    retried = client.post(endpoint, headers=headers)
    assert retried.json()["status"] == "delivered"
    assert retried.json()["delivery_attempts"] == 2


def test_delivery_requires_result_and_id(client_factory, bitrix_config, headers):
    client = client_factory(**bitrix_config)
    request_id = create_request(client, headers)
    missing_result = client.post(f"/api/v1/requests/{request_id}/send-to-bitrix", headers=headers)
    assert missing_result.status_code == 409
    assert missing_result.json()["detail"]["code"] == "result_missing"
    request_id = ready_request(client, headers, external_id=None)
    missing_id = client.post(f"/api/v1/requests/{request_id}/send-to-bitrix", headers=headers)
    assert missing_id.status_code == 409
    assert missing_id.json()["detail"]["code"] == "external_id_missing"


def test_concurrent_delivery_claim(client_factory, bitrix_config, headers):
    started, release = Event(), Event()
    calls = []

    async def handler(request):
        import asyncio

        calls.append(request)
        started.set()
        await asyncio.to_thread(release.wait, 5)
        return httpx.Response(200, json={"result": True})

    client = client_factory(handler, **bitrix_config)
    request_id = ready_request(client, headers)
    endpoint = f"/api/v1/requests/{request_id}/send-to-bitrix"
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(client.post, endpoint, headers=headers)
        try:
            assert started.wait(5)
            assert client.post(endpoint, headers=headers).status_code == 409
        finally:
            release.set()
        assert future.result(timeout=5).status_code == 200
    assert len(calls) == 1


def test_interrupted_delivery_recovered_on_restart(client_factory, bitrix_config, headers):
    client = client_factory(**bitrix_config)
    request_id = ready_request(client, headers)
    client.app.state.repository.claim_delivery(request_id)
    with TestClient(create_app(client.app.state.settings)) as restarted:
        record = restarted.get(f"/api/v1/requests/{request_id}", headers=headers).json()
    assert record["status"] == "delivery_failed"
    assert record["last_error"] == "delivery_interrupted_check_bitrix_before_retry"
