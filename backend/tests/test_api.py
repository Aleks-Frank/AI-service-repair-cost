import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import create_app


def create_request(client, headers, **overrides):
    body = {"external_id": "42", "data": {"description": "Протечка трубы"}, **overrides}
    response = client.post("/api/v1/requests", headers=headers, json=body)
    assert response.status_code == 202, response.text
    return response.json()["request_id"]


def test_health_and_openapi(client):
    assert client.get("/health").json() == {"status": "ok", "database": "ok"}
    schema = client.get("/openapi.json").json()
    assert "/api/v1/webhooks/bitrix" in schema["paths"]
    assert client.get("/docs").status_code == 200


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "incorrect"}])
def test_internal_authentication(client, headers):
    assert client.get("/api/v1/requests", headers=headers).status_code == 401


def test_unconfigured_authentication_fails_closed(client_factory, event):
    client = client_factory(api_key="", bitrix_outbound_token="")
    assert client.get("/api/v1/requests").status_code == 503
    assert client.post("/api/v1/webhooks/bitrix", json=event).status_code == 503


def test_create_get_filter_save_result(client, headers):
    request_id = create_request(client, headers)
    response = client.get(f"/api/v1/requests/{request_id}", headers=headers)
    assert response.json()["payload"]["description"] == "Протечка трубы"
    result = {"works": [], "requires_review": True, "notes": ["Test result only"]}
    client.app.state.repository.save_result(request_id, result)
    saved = client.get(f"/api/v1/requests/{request_id}", headers=headers)
    assert saved.status_code == 200
    assert saved.json()["status"] == "result_ready"
    assert saved.json()["result"] == result
    assert client.get("/api/v1/requests?status=received", headers=headers).json()["items"] == []
    page = client.get("/api/v1/requests?status=result_ready&limit=1", headers=headers).json()
    assert page["items"][0]["id"] == request_id


def test_idempotency_and_conflict(client, headers):
    headers = {**headers, "Idempotency-Key": "same-operation"}
    body = {"external_id": "42", "data": {"new_field": {"nested": [1, "two"]}}}
    first = client.post("/api/v1/requests", headers=headers, json=body)
    second = client.post("/api/v1/requests", headers=headers, json=body)
    assert second.json()["duplicate"] is True
    assert first.json()["request_id"] == second.json()["request_id"]
    body["external_id"] = "43"
    assert client.post("/api/v1/requests", headers=headers, json=body).status_code == 409


def test_validation_not_found_and_missing_mapping(client, headers):
    assert client.get(f"/api/v1/requests/{uuid4()}", headers=headers).status_code == 404
    assert client.get("/api/v1/requests/not-a-uuid", headers=headers).status_code == 422
    assert client.get("/api/v1/requests?limit=0", headers=headers).status_code == 422
    assert client.post("/api/v1/requests", headers=headers, json={"data": {}}).status_code == 422
    request_id = create_request(client, headers)
    response = client.post(f"/api/v1/requests/{request_id}/send-to-bitrix", headers=headers)
    assert response.status_code == 503
    assert (
        client.get(f"/api/v1/requests/{request_id}", headers=headers).json()["status"] == "received"
    )


def test_validation_does_not_echo_input(client, headers):
    response = client.post(
        "/api/v1/requests",
        headers=headers,
        json={"external_id": {"secret": "never-echo-me"}, "data": {"x": 1}},
    )
    assert response.status_code == 422
    assert "never-echo-me" not in response.text


def test_nonfinite_result_rejected(client, headers):
    request_id = create_request(client, headers)
    with pytest.raises(ValidationError):
        client.app.state.repository.save_result(request_id, {"total": float("nan")})
    record = client.get(f"/api/v1/requests/{request_id}", headers=headers).json()
    assert record["status"] == "received"
    assert record["result"] is None


def test_http_result_submission_removed(client, headers):
    request_id = create_request(client, headers)
    schema = client.get("/openapi.json").json()
    assert "/api/v1/requests/{request_id}/result" not in schema["paths"]
    assert "SubmitResult" not in schema["components"]["schemas"]
    response = client.post(
        f"/api/v1/requests/{request_id}/result",
        headers=headers,
        json={"result": {"demo": True}},
    )
    assert response.status_code == 404


def test_body_size_limit_including_no_content_length(client_factory, headers):
    client = client_factory(max_request_bytes=1024)
    body = json.dumps({"data": {"description": "x" * 2048}}).encode()
    response = client.post(
        "/api/v1/requests",
        headers={**headers, "Content-Type": "application/json"},
        content=iter([body[:1000], body[1000:]]),
    )
    assert response.status_code == 413


def test_persistence_after_restart(client_factory, headers):
    client = client_factory()
    request_id = create_request(client, headers)
    settings = client.app.state.settings
    with TestClient(create_app(settings)) as restarted:
        assert restarted.get(f"/api/v1/requests/{request_id}", headers=headers).status_code == 200
