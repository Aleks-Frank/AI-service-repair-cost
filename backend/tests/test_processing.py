import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import Settings
from app.core.errors import ServiceError
from app.main import create_app
from app.models import RequestStatus
from app.repository import RequestRepository
from tests.conftest import API_KEY


class RecordingAgent:
    def __init__(self, *, error=None, block=False):
        self.calls = []
        self.error = error
        self.started = Event()
        self.release = Event()
        self.closed = False
        if not block:
            self.release.set()

    def __call__(self, data):
        self.calls.append(data)
        self.started.set()
        if not self.release.wait(5):
            raise RuntimeError("Test agent was not released")
        if self.error:
            raise self.error
        return {"status": "draft", "works": [], "summary": "Тестовый результат"}

    def close(self):
        self.closed = True


def wait_for_status(client, headers, request_id, expected):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/requests/{request_id}", headers=headers)
        assert response.status_code == 200, response.text
        record = response.json()
        if record["status"] == expected:
            return record
        time.sleep(0.01)
    pytest.fail(f"Expected {expected}, got {record}")


def test_background_processing_keeps_api_available_and_deduplicates(client_factory, headers):
    agent = RecordingAgent(block=True)
    client = client_factory(agent=agent, ai_enabled=True, ai_poll_seconds=0.01)
    headers = {**headers, "Idempotency-Key": "case-1"}
    body = {"external_id": "42", "data": {"description": "Протечка трубы"}}
    try:
        response = client.post("/api/v1/requests", headers=headers, json=body)
        assert response.status_code == 202
        request_id = response.json()["request_id"]
        assert agent.started.wait(3)
        assert client.get("/health").status_code == 200
        wait_for_status(client, headers, request_id, "processing")
        duplicate = client.post("/api/v1/requests", headers=headers, json=body)
        assert duplicate.json()["duplicate"] is True
        assert duplicate.json()["request_id"] == request_id
        assert (
            client.post(f"/api/v1/requests/{request_id}/retry", headers=headers).status_code == 409
        )
    finally:
        agent.release.set()
    record = wait_for_status(client, headers, request_id, "result_ready")
    assert record["result"]["status"] == "draft"
    assert len(agent.calls) == 1
    assert record["delivery_attempts"] == 0


def test_images_and_object_context_reach_agent(client_factory, headers):
    image = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(image, format="PNG")
    encoded = base64.b64encode(image.getvalue()).decode()
    agent = RecordingAgent()
    client = client_factory(agent=agent, ai_enabled=True, ai_poll_seconds=0.01)
    response = client.post(
        "/api/v1/requests",
        headers=headers,
        json={
            "data": {
                "description": "Протечка",
                "object": {"system": "ХВС"},
                "images": [encoded],
            }
        },
    )
    assert response.status_code == 202
    wait_for_status(client, headers, response.json()["request_id"], "result_ready")
    assert agent.calls[0].image_bytes() == [image.getvalue()]
    assert "ХВС" in agent.calls[0].description_with_context()


@pytest.mark.parametrize(
    "data,code",
    [
        ({"unconfirmed_field": "x"}, "invalid_agent_input"),
        ({"description": "   "}, "invalid_agent_input"),
        ({"description": 42}, "invalid_agent_input"),
        ({"description": "x" * 4001}, "invalid_agent_input"),
        ({"description": "x", "object": {"value": "x" * 4000}}, "invalid_agent_input"),
        ({"description": "x", "images": ["a"] * 5}, "invalid_agent_input"),
        ({"description": "x", "images": ["https://example.com/image.jpg"]}, "invalid_image"),
        ({"description": "x", "images": ["bm90IGEgcGhvdG8="]}, "invalid_image"),
        ({"description": "x", "images": ["%%%"]}, "invalid_image"),
    ],
)
def test_invalid_input_does_not_enter_queue(client_factory, headers, data, code):
    agent = RecordingAgent()
    client = client_factory(agent=agent, ai_enabled=True)
    response = client.post("/api/v1/requests", headers=headers, json={"data": data})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == code
    assert client.get("/api/v1/requests", headers=headers).json()["items"] == []
    assert not agent.calls


@pytest.mark.parametrize(
    "error,code",
    [
        (ServiceError(502, "llm_failed", "sensitive text"), "llm_failed"),
        (RuntimeError("sensitive text"), "processing_internal_error"),
    ],
)
def test_failure_and_retry(client_factory, headers, caplog, error, code):
    agent = RecordingAgent(error=error)
    client = client_factory(agent=agent, ai_enabled=True, ai_poll_seconds=0.01)
    response = client.post(
        "/api/v1/requests", headers=headers, json={"data": {"description": "Протечка"}}
    )
    request_id = response.json()["request_id"]
    record = wait_for_status(client, headers, request_id, "processing_failed")
    assert record["last_error"] == code
    assert record["result"] is None
    assert "sensitive text" not in caplog.text
    agent.error = None
    assert client.post(f"/api/v1/requests/{request_id}/retry", headers=headers).status_code == 202
    record = wait_for_status(client, headers, request_id, "result_ready")
    assert record["last_error"] is None
    assert len(agent.calls) == 2


def test_queue_capacity_and_duplicate_at_capacity(client_factory, headers):
    agent = RecordingAgent(block=True)
    client = client_factory(agent=agent, ai_enabled=True, ai_queue_limit=1, ai_poll_seconds=0.01)
    headers = {**headers, "Idempotency-Key": "one"}
    body = {"data": {"description": "Протечка"}}
    try:
        assert client.post("/api/v1/requests", headers=headers, json=body).status_code == 202
        assert agent.started.wait(3)
        duplicate = client.post("/api/v1/requests", headers=headers, json=body)
        assert duplicate.status_code == 202 and duplicate.json()["duplicate"]
        response = client.post("/api/v1/requests", headers={"X-API-Key": API_KEY}, json=body)
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "queue_full"
    finally:
        agent.release.set()


def test_bitrix_pull_flow_no_outbound_call(client_factory, event, headers):
    agent = RecordingAgent()
    client = client_factory(
        agent=agent, ai_enabled=True, ai_poll_seconds=0.01, bitrix_input_path="data.case"
    )
    event["data"]["case"] = {"description": "Протечка трубы"}
    first = client.post("/api/v1/webhooks/bitrix", json=event)
    assert first.status_code == 202
    request_id = first.json()["request_id"]
    result = wait_for_status(client, headers, request_id, "result_ready")
    assert result["external_id"] == "42"
    assert result["source"] == "bitrix"
    assert "auth" not in result["payload"]
    assert result["delivery_attempts"] == 0
    second = client.post("/api/v1/webhooks/bitrix", json=event)
    assert second.json()["duplicate"] is True
    assert len(agent.calls) == 1


def test_id_only_webhook_does_not_invent_case(client_factory, event):
    client = client_factory(agent=RecordingAgent(), ai_enabled=True)
    response = client.post("/api/v1/webhooks/bitrix", json=event)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_agent_input"


def test_atomic_processing_claim_and_recovery(tmp_path, headers):
    repository = RequestRepository(tmp_path / "requests.sqlite3")
    repository.initialize()
    interrupted, _ = repository.create("api", None, {"description": "Первый"}, None, queue_limit=10)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda _: repository.claim_processing(), range(2)))
    assert sum(record is not None for record in claims) == 1
    queued, _ = repository.create("api", None, {"description": "Второй"}, None, queue_limit=10)
    agent = RecordingAgent()
    settings = Settings(
        _env_file=None,
        api_key=API_KEY,
        database_path=repository.path,
        ai_enabled=True,
        ai_poll_seconds=0.01,
    )
    with TestClient(create_app(settings, agent=agent)) as client:
        record = wait_for_status(client, headers, str(interrupted.id), "processing_failed")
        assert record["last_error"] == "processing_interrupted"
        wait_for_status(client, headers, str(queued.id), "result_ready")
    assert agent.closed
    assert [data.description for data in agent.calls] == ["Второй"]


def test_invalid_agent_result_is_not_saved(client_factory, headers):
    class InvalidAgent(RecordingAgent):
        def __call__(self, data):
            return {"amount": float("nan")}

    client = client_factory(agent=InvalidAgent(), ai_enabled=True, ai_poll_seconds=0.01)
    response = client.post(
        "/api/v1/requests", headers=headers, json={"data": {"description": "Протечка"}}
    )
    record = wait_for_status(client, headers, response.json()["request_id"], "processing_failed")
    assert record["last_error"] == "agent_invalid_result"
    assert record["result"] is None


def test_retry_disabled(client, headers):
    response = client.post(
        "/api/v1/requests", headers=headers, json={"data": {"description": "Протечка"}}
    )
    request_id = response.json()["request_id"]
    assert client.post(f"/api/v1/requests/{request_id}/retry", headers=headers).status_code == 503


def test_existing_received_request_can_be_queued(tmp_path):
    repository = RequestRepository(tmp_path / "requests.sqlite3")
    repository.initialize()
    record, _ = repository.create("api", None, {"description": "Протечка"}, None)
    queued = repository.retry_processing(str(record.id), 1)
    assert queued.status == RequestStatus.QUEUED
