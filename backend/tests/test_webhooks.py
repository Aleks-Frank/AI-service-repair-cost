from urllib.parse import urlencode

import pytest

from tests.conftest import OUTBOUND_TOKEN


def test_json_event_deduplicated_and_auth_removed(client, event, headers):
    first = client.post("/api/v1/webhooks/bitrix", json=event)
    assert first.status_code == 202
    second = client.post("/api/v1/webhooks/bitrix", json=event)
    assert second.json()["duplicate"] is True
    assert second.json()["request_id"] == first.json()["request_id"]
    record = client.get(f"/api/v1/requests/{first.json()['request_id']}", headers=headers).json()
    assert record["external_id"] == "42"
    assert "auth" not in record["payload"]
    assert record["payload"]["data"]["unconfirmed_field"] == "kept"
    assert OUTBOUND_TOKEN not in str(record)
    event["ts"] = "1791500001"
    next_event = client.post("/api/v1/webhooks/bitrix", json=event)
    assert next_event.json()["request_id"] != first.json()["request_id"]


def test_bitrix_form_with_nested_fields(client, headers):
    body = urlencode(
        [
            ("event", "ONCRMDEALADD"),
            ("data[FIELDS][ID]", "42"),
            ("data[DESCRIPTION]", "Протечка + фото"),
            ("data[PHOTOS][]", "first"),
            ("data[PHOTOS][]", "second"),
            ("auth[application_token]", OUTBOUND_TOKEN),
            ("ts", "1791500000"),
        ]
    )
    response = client.post(
        "/api/v1/webhooks/bitrix",
        content=body,
        headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"},
    )
    assert response.status_code == 202, response.text
    record = client.get(f"/api/v1/requests/{response.json()['request_id']}", headers=headers).json()
    assert record["external_id"] == "42"
    assert record["payload"]["data"]["DESCRIPTION"] == "Протечка + фото"
    assert record["payload"]["data"]["PHOTOS"] == ["first", "second"]


@pytest.mark.parametrize("token", [None, "wrong-token", [OUTBOUND_TOKEN, "wrong"]])
def test_bad_webhook_auth(client, event, token):
    event["auth"]["application_token"] = token
    assert client.post("/api/v1/webhooks/bitrix", json=event).status_code == 401


def test_duplicate_form_token_rejected(client):
    response = client.post(
        "/api/v1/webhooks/bitrix",
        content=urlencode(
            [
                ("event", "ONCRMDEALADD"),
                ("data[FIELDS][ID]", "42"),
                ("auth[application_token]", "wrong"),
                ("auth[application_token]", OUTBOUND_TOKEN),
            ]
        ),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 401


@pytest.mark.parametrize(
    "body,content_type,status",
    [
        ("[1,2]", "application/json", 400),
        ("{bad", "application/json", 400),
        ('{"value":NaN}', "application/json", 400),
        ("event=x&data=1&data[ID]=2", "application/x-www-form-urlencoded", 400),
        ("data[][ID]=2", "application/x-www-form-urlencoded", 400),
        ("anything", "text/plain", 415),
    ],
)
def test_malformed_webhooks(client, body, content_type, status):
    response = client.post(
        "/api/v1/webhooks/bitrix", content=body, headers={"Content-Type": content_type}
    )
    assert response.status_code == status


def test_event_without_timestamp_is_not_deduplicated(client, event):
    event.pop("ts")
    first = client.post("/api/v1/webhooks/bitrix", json=event).json()
    second = client.post("/api/v1/webhooks/bitrix", json=event).json()
    assert first["request_id"] != second["request_id"]


def test_configurable_id_path_and_unknown_event(client_factory, event, headers):
    client = client_factory(bitrix_entity_id_path="data.item.id")
    event["event"] = "FUTURE_EVENT"
    event["data"] = {"item": {"id": 100}}
    response = client.post("/api/v1/webhooks/bitrix", json=event)
    record = client.get(f"/api/v1/requests/{response.json()['request_id']}", headers=headers).json()
    assert record["external_id"] == "100"


def test_unknown_id_still_preserves_event(client, event, headers):
    event["data"] = {"custom": "data"}
    response = client.post("/api/v1/webhooks/bitrix", json=event)
    assert response.status_code == 202
    record = client.get(f"/api/v1/requests/{response.json()['request_id']}", headers=headers).json()
    assert record["external_id"] is None
    assert record["payload"]["data"] == {"custom": "data"}


@pytest.mark.parametrize("patch", [{"event": ""}, {"data": []}, {"event": 15}])
def test_invalid_event_envelope(client, event, patch):
    event.update(patch)
    assert client.post("/api/v1/webhooks/bitrix", json=event).status_code == 422
