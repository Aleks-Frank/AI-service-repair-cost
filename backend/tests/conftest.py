from contextlib import ExitStack

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

API_KEY = "test-internal-key-do-not-use-in-production"
OUTBOUND_TOKEN = "test-bitrix-token-do-not-use-in-production"


@pytest.fixture
def headers():
    return {"X-API-Key": API_KEY}


@pytest.fixture
def client_factory(tmp_path):
    with ExitStack() as stack:

        def make(handler=None, **overrides):
            values = {
                "_env_file": None,
                "database_path": tmp_path / "requests.sqlite3",
                "api_key": API_KEY,
                "bitrix_outbound_token": OUTBOUND_TOKEN,
                "bitrix_webhook_url": "",
                "bitrix_result_method": "",
                "bitrix_result_field": "",
                "bitrix_result_static_params": {},
            }
            values.update(overrides)
            if handler is None:

                def handler(request):
                    raise AssertionError("Tests must never access a real Bitrix portal")

            app = create_app(Settings(**values), bitrix_transport=httpx.MockTransport(handler))
            return stack.enter_context(TestClient(app))

        yield make


@pytest.fixture
def client(client_factory):
    return client_factory()


@pytest.fixture
def event():
    return {
        "event": "ONCRMDEALADD",
        "data": {"FIELDS": {"ID": "42"}, "unconfirmed_field": "kept"},
        "ts": "1791500000",
        "auth": {
            "application_token": OUTBOUND_TOKEN,
            "domain": "example.bitrix24.ru",
            "client_endpoint": "https://untrusted.example/rest/",
        },
    }


@pytest.fixture
def bitrix_config():
    return {
        "bitrix_webhook_url": "https://example.bitrix24.ru/rest/1/secret-token/",
        "bitrix_result_method": "crm.item.update",
        "bitrix_result_field": "ufCrmExampleResult",
        "bitrix_result_static_params": {"entityTypeId": 2},
    }
