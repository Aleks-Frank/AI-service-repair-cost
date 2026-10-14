import pytest
from pydantic import ValidationError

from app.core.config import Settings


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/rest/1/secret/",
        "https://example.com/rest/1/secret/?query=bad",
        "https://example.com/rest/1/secret/crm.item.update.json",
        "https://user:password@example.com/rest/1/secret/",
    ],
)
def test_invalid_webhook_url(url):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, bitrix_webhook_url=url)


@pytest.mark.parametrize("method", ["https://evil.example", "../update", "crm.item.update.json"])
def test_invalid_method(method):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, bitrix_result_method=method)


def test_static_parameters_cannot_override_destination():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, bitrix_result_static_params={"id": "other-object"})
