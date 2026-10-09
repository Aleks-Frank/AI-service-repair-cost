import re
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    app_name: str = "Repair Agent API"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    database_path: Path = PROJECT_ROOT / "data" / "requests.sqlite3"
    max_request_bytes: int = Field(default=1_048_576, ge=1024, le=10_485_760)
    api_key: SecretStr = SecretStr("")
    bitrix_outbound_token: SecretStr = SecretStr("")
    bitrix_entity_id_path: str = "data.FIELDS.ID"
    bitrix_webhook_url: SecretStr = SecretStr("")
    bitrix_timeout_seconds: float = Field(default=10, gt=0, le=60)
    bitrix_result_method: str = ""
    bitrix_result_field: str = ""
    bitrix_result_static_params: dict[str, Any] = Field(default_factory=dict)
    bitrix_result_id_param: str = "id"

    @field_validator("database_path")
    @classmethod
    def resolve_database_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @field_validator("bitrix_webhook_url")
    @classmethod
    def validate_webhook_url(cls, value: SecretStr) -> SecretStr:
        raw = value.get_secret_value()
        if not raw:
            return value
        parsed = urlsplit(raw)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or not re.fullmatch(r"/rest/\d+/[^/]+/?", parsed.path)
        ):
            raise ValueError("Expected an HTTPS inbound webhook base URL /rest/user/secret/")
        return value

    @field_validator("bitrix_result_method")
    @classmethod
    def validate_method(cls, value: str) -> str:
        if value and not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9]*(?:\.[a-zA-Z0-9]+)+", value):
            raise ValueError("Expected a REST method name, without URL or .json suffix")
        if value.endswith(".json"):
            raise ValueError("Do not include the .json suffix")
        return value

    @model_validator(mode="after")
    def validate_mapping(self) -> "Settings":
        if self.bitrix_result_id_param in {"fields", "auth", ""}:
            raise ValueError("Invalid result ID parameter")
        reserved = {"fields", "auth", self.bitrix_result_id_param}
        if reserved.intersection(self.bitrix_result_static_params):
            raise ValueError("Static parameters cannot override fields, auth or the object ID")
        return self
