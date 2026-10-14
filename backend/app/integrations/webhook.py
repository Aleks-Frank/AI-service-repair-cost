import json
import re
from typing import Any
from urllib.parse import parse_qsl

from fastapi import Request

from app.core.errors import ServiceError


def parse_form(body: str) -> dict[str, Any]:
    try:
        pairs = parse_qsl(body, keep_blank_values=True, max_num_fields=2000, errors="strict")
    except (ValueError, UnicodeError):
        raise ServiceError(
            400, "invalid_form", "Invalid form encoding or too many fields"
        ) from None
    result: dict[str, Any] = {}
    for key, value in pairs:
        if not re.fullmatch(r"[^\[\]]+(?:\[[^\[\]]*\])*", key):
            raise ServiceError(400, "invalid_form", "Invalid bracket notation")
        parts = [key.split("[", 1)[0], *re.findall(r"\[([^\[\]]*)\]", key)]
        append = parts[-1] == ""
        if append:
            parts.pop()
        if len(parts) > 20 or any(not part for part in parts):
            raise ServiceError(400, "invalid_form", "Unsupported nested form array")
        target = result
        for part in parts[:-1]:
            node = target.setdefault(part, {})
            if not isinstance(node, dict):
                raise ServiceError(400, "invalid_form", "Conflicting form fields")
            target = node
        leaf = parts[-1]
        if leaf not in target:
            target[leaf] = [value] if append else value
        elif isinstance(target[leaf], dict):
            raise ServiceError(400, "invalid_form", "Conflicting form fields")
        elif isinstance(target[leaf], list):
            target[leaf].append(value)
        else:
            target[leaf] = [target[leaf], value]
    return result


def reject_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


async def read_webhook(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    body = await request.body()
    try:
        if content_type == "application/json":
            data = json.loads(body, parse_constant=reject_constant)
        elif content_type == "application/x-www-form-urlencoded":
            data = parse_form(body.decode("utf-8"))
        else:
            raise ServiceError(415, "unsupported_media_type", "Use JSON or a URL-encoded form")
    except (ValueError, UnicodeError, RecursionError):
        raise ServiceError(400, "invalid_body", "Invalid request body") from None
    if not isinstance(data, dict):
        raise ServiceError(400, "invalid_body", "Expected an object")
    return data


def get_path(payload: dict, path: str) -> str | None:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    result = str(value).strip()
    return result if result and len(result) <= 200 else None
