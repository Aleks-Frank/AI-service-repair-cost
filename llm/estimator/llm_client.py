"""HTTP-клиент к LLM-сервису (контейнер llm_service) и получение от модели проверенного JSON."""

import base64
import json
from collections.abc import Callable
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from llm.estimator.schemas import parse_json_object

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMServiceError(RuntimeError):
    """LLM-сервис недоступен, ответил ошибкой или модель дважды нарушила схему ответа."""


class InputTooLong(ValueError):
    """Вход (фото + текст + кандидаты) длиннее лимита модели — нужно сократить список кандидатов."""


class LLMServiceClient:
    """Клиент LLM-сервиса поверх httpx."""

    def __init__(self, base_url: str, timeout_s: float = 900.0):
        """Создаёт HTTP-клиент.

        Принимает:
            base_url — адрес LLM-сервиса, например http://localhost:8002;
            timeout_s — таймаут одного запроса (генерация длинного JSON занимает минуты).
        Возвращает: ничего.
        """
        self._http = httpx.Client(base_url=base_url, timeout=timeout_s)

    def close(self) -> None:
        self._http.close()

    def generate(self, system_prompt: str, user_prompt: str, images: list[bytes], max_new_tokens: int) -> str:
        """Отправляет промпт и фотографии модели и возвращает её текстовый ответ.

        Принимает:
            system_prompt — инструкции модели;
            user_prompt — данные заявки;
            images — фотографии (байты JPEG/PNG);
            max_new_tokens — максимальная длина ответа в токенах.
        Возвращает: текст ответа модели.
        Исключения: InputTooLong (HTTP 413), LLMServiceError (прочие ошибки).
        """
        payload = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            # Картинки передаются в JSON как base64-строки.
            "images": [base64.b64encode(image).decode("ascii") for image in images],
            "max_new_tokens": max_new_tokens,
        }
        try:
            response = self._http.post("/generate", json=payload)
        except httpx.HTTPError as error:
            raise LLMServiceError(
                f"LLM-сервис {self._http.base_url} недоступен: {error}. Запустите: docker compose up -d llm_service"
            ) from error
        if response.status_code == 413:
            raise InputTooLong(response.json().get("detail", "Вход слишком длинный"))
        if response.is_error:
            raise LLMServiceError(f"LLM-сервис ответил {response.status_code}: {response.text[:500]}")
        return response.json()["text"]

    def ask_json(
        self,
        system_prompt: str,
        payload: dict,
        images: list[bytes],
        schema: type[SchemaT],
        validator: Callable[[SchemaT], None] | None = None,
        max_new_tokens: int = 1700,
    ) -> tuple[SchemaT, list[str]]:
        """Получает от модели ответ строго по схеме; при ошибке даёт модели одну повторную попытку.

        В промпт кладутся данные заявки и JSON-схема ответа. Ответ разбирается и проверяется
        Pydantic-схемой и (опционально) дополнительным валидатором. Если проверка не прошла,
        модель получает текст ошибки и отвечает заново; вторая ошибка — отказ, а не «тихое» принятие.

        Принимает:
            system_prompt — инструкции этапа;
            payload — данные заявки (будут переданы как JSON);
            images — фотографии;
            schema — Pydantic-класс ожидаемого ответа;
            validator — доп. проверка (например, номера фото в допустимых пределах);
            max_new_tokens — лимит длины ответа.
        Возвращает: кортеж (проверенный ответ, список сырых ответов модели — для трассировки).
        """
        schema_text = json.dumps(schema.model_json_schema(), ensure_ascii=False, separators=(",", ":"))
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        prompt = f"ДАННЫЕ ЗАЯВКИ:\n{body}\nJSON_SCHEMA:\n{schema_text}"
        attempts: list[str] = []
        for attempt in range(2):
            answer = self.generate(system_prompt, prompt, images, max_new_tokens)
            attempts.append(answer)
            try:
                value = schema.model_validate(parse_json_object(answer))
                if validator:
                    validator(value)
                return value, attempts
            except (ValueError, TypeError, ValidationError) as error:
                if attempt == 1:
                    raise LLMServiceError(
                        f"Модель дважды вернула некорректный ответ. Ошибка проверки: {str(error)[:500]}"
                    ) from error
                # Повтор с теми же данными + описание ошибки: модель исправляет формат.
                prompt = (
                    f"ДАННЫЕ ЗАЯВКИ:\n{body}\nJSON_SCHEMA:\n{schema_text}\n"
                    f"Предыдущий ответ нарушил схему. Ошибка: {str(error)[:700]}. "
                    "Сформируй заново полный JSON без дополнительных полей. "
                    "При неизвестном количестве используй null."
                )
        raise AssertionError("недостижимо")  # цикл всегда завершается return или raise
