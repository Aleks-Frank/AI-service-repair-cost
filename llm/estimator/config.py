"""Настройки конвейера составления сметы (читаются из .env, как и настройки RAG-модуля)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class EstimatorSettings(BaseSettings):
    """Параметры обращения к LLM и подготовки фотографий."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_service_url: str = "http://localhost:8002"
    # Генерация длинного JSON 4-битной моделью занимает до нескольких минут.
    llm_timeout_s: float = 900.0
    # Фото уменьшаются до этого числа пикселей перед отправкой: меньше токенов и видеопамяти.
    max_image_pixels: int = 524_288
    max_images: int = 4
    # Сколько кандидатов RAG оставлять на один дефект (при слишком длинном входе уменьшается).
    candidates_per_finding: int = 6
    # Лимиты длины ответа модели (в токенах) для двух этапов. Для Thinking-моделей их нужно
    # увеличить: рассуждение перед JSON занимает тысячи токенов.
    observation_max_tokens: int = 1500
    selection_max_tokens: int = 2200


@lru_cache
def get_estimator_settings() -> EstimatorSettings:
    """Возвращает единственный экземпляр настроек.

    Принимает: ничего.
    Возвращает: EstimatorSettings.
    """
    return EstimatorSettings()
