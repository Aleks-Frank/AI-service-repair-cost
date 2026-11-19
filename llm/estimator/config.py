"""Настройки конвейера составления сметы (читаются из .env, как и настройки RAG-модуля)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from llm.estimator.llm_client import LLMServiceClient


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
    # Поиск работ и по исходному тексту заявки (ветка description и общий запрос переранжирования
    # в find_candidates). Включён по EXP-C1: текст жильца находит работы лучше запросов модели (5 против 3 из 11).
    search_by_description: bool = True
    # Решение владельца (эксперимент B6): в местах общего пользования при неизвестном типе лампы
    # вместо 2-218/2-219 (цоколь E27) ставится 2-220 (T8/G13). См. llm/estimator/lamp_defaults.py.
    default_linear_lamps_common_areas: bool = False
    # Тариф автовышки — решение владельца (06.11.2026): 9 000 руб. за 1 шт (выезд), как в итоговых
    # калькуляциях АВР. В ПЗР и прайсах позиции нет. Пустое значение в .env отключает расчёт техники.
    aerial_platform_tariff_rub: float | None = 9000.0
    aerial_platform_tariff_unit: str = "шт"
    # Лимиты длины ответа модели (в токенах) для двух этапов. Для Thinking-моделей их нужно
    # увеличить: рассуждение перед JSON занимает тысячи токенов.
    observation_max_tokens: int = 1500
    # Этап 2 возвращает и работы, и материалы к ним — ответ длиннее, чем раньше.
    selection_max_tokens: int = 3000
    # Этап 3: выбор товаров из прайсов — сколько кандидатов на потребность и лимит ответа.
    material_candidates: int = 5
    material_selection_max_tokens: int = 1500


def build_llm_client(settings: EstimatorSettings) -> LLMServiceClient:
    """Создаёт клиента модели — локального llm_service (Qwen3-VL).

    Принимает: settings — настройки конвейера.
    Возвращает: LLMServiceClient.
    """
    return LLMServiceClient(settings.llm_service_url, settings.llm_timeout_s)


@lru_cache
def get_estimator_settings() -> EstimatorSettings:
    """Возвращает единственный экземпляр настроек.

    Принимает: ничего.
    Возвращает: EstimatorSettings.
    """
    return EstimatorSettings()
