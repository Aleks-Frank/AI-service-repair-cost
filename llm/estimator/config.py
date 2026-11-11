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
    # Отдельный поиск работ по исходному тексту заявки (ветка description в find_candidates).
    # Выключен, пока офлайн-оценка поиска не подтвердит пользу (llm/estimator/retrieval_eval.py).
    search_by_description: bool = False
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
    # Какая модель отвечает: local — свой llm_service (Qwen3-VL), gigachat — GigaChat API.
    llm_provider: str = "local"
    # GigaChat API (используются при llm_provider=gigachat; см. llm/estimator/gigachat_client.py).
    gigachat_credentials: str = ""
    gigachat_scope: str = "GIGACHAT_API_PERS"
    # Адрес API. Новый api.giga.chat (по умолчанию в библиотеке) отвечает не из всех сетей.
    gigachat_base_url: str = "https://gigachat.devices.sberbank.ru/api/v1"
    gigachat_model: str = "GigaChat-2-Max"
    gigachat_ca_bundle_file: str = ""
    gigachat_verify_ssl_certs: bool = True


def build_llm_client(settings: EstimatorSettings) -> LLMServiceClient:
    """Создаёт клиента модели по настройке llm_provider.

    Принимает: settings — настройки конвейера.
    Возвращает: LLMServiceClient (локальный llm_service) или GigaChatClient (GigaChat API).
    """
    if settings.llm_provider == "gigachat":
        from llm.estimator.gigachat_client import GigaChatClient

        return GigaChatClient(
            settings.gigachat_credentials, settings.gigachat_scope, settings.gigachat_model,
            settings.llm_timeout_s, settings.gigachat_ca_bundle_file, settings.gigachat_verify_ssl_certs,
            settings.gigachat_base_url,
        )
    if settings.llm_provider != "local":
        raise ValueError(f"Неизвестный LLM_PROVIDER={settings.llm_provider!r}: допустимо local или gigachat.")
    return LLMServiceClient(settings.llm_service_url, settings.llm_timeout_s)


@lru_cache
def get_estimator_settings() -> EstimatorSettings:
    """Возвращает единственный экземпляр настроек.

    Принимает: ничего.
    Возвращает: EstimatorSettings.
    """
    return EstimatorSettings()
