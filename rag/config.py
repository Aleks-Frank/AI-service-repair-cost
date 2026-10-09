"""Настройки RAG-модуля: читаются из переменных окружения и файла .env (см. .env.example)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки с типами и значениями по умолчанию.

    Pydantic сам приводит строки из окружения к нужным типам и падает
    с понятной ошибкой, если значение некорректно.
    """

    # Читаем .env из текущей директории; лишние переменные (например, для docker compose) игнорируем.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- База данных ---
    postgres_user: str = "smeta"
    postgres_password: str = "smeta"
    postgres_db: str = "pzr_pricing"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # --- ML-сервис ---
    ml_service_url: str = "http://localhost:8001"
    # Таймаут HTTP-запроса к ML-сервису, секунды (загрузка всего справочника идёт пачками).
    ml_timeout_s: float = 120.0
    # Сколько текстов отправлять в /embed за один запрос.
    ml_embed_batch: int = 64
    # Размерность векторов модели эмбеддингов (bge-m3 — 1024). Зашита в колонку vector(1024).
    embedding_dim: int = 1024

    # --- Параметры гибридного поиска ---
    search_candidates_per_branch: int = 50
    rrf_k: int = 60
    rerank_top_n: int = 30

    @property
    def database_url(self) -> str:
        """Собирает строку подключения SQLAlchemy.

        Принимает: ничего (использует поля настроек).
        Возвращает: строку вида postgresql+psycopg://user:pass@host:port/db.
        """
        # postgresql+psycopg — драйвер psycopg 3.
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Возвращает единственный экземпляр настроек (создаётся при первом вызове).

    Принимает: ничего.
    Возвращает: объект Settings.
    """
    return Settings()
