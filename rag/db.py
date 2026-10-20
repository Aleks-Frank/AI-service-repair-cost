"""База pzr_pricing: таблица расценок с вектором и полнотекстовым индексом.

Одна таблица rates: в одной строке лежат и данные расценки (наименование, единица, цена),
и её поисковые представления — вектор (pgvector) и tsvector (полнотекстовый поиск).
Поэтому индекс не может «разойтись» с ценами: это одна и та же строка.

Схема создаётся функцией init_db() — без миграций (для прототипа этого достаточно;
при изменении структуры таблицу проще пересоздать и перезагрузить справочник).
"""

from decimal import Decimal
from functools import lru_cache
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, Engine, Index, Integer, Numeric, String, Text, create_engine, text
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from rag.config import get_settings

EMBEDDING_DIM = get_settings().embedding_dim

# Генерируемая колонка tsvector. setweight задаёт «важность» частей текста:
# совпадение в наименовании (A) весит больше, чем в составе работ (B) и в названии раздела (C).
# Конфигурация 'russian' включает стемминг: «трещины», «трещину» -> «трещин».
TSVECTOR_EXPRESSION = (
    "setweight(to_tsvector('russian', coalesce(name, '')), 'A') || "
    "setweight(to_tsvector('russian', coalesce(work_composition, '')), 'B') || "
    "setweight(to_tsvector('russian', coalesce(section_title, '')), 'C')"
)


class Base(DeclarativeBase):
    """Базовый класс ORM-моделей."""


class Rate(Base):
    """Расценка справочника ПЗР."""

    __tablename__ = "rates"
    __table_args__ = (
        # HNSW — приближённый индекс для быстрого поиска ближайших векторов;
        # vector_cosine_ops — расстояние косинусное (оператор <=>).
        Index(
            "ix_rates_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # GIN — инвертированный индекс для полнотекстового поиска.
        Index("ix_rates_tsv_gin", "tsv", postgresql_using="gin"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    # Код расценки: «1-22» = справочник № 1, позиция 22; для строк без номера — «2-17.1».
    code: Mapped[str] = mapped_column(String(32), unique=True)
    # Номер внутреннего справочника: 1 — строительно-ремонтные, 2 — инженерные работы.
    book_number: Mapped[int] = mapped_column(Integer, index=True)
    section_title: Mapped[str] = mapped_column(String(512))
    name: Mapped[str] = mapped_column(Text)
    work_composition: Mapped[str | None] = mapped_column(Text)

    # Единица после нормализации («мп», «м/п», «пм» -> «м») и как в исходном файле.
    unit: Mapped[str] = mapped_column(String(32), index=True)
    unit_raw: Mapped[str] = mapped_column(String(64))
    quantity: Mapped[Decimal] = mapped_column(Numeric(12, 4))

    # Деньги и трудозатраты — Numeric, а не float: float даёт ошибки округления.
    labor_hours: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    labor_rate: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    unit_price: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))

    # Ссылка на норматив из примечаний (например, «ГЭСНр 65-01-006-25»).
    norm_reference: Mapped[str | None] = mapped_column(String(128))
    notes: Mapped[str | None] = mapped_column(Text)
    # Проблемы в данных, найденные при загрузке: [{"code", "severity", "message"}, ...].
    issues: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)

    # Откуда строка — чтобы сметчик мог открыть файл и проверить её.
    source_sheet: Mapped[str] = mapped_column(String(255))
    source_row: Mapped[int] = mapped_column(Integer)

    # Текст, по которому посчитан вектор, и сам вектор.
    search_text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    # Вычисляется самой БД при вставке (GENERATED ALWAYS AS ... STORED).
    tsv: Mapped[Any] = mapped_column(TSVECTOR, Computed(TSVECTOR_EXPRESSION, persisted=True))


def build_engine(database_url: str) -> Engine:
    """Создаёт движок SQLAlchemy.

    Принимает: database_url — строку подключения.
    Возвращает: Engine с пулом соединений.
    """
    # pool_pre_ping: проверять соединение перед использованием (переживает перезапуск контейнера БД).
    return create_engine(database_url, pool_pre_ping=True)


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    """Возвращает фабрику сессий для базы из настроек (создаётся один раз).

    Принимает: ничего.
    Возвращает: sessionmaker.
    """
    return sessionmaker(bind=build_engine(get_settings().database_url), expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Создаёт расширение pgvector и таблицу rates, если их ещё нет.

    Принимает: engine — движок нужной базы.
    Возвращает: ничего.
    """
    with engine.begin() as connection:
        # Расширение vector добавляет тип vector и операторы расстояния (<=> и др.).
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
