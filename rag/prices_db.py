"""База supplier_prices: прайс-листы поставщиков материалов с векторами и полнотекстовым индексом.

Отдельная база (контейнер prices-db), чтобы справочник ПЗР и цены материалов не смешивались.
Две таблицы:
  * suppliers — поставщики: код по названию компании («ros_elektro»), название, дата цен;
  * materials — позиции прайсов всех поставщиков; поставщик — поле supplier_code.
Поиск идёт одним запросом по всем поставщикам, при необходимости — с фильтром по коду.

Схема создаётся функцией init_prices_db() — без миграций, как и база ПЗР.
"""

from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean, Computed, Date, DateTime, Engine, ForeignKey, Index, Integer, Numeric, String, Text,
    UniqueConstraint, func, text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from rag.config import get_settings
from rag.db import EMBEDDING_DIM, build_engine

# Полнотекстовое представление строится по search_text (наименование с раскрытыми сокращениями
# + категория), а не по name: иначе «Роз. с з/к» не нашлась бы по запросу «розетка заземление».
TSVECTOR_EXPRESSION = "to_tsvector('russian', coalesce(search_text, ''))"


class PricesBase(DeclarativeBase):
    """Базовый класс ORM-моделей базы прайсов (отдельный от базы ПЗР)."""


class Supplier(PricesBase):
    """Поставщик и сведения о последнем загруженном прайсе."""

    __tablename__ = "suppliers"

    # Код по названию компании, латиницей: «ros_elektro». Задаётся при загрузке прайса.
    code: Mapped[str] = mapped_column(String(64), primary_key=True)
    # Название из шапки прайса: «ООО "Р.О.С. электро"».
    name: Mapped[str] = mapped_column(String(255))
    # Дата, на которую действуют цены («Цены указаны на 16.10.2026»).
    price_date: Mapped[date | None] = mapped_column(Date)
    source_file: Mapped[str] = mapped_column(String(512))
    loaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Material(PricesBase):
    """Позиция прайс-листа конкретного поставщика."""

    __tablename__ = "materials"
    __table_args__ = (
        # Артикул уникален в пределах поставщика; у разных поставщиков артикулы могут совпасть.
        UniqueConstraint("supplier_code", "article", name="uq_materials_supplier_article"),
        Index(
            "ix_materials_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_materials_tsv_gin", "tsv", postgresql_using="gin"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    # При удалении поставщика удаляются и его позиции.
    supplier_code: Mapped[str] = mapped_column(ForeignKey("suppliers.code", ondelete="CASCADE"), index=True)
    article: Mapped[str] = mapped_column(String(128))
    name: Mapped[str] = mapped_column(Text)
    # Бренд — группа верхнего уровня прайса («ИЕК», «ITK»), может отсутствовать.
    brand: Mapped[str | None] = mapped_column(String(128))
    # Путь групп прайса: «Кабеленесущие системы / Кабель-каналы и аксессуары / …».
    category: Mapped[str] = mapped_column(Text)

    # Единица после нормализации и как в файле; цена — за эту единицу.
    unit: Mapped[str] = mapped_column(String(32), index=True)
    unit_raw: Mapped[str] = mapped_column(String(64))
    price: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    # Остаток на складе поставщика (None — не указан) и признак распродажи.
    stock: Mapped[Decimal | None] = mapped_column(Numeric(14, 3))
    on_sale: Mapped[bool] = mapped_column(Boolean, default=False)
    # Фасовка из наименования: «(8м)» -> 8 м, «(4шт/компл)» -> 4 шт. Пока только справочно.
    pack_size: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    pack_unit: Mapped[str | None] = mapped_column(String(32))

    source_row: Mapped[int] = mapped_column(Integer)
    search_text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    tsv: Mapped[Any] = mapped_column(TSVECTOR, Computed(TSVECTOR_EXPRESSION, persisted=True))


@lru_cache
def get_prices_session_factory() -> sessionmaker[Session]:
    """Возвращает фабрику сессий базы прайсов (создаётся один раз).

    Принимает: ничего.
    Возвращает: sessionmaker.
    """
    return sessionmaker(bind=build_engine(get_settings().prices_database_url), expire_on_commit=False)


def init_prices_db(engine: Engine) -> None:
    """Создаёт расширение pgvector и таблицы suppliers и materials, если их ещё нет.

    Принимает: engine — движок базы прайсов.
    Возвращает: ничего.
    """
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    PricesBase.metadata.create_all(engine)
