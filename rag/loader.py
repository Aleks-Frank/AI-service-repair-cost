"""Загрузка справочника ПЗР (база pzr_pricing) и прайс-листов поставщиков (база supplier_prices).

Порядок: разбор файла -> векторы для всех строк (через ML-сервис) -> замена данных
одной транзакцией. Версий нет: каждая загрузка заменяет предыдущую (для прайса — только
позиции этого поставщика). Если что-то упадёт, транзакция откатится и старые данные останутся.
"""

import logging
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func
from sqlalchemy.orm import Session

from rag.db import Rate
from rag.ml_client import MLServiceClient
from rag.parser import ParsedRate, parse_price_book
from rag.price_list_parser import extract_params, normalize_article, parse_price_list
from rag.prices_db import Material, Supplier

logger = logging.getLogger(__name__)


def _to_row(parsed: ParsedRate, vector: list[float]) -> Rate:
    """Превращает разобранную расценку и её вектор в строку таблицы.

    Принимает: parsed — расценка из парсера; vector — её эмбеддинг.
    Возвращает: ORM-объект Rate (ещё не сохранённый).
    """
    return Rate(
        code=parsed.code,
        book_number=parsed.section.book_number,
        section_title=parsed.section.title,
        name=parsed.name,
        work_composition=parsed.work_composition,
        unit=parsed.unit,
        unit_raw=parsed.unit_raw,
        quantity=parsed.quantity,
        labor_hours=parsed.labor_hours,
        labor_rate=parsed.labor_rate,
        unit_price=parsed.unit_price,
        norm_reference=parsed.norm_reference,
        notes=parsed.notes,
        issues=[issue.to_dict() for issue in parsed.issues],
        source_sheet=parsed.source_sheet,
        source_row=parsed.source_row,
        search_text=parsed.search_text,
        embedding=vector,
    )


def load_price_book(session: Session, path: str | Path, models: MLServiceClient) -> dict[str, Any]:
    """Загружает справочник в БД, полностью заменяя прежнее содержимое.

    Принимает:
        session — сессия БД;
        path — путь к xlsx-файлу;
        models — ML-сервис (нужен метод embed).
    Возвращает: сводку разбора (сколько расценок, какие проблемы найдены).
    """
    # --- 1. Разбор файла. ---
    result = parse_price_book(path)
    summary = result.summary()
    logger.info("Разобрано: %s расценок, %s разделов", summary["rates"], summary["sections"])

    # --- 2. Векторы для всех расценок одним проходом (клиент сам режет на пачки). ---
    vectors = models.embed([rate.search_text for rate in result.rates])

    # --- 3. Замена содержимого таблицы в одной транзакции. ---
    session.execute(delete(Rate))
    session.add_all(_to_row(rate, vector) for rate, vector in zip(result.rates, vectors, strict=True))
    session.commit()
    return summary


def load_price_list(
    session: Session,
    path: str | Path,
    supplier_code: str,
    models: MLServiceClient,
    article_is_manufacturer: bool = True,
) -> dict[str, Any]:
    """Загружает прайс-лист поставщика в базу supplier_prices.

    Заменяются позиции только этого поставщика: прайсы остальных не трогаются.
    Настройки поставщика (активность, приоритет, НДС) при повторной загрузке сохраняются.

    Принимает:
        session — сессия базы прайсов;
        path — путь к .xls/.xlsx;
        supplier_code — код поставщика по названию компании латиницей («ros_elektro»);
        models — ML-сервис (нужен метод embed);
        article_is_manufacturer — артикулы в прайсе — коды производителя (False — свои коды поставщика).
    Возвращает: сводку разбора.
    """
    # --- 1. Разбор файла. ---
    result = parse_price_list(path)
    summary = result.summary()
    logger.info("Разобрано: %s позиций, %s категорий", summary["materials"], summary["categories"])

    # --- 2. Векторы для всех позиций. ---
    vectors = models.embed([item.search_text for item in result.materials])

    # --- 3. Поставщик (создаётся или обновляется) и замена его позиций одной транзакцией. ---
    session.merge(Supplier(
        code=supplier_code,
        name=result.supplier_name or supplier_code,
        price_date=result.price_date,
        source_file=Path(path).name,
        loaded_at=func.now(),
        article_is_manufacturer=article_is_manufacturer,
    ))
    session.execute(delete(Material).where(Material.supplier_code == supplier_code))
    session.add_all(
        Material(
            supplier_code=supplier_code,
            article=item.article,
            article_norm=normalize_article(item.article),
            name=item.name,
            brand=item.brand,
            category=item.category,
            unit=item.unit,
            unit_raw=item.unit_raw,
            price=item.price,
            base_unit=item.pricing[0],
            units_per_price_unit=item.pricing[1],
            base_price=item.base_price,
            sold_by_pack=item.pricing[2],
            stock=item.stock,
            on_sale=item.on_sale,
            pack_size=item.pack_size,
            pack_unit=item.pack_unit,
            params=extract_params(item.name),
            source_row=item.source_row,
            search_text=item.search_text,
            embedding=vector,
        )
        for item, vector in zip(result.materials, vectors, strict=True)
    )
    session.commit()
    return summary
