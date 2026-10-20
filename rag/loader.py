"""Загрузка справочника ПЗР в базу pzr_pricing.

Порядок: разбор xlsx -> векторы для всех расценок (через ML-сервис) -> полная замена
содержимого таблицы rates одной транзакцией. Версий справочника нет: каждая загрузка
заменяет предыдущую. Если что-то упадёт, транзакция откатится и старые данные останутся.
"""

import logging
from pathlib import Path
from typing import Any

from sqlalchemy import delete
from sqlalchemy.orm import Session

from rag.db import Rate
from rag.ml_client import MLServiceClient
from rag.parser import ParsedRate, parse_price_book

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
