"""Командная строка RAG-модуля.

Примеры:
    python -m rag.cli validate "Справочник ПЗР.xlsx"     # проверить файл, ничего не записывая
    python -m rag.cli load "Справочник ПЗР.xlsx"         # создать таблицу и загрузить справочник
    python -m rag.cli search "трещина на стене в подъезде" --top-k 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from rag.config import get_settings


def cmd_validate(args: argparse.Namespace) -> int:
    """Разбирает xlsx и печатает найденные проблемы в данных, не трогая БД.

    Принимает: args.file — путь к xlsx; args.json — куда сохранить полный отчёт (опционально).
    Возвращает: код выхода (0 — успех).
    """
    from rag.parser import parse_price_book

    result = parse_price_book(args.file)
    print(json.dumps(result.summary(), ensure_ascii=False, indent=2))
    print("\nСтроки с проблемами:")
    for rate in result.rates:
        for issue in rate.issues:
            print(f"  [{issue.severity:7}] {rate.source_sheet!r} стр.{rate.source_row:<4} {rate.code:<10} {issue.message}")
    for row in result.skipped:
        print(f"  [skipped] {row.sheet!r} стр.{row.row}: {row.reason} — {row.content}")

    if args.json:
        report = {
            "summary": result.summary(),
            "issues": [
                {"code": r.code, "sheet": r.source_sheet, "row": r.source_row, "name": r.name,
                 "issues": [i.to_dict() for i in r.issues]}
                for r in result.rates if r.issues
            ],
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nОтчёт сохранён: {args.json}")
    return 0


def cmd_load(args: argparse.Namespace) -> int:
    """Создаёт таблицу (если её нет) и загружает справочник, заменяя прежние данные.

    Принимает: args.file — путь к xlsx.
    Возвращает: код выхода (0 — успех).
    """
    from rag.db import get_session_factory, init_db
    from rag.loader import load_price_book
    from rag.ml_client import build_ml_client

    factory = get_session_factory()
    init_db(factory.kw["bind"])
    with factory() as session:
        summary = load_price_book(session, args.file, build_ml_client(get_settings()))
    print("Справочник загружен в pzr_pricing.rates")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Ищет расценки и печатает их таблицей.

    Принимает: args с полями query, top_k, no_rerank, book.
    Возвращает: код выхода (0 — успех).
    """
    from rag.db import get_session_factory
    from rag.ml_client import build_ml_client
    from rag.retriever import HybridRetriever

    settings = get_settings()
    with get_session_factory()() as session:
        retriever = HybridRetriever(session, build_ml_client(settings), settings)
        result = retriever.search(
            args.query,
            top_k=args.top_k,
            use_reranker=not args.no_rerank,
            book_numbers=args.book,
        )
    print(f"reranker: {'да' if result.reranked else 'нет'}, время, мс: {result.timings_ms}")
    for position, hit in enumerate(result.hits, start=1):
        rate = hit.rate
        price = f"{rate.unit_price} руб./{rate.unit}" if rate.unit_price is not None else "нет цены"
        rerank = f"{hit.rerank_score:.3f}" if hit.rerank_score is not None else "  -  "
        print(f"{position:>2}. [{rate.code:<7}] rerank={rerank} rrf={hit.rrf_score:.4f} "
              f"dense#{hit.dense_rank or '-'} lex#{hit.lexical_rank or '-'} | {rate.name[:80]} | {price}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Описывает команды и их аргументы.

    Принимает: ничего.
    Возвращает: настроенный ArgumentParser.
    """
    parser = argparse.ArgumentParser(prog="python -m rag.cli", description="RAG-модуль поиска расценок ПЗР")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="проверить xlsx без загрузки")
    p.add_argument("file")
    p.add_argument("--json", help="сохранить полный отчёт в JSON")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("load", help="загрузить справочник в БД (заменяет прежние данные)")
    p.add_argument("file")
    p.set_defaults(func=cmd_load)

    p = sub.add_parser("search", help="найти расценки")
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--no-rerank", action="store_true", help="без переранжирования")
    p.add_argument("--book", type=int, action="append", help="номер справочника (1, 2); можно несколько раз")
    p.set_defaults(func=cmd_search)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI.

    Принимает: argv — аргументы (по умолчанию из sys.argv).
    Возвращает: код выхода.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    # Каждый HTTP-запрос к ML-сервису логируется на уровне INFO — для нас это шум.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
