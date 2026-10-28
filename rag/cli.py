"""Командная строка RAG-модуля.

Примеры:
    python -m rag.cli validate "Справочник ПЗР.xlsx"     # проверить файл, ничего не записывая
    python -m rag.cli load "Справочник ПЗР.xlsx"         # создать таблицу и загрузить справочник
    python -m rag.cli search "трещина на стене в подъезде" --top-k 5

    # Прайс-листы поставщиков материалов (база supplier_prices):
    python -m rag.cli validate-prices "data/xlsx/прайс для сервисной компании.xls"
    python -m rag.cli load-prices "data/xlsx/прайс для сервисной компании.xls" --supplier ros_elektro
    python -m rag.cli load-prices data/shop/shop/sdvor/SdvorPoisk.json --supplier sdvor --name "Строительный двор" --own-articles
    python -m rag.cli search-prices "розетка с заземлением" --top-k 5
    python -m rag.cli list-suppliers
    python -m rag.cli offers CKMP10D-N-040-016-K01 --qty 3       # подбор поставщика для товара
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
            print(f"  {rate.source_sheet!r} стр.{rate.source_row:<4} {rate.code:<10} {issue.message}")

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
        rate = hit.item
        price = f"{rate.unit_price} руб./{rate.unit}" if rate.unit_price is not None else "нет цены"
        rerank = f"{hit.rerank_score:.3f}" if hit.rerank_score is not None else "  -  "
        print(f"{position:>2}. [{rate.code:<7}] rerank={rerank} rrf={hit.rrf_score:.4f} "
              f"dense#{hit.dense_rank or '-'} lex#{hit.lexical_rank or '-'} | {rate.name[:80]} | {price}")
    return 0


def cmd_validate_prices(args: argparse.Namespace) -> int:
    """Разбирает прайс-лист и печатает сводку и примеры позиций, не трогая БД.

    Принимает: args.file — путь к .xls/.xlsx.
    Возвращает: код выхода (0 — успех).
    """
    from rag.price_list_parser import parse_price_list

    result = parse_price_list(args.file)
    print(json.dumps(result.summary(), ensure_ascii=False, indent=2))
    for row, reason in result.skipped:
        print(f"  пропущена стр.{row}: {reason}")
    print("\nПервые позиции:")
    for item in result.materials[:5]:
        base_unit, per_unit, by_pack = item.pricing
        pack = f" | упаковка {per_unit.normalize():f} {base_unit}, покупка упаковками" if by_pack else ""
        print(f"  стр.{item.source_row:<5} [{item.article}] {item.name[:70]} | {item.price} руб./{item.unit} "
              f"= {item.base_price.normalize():f} руб./{base_unit}{pack}")
    return 0


def cmd_load_prices(args: argparse.Namespace) -> int:
    """Создаёт таблицы базы прайсов (если их нет) и загружает прайс поставщика.

    Принимает: args.file — путь к прайсу; args.supplier — код поставщика латиницей;
        args.own_articles — артикулы в прайсе свои, а не производителя;
        args.name — название поставщика (если в файле его нет или оно служебное);
        args.recreate — пересоздать таблицы (после изменения их структуры).
    Возвращает: код выхода (0 — успех).
    """
    from rag.loader import load_price_list
    from rag.ml_client import build_ml_client
    from rag.prices_db import get_prices_session_factory, init_prices_db

    factory = get_prices_session_factory()
    init_prices_db(factory.kw["bind"], recreate=args.recreate)
    if args.recreate:
        print("Таблицы базы прайсов пересозданы: прайсы остальных поставщиков нужно загрузить заново.")
    with factory() as session:
        summary = load_price_list(
            session, args.file, args.supplier, build_ml_client(get_settings()),
            article_is_manufacturer=not args.own_articles,
            supplier_name=args.name,
        )
    print(f"Прайс загружен в supplier_prices.materials, поставщик {args.supplier}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_search_prices(args: argparse.Namespace) -> int:
    """Ищет материалы в прайсах поставщиков и печатает их таблицей.

    Принимает: args с полями query, top_k, no_rerank, supplier, in_stock.
    Возвращает: код выхода (0 — успех).
    """
    from rag.ml_client import build_ml_client
    from rag.prices_db import get_prices_session_factory
    from rag.retriever import MaterialRetriever

    settings = get_settings()
    with get_prices_session_factory()() as session:
        retriever = MaterialRetriever(session, build_ml_client(settings), settings)
        result = retriever.search(
            args.query,
            top_k=args.top_k,
            use_reranker=not args.no_rerank,
            suppliers=args.supplier,
            in_stock_only=args.in_stock,
        )
    print(f"reranker: {'да' if result.reranked else 'нет'}, время, мс: {result.timings_ms}")
    for position, hit in enumerate(result.hits, start=1):
        item = hit.item
        rerank = f"{hit.rerank_score:.3f}" if hit.rerank_score is not None else "  -  "
        stock = "нет данных" if item.stock is None else f"{item.stock.normalize():f}"
        print(f"{position:>2}. [{item.supplier_code} {item.article}] rerank={rerank} "
              f"dense#{hit.dense_rank or '-'} lex#{hit.lexical_rank or '-'} | {item.name[:70]} | "
              f"{item.price} руб./{item.unit} | остаток {stock}")
    return 0


def cmd_list_suppliers(args: argparse.Namespace) -> int:
    """Печатает поставщиков: число позиций, дата цен и настройки подбора.

    Принимает: args (параметров нет).
    Возвращает: код выхода (0 — успех).
    """
    from sqlalchemy import func, select

    from rag.prices_db import Material, Supplier, get_prices_session_factory

    with get_prices_session_factory()() as session:
        counts = dict(session.execute(select(Material.supplier_code, func.count()).group_by(Material.supplier_code)).all())
        for supplier in session.scalars(select(Supplier).order_by(Supplier.code)):
            date_text = f"{supplier.price_date:%d.%m.%Y}" if supplier.price_date else "—"
            print(f"{supplier.code:<16} {'активен' if supplier.active else 'отключён':<9} позиций {counts.get(supplier.code, 0):<6} "
                  f"цены на {date_text:<10} артикулы {'производителя' if supplier.article_is_manufacturer else 'свои':<14} "
                  f"приоритет {supplier.priority} | {supplier.name}")
    return 0


def cmd_delete_prices(args: argparse.Namespace) -> int:
    """Удаляет поставщика и все его позиции из базы прайсов.

    Принимает: args.supplier — код поставщика.
    Возвращает: код выхода (0 — успех, 1 — поставщик не найден).
    """
    from rag.prices_db import Supplier, get_prices_session_factory

    with get_prices_session_factory()() as session:
        supplier = session.get(Supplier, args.supplier)
        if supplier is None:
            print(f"Поставщик {args.supplier} не найден")
            return 1
        # Позиции удаляются каскадно (ondelete="CASCADE" у materials.supplier_code).
        session.delete(supplier)
        session.commit()
    print(f"Поставщик {args.supplier} и его позиции удалены")
    return 0


def cmd_offers(args: argparse.Namespace) -> int:
    """Показывает предложения одного товара у всех поставщиков и выбор самого дешёвого.

    Принимает: args.article — артикул эталона; args.supplier — поставщик эталона (если артикул
        встречается у нескольких); args.qty — потребность в базовых единицах.
    Возвращает: код выхода (0 — успех, 1 — артикул не найден).
    """
    from decimal import Decimal

    from sqlalchemy import select

    from rag.ml_client import build_ml_client
    from rag.offers import choose_offer, find_offers, purchase
    from rag.price_list_parser import normalize_article
    from rag.prices_db import Material, get_prices_session_factory
    from rag.retriever import MaterialRetriever

    settings = get_settings()
    need = Decimal(str(args.qty)) if args.qty is not None else None
    with get_prices_session_factory()() as session:
        query = select(Material).where(Material.article_norm == normalize_article(args.article))
        if args.supplier:
            query = query.where(Material.supplier_code == args.supplier)
        reference = session.scalars(query.order_by(Material.supplier_code)).first()
        if reference is None:
            print(f"Артикул {args.article} не найден")
            return 1
        offers = find_offers(MaterialRetriever(session, build_ml_client(settings), settings), reference)
        if not offers:
            print("Поставщик эталона отключён — предложений нет")
            return 1
        best, warnings = choose_offer(offers, need)
        print(f"Эталон: [{reference.supplier_code} {reference.article}] {reference.name} | параметры {reference.params}")
        for offer in sorted(offers, key=lambda o: o.material.base_price):
            m = offer.material
            mark = "→" if offer is best else " "
            stock = "нет данных" if m.stock is None else f"{m.stock.normalize():f}"
            score = f" {offer.match_score}" if offer.match_score is not None else ""
            print(f" {mark} {offer.supplier.code:<14} [{m.article}] {m.base_price.normalize():f} руб./{m.base_unit} "
                  f"({m.price} руб./{m.unit}) остаток {stock} | {offer.match_type}{score} | {m.name[:60]}")
        if need is not None:
            buy, amount = purchase(best.material, need)
            print(f"Нужно {need} {best.material.base_unit} → купить {buy.normalize():f} {best.material.unit} "
                  f"у {best.supplier.code} на {amount.quantize(Decimal('0.01'))} руб.")
        for warning in warnings:
            print(f"  ! {warning}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Описывает команды и их аргументы.

    Принимает: ничего.
    Возвращает: настроенный ArgumentParser.
    """
    parser = argparse.ArgumentParser(prog="python -m rag.cli", description="RAG-модуль: поиск расценок ПЗР и материалов поставщиков")
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

    p = sub.add_parser("validate-prices", help="проверить прайс-лист поставщика без загрузки")
    p.add_argument("file")
    p.set_defaults(func=cmd_validate_prices)

    p = sub.add_parser("load-prices", help="загрузить прайс поставщика (заменяет только его позиции)")
    p.add_argument("file")
    p.add_argument("--supplier", required=True, help="код поставщика латиницей по названию компании: ros_elektro")
    p.add_argument("--name", help="название поставщика, например «Строительный двор»")
    p.add_argument("--own-articles", action="store_true",
                   help="артикулы в прайсе — свои коды поставщика, а не производителя")
    p.add_argument("--recreate", action="store_true",
                   help="пересоздать таблицы (после изменения структуры; прайсы остальных удалятся)")
    p.set_defaults(func=cmd_load_prices)

    p = sub.add_parser("search-prices", help="найти материалы в прайсах поставщиков")
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--no-rerank", action="store_true", help="без переранжирования")
    p.add_argument("--supplier", action="append", help="код поставщика; можно несколько раз (по умолчанию все)")
    p.add_argument("--in-stock", action="store_true", help="только позиции с остатком")
    p.set_defaults(func=cmd_search_prices)

    p = sub.add_parser("list-suppliers", help="поставщики и их прайсы")
    p.set_defaults(func=cmd_list_suppliers)

    p = sub.add_parser("delete-prices", help="удалить поставщика и его позиции")
    p.add_argument("--supplier", required=True)
    p.set_defaults(func=cmd_delete_prices)

    p = sub.add_parser("offers", help="предложения одного товара у всех поставщиков и выбор самого дешёвого")
    p.add_argument("article", help="артикул эталона")
    p.add_argument("--supplier", help="поставщик эталона (если артикул есть у нескольких)")
    p.add_argument("--qty", type=float, help="потребность в базовых единицах (шт, м)")
    p.set_defaults(func=cmd_offers)
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
