"""Каталоги интернет-магазинов в JSON — как «LemanaproPoisk.json» («Лемана Про»).

Формат:
    {"generation_date", "shop": {...}, "products": [{"article_code", "vendor_code"?, "name",
     "breadcrumbs"?, "brand"?, "specifications"?, "cities": [{"city_name",
     "price": {"date", "price"}, "stock"?: [{"type": "all", "count"}, ...]}]}]}
Единиц измерения в каталогах нет — цена считается за штуку; остаток берётся из записи
«all», если магазин его передаёт, иначе остаток неизвестен.
"""

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from rag.normalizers import clean_text, to_decimal
from rag.price_lists.common import DETAILS_LIMIT, ParsedMaterial, PriceListResult, cell_text, pack_from_name

# Служебные пункты «хлебных крошек» каталога, которые не являются категориями.
_BREADCRUMB_SKIP = {"главная", "каталог"}


def _city_offer(cities: list[dict]) -> tuple[Decimal | None, Decimal | None]:
    """Выбирает цену и остаток из записей по городам: запись с ценой и самой свежей датой цены.

    Принимает: cities — список записей товара по городам из каталога.
    Возвращает: (цена или None, остаток или None — если магазин остатки не передаёт).
    """
    priced = [c for c in cities if (c.get("price") or {}).get("price")]
    if not priced:
        return None, None
    best = max(priced, key=lambda c: c["price"].get("date") or "")
    stock_rows = best.get("stock") or []
    total = next((s.get("count") for s in stock_rows if s.get("type") == "all"), None)
    if total is None and stock_rows:
        total = sum(s.get("count") or 0 for s in stock_rows)
    return to_decimal(best["price"]["price"]), to_decimal(total) if stock_rows else None


def parse(path: Path) -> PriceListResult:
    """Разбирает каталог интернет-магазина в формате JSON (структура описана в начале модуля).

    Артикул — vendor_code, если он есть, иначе код товара в магазине (article_code). Это коды
    магазина, а не производителя, поэтому такие каталоги загружаются с флагом --own-articles.
    Категория — «хлебные крошки» без служебных пунктов и бренда.

    Принимает: path — путь к .json.
    Возвращает: PriceListResult.
    """
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    generated = data.get("generation_date")
    price_date = datetime.fromisoformat(generated.replace("Z", "+00:00")).date() if generated else None
    result = PriceListResult(
        source_file=str(path),
        supplier_name=(data.get("shop") or {}).get("shop_name"),
        price_date=price_date,
    )
    seen_articles: set[str] = set()
    for number, product in enumerate(data.get("products") or [], start=1):
        name = clean_text(product.get("name"))
        article = cell_text(product.get("vendor_code")) or cell_text(product.get("article_code"))
        if not name:
            continue
        # Магазин дублирует свой код в конце названия: «Герметик Tytan, белый [тов-213048]».
        if article:
            name = clean_text(name.replace(f"[{article}]", "")) or name
        price, stock = _city_offer(product.get("cities") or [])
        if price is None:
            result.skipped.append((number, "no_price"))
            continue
        if not article:
            result.skipped.append((number, "no_article"))
            continue
        if article in seen_articles:
            result.skipped.append((number, "duplicate_article"))
            continue
        seen_articles.add(article)

        brand = clean_text(product.get("brand"))
        crumbs = [clean_text(c) for c in product.get("breadcrumbs") or []]
        category = " / ".join(dict.fromkeys(
            c for c in crumbs if c and c.lower() not in _BREADCRUMB_SKIP and c != brand
        ))
        specs = product.get("specifications") or {}
        details = "; ".join(f"{clean_text(k)}: {clean_text(v)}" for k, v in specs.items() if clean_text(v))
        pack_size, pack_unit = pack_from_name(name)
        result.materials.append(ParsedMaterial(
            article=article,
            name=name,
            brand=brand,
            category=category,
            # Единиц измерения в каталогах нет: цена магазина — за штуку товара.
            unit="шт",
            unit_raw="",
            price=price,
            stock=stock,
            on_sale=False,
            pack_size=pack_size,
            pack_unit=pack_unit,
            source_row=number,
            details=details[:DETAILS_LIMIT],
        ))
    return result
