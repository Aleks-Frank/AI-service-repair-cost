"""Выгрузки каталогов интернет-магазинов таблицей (.csv/.xlsx) — «Строительный двор», «Рубильник», «Сантехкомплект».

Формат: одна строка — один товар; первая строка — заголовки. Обязательные колонки:
    «Название», «Артикул» (код товара в магазине), «Цена».
Код товара по умолчанию — «Артикул»; если там полное название (как у «Сантехкомплекта»),
колонка кода задаётся параметром article_column («Номенклатурный номер»).
Необязательные (у разных магазинов набор разный):
    «Бренд», «Категория», «Подкатегория», «Раздел» — путь категории;
    «Единица измерения» — по умолчанию «шт»;
    «Остаток» или «Центральный склад» («2 284 м») — остаток, первое число ячейки;
    «Наличие» («Да»/«Нет») — используется, только если колонки остатка нет.
Все остальные заполненные колонки — характеристики товара («Цвет», «Диаметр», «Вес» …);
они идут в текст для поиска, первыми — колонки назначения («Тип работ», «Область применения» …),
чтобы их не отрезал лимит длины. «Описание» (HTML) очищается от разметки, и его начало
тоже идёт в поиск. Служебные колонки (ссылки, коды учёта) пропускаются.

CSV — с разделителем «;», кодировка UTF-8 или Windows-1251 (определяется сама).
"""

import csv
import re
from collections.abc import Iterator
from datetime import date
from decimal import Decimal
from pathlib import Path

from rag.normalizers import clean_text, normalize_unit, to_decimal
from rag.price_lists.common import DESCRIPTION_LIMIT, DETAILS_LIMIT, ParsedMaterial, PriceListResult, pack_from_name

_REQUIRED = ("Название", "Артикул", "Цена")
_CATEGORY_COLUMNS = ("Категория", "Подкатегория", "Раздел")
_STOCK_COLUMNS = ("Остаток", "Центральный склад")
# Колонки, которые не описывают сам товар: ссылки, служебные коды, цены, наличие.
_SERVICE_COLUMNS = {
    *_REQUIRED, *_CATEGORY_COLUMNS, *_STOCK_COLUMNS,
    "Изображения", "URL", "Описание", "Файлы", "Валюта", "Наличие", "Бренд", "Единица измерения",
    "Старая цена", "Арт", "ЭМКод", "Номенклатурный номер", "Штрих-код на одну ТМЦ",
    "Запись в реестре Росаккредитации",
}
# Колонки назначения товара: по ним ищут «штукатурку для фасада», «шпаклёвку для влажных помещений».
_USAGE_COLUMNS = ("Тип работ", "Область применения", "Тип помещения", "Основание", "Назначение", "Инженерная система")
_TAG_RE = re.compile(r"<[^>]+>")
# Длина артикула в базе (materials.article — VARCHAR(128)).
_ARTICLE_LIMIT = 128
# Первое число ячейки остатка: «2 284 м» -> 2284, «176» -> 176.
_NUMBER_RE = re.compile(r"\d[\d\s]*(?:[.,]\d+)?")


def is_shop_export(path: Path) -> bool:
    """Проверяет по заголовкам, что файл — выгрузка каталога магазина (а не прайс 1С).

    Принимает: path — путь к .csv/.xlsx.
    Возвращает: True, если в первой строке есть «Название», «Артикул» и «Цена».
    """
    header = next(_read_rows(path), {})
    return all(column in header for column in _REQUIRED)


def _read_rows(path: Path) -> Iterator[dict[str, object]]:
    """Читает строки таблицы как словари {заголовок: значение}.

    Принимает: path — путь к .csv или .xlsx.
    Возвращает: итератор словарей (по одному на товар).
    """
    if path.suffix.lower() == ".csv":
        # Ячейки характеристик бывают длинными — снимаем ограничение csv на размер поля.
        csv.field_size_limit(2**31 - 1)
        encoding = "utf-8-sig"
        try:
            with path.open(encoding=encoding) as file:
                file.read(1 << 20)
        except UnicodeDecodeError:
            encoding = "cp1251"
        with path.open(encoding=encoding, newline="") as file:
            yield from csv.DictReader(file, delimiter=";")
        return

    from openpyxl import load_workbook

    # read_only — файл в сотню мегабайт читается потоком, без загрузки целиком в память.
    sheet = load_workbook(path, read_only=True, data_only=True).worksheets[0]
    # В выгрузке не записан размер листа — без сброса openpyxl прочитает одну ячейку.
    sheet.reset_dimensions()
    rows = sheet.iter_rows(values_only=True)
    header = [clean_text(value) or "" for value in next(rows, ())]
    for values in rows:
        yield dict(zip(header, values))


def _stock(row: dict[str, object], trust_availability: bool) -> Decimal | None:
    """Остаток товара: число из колонки остатка; без неё — по «Наличию».

    Принимает:
        row — строка выгрузки;
        trust_availability — верить ли колонке «Наличие» (у части магазинов она заполнена «Нет»
            у всех товаров — ошибка выгрузки, тогда остаток считается неизвестным).
    Возвращает: остаток; 0 — нет в наличии; None — неизвестно (считается в наличии).
    """
    for column in _STOCK_COLUMNS:
        if column in row:
            match = _NUMBER_RE.search(str(row[column] or ""))
            return to_decimal(match.group().replace(" ", "")) if match else Decimal(0)
    if trust_availability and clean_text(row.get("Наличие")) == "Нет":
        return Decimal(0)
    return None


def parse(
    path: Path,
    trust_availability: bool = True,
    price_date: date | None = None,
    article_column: str = "Артикул",
) -> PriceListResult:
    """Разбирает выгрузку каталога магазина (структура описана в начале модуля).

    Артикул — код товара в магазине, поэтому такие прайсы загружаются с флагом --own-articles.
    Повтор артикула (магазин показывает товар в двух разделах) пропускается.

    Принимает:
        path — путь к .csv/.xlsx;
        trust_availability — учитывать ли колонку «Наличие» (см. _stock);
        price_date — дата цен; если не задана — дата изменения файла (дата выгрузки);
        article_column — колонка с кодом товара.
    Возвращает: PriceListResult.
    """
    result = PriceListResult(
        source_file=str(path),
        supplier_name=None,
        price_date=price_date or date.fromtimestamp(path.stat().st_mtime),
    )
    seen_articles: set[str] = set()
    for number, row in enumerate(_read_rows(path), start=2):
        name = clean_text(row.get("Название"))
        if not name:
            continue
        article = clean_text(row.get(article_column))
        price = to_decimal(row.get("Цена"))
        if price is None:
            result.skipped.append((number, "no_price"))
            continue
        if not article:
            result.skipped.append((number, "no_article"))
            continue
        if len(article) > _ARTICLE_LIMIT:
            result.skipped.append((number, "article_too_long"))
            continue
        if article in seen_articles:
            result.skipped.append((number, "duplicate_article"))
            continue
        seen_articles.add(article)

        brand = clean_text(row.get("Бренд"))
        category = " / ".join(dict.fromkeys(
            text for text in (clean_text(row.get(column)) for column in _CATEGORY_COLUMNS) if text
        ))
        # «Нет» в признаках («Без подсветки: Нет») товар не описывает — такие пары пропускаем.
        specs = [f"{column}: {text}" for column, value in row.items()
                 if column and column not in _SERVICE_COLUMNS
                 and (text := clean_text(value)) and text.lower() != "нет"]
        # Назначение — в начало, остальные характеристики — в порядке колонок файла.
        specs.sort(key=lambda spec: not spec.startswith(_USAGE_COLUMNS))
        # Артикул-обозначение («ВВГ-Пнг(А)-LS 3х1,5 (бух 5м)») отличает фасовки с одинаковым
        # названием — добавляем его к характеристикам; числовой код магазина поиску не нужен.
        shop_article = clean_text(row.get("Артикул")) or ""
        if re.search(r"[A-Za-zА-Яа-я]", shop_article) and shop_article not in name:
            specs.insert(0, f"Артикул: {shop_article}")
        details = "; ".join(specs)
        unit_raw = clean_text(row.get("Единица измерения")) or ""
        pack_size, pack_unit = pack_from_name(name)
        result.materials.append(ParsedMaterial(
            article=article,
            name=name,
            brand=brand,
            category=category,
            unit=normalize_unit(unit_raw) if unit_raw else "шт",
            unit_raw=unit_raw,
            price=price,
            stock=_stock(row, trust_availability),
            on_sale=False,
            pack_size=pack_size,
            pack_unit=pack_unit,
            source_row=number,
            details=details[:DETAILS_LIMIT],
            description=(clean_text(_TAG_RE.sub(" ", str(row.get("Описание") or ""))) or "")[:DESCRIPTION_LIMIT],
        ))
    return result
