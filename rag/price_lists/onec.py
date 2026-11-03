"""Прайс-листы, выгруженные из 1С (.xls/.xlsx), — как «прайс для сервисной компании.xls» от «Р.О.С. электро».

Ожидаемая структура:
    шапка    — название поставщика («ООО "…"») и строка «Цены указаны на ДД.ММ.ГГГГ»;
    заголовок — строка с колонкой «…Артикул»; колонки ищутся по словам, а не по буквам:
               «Номенклатура», «Артикул», «Остаток», «Распродажа», «Цена», «Ед.»;
    группы   — строки без цены; вложенность задаёт уровень группировки строк Excel
               (outline level): 0 — бренд («ИЕК», «ITK»), 1 и глубже — категории;
    позиции  — строки с ценой: наименование, артикул, остаток, цена, единица.
Другой поставщик с другими заголовками колонок потребует дописать слова в _COLUMNS.
"""

import re
from datetime import date
from decimal import Decimal
from pathlib import Path

from rag.normalizers import clean_text, normalize_unit, to_decimal
from rag.price_lists.common import ParsedMaterial, PriceListResult, cell_text, pack_from_name

# Колонки прайса: ключ -> (слова, одно из которых должно быть в заголовке; слова-исключения).
# Порядок важен: «Номенклатура.Артикул» содержит и «номенклатура», поэтому артикул ищем первым
# и исключаем его колонку при поиске наименования.
_COLUMNS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "article": (("артикул",), ()),
    "name": (("номенклатура", "наименование"), ("артикул",)),
    "stock": (("остаток",), ()),
    "sale": (("распродажа",), ()),
    "price": (("цена",), ()),
    "unit": (("ед.", "ед ", "единица"), ()),
}
_REQUIRED_COLUMNS = ("article", "name", "price", "unit")

# Организационно-правовая форма в начале строки шапки — признак названия поставщика.
_SUPPLIER_RE = re.compile(r"^(ООО|АО|ПАО|ЗАО|ОАО|ИП)\b")
_DATE_RE = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})")


def _read_rows(path: Path) -> list[tuple[int, int, list]]:
    """Читает первый лист файла вместе с уровнями группировки строк.

    Принимает: path — путь к .xls или .xlsx.
    Возвращает: список (номер строки с 1, уровень группировки, значения ячеек).
    """
    if path.suffix.lower() == ".xls":
        import xlrd

        # formatting_info=True нужен, чтобы xlrd прочитал уровни группировки строк (rowinfo_map).
        sheet = xlrd.open_workbook(path, formatting_info=True).sheet_by_index(0)
        return [
            (index + 1, sheet.rowinfo_map[index].outline_level if index in sheet.rowinfo_map else 0, sheet.row_values(index))
            for index in range(sheet.nrows)
        ]

    from openpyxl import load_workbook

    # Обычный режим (не read_only): только в нём доступны row_dimensions с уровнями группировки.
    sheet = load_workbook(path, data_only=True).worksheets[0]
    return [
        (index, sheet.row_dimensions[index].outline_level or 0, list(values))
        for index, values in enumerate(sheet.iter_rows(values_only=True), start=1)
    ]


def _find_header(rows: list[tuple[int, int, list]]) -> tuple[int, dict[str, int]]:
    """Находит строку заголовка и номера колонок по словам в заголовках.

    Заголовок бывает в две строки («6. для тендеров» над «Цена»), поэтому текст колонки
    склеивается из строки с «Артикул» и следующей за ней.

    Заголовок — первая строка со словом «артикул», в которой нашлись все обязательные колонки
    (слово может встретиться и в шапке, например в названии поставщика).

    Принимает: rows — строки листа.
    Возвращает: (индекс строки заголовка в rows, {ключ колонки: номер колонки}).
    Исключения: ValueError, если заголовок или обязательные колонки не найдены.
    """
    missing: list[str] = []
    for index, (_, _, values) in enumerate(rows):
        texts = [(clean_text(value) or "").lower() for value in values]
        if not any("артикул" in text for text in texts):
            continue
        below = rows[index + 1][2] if index + 1 < len(rows) else []
        headers = [f"{text} {(clean_text(below[col]) or '').lower() if col < len(below) else ''} "
                   for col, text in enumerate(texts)]
        columns: dict[str, int] = {}
        for key, (words, excluded) in _COLUMNS.items():
            # next() берёт первую подходящую колонку, которую ещё не заняла другая.
            column = next(
                (col for col, header in enumerate(headers)
                 if col not in columns.values()
                 and any(word in header for word in words)
                 and not any(word in header for word in excluded)),
                None,
            )
            if column is not None:
                columns[key] = column
        missing = [key for key in _REQUIRED_COLUMNS if key not in columns]
        if not missing:
            return index, columns
    if missing:
        raise ValueError(f"В заголовке прайса не найдены колонки: {', '.join(missing)}")
    raise ValueError("Не найдена строка заголовка с колонкой «Артикул».")


def _header_info(rows: list[tuple[int, int, list]]) -> tuple[str | None, date | None]:
    """Достаёт из шапки название поставщика и дату цен.

    Принимает: rows — строки шапки (до заголовка таблицы).
    Возвращает: (название или None, дата или None).
    """
    supplier, price_date = None, None
    for _, _, values in rows:
        for value in values:
            text = clean_text(value)
            if not text:
                continue
            if supplier is None and _SUPPLIER_RE.match(text):
                supplier = text
            match = _DATE_RE.search(text)
            if price_date is None and match and "цен" in text.lower():
                day, month, year = map(int, match.groups())
                price_date = date(year, month, day)
    return supplier, price_date


def parse(path: Path) -> PriceListResult:
    """Разбирает выгрузку прайса из 1С (структура описана в начале модуля).

    Принимает: path — путь к .xls/.xlsx.
    Возвращает: PriceListResult.
    """
    rows = _read_rows(path)
    header_index, columns = _find_header(rows)
    supplier, price_date = _header_info(rows[:header_index])
    result = PriceListResult(source_file=str(path), supplier_name=supplier, price_date=price_date)

    def cell(values: list, key: str) -> object:
        """Значение колонки key в строке (None, если колонки нет в прайсе)."""
        column = columns.get(key)
        return values[column] if column is not None and column < len(values) else None

    # Текущие группы по уровням: {уровень: название}. Уровень 0 — бренд.
    groups: dict[int, str] = {}
    seen_articles: set[str] = set()
    # Данные начинаются после двухстрочного заголовка.
    for row_number, level, values in rows[header_index + 2:]:
        name = clean_text(cell(values, "name"))
        if not name:
            continue
        price = to_decimal(cell(values, "price"))
        unit_raw = clean_text(cell(values, "unit"))

        # --- Строка группы: запоминаем на её уровне и забываем более глубокие уровни. ---
        if price is None and not unit_raw:
            groups = {lvl: title for lvl, title in groups.items() if lvl < level}
            groups[level] = name
            continue

        # --- Позиция прайса. ---
        article = cell_text(cell(values, "article"))
        if price is None:
            result.skipped.append((row_number, "no_price"))
            continue
        if not article:
            result.skipped.append((row_number, "no_article"))
            continue
        if article in seen_articles:
            result.skipped.append((row_number, "duplicate_article"))
            continue
        seen_articles.add(article)

        # Путь категории — группы выше позиции, кроме бренда; соседние повторы (в 1С группа
        # и подгруппа бывают названы одинаково) склеиваем через dict.fromkeys.
        path_titles = [title for lvl, title in sorted(groups.items()) if 0 < lvl < level]
        pack_size, pack_unit = pack_from_name(name)
        result.materials.append(ParsedMaterial(
            article=article,
            name=name,
            brand=groups.get(0),
            category=" / ".join(dict.fromkeys(path_titles)),
            unit=normalize_unit(unit_raw),
            unit_raw=unit_raw or "",
            price=price,
            # Колонка «Остаток» есть, но ячейка пуста — товара нет на складе (0).
            # Колонки нет вовсе — остаток неизвестен (None).
            stock=(to_decimal(cell(values, "stock")) or Decimal(0)) if "stock" in columns else None,
            # В колонке «Распродажа» стоит «***», если позиция продаётся только из наличия.
            on_sale=bool(clean_text(cell(values, "sale"))),
            pack_size=pack_size,
            pack_unit=pack_unit,
            source_row=row_number,
        ))
    return result
