"""Парсер прайс-листа поставщика материалов (выгрузка из 1С, xls или xlsx).

Как и парсер ПЗР, ничего не знает о базе: на входе файл, на выходе PriceListResult.

Ожидаемая структура (как в «прайс для сервисной компании.xls» от «Р.О.С. электро»):
    шапка    — название поставщика («ООО "…"») и строка «Цены указаны на ДД.ММ.ГГГГ»;
    заголовок — строка с колонкой «…Артикул»; колонки ищутся по словам, а не по буквам:
               «Номенклатура», «Артикул», «Остаток», «Распродажа», «Цена», «Ед.»;
    группы   — строки без цены; вложенность задаёт уровень группировки строк Excel
               (outline level): 0 — бренд («ИЕК», «ITK»), 1 и глубже — категории;
    позиции  — строки с ценой: наименование, артикул, остаток, цена, единица.
Другой поставщик с другими заголовками колонок потребует дописать слова в _COLUMNS.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path

from rag.normalizers import clean_text, normalize_unit, to_decimal

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
# Фасовка в наименовании: «(8м)», «(120м)», «(4шт/компл)», «(20шт)».
_PACK_RE = re.compile(r"\((\d+(?:[.,]\d+)?)\s*(м|шт)(?=[\s/)])")

# Частые сокращения в наименованиях прайса. Модели эмбеддингов и reranker их не понимают
# («Роз. с з/к о/у» не похоже на «розетка с заземлением открытой установки»), поэтому в тексте
# для поиска они раскрываются. Наименование в базе остаётся как в прайсе.
_ABBREVIATIONS = {
    "Роз.": "Розетка", "Розет.": "Розетка", "Выключ.": "Выключатель", "Выкл.": "Выключатель",
    "з/к": "заземляющий контакт", "з/ш": "защитные шторки",
    "о/у": "открытой установки", "с/у": "скрытой установки", "откр. уст.": "открытой установки",
    "1-кл.": "одноклавишный", "2-кл.": "двухклавишный", "3-кл.": "трехклавишный",
    "1-мест.": "одноместная", "2-мест.": "двухместная", "3-мест.": "трехместная", "4-мест.": "четырехместная",
    "прох.": "проходной", "перекр.": "перекрестный", "перенос.": "переносная", "встр.": "встраиваемая",
    "стац.": "стационарная", "расп.": "распределительная", "гофр.": "гофрированная", "сб.": "сборе",
    "бел.": "белый", "черн.": "черный", "граф.": "графит", "крем.": "кремовый", "Пож.": "Пожарный",
}
# Сокращение заменяется только целиком: слева не буква/цифра/дефис, справа не буква.
_ABBREVIATION_RE = re.compile(
    r"(?<![\w-])(" + "|".join(re.escape(key) for key in sorted(_ABBREVIATIONS, key=len, reverse=True)) + r")(?!\w)"
)


def expand_abbreviations(text: str) -> str:
    """Раскрывает частые сокращения прайса: «Роз. с з/к о/у» -> «Розетка с заземляющий контакт открытой установки».

    Принимает: text — наименование позиции.
    Возвращает: текст с раскрытыми сокращениями (для эмбеддинга и полнотекстового поиска).
    """
    return _ABBREVIATION_RE.sub(lambda match: _ABBREVIATIONS[match.group(1)], text)


@dataclass
class ParsedMaterial:
    """Одна позиция прайс-листа."""

    article: str
    name: str
    brand: str | None
    category: str
    unit: str
    unit_raw: str
    price: Decimal
    stock: Decimal | None
    on_sale: bool
    pack_size: Decimal | None
    pack_unit: str | None
    source_row: int

    @property
    def search_text(self) -> str:
        """Текст для поиска: наименование с раскрытыми сокращениями и путь категории.

        Цена и поставщик не входят: они не описывают сам материал.

        Принимает: ничего.
        Возвращает: строку для модели эмбеддингов и reranker'а.
        """
        name = expand_abbreviations(self.name)
        return f"{name}\nКатегория: {self.category}" if self.category else name


@dataclass
class PriceListResult:
    """Результат разбора прайс-листа."""

    source_file: str
    supplier_name: str | None
    price_date: date | None
    materials: list[ParsedMaterial] = field(default_factory=list)
    # Пропущенные строки: (номер строки, причина) — без цены, без артикула, дубль артикула.
    skipped: list[tuple[int, str]] = field(default_factory=list)

    def summary(self) -> dict:
        """Сводка для печати: сколько позиций, категорий, единиц и пропусков.

        Принимает: ничего.
        Возвращает: словарь со счётчиками.
        """
        return {
            "source_file": self.source_file,
            "supplier_name": self.supplier_name,
            "price_date": self.price_date.isoformat() if self.price_date else None,
            "materials": len(self.materials),
            "categories": len({m.category for m in self.materials}),
            "units": dict(Counter(m.unit for m in self.materials).most_common()),
            "with_pack": sum(m.pack_size is not None for m in self.materials),
            "skipped": dict(Counter(reason for _, reason in self.skipped)),
        }


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

    Принимает: rows — строки листа.
    Возвращает: (индекс строки заголовка в rows, {ключ колонки: номер колонки}).
    Исключения: ValueError, если заголовок или обязательные колонки не найдены.
    """
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
        if missing:
            raise ValueError(f"В заголовке прайса не найдены колонки: {', '.join(missing)}")
        return index, columns
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


def _cell_text(value: object) -> str | None:
    """Текст ячейки; числа без «.0» (артикул 12345 xlrd отдаёт как 12345.0).

    Принимает: value — значение ячейки.
    Возвращает: строку или None.
    """
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return clean_text(value)


def _pack(name: str) -> tuple[Decimal | None, str | None]:
    """Извлекает фасовку из наименования: «Кабель-канал 16х16 (84м)» -> (84, «м»).

    Принимает: name — наименование позиции.
    Возвращает: (количество, единица) или (None, None).
    """
    match = _PACK_RE.search(name)
    if not match:
        return None, None
    return to_decimal(match.group(1)), match.group(2)


def parse_price_list(path: str | Path) -> PriceListResult:
    """Разбирает прайс-лист поставщика.

    Принимает: path — путь к .xls/.xlsx.
    Возвращает: PriceListResult с позициями и пропущенными строками.
    """
    path = Path(path)
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
        article = _cell_text(cell(values, "article"))
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
        pack_size, pack_unit = _pack(name)
        result.materials.append(ParsedMaterial(
            article=article,
            name=name,
            brand=groups.get(0),
            category=" / ".join(dict.fromkeys(path_titles)),
            unit=normalize_unit(unit_raw),
            unit_raw=unit_raw or "",
            price=price,
            stock=to_decimal(cell(values, "stock")),
            # В колонке «Распродажа» стоит «***», если позиция продаётся только из наличия.
            on_sale=bool(clean_text(cell(values, "sale"))),
            pack_size=pack_size,
            pack_unit=pack_unit,
            source_row=row_number,
        ))
    return result
