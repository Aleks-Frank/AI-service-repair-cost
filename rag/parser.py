"""Парсер xlsx-справочника ПЗР в типизированные структуры Python.

Парсер ничего не знает о базе данных: на входе путь к файлу, на выходе ParseResult
со списком разделов, расценок и найденных проблем. Благодаря этому его можно
запускать отдельно как «валидатор» справочника (python -m rag.cli validate).

Ожидаемая структура листа (как в «Справочник ПЗР.xlsx»):
    A — № п/п, B — наименование, C — состав работ, D — ед. изм., E — кол-во,
    F — затраты труда (чел.-ч), G — стоимость чел.-ч, H — стоимость за единицу,
    I и далее — служебные примечания (ссылки на ГЭСН, даты добавления, пометки).
Первые строки — шапка; данные начинаются с первой строки «Раздел …».
"""

from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from rag.normalizers import (
    clean_text,
    extract_book_number,
    extract_norm_reference,
    is_section_title,
    normalize_for_match,
    normalize_unit,
    parse_item_number,
    parse_quantity,
    to_decimal,
)

# Индексы колонок (с нуля), соответствуют буквам A..H.
COL_NUMBER, COL_NAME, COL_COMPOSITION, COL_UNIT, COL_QTY, COL_LABOR, COL_LABOR_RATE, COL_PRICE = range(8)
# Всё, что правее H, считаем примечаниями.
FIRST_NOTES_COL = 8

# Допустимое расхождение «цена из файла» vs «трудозатраты × ставка», в рублях.
PRICE_TOLERANCE_RUB = Decimal("1.00")

# Сколько первых строк листа просматриваем в поисках «Справочник № N».
HEADER_SCAN_ROWS = 10


@dataclass
class DataIssue:
    """Проблема в данных справочника: нет цены, цена не сходится или дубль наименования."""

    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        """Преобразует проблему в словарь для сохранения в JSONB.

        Принимает: ничего.
        Возвращает: словарь {"code", "message"}.
        """
        return {"code": self.code, "message": self.message}


@dataclass
class ParsedSection:
    """Раздел справочника, найденный на листе."""

    book_number: int
    sheet_name: str
    title: str
    position: int


@dataclass
class ParsedRate:
    """Одна расценка, разобранная из строки листа."""

    code: str
    item_number: int | None
    section: ParsedSection
    name: str
    work_composition: str | None
    unit: str
    unit_raw: str
    quantity: Decimal
    labor_hours: Decimal | None
    labor_rate: Decimal | None
    unit_price: Decimal | None
    norm_reference: str | None
    notes: str | None
    source_sheet: str
    source_row: int
    issues: list[DataIssue] = field(default_factory=list)

    @property
    def search_text(self) -> str:
        """Формирует текст, по которому считается эмбеддинг расценки.

        Включаем раздел (даёт контекст: «Кровля», «Полы»), наименование, состав работ
        и единицу измерения. Цены в текст не включаем — они не помогают искать по смыслу.

        Принимает: ничего.
        Возвращает: одну строку для подачи в модель эмбеддингов.
        """
        parts = [f"Раздел: {self.section.title}.", f"Работа: {self.name}."]
        if self.work_composition:
            # Пункты состава работ склеиваем через «; », чтобы получился связный текст.
            parts.append("Состав работ: " + "; ".join(self.work_composition.splitlines()) + ".")
        parts.append(f"Единица измерения: {self.unit}.")
        return " ".join(parts)


@dataclass
class ParseResult:
    """Итог разбора файла: разделы и расценки."""

    source_file: str
    sections: list[ParsedSection]
    rates: list[ParsedRate]

    def summary(self) -> dict[str, Any]:
        """Считает сводную статистику разбора (для логов и CLI).

        Принимает: ничего.
        Возвращает: словарь с количеством разделов, расценок и проблем по типам.
        """
        issue_counts = Counter(issue.code for rate in self.rates for issue in rate.issues)
        return {
            "source_file": self.source_file,
            "sections": len(self.sections),
            "rates": len(self.rates),
            "issues_by_code": dict(sorted(issue_counts.items())),
        }


def _cell(row: tuple[Any, ...], index: int) -> Any:
    """Безопасно достаёт значение ячейки по индексу (строка может быть короче).

    Принимает: row — кортеж значений строки; index — индекс колонки.
    Возвращает: значение ячейки или None.
    """
    return row[index] if index < len(row) else None


def _price_issues(rate: ParsedRate) -> list[DataIssue]:
    """Проверяет цену расценки: указана ли она и сходится ли с трудозатратами.

    Цена в справочнике должна равняться: трудозатраты × стоимость чел.-ч × количество.
    Расхождение — повод для сметчика проверить строку (это и есть «аномалия» в данных).

    Принимает: rate — разобранная расценка.
    Возвращает: список найденных проблем (пустой, если всё в порядке).
    """
    if rate.unit_price is None:
        return [DataIssue("no_price", "Не указана стоимость за единицу")]
    # Сверяем цену, только если в строке есть обе составляющие расчёта.
    if rate.labor_hours is None or rate.labor_rate is None:
        return []
    expected = rate.labor_hours * rate.labor_rate * rate.quantity
    difference = abs(expected - rate.unit_price)
    if difference <= PRICE_TOLERANCE_RUB:
        return []
    return [DataIssue(
        "price_mismatch",
        f"Цена {rate.unit_price} ≠ трудозатраты × ставка × кол-во = {expected:.2f} (расхождение {difference:.2f} руб.)",
    )]


def _parse_sheet(sheet: Any, sheet_index: int) -> tuple[list[ParsedSection], list[ParsedRate]]:
    """Разбирает один лист справочника.

    Принимает:
        sheet — лист openpyxl;
        sheet_index — порядковый номер листа (запасной вариант номера справочника).
    Возвращает: кортеж (разделы, расценки) этого листа.
    """
    sheet_name = sheet.title.strip()
    rows = list(sheet.iter_rows(values_only=True))

    # --- 1. Номер справочника из шапки («Справочник № 2»), иначе — номер листа. ---
    header_numbers = (extract_book_number(clean_text(value)) for row in rows[:HEADER_SCAN_ROWS] for value in row)
    book_number = next((number for number in header_numbers if number), sheet_index + 1)

    sections: list[ParsedSection] = []
    rates: list[ParsedRate] = []
    seen_codes: set[str] = set()
    current_section: ParsedSection | None = None
    # Последний встреченный номер позиции и счётчик ненумерованных строк после него.
    last_number = 0
    unnumbered_after_last = 0

    # enumerate с 1, чтобы номер строки совпадал с номером в Excel.
    for row_number, row in enumerate(rows, start=1):
        number_cell = _cell(row, COL_NUMBER)
        name = clean_text(_cell(row, COL_NAME))
        unit_cell = clean_text(_cell(row, COL_UNIT))
        first_text = clean_text(number_cell) if isinstance(number_cell, str) else None

        # --- 2. Заголовок раздела: «Раздел …» в колонке A или B и пустая единица. ---
        section_title = next((text for text in (name, first_text) if is_section_title(text)), None)
        if section_title and not unit_cell:
            current_section = ParsedSection(book_number, sheet_name, section_title, len(sections) + 1)
            sections.append(current_section)
            continue

        # Шапка до первого раздела, пустые строки и строки без наименования/единицы — не расценки.
        if current_section is None or not name or not unit_cell:
            continue

        # --- 3. Код расценки. ---
        item_number = parse_item_number(number_cell)
        if item_number is not None:
            last_number, unnumbered_after_last = item_number, 0
            code = f"{book_number}-{item_number}"
        else:
            # Строку добавили в файл позже, и формула нумерации её не учла.
            # Даём ей код от предыдущей позиции: «2-17.1», «2-17.2», ...
            unnumbered_after_last += 1
            code = f"{book_number}-{last_number}.{unnumbered_after_last}"
        if code in seen_codes:
            # Защита от повторной нумерации в файле: код обязан быть уникальным.
            code = f"{code}-r{row_number}"
        seen_codes.add(code)

        # --- 4. Примечания из служебных колонок (I и правее). ---
        notes = "; ".join(text for text in map(clean_text, row[FIRST_NOTES_COL:]) if text) or None

        # Количество: цена в справочнике всегда «за единицу», поэтому без значения берём 1.
        quantity = parse_quantity(_cell(row, COL_QTY))
        rate = ParsedRate(
            code=code,
            item_number=item_number,
            section=current_section,
            name=name,
            work_composition=clean_text(_cell(row, COL_COMPOSITION), keep_newlines=True),
            unit=normalize_unit(unit_cell),
            unit_raw=unit_cell,
            quantity=quantity if quantity and quantity > 0 else Decimal("1"),
            labor_hours=to_decimal(_cell(row, COL_LABOR)),
            labor_rate=to_decimal(_cell(row, COL_LABOR_RATE)),
            unit_price=to_decimal(_cell(row, COL_PRICE)),
            norm_reference=extract_norm_reference(notes, name),
            notes=notes,
            source_sheet=sheet_name,
            source_row=row_number,
        )
        # --- 5. Проверка цены. ---
        rate.issues.extend(_price_issues(rate))
        rates.append(rate)

    return sections, rates


def _mark_duplicate_names(rates: list[ParsedRate]) -> None:
    """Помечает расценки, у которых совпадают наименование и единица измерения.

    Дубликаты мешают поиску (две одинаковые строки с разной ценой) и обычно означают,
    что позицию добавили в справочник повторно.

    Принимает: rates — все расценки справочника.
    Возвращает: ничего (дописывает проблемы в rate.issues).
    """
    groups: dict[tuple[str, str], list[ParsedRate]] = {}
    for rate in rates:
        groups.setdefault((normalize_for_match(rate.name), rate.unit), []).append(rate)
    for group in groups.values():
        if len(group) < 2:
            continue
        for rate in group:
            others = ", ".join(other.code for other in group if other is not rate)
            rate.issues.append(DataIssue("duplicate_name", f"Такое же наименование у позиций: {others}"))


def parse_price_book(path: str | Path) -> ParseResult:
    """Разбирает xlsx-справочник целиком (все листы).

    Принимает: path — путь к xlsx-файлу.
    Возвращает: ParseResult с разделами и расценками.
    """
    path = Path(path)
    # data_only=True: читаем не формулы («=F9*G9»), а значения, которые Excel
    # сохранил при последнем пересчёте. read_only ускоряет чтение больших файлов.
    workbook = load_workbook(path, data_only=True, read_only=True)
    try:
        sections: list[ParsedSection] = []
        rates: list[ParsedRate] = []
        for index, sheet in enumerate(workbook.worksheets):
            sheet_sections, sheet_rates = _parse_sheet(sheet, index)
            sections += sheet_sections
            rates += sheet_rates
    finally:
        workbook.close()
    _mark_duplicate_names(rates)
    return ParseResult(str(path), sections, rates)
