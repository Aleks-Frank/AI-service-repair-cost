"""Парсер xlsx-справочника ПЗР в типизированные структуры Python.

Парсер ничего не знает о базе данных: на входе путь к файлу, на выходе ParseResult
со списком разделов, расценок и найденных проблем. Благодаря этому его можно
тестировать без БД и запускать отдельно как «валидатор» справочника.

Ожидаемая структура листа (как в «Справочник ПЗР.xlsx»):
    A — № п/п, B — наименование, C — состав работ, D — ед. изм., E — кол-во,
    F — затраты труда (чел.-ч), G — стоимость чел.-ч, H — стоимость за единицу,
    I и далее — служебные примечания (ссылки на ГЭСН, даты добавления, пометки).
Первые строки — шапка; данные начинаются с первой строки «Раздел …».
"""

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
    """Проблема в данных, найденная при разборе строки.

    severity: "error" — расценку нельзя использовать в смете без исправления;
              "warning" — расценка пригодна, но данные стоит проверить.
    """

    code: str
    severity: str
    message: str

    def to_dict(self) -> dict[str, str]:
        """Преобразует проблему в словарь для сохранения в JSONB.

        Принимает: ничего.
        Возвращает: словарь {"code", "severity", "message"}.
        """
        return {"code": self.code, "severity": self.severity, "message": self.message}


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

    @property
    def has_errors(self) -> bool:
        """Проверяет, есть ли у расценки проблемы уровня error.

        Принимает: ничего.
        Возвращает: True, если хотя бы одна проблема имеет severity == "error".
        """
        return any(issue.severity == "error" for issue in self.issues)


@dataclass
class SkippedRow:
    """Строка, которую парсер не смог отнести ни к разделу, ни к расценке."""

    sheet: str
    row: int
    reason: str
    content: str


@dataclass
class ParseResult:
    """Итог разбора файла: разделы, расценки и пропущенные строки."""

    source_file: str
    sections: list[ParsedSection]
    rates: list[ParsedRate]
    skipped: list[SkippedRow]

    def summary(self) -> dict[str, Any]:
        """Считает сводную статистику разбора (удобно для логов и CLI).

        Принимает: ничего.
        Возвращает: словарь с количеством разделов, расценок, проблем по кодам и т.п.
        """
        issue_counts: dict[str, int] = {}
        for rate in self.rates:
            for issue in rate.issues:
                issue_counts[issue.code] = issue_counts.get(issue.code, 0) + 1
        return {
            "source_file": self.source_file,
            "sections": len(self.sections),
            "rates": len(self.rates),
            "rates_with_errors": sum(1 for r in self.rates if r.has_errors),
            "rates_without_price": sum(1 for r in self.rates if r.unit_price is None),
            "skipped_rows": len(self.skipped),
            "issues_by_code": dict(sorted(issue_counts.items())),
        }


def _cell(row: tuple[Any, ...], index: int) -> Any:
    """Безопасно достаёт значение ячейки по индексу (строка может быть короче).

    Принимает: row — кортеж значений строки; index — индекс колонки.
    Возвращает: значение ячейки или None.
    """
    return row[index] if index < len(row) else None


def _validate_prices(rate: ParsedRate) -> None:
    """Проверяет согласованность цены, трудозатрат и ставки; дописывает проблемы в rate.issues.

    Цена в справочнике должна равняться: трудозатраты × стоимость чел.-ч × количество.
    Расхождение — повод для сметчика проверить строку (это и есть «аномалия» в данных).

    Принимает: rate — разобранная расценка.
    Возвращает: ничего (изменяет rate на месте).
    """
    if rate.unit_price is None:
        rate.issues.append(DataIssue("no_price", "error", "Не указана стоимость за единицу"))
    if rate.labor_hours is None:
        rate.issues.append(DataIssue("no_labor_hours", "warning", "Не указаны затраты труда (чел.-ч)"))
    if rate.labor_rate is None:
        rate.issues.append(DataIssue("no_labor_rate", "warning", "Не указана стоимость чел.-ч"))

    # Сверяем цену, только если есть все три составляющие.
    if None in (rate.unit_price, rate.labor_hours, rate.labor_rate):
        return
    expected = rate.labor_hours * rate.labor_rate * rate.quantity  # type: ignore[operator]
    difference = abs(expected - rate.unit_price)  # type: ignore[operator]
    if difference > PRICE_TOLERANCE_RUB:
        rate.issues.append(
            DataIssue(
                "price_mismatch",
                "warning",
                f"Цена {rate.unit_price} ≠ трудозатраты × ставка × кол-во = {expected:.2f} "
                f"(расхождение {difference:.2f} руб.)",
            )
        )


def _parse_sheet(sheet: Any, sheet_index: int) -> tuple[list[ParsedSection], list[ParsedRate], list[SkippedRow]]:
    """Разбирает один лист справочника.

    Принимает:
        sheet — лист openpyxl (открытый в режиме values_only);
        sheet_index — порядковый номер листа (запасной вариант номера справочника).
    Возвращает: кортеж (разделы, расценки, пропущенные строки) этого листа.
    """
    sheet_name = sheet.title.strip()
    rows = list(sheet.iter_rows(values_only=True))

    # --- 1. Номер справочника из шапки («Справочник № 2»), иначе — номер листа. ---
    book_number = None
    for row in rows[:HEADER_SCAN_ROWS]:
        for value in row:
            book_number = extract_book_number(clean_text(value))
            if book_number:
                break
        if book_number:
            break
    book_number = book_number or sheet_index + 1

    sections: list[ParsedSection] = []
    rates: list[ParsedRate] = []
    skipped: list[SkippedRow] = []
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
        section_title = name if is_section_title(name) else first_text if is_section_title(first_text) else None
        if section_title and not unit_cell:
            current_section = ParsedSection(book_number, sheet_name, section_title, len(sections) + 1)
            sections.append(current_section)
            continue

        # До первого раздела идёт шапка таблицы — пропускаем её молча.
        if current_section is None:
            continue

        # Полностью пустые строки пропускаем молча.
        if all(value is None or (isinstance(value, str) and not value.strip()) for value in row):
            continue

        # --- 3. Строка без наименования или единицы — это не расценка. ---
        if not name or not unit_cell:
            content = " | ".join(str(v) for v in row if v is not None)
            skipped.append(SkippedRow(sheet_name, row_number, "нет наименования или единицы измерения", content[:200]))
            continue

        issues: list[DataIssue] = []

        # --- 4. Код расценки. ---
        item_number = parse_item_number(number_cell)
        if item_number is not None:
            last_number, unnumbered_after_last = item_number, 0
            code = f"{book_number}-{item_number}"
        else:
            # Строку добавили в файл позже, и формула нумерации её не учла.
            # Даём ей код от предыдущей позиции: «2-17.1», «2-17.2», ...
            unnumbered_after_last += 1
            code = f"{book_number}-{last_number}.{unnumbered_after_last}"
            issues.append(DataIssue("no_item_number", "warning", "В файле нет номера позиции, код сгенерирован"))
        if code in seen_codes:
            # Защита от повторной нумерации в файле: код обязан быть уникальным.
            code = f"{code}-r{row_number}"
            issues.append(DataIssue("duplicate_number", "warning", "Номер позиции повторяется в файле"))
        seen_codes.add(code)

        # --- 5. Единица измерения и количество. ---
        unit, unit_known = normalize_unit(unit_cell)
        if not unit_known:
            issues.append(DataIssue("unknown_unit", "warning", f"Неизвестная единица измерения «{unit_cell}»"))
        quantity = parse_quantity(_cell(row, COL_QTY))
        if quantity is None or quantity <= 0:
            # В справочнике цена всегда дана «за единицу», поэтому по умолчанию берём 1.
            quantity = Decimal("1")
            issues.append(DataIssue("no_quantity", "warning", "Не указано количество, принято 1"))

        # --- 6. Примечания из служебных колонок (I и правее). ---
        note_values = [clean_text(v) for v in row[FIRST_NOTES_COL:]]
        notes = "; ".join(v for v in note_values if v) or None
        if notes and "исключить" in notes.lower():
            issues.append(DataIssue("marked_for_exclusion", "warning", "В примечаниях строка помечена «исключить»"))

        rate = ParsedRate(
            code=code,
            item_number=item_number,
            section=current_section,
            name=name,
            work_composition=clean_text(_cell(row, COL_COMPOSITION), keep_newlines=True),
            unit=unit,
            unit_raw=unit_cell,
            quantity=quantity,
            labor_hours=to_decimal(_cell(row, COL_LABOR)),
            labor_rate=to_decimal(_cell(row, COL_LABOR_RATE)),
            unit_price=to_decimal(_cell(row, COL_PRICE)),
            norm_reference=extract_norm_reference(notes, name),
            notes=notes,
            source_sheet=sheet_name,
            source_row=row_number,
            issues=issues,
        )
        # --- 7. Проверка согласованности цены. ---
        _validate_prices(rate)
        rates.append(rate)

    return sections, rates, skipped


def parse_price_book(path: str | Path) -> ParseResult:
    """Разбирает xlsx-справочник целиком (все листы).

    Принимает: path — путь к xlsx-файлу.
    Возвращает: ParseResult с разделами, расценками и пропущенными строками.
    """
    path = Path(path)
    # data_only=True: читаем не формулы («=F9*G9»), а значения, которые Excel
    # сохранил при последнем пересчёте. read_only ускоряет чтение больших файлов.
    workbook = load_workbook(path, data_only=True, read_only=True)
    try:
        all_sections: list[ParsedSection] = []
        all_rates: list[ParsedRate] = []
        all_skipped: list[SkippedRow] = []
        for index, sheet in enumerate(workbook.worksheets):
            sections, rates, skipped = _parse_sheet(sheet, index)
            all_sections.extend(sections)
            all_rates.extend(rates)
            all_skipped.extend(skipped)
    finally:
        workbook.close()
    _mark_duplicate_names(all_rates)
    return ParseResult(str(path), all_sections, all_rates, all_skipped)


def _mark_duplicate_names(rates: list[ParsedRate]) -> None:
    """Помечает расценки, у которых совпадают наименование и единица измерения.

    Дубликаты мешают поиску (две одинаковые строки с разной ценой) и обычно означают,
    что позицию добавили в справочник повторно.

    Принимает: rates — все расценки справочника.
    Возвращает: ничего (дописывает проблемы в rate.issues).
    """
    # Группируем по ключу (наименование в нижнем регистре, единица).
    groups: dict[tuple[str, str], list[ParsedRate]] = {}
    for rate in rates:
        groups.setdefault((rate.name.lower(), rate.unit), []).append(rate)
    for group in groups.values():
        if len(group) < 2:
            continue
        for rate in group:
            others = ", ".join(other.code for other in group if other is not rate)
            rate.issues.append(DataIssue("duplicate_name", "warning", f"Такое же наименование у позиций: {others}"))
