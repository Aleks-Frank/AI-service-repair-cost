"""Функции нормализации «грязных» значений из xlsx-справочника.

В реальном файле одна и та же единица записана по-разному («мп», «м/п», «пм»),
в числовых ячейках встречаются строки («1 узел», « »), а в тексте — табуляции
и повторяющиеся пробелы. Здесь собраны чистые функции, которые приводят всё
к единому виду. Они не зависят от БД и легко покрываются юнит-тестами.
"""

import re
from decimal import Decimal, InvalidOperation

# Словарь синонимов единиц измерения -> каноническое обозначение.
# Ключи записаны в «упрощённом» виде (нижний регистр, без пробелов и точек на конце),
# см. функцию _unit_key ниже.
_UNIT_SYNONYMS: dict[str, str] = {
    # Погонные метры: в справочнике встречаются все четыре варианта.
    "м": "м",
    "мп": "м",
    "м/п": "м",
    "пм": "м",
    "м.п": "м",
    "п.м": "м",
    # Площадь и объём.
    "м2": "м2",
    "м²": "м2",
    "кв.м": "м2",
    "м3": "м3",
    "м³": "м3",
    "куб.м": "м3",
    # Штучные единицы.
    "шт": "шт",
    "шт.": "шт",
    # Трудозатраты (работа, оплачиваемая по часам).
    "чел.ч": "чел.-ч",
    "чел.-ч": "чел.-ч",
    "чел/ч": "чел.-ч",
    "ч/ч": "чел.-ч",
    "ч": "чел.-ч",
    # Специфические единицы — оставляем как есть, но в едином написании.
    "выход": "выход",
    "место": "место",
    "прибор": "прибор",
    "стык": "стык",
    "итп": "ИТП",
    "узел": "узел",
    "счетчик": "счетчик",
    "счётчик": "счетчик",
    "лампа": "лампа",
    "стояк": "стояк",
    "щит": "щит",
    "мкд": "МКД",
    "соединение": "соединение",
    "система": "система",
}

# Шаблон ссылки на норматив: ГЭСН / ГЭСНр / ГЭСНм / ФЕР / ТЕР + код вида 65-01-006-25.
_NORM_REFERENCE_RE = re.compile(
    r"\b(ГЭСН[а-я]*|ФЕР[а-я]*|ТЕР[а-я]*)\s*(\d{2}-\d{2}-\d{3}-\d{2})",
    flags=re.IGNORECASE,
)

# Шаблон «Справочник № 2» в шапке листа.
_BOOK_NUMBER_RE = re.compile(r"Справочник\s*№\s*(\d+)", flags=re.IGNORECASE)

# Шаблон заголовка раздела: «Раздел 1. Фундаменты», «Раздел 4: Проемы», «Раздел.  Техническое…».
_SECTION_RE = re.compile(r"^\s*Раздел\b", flags=re.IGNORECASE)


def clean_text(value: object, keep_newlines: bool = False) -> str | None:
    """Очищает текст ячейки от лишних пробелов, табуляций и пустых строк.

    Принимает:
        value — значение ячейки (строка, число или None);
        keep_newlines — сохранять ли переводы строк (нужно для «состава работ»,
            где каждый пункт с новой строки).
    Возвращает: очищенную строку или None, если после очистки ничего не осталось.
    """
    if value is None:
        return None
    text = str(value).replace("\t", " ").replace("\xa0", " ")
    if keep_newlines:
        # Чистим каждую строку отдельно и выбрасываем пустые строки.
        lines = [re.sub(r" {2,}", " ", line).strip() for line in text.splitlines()]
        text = "\n".join(line for line in lines if line)
    else:
        # Любые последовательности пробельных символов (в т.ч. \n) -> один пробел.
        text = re.sub(r"\s+", " ", text).strip()
    return text or None


def to_decimal(value: object) -> Decimal | None:
    """Преобразует значение ячейки в Decimal.

    Принимает: value — число, строку («12,5», « 1500 ») или None.
    Возвращает: Decimal или None, если значение пустое или не является числом.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, Decimal)):
        return Decimal(value)
    if isinstance(value, float):
        # Через str(), чтобы получить «0.27», а не 0.270000000000000017763568394002504646778106689453125.
        return Decimal(str(value))
    text = str(value).strip().replace(",", ".").replace(" ", "")
    if not text:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def parse_quantity(value: object) -> Decimal | None:
    """Извлекает число из ячейки «Кол.», где бывают значения вроде «1 узел».

    Принимает: value — содержимое ячейки количества.
    Возвращает: Decimal с количеством или None, если число найти не удалось.
    """
    direct = to_decimal(value)
    if direct is not None:
        return direct
    if value is None:
        return None
    # Ищем первое число в строке: «1 узел» -> 1, «2,5 м» -> 2.5.
    match = re.search(r"\d+(?:[.,]\d+)?", str(value))
    return to_decimal(match.group(0)) if match else None


def parse_item_number(value: object) -> int | None:
    """Достаёт номер позиции из колонки «№ пп».

    Принимает: value — значение ячейки (число, строка или None).
    Возвращает: целый номер позиции или None, если номера нет.
    """
    number = to_decimal(value)
    if number is None or number != number.to_integral_value():
        return None
    return int(number)


def _unit_key(raw: str) -> str:
    """Приводит единицу к ключу словаря синонимов.

    Принимает: raw — единицу измерения как в файле («1 выход», «м/п », «Шт.»).
    Возвращает: упрощённую строку для поиска в _UNIT_SYNONYMS.
    """
    key = raw.strip().lower()
    # Убираем ведущее количество: «1 выход» -> «выход», «1ч.» -> «ч.».
    key = re.sub(r"^\d+(?:[.,]\d+)?\s*", "", key)
    # Убираем пробелы внутри и точку на конце: «чел. ч.» -> «чел.ч».
    key = key.replace(" ", "").rstrip(".")
    return key


def normalize_unit(raw: object) -> str:
    """Нормализует единицу измерения к каноническому виду («мп», «м/п», «пм» -> «м»).

    Принимает: raw — значение ячейки «Ед. изм.» или единица из ответа модели.
    Возвращает: каноническую единицу; неизвестная единица возвращается очищенной, как есть.
    """
    text = clean_text(raw) or ""
    return _UNIT_SYNONYMS.get(_unit_key(text), text)


def normalize_for_match(text: str) -> str:
    """Приводит текст к виду для сравнения: нижний регистр, «ё» -> «е», одиночные пробелы.

    Единая нормализация для всех сравнений текста в проекте (дубли наименований,
    поиск цитаты в описании, определение типа операции).

    Принимает: text — исходный текст.
    Возвращает: нормализованную строку.
    """
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


def extract_norm_reference(*texts: str | None) -> str | None:
    """Ищет ссылку на норматив (ГЭСН/ФЕР/ТЕР) в переданных текстах.

    Принимает: произвольное число строк (примечания, наименование и т.п.), None пропускаются.
    Возвращает: первую найденную ссылку в виде «ГЭСНр 65-01-006-25» или None.
    """
    for text in texts:
        if not text:
            continue
        match = _NORM_REFERENCE_RE.search(text)
        if match:
            return f"{match.group(1)} {match.group(2)}"
    return None


def extract_book_number(text: str | None) -> int | None:
    """Достаёт номер справочника из строки шапки «Справочник № 1».

    Принимает: text — текст ячейки.
    Возвращает: номер справочника или None.
    """
    if not text:
        return None
    match = _BOOK_NUMBER_RE.search(text)
    return int(match.group(1)) if match else None


def is_section_title(text: str | None) -> bool:
    """Проверяет, является ли текст заголовком раздела («Раздел 2. Стены…»).

    Принимает: text — текст ячейки.
    Возвращает: True, если строка начинается со слова «Раздел».
    """
    return bool(text) and bool(_SECTION_RE.match(text))
