"""Парсеры прайс-листов поставщиков: один модуль — один формат файла.

    common.py       — общее: позиция прайса (ParsedMaterial), результат разбора, сокращения, артикулы;
    matching.py     — сопоставление одного товара у разных поставщиков по наименованию;
    onec.py         — выгрузки из 1С (.xls/.xlsx): «Р.О.С. электро»;
    shop_json.py    — каталоги магазинов в JSON: «Лемана Про»;
    shop_export.py  — выгрузки каталогов магазинов таблицей (.csv/.xlsx): «Строительный двор»,
                      «Рубильник», «Сантехкомплект».

Парсеры ничего не знают о базе: на входе файл, на выходе PriceListResult.
Новый формат = новый модуль с функцией parse(path, **options) и строка в FORMATS.
Какой файл каким парсером читать, задаётся в data_sources.toml (поле format); без него формат
определяется по расширению и заголовкам (detect_format).
"""

from pathlib import Path

from rag.price_lists import onec, shop_export, shop_json
from rag.price_lists.common import ParsedMaterial, PriceListResult, normalize_article
from rag.price_lists.matching import extract_params

FORMATS = {
    "onec": onec.parse,
    "shop_json": shop_json.parse,
    "shop_export": shop_export.parse,
}

__all__ = ["FORMATS", "ParsedMaterial", "PriceListResult", "detect_format", "extract_params",
           "normalize_article", "parse_price_list"]


def detect_format(path: Path) -> str:
    """Определяет формат прайса по расширению и заголовкам.

    Принимает: path — путь к файлу.
    Возвращает: ключ FORMATS: .json — shop_json; .csv — shop_export; .xls/.xlsx — shop_export,
        если в первой строке заголовки выгрузки магазина, иначе onec.
    """
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "shop_json"
    if suffix == ".csv" or (suffix == ".xlsx" and shop_export.is_shop_export(path)):
        return "shop_export"
    return "onec"


def parse_price_list(path: str | Path, fmt: str | None = None, **options) -> PriceListResult:
    """Разбирает прайс-лист нужным парсером.

    Принимает:
        path — путь к файлу;
        fmt — формат (ключ FORMATS); None — определить автоматически;
        options — параметры конкретного парсера (например, trust_availability для shop_export).
    Возвращает: PriceListResult с позициями и пропущенными строками.
    Исключения: ValueError — неизвестный формат.
    """
    path = Path(path)
    fmt = fmt or detect_format(path)
    if fmt not in FORMATS:
        raise ValueError(f"Неизвестный формат прайса «{fmt}». Доступны: {', '.join(FORMATS)}.")
    return FORMATS[fmt](path, **options)
