"""Общее для всех парсеров прайсов: позиция прайса, результат разбора и обработка наименований.

Здесь только то, что нужно каждому формату: ParsedMaterial (одна позиция), PriceListResult
(весь прайс), раскрытие сокращений для поиска, нормализация артикула, единицы цены и фасовка.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from rag.normalizers import clean_text, to_decimal

# Сколько символов характеристик и описания включать в текст для поиска: модель эмбеддингов
# читает до 512 токенов, а длинный текст размывает смысл.
DETAILS_LIMIT = 400
DESCRIPTION_LIMIT = 300

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


# Кириллические буквы, которые при наборе артикула легко спутать с латинскими.
# «СКМР10D» (часть букв кириллицей) и «CKMP10D» — один артикул.
LOOKALIKE = str.maketrans("АВЕКМНОРСТУХ", "ABEKMHOPCTYX")


def normalize_article(article: str) -> str:
    """Приводит артикул к виду для сравнения между поставщиками.

    Верхний регистр, без пробелов, похожие кириллические буквы -> латинские.
    Суффиксы («-R», «-K01») не отрезаются: это могут быть разные модификации товара.

    Принимает: article — артикул из прайса.
    Возвращает: нормализованный артикул.
    """
    return re.sub(r"\s+", "", article).upper().translate(LOOKALIKE)



def pricing_units(unit: str, pack_size: Decimal | None, pack_unit: str | None) -> tuple[str, Decimal, bool]:
    """Определяет базовую единицу, сколько её в единице цены и обязательна ли покупка упаковками.

    Правило: кратность обязательна, только если цена указана за упаковку («компл», «уп»).
        «компл» + «(4шт/компл)» -> базовая «шт», 4 шт в единице цены, покупка комплектами;
        «уп» без фасовки в названии -> базовая «уп», состав упаковки неизвестен;
        «шт» + «(6шт)», «м» + «(8м)» -> цена уже за штуку/метр, коробка справочно.

    Принимает: unit — единица цены; pack_size, pack_unit — фасовка из наименования.
    Возвращает: (базовая единица, базовых единиц в единице цены, покупать ли целыми упаковками).
    """
    if unit in {"компл", "уп"}:
        if pack_size and pack_unit:
            return pack_unit, pack_size, True
        return unit, Decimal(1), True
    return unit, Decimal(1), False


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
    # Характеристики из каталога магазина («Цвет: белый; Диаметр: 25 мм»); у прайсов 1С пусто.
    details: str = ""
    # Начало описания товара из каталога: для чего и где применяется («Для выравнивания стен
    # во влажных помещениях…»). Помогает находить товар по назначению, а не только по названию.
    description: str = ""

    @property
    def search_text(self) -> str:
        """Текст для поиска: наименование с раскрытыми сокращениями, категория, характеристики и описание.

        Цена и поставщик не входят: они не описывают сам материал.

        Принимает: ничего.
        Возвращает: строку для модели эмбеддингов и reranker'а.
        """
        parts = [expand_abbreviations(self.name)]
        if self.category:
            parts.append(f"Категория: {self.category}")
        if self.details:
            parts.append(f"Характеристики: {self.details}")
        if self.description:
            parts.append(f"Описание: {self.description}")
        return "\n".join(parts)

    @property
    def pricing(self) -> tuple[str, Decimal, bool]:
        """Базовая единица, базовых единиц в единице цены и обязательность покупки упаковками.

        Принимает: ничего.
        Возвращает: кортеж из pricing_units().
        """
        return pricing_units(self.unit, self.pack_size, self.pack_unit)

    @property
    def base_price(self) -> Decimal:
        """Цена за одну базовую единицу (за 1 шт или 1 м) — по ней сравниваются поставщики.

        Принимает: ничего.
        Возвращает: цену с точностью до 0,0001 руб.
        """
        return (self.price / self.pricing[1]).quantize(Decimal("0.0001"))


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
            # Позиции, которые покупаются только целыми упаковками (цена за «компл»/«уп»).
            "sold_by_pack": sum(m.pricing[2] for m in self.materials),
            "skipped": dict(Counter(reason for _, reason in self.skipped)),
        }


def cell_text(value: object) -> str | None:
    """Текст ячейки; числа без «.0» (артикул 12345 xlrd отдаёт как 12345.0).

    Принимает: value — значение ячейки.
    Возвращает: строку или None.
    """
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return clean_text(value)


def pack_from_name(name: str) -> tuple[Decimal | None, str | None]:
    """Извлекает фасовку из наименования: «Кабель-канал 16х16 (84м)» -> (84, «м»).

    Принимает: name — наименование позиции.
    Возвращает: (количество, единица) или (None, None).
    """
    match = _PACK_RE.search(name)
    if not match:
        return None, None
    return to_decimal(match.group(1)), match.group(2)
