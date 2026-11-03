"""Сопоставление одного товара у разных поставщиков по наименованию.

Используется при загрузке (параметры из наименования сохраняются в базе) и при подборе
поставщика (rag/offers.py): у товара другого поставщика должны совпасть тип изделия,
ключевые параметры (размер, ток, мощность, цоколь) и обозначение модели.
"""

import re

from rag.price_lists.common import LOOKALIKE, expand_abbreviations

# Ключевые параметры товара из наименования. Если у эталона есть такой параметр, у товара
# другого поставщика он должен совпасть — иначе «Угол 40х16» спутается с «Углом 40х25».
_PARAM_PATTERNS: dict[str, re.Pattern[str]] = {
    "size": re.compile(r"(?<!\d)(\d+(?:[.,]\d+)?(?:\s*[хxХX×*/]\s*\d+(?:[.,]\d+)?)+)(?!\d)"),
    "current": re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)\s?[АA](?![\w])"),
    "power": re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)\s?Вт(?![\w])"),
    "color_temp": re.compile(r"(?<![\w.,])(\d{4})\s?[КK](?![\w])"),
    "socket": re.compile(r"(?<![\w])([ЕE]\d{2}|G[UX]?\d{1,2}(?:[.,]\d)?)(?![\w])"),
    "ip": re.compile(r"(?<![\w])IP\s?(\d{2})(?![\w])"),
}
_COLORS = {
    "бел": "белый", "черн": "черный", "сер": "серый", "крем": "кремовый", "беж": "бежевый",
    "граф": "графит", "син": "синий", "красн": "красный", "желт": "желтый", "зелен": "зеленый",
    "коричн": "коричневый", "алюм": "алюминий", "сереб": "серебро", "золот": "золото",
}
# Исполнение, которое отличает разные товары с одинаковыми размерами: «Угол внутренний 40х40»
# и «Угол внешний 40х40». «Наружный» — синоним «внешнего» (так пишут у разных поставщиков).
_KINDS = {"внутрен": "внутренний", "внешн": "внешний", "наружн": "внешний"}
# Параметры, без которых товар считается другим (если они есть у эталона, а у кандидата нет).
CRITICAL_PARAMS = ("size", "current", "power", "socket")


def extract_params(name: str) -> dict[str, list[str]]:
    """Извлекает ключевые параметры из наименования: размер, ток, мощность, цоколь, IP, цвет.

    Принимает: name — наименование позиции.
    Возвращает: словарь {параметр: отсортированный список значений}; пример:
        «Угол внешний КМН 40х16 белый (4шт/компл)» -> {"size": ["40x16"], "color": ["белый"], "kind": ["внешний"]}.
    """
    params: dict[str, list[str]] = {}
    for key, pattern in _PARAM_PATTERNS.items():
        values = set()
        for match in pattern.finditer(name):
            # Единый вид значения: латиница, точка вместо запятой, «х» -> «x», без пробелов.
            value = re.sub(r"\s+", "", match.group(1)).upper().translate(LOOKALIKE).replace(",", ".")
            values.add(re.sub(r"[XХ×*]", "x", value) if key == "size" else value)
        if values:
            params[key] = sorted(values)
    words = re.findall(r"[а-яё]+", name.lower())
    colors = {color for stem, color in _COLORS.items() for word in words if word.startswith(stem)}
    if colors:
        params["color"] = sorted(colors)
    kinds = {kind for stem, kind in _KINDS.items() for word in words if word.startswith(stem)}
    if kinds:
        params["kind"] = sorted(kinds)
    return params


# Обозначение модели: слово с латинскими буквами («Colour», «Como», «C35», «VTM.303»).
_MODEL_TOKEN_RE = re.compile(r"(?<![\w])[A-Za-z0-9][A-Za-z0-9.\-/]*[A-Za-z][A-Za-z0-9.\-/]*|(?<![\w])[A-Za-z][A-Za-z0-9.\-/]*")
# Латинские слова, которые не отличают модели: единицы, цоколи, степень защиты, общие слова.
_NOT_MODEL = {"led", "ip", "w", "v", "k", "mm", "cm", "rgb", "white", "black", "pro", "max", "mini"}


def model_tokens(name: str, brand: str | None) -> set[str]:
    """Обозначения модели в наименовании — латинские слова, кроме бренда, цоколя и общих слов.

    Пример: «Раковина Cersanit Colour керамика 50 см» (бренд Cersanit) -> {"colour"}.

    Принимает: name — наименование; brand — бренд позиции (может быть None).
    Возвращает: множество обозначений в нижнем регистре.
    """
    params = extract_params(name)
    skip = {value.lower() for key in ("socket", "ip") for value in params.get(key, [])} | _NOT_MODEL
    if brand:
        skip |= {word.lower() for word in re.findall(r"[A-Za-z0-9]+", brand)}
    tokens = {token.lower().strip(".-/") for token in _MODEL_TOKEN_RE.findall(name)}
    return {token for token in tokens if token and token not in skip and not token.startswith("ip")}


def product_type(name: str) -> str | None:
    """Тип изделия — первое русское слово наименования (без аббревиатур в верхнем регистре).

    «РКС-20-30-П-К Розетка с з/к» -> «розет»; «Разъем РШ-ВШ 32-002» -> «разъе»; «VEGA Роз. …» -> «розет».
    Сравниваются первые 5 букв, чтобы «розетка» и «розетки» совпадали.

    Принимает: name — наименование позиции.
    Возвращает: начало слова-типа в нижнем регистре или None.
    """
    for word in re.findall(r"[А-ЯЁа-яё]{3,}", expand_abbreviations(name)):
        if not word.isupper():
            return word.lower().replace("ё", "е")[:5]
    return None


def models_match(reference: set[str], candidate: set[str]) -> bool:
    """Проверяет, что обозначения моделей не противоречат друг другу.

    Если у обоих товаров есть обозначения модели, нужно хотя бы одно общее: «Colour» и «Como»
    одного бренда — разные раковины. Если у одного из товаров обозначений нет, проверка не
    применяется (название могло быть сокращено).

    Принимает: reference, candidate — результаты model_tokens.
    Возвращает: True, если товары можно считать одной моделью.
    """
    return not (reference and candidate) or bool(reference & candidate)


def params_match(reference: dict[str, list[str]], candidate: dict[str, list[str]]) -> bool:
    """Проверяет, что у кандидата те же ключевые параметры, что и у эталона.

    Правила: параметр, который есть у обоих, должен совпасть; критичный параметр эталона
    (размер, ток, мощность, цоколь) обязан быть и у кандидата. Цвет или IP, не указанные
    у кандидата, не считаются расхождением — их часто опускают в названии.

    Принимает: reference, candidate — результаты extract_params.
    Возвращает: True, если товары можно считать одинаковыми по параметрам.
    """
    for key, values in reference.items():
        if key in candidate:
            if candidate[key] != values:
                return False
        elif key in CRITICAL_PARAMS:
            return False
    return True
