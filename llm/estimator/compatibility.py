"""Проверка соответствия работы дефекту: объект, операция, масштаб (деталь/узел), среда.

Тематическая близость («освещение», «сантехника») не делает работу подходящей. Модель оценивает
смысл, а код проверяет явные противоречия между операцией дефекта и наименованием расценки:
    - операция: ремонт/очистка существующего ≠ установка (устройство) нового изделия;
      очистка ≠ замена/демонтаж;
    - масштаб: деталь ≠ узел целиком (лампа ≠ светильник, прокладка ≠ кран, …) — таблица PART_WHOLE;
    - среда: наружная работа ≠ работа в помещении (по явным словам в наименовании).
Проверка только отклоняет. Она не подбирает и не подставляет другую работу: разные расценки
имеют разный состав и не заменяют друг друга (например, «Смена ламп …» не равна паре
«Снятие светильников» + «Установка светильников»). Если после отклонения работ не осталось,
решает сметчик.

Правила срабатывают только при известной операции (operation_status confirmed/inferred).
Наименование расценки разбирается так: первое слово-действие («Смена», «Установка», «Временная
заделка») и следующая за ним именная группа — объект работы, до предлога или знака препинания, не
длиннее трёх слов («ламп», «дверного замка врезного»). Если в группе названа деталь, работа — над
деталью, даже если рядом стоит узел («дверного замка» — замок, а не дверь).
"""

import re

# Виды операций по корню слова-действия.
ACTIONS = (
    ("replacement", r"смен|замен"),
    ("removal", r"демонтаж|сняти|снять|разборк"),
    ("installation", r"установ|монтаж|устройств|устро"),
    ("cleaning", r"очист|прочист|промыв|почист"),
    ("painting", r"окраск|покраск|окрас|покрас"),
    ("repair", r"ремонт|восстанов|заделк|герметиз|закреплен|закреп|регулир|модерниз|почин|отремонт"),
)
# Несовместимые пары (операция дефекта, действие расценки).
CONFLICTS = {
    ("repair", "installation"): "установка нового изделия вместо ремонта существующего",
    ("cleaning", "installation"): "установка вместо очистки",
    ("cleaning", "replacement"): "замена вместо очистки",
    ("cleaning", "removal"): "демонтаж вместо очистки",
    ("painting", "installation"): "установка вместо окраски",
}
# Деталь и узел, в который она входит (версионируемая таблица; дополняется по подтверждённым случаям).
PART_WHOLE = (
    (r"ламп", r"светильник|прожектор|фонар"),
    (r"прокладк|картридж|букс", r"кран|смесител"),
    (r"ручк|петл|доводчик|замк|замок|личинк", r"двер"),
    (r"крышк", r"люк"),
)
_OUTDOOR = re.compile(r"наружн|уличн|фасад|придомов", re.I)
_INDOOR = re.compile(r"внутренн|в помещени|подвесн\w* потолк|в подъезд", re.I)
_WORD = re.compile(r"[а-яё]+", re.I)


def action_of(text: str) -> str | None:
    """Вид операции по первому слову-действию текста.

    Принимает: text — наименование расценки или операция дефекта.
    Возвращает: replacement / removal / installation / cleaning / painting / repair или None.
    """
    action, _ = _action_and_object(text)
    return action


_PREPOSITIONS = {"в", "во", "на", "с", "со", "из", "для", "по", "под", "над", "при", "у", "и", "до", "от"}


def _action_and_object(text: str) -> tuple[str | None, str]:
    """Первое слово-действие и следующая за ним именная группа (объект).

    Принимает: text — наименование расценки или операция.
    Возвращает: (вид операции или None, объект — слова в нижнем регистре через пробел или "").
    """
    # Знаки препинания и скобки обрывают именную группу («Смена светильников: с лампами …»).
    tokens = re.findall(r"[а-яё]+|[,:;()/]", text.lower().replace("ё", "е"))
    for index, word in enumerate(tokens):
        for kind, pattern in ACTIONS:
            if re.match(pattern, word):
                phrase = []
                for token in tokens[index + 1:]:
                    if not token.isalpha() or token in _PREPOSITIONS or len(phrase) == 3:
                        break
                    phrase.append(token)
                return kind, " ".join(phrase)
    return None, ""


def work_object(name: str) -> str:
    """Объект работы — слово после действия («Смена ламп в светильниках» -> «ламп»).

    Принимает: name — наименование расценки.
    Возвращает: слово-объект или "" (действие не распознано).
    """
    return _action_and_object(name)[1]


def _scale(text: str) -> tuple[set[int], set[int]]:
    """Какие детали и узлы из PART_WHOLE названы в тексте.

    Если в одной строке таблицы названы и деталь, и узел («дверного замка»), текст считается
    про деталь: узел здесь — уточнение, чья это деталь.

    Принимает: text — текст.
    Возвращает: (номера строк PART_WHOLE, где названа деталь; … где назван только узел).
    """
    text = text.lower()
    parts = {i for i, (part, _) in enumerate(PART_WHOLE) if re.search(part, text)}
    wholes = {i for i, (_, whole) in enumerate(PART_WHOLE) if re.search(whole, text)}
    return parts, wholes - parts


def work_conflicts(operation: str, environment: str, work_name: str) -> list[str]:
    """Явные противоречия между операцией дефекта и расценкой.

    Принимает:
        operation — что нужно сделать по дефекту («замена светильника целиком»);
        environment — indoor / outdoor / unknown;
        work_name — наименование расценки.
    Возвращает: список причин отклонения (пустой — противоречий не найдено).
    """
    reasons: list[str] = []
    op_action = action_of(operation)
    work_action, obj = _action_and_object(work_name)
    if op_action and work_action and (op_action, work_action) in CONFLICTS:
        reasons.append(CONFLICTS[(op_action, work_action)])
    # Масштаб: объект работы — деталь, а операция про узел целиком (и наоборот).
    if obj:
        op_parts, op_wholes = _scale(operation)
        obj_parts, obj_wholes = _scale(obj)
        if obj_parts & op_wholes and not obj_parts & op_parts:
            reasons.append(f"работа над деталью («{obj}»), а нужна операция над узлом целиком")
        # Узел вместо детали — противоречие только для замены/установки/демонтажа («замена прокладки» ≠
        # «смена крана»). Регулировка или ремонт узла целиком включает его детали («регулировка петли» —
        # «Комплексная регулировка двери»), поэтому для них правило не применяется.
        replacing = op_action in ("replacement", "installation", "removal")
        if replacing and obj_wholes & op_parts and not obj_wholes & op_wholes:
            reasons.append(f"работа над узлом целиком («{obj}»), а нужна операция над деталью")
    if environment == "outdoor" and _INDOOR.search(work_name):
        reasons.append("работа в помещении, а объект снаружи")
    if environment == "indoor" and _OUTDOOR.search(work_name):
        reasons.append("наружная работа, а объект в помещении")
    return reasons


def model_mismatches(choice) -> list[str]:
    """Оси, по которым модель сама отметила несоответствие (значение «no»).

    Принимает: choice — WorkChoice.
    Возвращает: список названий осей.
    """
    axes = {"object_match": "объект", "operation_match": "операция", "scale_match": "масштаб",
            "environment_match": "среда"}
    return [label for field, label in axes.items() if getattr(choice, field, "unclear") == "no"]


def number_counts_object(quote: str, quantity: float, work_name: str) -> bool:
    """Относится ли число в цитате к объекту работы, а не к другому предмету.

    «2-х столбах» не даёт количества ламп: число стоит при слове «столб», а объект работы — «ламп».
    Число принимается, если сразу за ним (в пределах двух слов) идёт единица (шт, м, м2 …) или слово
    с тем же корнем (4 буквы), что у объекта работы. Если объект в наименовании не распознан,
    проверка не применяется.

    Принимает: quote — цитата с числом; quantity — объём; work_name — наименование расценки.
    Возвращает: True, если число можно считать количеством объекта работы.
    """
    obj = work_object(work_name)
    if not obj:
        return True
    stems = {word[:4] for word in obj.split()}
    text = quote.lower().replace("ё", "е").replace(",", ".")
    for match in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)(?:-?х)?", text):
        if float(match.group(1)) != float(quantity):
            continue
        after = _WORD.findall(text[match.end():])[:2]
        if any(word in {"шт", "штук", "штуки", "м", "м2", "кв", "пог", "мп"} or word[:4] in stems for word in after):
            return True
    return False


def explicit_action(text: str) -> bool:
    """Названо ли в тексте действие («заменить», «замена», «почистить», «требуется ремонт»).

    Принимает: text — описание заявки или осмотра.
    Возвращает: True, если найдено слово-действие.
    """
    return action_of(text) is not None


def material_conflict(work_name: str, query: str, operation: str = "") -> str | None:
    """Противоречит ли материал масштабу работы или операции (лампа вместо светильника и наоборот).

    Принимает:
        work_name — наименование работы;
        query — что искать в прайсах;
        operation — операция дефекта («замена светильника целиком»), если известна.
    Возвращает: причину или None.
    """
    query_parts, query_wholes = _scale(query)
    obj = work_object(work_name)
    obj_parts, _ = _scale(obj) if obj else (set(), set())
    # Изделие-узел («светильник») для работы над деталью («Смена ламп»).
    if obj_parts & query_wholes and not query_parts & obj_parts:
        return f"для работы над «{obj}» подобран узел целиком"
    # Операция — замена узла целиком, а материал — только деталь («лампа» при замене светильника).
    op_parts, op_wholes = _scale(operation)
    if op_wholes & query_parts and not op_parts & query_parts and not op_wholes & query_wholes:
        return "операция — замена узла целиком, а подобрана только деталь"
    return None


def _item_kind(name: str) -> tuple[int, str] | None:
    """Тип товара по таблице PART_WHOLE: первое по порядку в наименовании слово детали или узла.

    В каталогах тип товара стоит первым («Светильник … с LED лампами G13» — светильник,
    «Лампа для светильника …» — лампа). Бренд в начале («ЭРА лента …») не мешает: ищется
    первое слово из таблицы, а не первое слово вообще.

    Принимает: name — наименование товара.
    Возвращает: (номер строки PART_WHOLE, "part" или "whole") или None, если тип не из таблицы.
    """
    text = name.lower().replace("ё", "е")
    found = []
    for index, (part, whole) in enumerate(PART_WHOLE):
        for kind, pattern in (("part", part), ("whole", whole)):
            match = re.search(pattern, text)
            if match:
                found.append((match.start(), index, kind))
    if not found:
        return None
    _, index, kind = min(found)
    return index, kind


def item_scale_conflict(need_query: str, item_name: str) -> str | None:
    """Противоречит ли тип товара потребности: светильник на запрос «лампа» и наоборот.

    Принимает: need_query — запрос потребности («лампа уличного освещения»); item_name — товар.
    Возвращает: причину или None.
    """
    kind = _item_kind(item_name)
    if kind is None:
        return None
    index, item_type = kind
    need_parts, need_wholes = _scale(need_query)
    if item_type == "whole" and index in need_parts:
        return "нужна деталь, а товар — узел целиком"
    if item_type == "part" and index in need_wholes:
        return "нужен узел целиком, а товар — деталь"
    return None
