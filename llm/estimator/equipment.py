"""Специальная техника для выполнения работ (автовышка и т. п.) — отдельно от работ и материалов.

Техника — не работа, не материал и не инструмент: это машина, которая нужна, чтобы работу можно
было выполнить (доступ к светильнику на высокой опоре). Поэтому она не попадает в materials и
в подбор товаров, а выводится отдельным необязательным блоком сметы equipment_requirements.

Что известно по данным проекта:
    - в справочнике ПЗР нет позиций на эксплуатацию машин и механизмов, аренду или подачу техники;
    - цена каждой расценки ПЗР — только труд (трудозатраты × стоимость чел.-ч), стоимость машин
      в неё не входит (например, 2-221 «Смена ламп в светильниках наружного освещения с приставных
      лестниц» — 0,5 чел.-ч × 1500 руб.);
    - в прайсах поставщиков аренды автовышки нет.
Стоимость техники считается по тарифу владельца (price_equipment; по умолчанию 9 000 руб. за 1 шт —
см. EstimatorSettings.aerial_platform_tariff_rub): названная в заявке или осмотре и вероятная при работе
на опоре/столбе — в итог сметы, прочая вероятная (высота, фасад) — в предварительную сумму. Без тарифа требование выводится с pricing_status «not_priced». Если в справочнике появится расценка,
в наименовании или составе которой уже есть техника, требование помечается «included_in_rate» —
без второй оплаты.

Правила (по работе сметы и тексту заявки, без вызова модели). Если в заявке несколько дефектов,
признаки берутся только из описания дефекта: общий текст заявки не переносит технику на соседний дефект.
    - в описании или дефекте прямо названа техника («автовышка», «АГП», «люлька») — требование
      установлено по данным заявки (needs_human_confirmation = False);
    - работа по освещению, и в тексте есть признак высоты или наружной опоры («столб», «опора»,
      «высота», «фасад») — требование вероятно, нужна проверка (needs_human_confirmation = True);
    - работа по освещению на улице без признаков высоты («фонарь», «наружное освещение») — требования
      нет, только предупреждение проверить способ доступа: одного слова «фонарь» недостаточно;
    - остальные работы — ничего.
Для «Ремонтных работ» (1-282) наименование ничего не говорит о работе, поэтому признак «работа по
освещению» берётся из описания самого дефекта (иначе автовышка терялась, когда работа ушла в fallback).
Количество (машино-часы, смены) не известно и не выдумывается: quantity и unit — None.
Запись осмотра мастера (inspection) учитывается как текст дефекта: техника, названная в осмотре, —
status «confirmed»; рабочая высота («автовышка 15–20 м») сохраняется как working_height_m строкой.
Порог высоты, с которого автовышка обязательна, не задаётся: это правило владельца процесса / охраны труда.
"""

import re
from decimal import Decimal

# Код «Ремонтных работ» (см. candidates.UNIVERSAL_WORK_ID; здесь без импорта, чтобы модуль не зависел от RAG).
UNIVERSAL_WORK_ID = "1-282"

AERIAL_PLATFORM = {
    "resource_type": "machinery",
    "name": "Автовышка",
    "technical_name": "Автогидроподъёмник (АГП) с рабочей платформой",
}
_EXPLICIT = re.compile(r"автовышк|автогидроподъ|\bагп\b|люльк", re.IGNORECASE)
_HEIGHT = re.compile(r"столб|опор[аеуыо]|высот|фасад|кровл", re.IGNORECASE)
# Рабочая высота рядом с названием техники: «автовышка 15-20 м», «АГП 18 м».
_WORKING_HEIGHT = re.compile(r"(?:автовышк|автогидроподъ|\bагп\b|люльк)\w*\D{0,15}?(\d+(?:[.,]\d+)?"
                             r"(?:\s*[-–—]\s*\d+(?:[.,]\d+)?)?)\s*м\b", re.IGNORECASE)
_OUTDOOR = re.compile(r"фонар|уличн|наружн|придомов|во дворе", re.IGNORECASE)
# Работа на опоре или столбе: вероятная автовышка входит в итог сметы (решение владельца, 06.11.2026).
_POLE = re.compile(r"столб|опор[аеуыо]", re.IGNORECASE)
_LIGHTING_WORK = re.compile(r"ламп|светильник|прожектор|освещени", re.IGNORECASE)


def _quote(text: str, pattern: re.Pattern) -> str:
    """Фрагмент текста вокруг первого совпадения — основание требования для сметчика."""
    match = pattern.search(text)
    if not match:
        return ""
    start, end = max(0, match.start() - 40), min(len(text), match.end() + 40)
    return text[start:end].strip()


def equipment_requirements(works: list[dict], findings: dict[str, dict], description: str,
                           inspection: str = "") -> tuple[list[dict], list[str]]:
    """Определяет технику для основных работ сметы.

    Принимает:
        works — основные строки работ сметы (work_id, name, composition, finding_id);
        findings — {finding_id: {"object_name", "visible_issue"}} из этапа 1;
        description — текст заявки;
        inspection — запись осмотра мастера ("" — осмотра нет).
    Возвращает: (требования к технике — по одному на дефект, предупреждения для сметчика).
    """
    requirements: list[dict] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for line in works:
        finding_id = line["finding_id"]
        if finding_id in seen:
            continue
        finding = findings.get(finding_id, {})
        own = f"{finding.get('object_name', '')} {finding.get('visible_issue', '')}"
        # Текст заявки общий для всех дефектов: при нескольких дефектах признаки высоты берутся только
        # из описания самого дефекта, чтобы техника не перешла на соседний дефект.
        text = f"{description} {inspection} {own}" if len(findings) <= 1 else own
        work_text = f"{line['name']} {line.get('composition') or ''}"
        if line["work_id"] == UNIVERSAL_WORK_ID:
            work_text += f" {own}"
        named_in_request = len(findings) > 1 and _EXPLICIT.search(f"{description} {inspection}")
        if _EXPLICIT.search(text):
            confirmed, evidence = False, f"В заявке названа техника: «{_quote(text, _EXPLICIT)}»"
        elif named_in_request and _LIGHTING_WORK.search(work_text) and (_HEIGHT.search(own) or _OUTDOOR.search(own)):
            confirmed, evidence = True, (f"В заявке названа техника («{_quote(description, _EXPLICIT)}»), "
                                         "к какому дефекту она относится, не указано — проверьте.")
        elif _LIGHTING_WORK.search(work_text) and _HEIGHT.search(text):
            confirmed, evidence = True, (f"Работа по освещению, в заявке признак высоты: «{_quote(text, _HEIGHT)}». "
                                         "Нужна ли автовышка, зависит от высоты и доступа — проверьте.")
        elif _LIGHTING_WORK.search(work_text) and _OUTDOOR.search(text):
            warnings.append(f"Для {finding_id} (наружное освещение) способ доступа не определён — "
                            "проверьте, нужна ли автовышка.")
            continue
        else:
            continue
        seen.add(finding_id)
        included = bool(_EXPLICIT.search(work_text))
        height = _WORKING_HEIGHT.search(text)
        requirements.append({
            **AERIAL_PLATFORM,
            "finding_id": finding_id,
            "work_id": line["work_id"],
            "purpose": f"Доступ для работы «{line['name'][:80]}»",
            "evidence": evidence,
            "quantity": None,
            "unit": None,
            "pricing_status": "included_in_rate" if included else "not_priced",
            "needs_human_confirmation": confirmed,
            # confirmed — техника прямо названа в заявке или осмотре; probable — нужна проверка.
            "status": "probable" if confirmed else "confirmed",
            "working_height_m": height.group(1).replace(" ", "") if height else None,
            # Работа на опоре/столбе: без автовышки её не выполнить — стоимость идёт в итог, даже если
            # техника в заявке прямо не названа (решение владельца, 06.11.2026).
            "on_pole": bool(_POLE.search(text)),
        })
        if not included:
            warnings.append(f"Требуется техника: {AERIAL_PLATFORM['name']} ({AERIAL_PLATFORM['technical_name']}) "
                            f"для {finding_id}; стоимость и единица тарификации не определены — в ПЗР и прайсах "
                            "нет позиции, нужен тариф/договор. В итог сметы не включена.")
    return requirements, warnings


def price_equipment(result: dict, tariff_rub: Decimal | None, unit: str = "смена",
                    units_per_order: Decimal = Decimal(1)) -> dict:
    """Включает технику в суммы сметы по тарифу из настроек (эксперимент B9).

    Тарифа в ПЗР и прайсах нет, поэтому он задаётся владельцем (AERIAL_PLATFORM_TARIFF_RUB).
    Пока он не задан (None), смета не меняется — как раньше, pricing_status «not_priced».
    С тарифом в подтверждённую сумму total_rub входит техника, прямо названная в заявке или осмотре
    (status «confirmed»), и вероятная при работе на опоре или столбе (on_pole); остальная вероятная
    (высота, фасад, кровля) — только в предварительную total_with_review_rub, как материалы REVIEW. Включённая в расценку (included_in_rate)
    второй раз не оплачивается.

    Принимает:
        result — смета (изменяется на месте);
        tariff_rub — цена за единицу тарификации или None;
        unit — единица тарификации («смена», «час», «рейс»);
        units_per_order — сколько единиц на одно требование (минимальный заказ).
    Возвращает: ту же смету с полями resources_total_rub и resources_review_total_rub.
    """
    from llm.estimator.estimate import money

    requirements = result.get("equipment_requirements") or []
    if tariff_rub is None or not requirements:
        return result
    confirmed = review = Decimal(0)
    for item in requirements:
        if item["pricing_status"] == "included_in_rate":
            continue
        amount = tariff_rub * units_per_order
        item.update(quantity=str(units_per_order), unit=unit, unit_price_rub=money(tariff_rub),
                    amount_rub=money(amount), pricing_status="priced_by_tariff")
        if item["status"] == "confirmed" or item.get("on_pole"):
            confirmed += amount
        else:
            review += amount
    base = Decimal(result["total_rub"]) if result.get("total_rub") is not None else Decimal(0)
    with_review = (Decimal(result["total_with_review_rub"]) if result.get("total_with_review_rub") is not None
                   else base)
    result.update(
        resources_total_rub=money(confirmed),
        resources_review_total_rub=money(review),
        total_rub=money(base + confirmed),
        total_with_review_rub=money(with_review + confirmed + review),
    )
    # Предупреждение «стоимость не определена» больше не относится к технике с тарифом.
    result["warnings"] = [w for w in result["warnings"] if "стоимость и единица тарификации не определены" not in w]
    return result
