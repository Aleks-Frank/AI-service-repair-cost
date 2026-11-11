"""Этап «Материалы»: от принятых работ к позициям прайсов поставщиков с ценами.

    принятые работы сметы + materials из ответа модели на этапе 2
        │
        ▼  collect_needs: потребности (что искать, сколько на единицу работы, объём работы)
        ▼  группы: одинаковый материал разных работ и дефектов складывается в одну покупку
        ▼  find_material_candidates: RAG по прайсам — кандидаты типа товара (без цен)
        ▼  LLM: какие товары подходят (все подходящие артикулы, разные фасовки — тоже)
        ▼  код: вариант с наименьшей суммой покупки под объём (фасовка из наименования)
        ▼  rag.offers: тот же товар у всех поставщиков -> самый дешёвый в наличии
    строки раздела «Материалы»

Норм расхода нет: что и сколько нужно, предлагает модель — это помечается в смете.
Цены, количество к покупке и суммы считает только код по данным базы supplier_prices.
"""

import logging
import math
import re
from decimal import Decimal

from rag.offers import choose_offer, find_offers, purchase
from rag.prices_db import Material
from rag.retriever import MaterialRetriever
from llm.estimator.candidates import UNIVERSAL_WORK_ID, find_material_candidates
from llm.estimator.estimate import money
from llm.estimator.llm_client import LLMServiceClient
from llm.estimator.prompts import MATERIAL_SELECTION_PROMPT
from llm.estimator.schemas import MaterialSelection, Selection

logger = logging.getLogger(__name__)


def collect_needs(selection: Selection, works: list[dict]) -> list[dict]:
    """Собирает потребности в материалах по работам, принятым в основную смету.

    Материалы альтернатив и отклонённых работ не берутся. Работа, перенесённая из альтернатив
    в основной вариант, тоже учитывается (ищется по паре «дефект + код»).

    Принимает: selection — ответ модели этапа 2; works — основные строки сметы (make_estimate).
    Возвращает: список потребностей-словарей с need_id «m1», «m2», …
    """
    accepted = {(line["finding_id"], line["work_id"]): line for line in works}
    needs: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for choice in [*selection.works, *selection.alternatives]:
        key = (choice.finding_id, choice.work_id)
        line = accepted.get(key)
        if line is None or key in seen:
            continue
        seen.add(key)
        for material in choice.materials:
            needs.append({
                "need_id": f"m{len(needs) + 1}",
                "finding_id": choice.finding_id,
                "work_id": choice.work_id,
                "work_name": line["name"],
                "work_unit": line["unit"],
                # Объём работы (строка) или None, если он не подтверждён.
                "work_quantity": line["quantity"],
                "query": material.query,
                "qty_per_work_unit": str(Decimal(str(material.qty_per_work_unit))),
                "unit": material.unit,
                "consumable": material.consumable,
                "reason": material.reason,
            })
    return needs


def _need_quantity(need: dict) -> Decimal | None:
    """Потребность в базовых единицах = объём работы × количество на единицу работы.

    Принимает: need — потребность из collect_needs.
    Возвращает: Decimal или None, если объём работы неизвестен.
    """
    if need["work_quantity"] is None:
        return None
    return Decimal(need["work_quantity"]) * Decimal(need["qty_per_work_unit"])


def _number(value: Decimal | None) -> str | None:
    """Число строкой без лишних нулей: Decimal("27.000") -> «27», Decimal("2.50") -> «2.5».

    Принимает: value — число или None.
    Возвращает: строку или None.
    """
    return None if value is None else f"{value.normalize():f}"


def _offer_summary(offer) -> dict:
    """Краткое описание предложения поставщика для сметы и трассировки.

    Принимает: offer — rag.offers.Offer.
    Возвращает: словарь с поставщиком, артикулом, ценами и остатком.
    """
    m = offer.material
    return {
        "supplier": offer.supplier.code,
        "article": m.article,
        "name": m.name,
        "base_price_rub": str(m.base_price),
        "price_rub": str(m.price),
        "price_unit": m.unit,
        "stock": _number(m.stock),
        "match_type": offer.match_type,
        "match_score": offer.match_score,
    }


# Фасовка в наименовании: «Смывка … 12 кг», «Спрей … 400 мл», «Грунтовка … 15кг», «… 2,5 л».
_PACKAGE_RE = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)\s*(кг|г|л|мл)(?![а-яё])", re.IGNORECASE)
_PACKAGE_UNITS = {"кг": ("кг", Decimal(1)), "г": ("кг", Decimal("0.001")), "л": ("л", Decimal(1)), "мл": ("л", Decimal("0.001"))}


def package_size(name: str) -> Decimal | None:
    """Сколько вещества в одной упаковке, по наименованию: «Смывка … 12 кг» -> 12, «Спрей … 400 мл» -> 0.4.

    Килограммы и литры не различаются: для смесей, грунтовок и очистителей плотность близка
    к единице, а точнее по прайсу всё равно не определить.

    Принимает: name — наименование позиции прайса.
    Возвращает: размер упаковки в кг (л) или None, если фасовки в наименовании нет.
    """
    matches = _PACKAGE_RE.findall(name)
    if not matches:
        return None
    value, unit = matches[-1]
    return Decimal(value.replace(",", ".")) * _PACKAGE_UNITS[unit.lower()][1]


def _group_key(need: dict) -> tuple[str, str]:
    """Ключ группы: одинаковый материал для разных работ и дефектов покупается одной позицией.

    Принимает: need — потребность из collect_needs.
    Возвращает: (запрос без регистра и лишних пробелов, единица).
    """
    return " ".join(need["query"].lower().split()), need["unit"]


def _consumable_purchase(material: Material, need_qty: Decimal | None) -> tuple[Decimal, str | None]:
    """Сколько упаковок расходника купить под потребность в кг/л.

    Принимает: material — позиция прайса; need_qty — потребность в кг/л (None — неизвестна).
    Возвращает: (число единиц цены к покупке, предупреждение или None).
    """
    if material.base_unit in ("кг", "л"):
        # Товар продаётся на вес или объём: покупаем сколько нужно, минимум одну единицу.
        return Decimal(max(1, math.ceil(need_qty))) if need_qty else Decimal(1), None
    size = package_size(material.name)
    if need_qty is None:
        return Decimal(1), "объём работы не определён — взята одна упаковка"
    if size is None or size <= 0:
        return Decimal(1), "фасовка в наименовании не указана — взята одна упаковка"
    return Decimal(math.ceil(need_qty / size)), None


def _buy(material: Material, need: dict, need_qty: Decimal | None) -> tuple[Decimal | None, list[str]]:
    """Сколько единиц цены этой позиции купить под потребность.

    Принимает: material — позиция прайса; need — потребность (единица, расходник ли);
        need_qty — суммарная потребность (None — неизвестна).
    Возвращает: (количество к покупке или None, предупреждения).
    """
    if need["unit"] in ("кг", "л"):
        buy, warning = _consumable_purchase(material, need_qty)
        return buy, [warning] if warning else []
    if need_qty is None:
        # Расходник в упаковках или изделие с неизвестным объёмом работы: минимум одна единица
        # только для расходника — у изделия количество без объёма работы не угадать.
        return (Decimal(1), ["объём работы не определён — взята одна упаковка"]) if need["consumable"] else (None, [])
    if need["consumable"]:
        return Decimal(max(1, math.ceil(need_qty))), []
    return purchase(material, need_qty)[0], []


def _in_stock(material: Material, buy: Decimal | None) -> bool:
    """Хватает ли остатка позиции на покупку (неизвестный остаток считается достаточным).

    Принимает: material — позиция прайса; buy — сколько единиц цены купить.
    Возвращает: True — можно купить.
    """
    if material.stock is None:
        return True
    return material.stock > 0 and (buy is None or material.stock >= buy)


def price_materials(
    needs: list[dict],
    description: str,
    images: list[bytes],
    llm: LLMServiceClient,
    retriever: MaterialRetriever,
    per_need: int = 5,
    max_new_tokens: int = 1500,
) -> tuple[list[dict], list[dict], dict]:
    """Подбирает товар и поставщика для потребностей и считает стоимость.

    Одинаковые потребности разных работ и дефектов (тот же запрос и единица — «очиститель граффити, л»
    для трёх надписей) объединяются: количество складывается, покупка считается один раз,
    а в строке сметы указано, на какие дефекты она делится. Группы с разными запросами, но общим
    подходящим товаром («очиститель граффити» и «состав для удаления граффити») тоже объединяются.
    Модель отмечает все подходящие товары; программа выбирает вариант с наименьшей суммой покупки
    среди тех, что есть в наличии. Для расходников в кг/л учитывается фасовка: на 0,4 л возьмётся
    спрей 400 мл, а не канистра 12 кг; на 50 л — канистра, если она выгоднее.

    Принимает:
        needs — потребности из collect_needs;
        description, images — данные заявки (модель сверяет с ними характеристики);
        llm — клиент LLM-сервиса;
        retriever — поиск по прайсам (база supplier_prices);
        per_need — сколько кандидатов типа товара показывать модели на потребность;
        max_new_tokens — лимит ответа модели.
    Возвращает: (строки материалов, ненайденные потребности, трассировка этапа).
    """
    trace: dict = {"material_needs": needs}
    if not needs:
        return [], [], trace

    # --- 1. Группы одинаковых потребностей; поиск кандидатов — один раз на группу. ---
    groups: dict[tuple[str, str], list[dict]] = {}
    for need in needs:
        groups.setdefault(_group_key(need), []).append(need)
    heads = [members[0] for members in groups.values()]
    filtered: dict[str, list[dict]] = {}
    candidates = find_material_candidates(retriever, heads, per_need, log=filtered)
    trace["material_candidates"] = candidates
    # Товары другого масштаба (светильник на запрос «лампа»), не показанные модели.
    trace["material_candidates_filtered"] = filtered
    with_candidates = [head for head in heads if candidates[head["need_id"]]]

    # --- 2. Модель отмечает все подходящие товары (цен она не видит). ---
    choices: dict[str, tuple[list[str], str]] = {}
    if with_candidates:
        allowed = {head["need_id"]: {c["article"] for c in candidates[head["need_id"]]} for head in with_candidates}

        def validate(value: MaterialSelection) -> None:
            """Артикулы должны быть среди кандидатов своей потребности."""
            for choice in value.choices:
                if choice.need_id not in allowed:
                    raise ValueError(f"Неизвестная потребность {choice.need_id}.")
                extra = set(choice.articles) - allowed[choice.need_id]
                if extra:
                    raise ValueError(f"Артикулы {', '.join(sorted(extra))} не из candidates потребности {choice.need_id}.")

        payload = {
            "description": description or None,
            "NEEDS": [
                {
                    "need_id": head["need_id"],
                    "work": head["work_name"],
                    "query": head["query"],
                    "unit": head["unit"],
                    "candidates": [{k: c[k] for k in ("article", "name", "category", "unit")}
                                   for c in candidates[head["need_id"]]],
                }
                for head in with_candidates
            ],
        }
        selection, raw = llm.ask_json(MATERIAL_SELECTION_PROMPT, payload, images, MaterialSelection,
                                      validate, max_new_tokens=max_new_tokens)
        trace.update(raw_material_selection=raw, material_selection=selection.model_dump())
        choices = {choice.need_id: (choice.articles, choice.reason) for choice in selection.choices}

    # --- 3. Слияние групп с общими подходящими товарами. ---
    # Модель называет один материал по-разному («очиститель граффити» и «состав для удаления
    # граффити»). Если у двух групп с одной единицей есть общий подходящий товар — это один
    # материал: группы сливаются, выбор идёт среди общих товаров.
    merged: list[tuple[list[dict], set[str], str]] = []
    for members in groups.values():
        head = members[0]
        articles, reason = choices.get(head["need_id"], ([], ""))
        target = next((group for group in merged
                       if articles and group[1] & set(articles) and group[0][0]["unit"] == head["unit"]), None)
        if target is None:
            merged.append((list(members), set(articles), reason))
        else:
            target[0].extend(members)
            target[1].intersection_update(articles)

    # --- 4. По группе: самый выгодный подходящий товар, тот же товар у других поставщиков, сумма. ---
    lines: list[dict] = []
    not_found: list[dict] = []
    offers_trace: dict[str, list[dict]] = {}
    material_candidates = {c["article"]: c for head in heads for c in candidates[head["need_id"]]}
    for members, articles, reason in merged:
        head = members[0]
        work_ids = ", ".join(dict.fromkeys(m["work_id"] for m in members))
        if not candidates[head["need_id"]]:
            reason = "в прайсах не найдено похожих товаров"
        elif not articles:
            reason = reason or "модель не выбрала подходящий товар среди кандидатов"
        if not articles:
            not_found.append({"need_id": head["need_id"], "work_id": work_ids, "query": head["query"],
                              "unit": head["unit"], "reason": reason})
            continue

        # Суммарная потребность группы; если у части работ объём неизвестен — считаем по известным.
        quantities = [_need_quantity(m) for m in members]
        known = [q for q in quantities if q is not None]
        need_qty = sum(known, Decimal(0)) if known else None

        # Вариант с наименьшей суммой покупки: сначала среди тех, что есть в наличии.
        options = []
        for article in sorted(articles):
            material = retriever.session.get(Material, material_candidates[article]["material_id"])
            buy, buy_warnings = _buy(material, head, need_qty)
            cost = buy * material.price if buy is not None else material.price
            options.append((not _in_stock(material, buy), cost, material.price, material, buy_warnings))
        _, _, _, reference, buy_warnings = min(options, key=lambda option: option[:3])

        # Тот же товар (та же фасовка) у других поставщиков — самый дешёвый в наличии.
        offers = find_offers(retriever, reference)
        if not offers:
            not_found.append({"need_id": head["need_id"], "work_id": work_ids, "query": head["query"],
                              "unit": head["unit"], "reason": "поставщик товара отключён"})
            continue
        buy, buy_warnings = _buy(reference, head, need_qty)
        best, warnings = choose_offer(offers, buy if head["consumable"] or head["unit"] in ("кг", "л") else need_qty)
        offers_trace[head["need_id"]] = [_offer_summary(offer) for offer in offers]
        material = best.material
        buy, buy_warnings = _buy(material, head, need_qty)
        warnings += buy_warnings
        amount = buy * material.price if buy is not None else None

        if need_qty is None and not buy_warnings:
            warnings.append("объём работы не определён — количество и сумма не рассчитаны")
        if len(known) < len(quantities) and known:
            warnings.append("у части работ объём не определён — количество посчитано по остальным")
        if len(members) > 1:
            findings = ", ".join(dict.fromkeys(m["finding_id"] for m in members))
            warnings.append(f"одна покупка на {len(members)} работы ({findings})")
        if any(m["work_id"] == UNIVERSAL_WORK_ID for m in members):
            warnings.append("материал определён по описанию дефекта (работа не определена) — проверьте")
        if head["consumable"]:
            warnings.append("расход оценён моделью (норм расхода нет) — проверьте количество")
        unit_ok = head["unit"] in ("кг", "л") or head["consumable"] or material.base_unit == head["unit"]
        if not unit_ok:
            warnings.append(f"единица товара ({material.base_unit}) не совпадает с потребностью ({head['unit']})")
        lines.append({
            "need_id": head["need_id"],
            "finding_id": head["finding_id"],
            "work_id": work_ids,
            # Работы и дефекты, на которые делится покупка (у одиночной — одна запись).
            "shared": [{"finding_id": m["finding_id"], "work_id": m["work_id"], "need_qty": _number(q)}
                       for m, q in zip(members, quantities)],
            "need": head["query"],
            "need_reason": head["reason"],
            "choice_reason": reason,
            # Расходник: количество оценено моделью.
            "consumable": head["consumable"],
            "name": material.name,
            # Категория прайса — для проверки совместимости материала с работой (material_validation).
            "category": material.category or "",
            "article": material.article,
            "supplier": best.supplier.code,
            "supplier_name": best.supplier.name,
            "price_date": best.supplier.price_date.isoformat() if best.supplier.price_date else None,
            "qty_per_work_unit": head["qty_per_work_unit"],
            "need_qty": _number(need_qty),
            # Для расходника в кг/л — единица вещества, иначе базовая единица товара.
            "base_unit": head["unit"] if head["unit"] in ("кг", "л") else material.base_unit,
            "package_size": _number(package_size(material.name)),
            "buy_qty": _number(buy),
            "price_unit": material.unit,
            "units_per_price_unit": _number(material.units_per_price_unit),
            "price_rub": str(material.price),
            "base_price_rub": str(material.base_price),
            "amount_rub": money(amount) if amount is not None else None,
            "in_stock": best.available(buy if head["consumable"] or head["unit"] in ("кг", "л") else need_qty),
            "match_type": best.match_type,
            # Норм расхода нет: потребность (что и сколько) предложена моделью.
            "source": "llm",
            "other_offers": [_offer_summary(o) for o in offers if o is not best],
            "warnings": list(dict.fromkeys(warnings)),
        })
    trace["material_offers"] = offers_trace
    return lines, not_found, trace
