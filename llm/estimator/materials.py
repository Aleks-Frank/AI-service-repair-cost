"""Этап «Материалы»: от принятых работ к позициям прайсов поставщиков с ценами.

    принятые работы сметы + materials из ответа модели на этапе 2
        │
        ▼  collect_needs: потребности (что искать, сколько на единицу работы, объём работы)
        ▼  find_material_candidates: RAG по прайсам — кандидаты типа товара (без цен)
        ▼  LLM: какой именно товар подходит каждой потребности (артикул только из кандидатов)
        ▼  rag.offers: тот же товар у всех поставщиков -> самый дешёвый в наличии -> упаковки
    строки раздела «Материалы»

Норм расхода нет: что и сколько нужно, предлагает модель — это помечается в смете.
Цены, количество к покупке и суммы считает только код по данным базы supplier_prices.
"""

import logging
from decimal import Decimal

from rag.offers import choose_offer, find_offers, purchase
from rag.prices_db import Material
from rag.retriever import MaterialRetriever
from llm.estimator.candidates import find_material_candidates
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


def price_materials(
    needs: list[dict],
    description: str,
    images: list[bytes],
    llm: LLMServiceClient,
    retriever: MaterialRetriever,
    per_need: int = 5,
    max_new_tokens: int = 1500,
) -> tuple[list[dict], list[dict], dict]:
    """Подбирает товар и поставщика для каждой потребности и считает стоимость.

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

    # --- 1. Кандидаты типа товара для каждой потребности. ---
    candidates = find_material_candidates(retriever, needs, per_need)
    trace["material_candidates"] = candidates
    with_candidates = [need for need in needs if candidates[need["need_id"]]]

    # --- 2. Модель выбирает, какой именно товар подходит (цен она не видит). ---
    choices: dict[str, tuple[str | None, str]] = {}
    if with_candidates:
        allowed = {need["need_id"]: {c["article"] for c in candidates[need["need_id"]]} for need in with_candidates}

        def validate(value: MaterialSelection) -> None:
            """Артикул должен быть среди кандидатов своей потребности."""
            for choice in value.choices:
                if choice.need_id not in allowed:
                    raise ValueError(f"Неизвестная потребность {choice.need_id}.")
                if choice.article is not None and choice.article not in allowed[choice.need_id]:
                    raise ValueError(f"Артикул {choice.article} не из candidates потребности {choice.need_id}.")

        payload = {
            "description": description or None,
            "NEEDS": [
                {
                    "need_id": need["need_id"],
                    "work": need["work_name"],
                    "query": need["query"],
                    "unit": need["unit"],
                    "candidates": [{k: c[k] for k in ("article", "name", "category", "unit")}
                                   for c in candidates[need["need_id"]]],
                }
                for need in with_candidates
            ],
        }
        selection, raw = llm.ask_json(MATERIAL_SELECTION_PROMPT, payload, images, MaterialSelection,
                                      validate, max_new_tokens=max_new_tokens)
        trace.update(raw_material_selection=raw, material_selection=selection.model_dump())
        choices = {choice.need_id: (choice.article, choice.reason) for choice in selection.choices}

    # --- 3. Тот же товар у всех поставщиков, самый дешёвый в наличии, количество и сумма. ---
    lines: list[dict] = []
    not_found: list[dict] = []
    offers_trace: dict[str, list[dict]] = {}
    for need in needs:
        article, reason = choices.get(need["need_id"], (None, ""))
        if not candidates[need["need_id"]]:
            reason = "в прайсах не найдено похожих товаров"
        elif article is None:
            reason = reason or "модель не выбрала подходящий товар среди кандидатов"
        if article is None:
            not_found.append({**{k: need[k] for k in ("need_id", "work_id", "query", "unit")}, "reason": reason})
            continue

        candidate = next(c for c in candidates[need["need_id"]] if c["article"] == article)
        reference = retriever.session.get(Material, candidate["material_id"])
        offers = find_offers(retriever, reference)
        if not offers:
            not_found.append({**{k: need[k] for k in ("need_id", "work_id", "query", "unit")},
                              "reason": "поставщик товара отключён"})
            continue
        need_qty = _need_quantity(need)
        best, warnings = choose_offer(offers, need_qty)
        offers_trace[need["need_id"]] = [_offer_summary(offer) for offer in offers]
        material = best.material

        buy_qty, amount = purchase(material, need_qty) if need_qty is not None else (None, None)
        if need_qty is None:
            warnings.append("объём работы не определён — количество и сумма не рассчитаны")
        if material.base_unit != need["unit"]:
            warnings.append(f"единица товара ({material.base_unit}) не совпадает с потребностью ({need['unit']})")
        lines.append({
            "need_id": need["need_id"],
            "finding_id": need["finding_id"],
            "work_id": need["work_id"],
            "need": need["query"],
            "need_reason": need["reason"],
            "choice_reason": reason,
            "name": material.name,
            "article": material.article,
            "supplier": best.supplier.code,
            "supplier_name": best.supplier.name,
            "price_date": best.supplier.price_date.isoformat() if best.supplier.price_date else None,
            "qty_per_work_unit": need["qty_per_work_unit"],
            "need_qty": _number(need_qty),
            "base_unit": material.base_unit,
            "buy_qty": _number(buy_qty),
            "price_unit": material.unit,
            "units_per_price_unit": _number(material.units_per_price_unit),
            "price_rub": str(material.price),
            "base_price_rub": str(material.base_price),
            "amount_rub": money(amount) if amount is not None else None,
            "in_stock": best.available(need_qty),
            "match_type": best.match_type,
            # Норм расхода нет: потребность (что и сколько) предложена моделью.
            "source": "llm",
            "other_offers": [_offer_summary(o) for o in offers if o is not best],
            "warnings": warnings,
        })
    trace["material_offers"] = offers_trace
    return lines, not_found, trace
