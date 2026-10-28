"""Подбор поставщика: один и тот же товар у всех поставщиков и самое дешёвое предложение.

    эталон (позиция прайса, выбранная как «нужный товар»)
        │
        ▼  find_offers: тот же товар у других поставщиков
        │     1) тот же артикул производителя (article_norm);
        │     2) иначе — по наименованию: совпали ключевые параметры И reranker ≥ порога
        ▼
    предложения (Offer)
        │
        ▼  choose_offer: самое дешёвое за базовую единицу среди тех, что есть в нужном количестве
        ▼
    purchase: сколько покупать (с учётом упаковки) и сумма

Модуль работает только с данными базы supplier_prices и не обращается к LLM.
"""

import math
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import select

from rag.price_list_parser import CRITICAL_PARAMS, model_tokens, models_match, params_match, product_type
from rag.prices_db import Material, Supplier
from rag.retriever import MaterialRetriever

# Минимальная оценка reranker'а, при которой товар без общего артикула считается тем же товаром.
# Порог высокий: лучше «не нашли у другого поставщика», чем подменить товар похожим.
NAME_MATCH_THRESHOLD = 0.9
# Пониженный порог, если у эталона есть критичные параметры (размер, ток, мощность, цоколь)
# и они совпали: тогда защиту от подмены даёт совпадение параметров, а сильно переписанные
# названия («Угол внешний КМН 40х16 белый» / «Угол наружный 40*16 бел.», оценка ~0,6) не теряются.
NAME_MATCH_THRESHOLD_WITH_PARAMS = 0.5
# Сколько позиций других поставщиков проверять при сопоставлении по наименованию.
NAME_MATCH_CANDIDATES = 10


@dataclass
class Offer:
    """Предложение поставщика: позиция прайса и то, как она сопоставлена с эталоном."""

    material: Material
    supplier: Supplier
    # reference — сам эталон; article — тот же артикул производителя; name — по наименованию.
    match_type: str
    match_score: float | None = None

    def available(self, need_base_qty: Decimal | None) -> bool:
        """Хватает ли остатка поставщика на потребность.

        Остаток в прайсе указан в единицах цены (комплектах), потребность — в базовых (штуках).
        Если поставщик остатки не передаёт (stock = None), товар не считается отсутствующим:
        неизвестный остаток — не то же самое, что нулевой.

        Принимает: need_base_qty — потребность в базовых единицах (None — неизвестна, хватит любого остатка).
        Возвращает: True, если остаток покрывает потребность или неизвестен.
        """
        stock = self.material.stock
        if stock is None:
            return True
        if stock <= 0:
            return False
        return need_base_qty is None or stock * self.material.units_per_price_unit >= need_base_qty

    def price_outdated(self, today: date | None = None) -> bool:
        """Устарела ли цена: дата прайса старше срока актуальности поставщика.

        Принимает: today — текущая дата (по умолчанию сегодня).
        Возвращает: True, если цена устарела; False, если актуальна или дата прайса неизвестна.
        """
        if self.supplier.price_date is None:
            return False
        return (today or date.today()) - self.supplier.price_date > timedelta(days=self.supplier.price_valid_days)


def find_offers(retriever: MaterialRetriever, reference: Material) -> list[Offer]:
    """Находит тот же товар у всех активных поставщиков.

    Принимает:
        retriever — поиск по прайсам (его сессия используется для запросов к базе);
        reference — эталон: позиция прайса, выбранная как нужный товар.
    Возвращает: список предложений (эталон всегда первый, если его поставщик активен).
    """
    session = retriever.session
    suppliers = {s.code: s for s in session.scalars(select(Supplier).where(Supplier.active))}
    if reference.supplier_code not in suppliers:
        return []
    offers = [Offer(reference, suppliers[reference.supplier_code], "reference")]

    # --- 1. Тот же артикул производителя у других поставщиков. ---
    # Сравнивать можно, только если и у эталона, и у другого поставщика артикул — код производителя.
    if suppliers[reference.supplier_code].article_is_manufacturer:
        same_article = session.scalars(
            select(Material).where(
                Material.article_norm == reference.article_norm,
                Material.supplier_code != reference.supplier_code,
            )
        )
        for material in same_article:
            supplier = suppliers.get(material.supplier_code)
            if supplier and supplier.article_is_manufacturer and material.base_unit == reference.base_unit:
                offers.append(Offer(material, supplier, "article", 1.0))

    # --- 2. По наименованию — у поставщиков, где товар по артикулу не нашёлся. ---
    found = {offer.supplier.code for offer in offers}
    remaining = [code for code in suppliers if code not in found]
    if remaining:
        has_critical = any(key in reference.params for key in CRITICAL_PARAMS)
        threshold = NAME_MATCH_THRESHOLD_WITH_PARAMS if has_critical else NAME_MATCH_THRESHOLD
        hits = retriever.search(reference.name, top_k=NAME_MATCH_CANDIDATES, suppliers=remaining).hits
        reference_models = model_tokens(reference.name, reference.brand)
        reference_type = product_type(reference.name)
        best: dict[str, Offer] = {}
        for hit in hits:
            material: Material = hit.item  # type: ignore[assignment]
            score = hit.rerank_score or 0.0
            # Тот же товар: высокая оценка, тот же тип изделия, та же единица, те же размеры/ток/мощность
            # и та же модель.
            if (score >= threshold and material.base_unit == reference.base_unit
                    and product_type(material.name) == reference_type
                    and params_match(reference.params, material.params)
                    and models_match(reference_models, model_tokens(material.name, material.brand))):
                current = best.get(material.supplier_code)
                if current is None or score > (current.match_score or 0):
                    best[material.supplier_code] = Offer(material, suppliers[material.supplier_code], "name", round(score, 3))
        offers.extend(best.values())
    return offers


def choose_offer(offers: list[Offer], need_base_qty: Decimal | None) -> tuple[Offer, list[str]]:
    """Выбирает самое дешёвое предложение за базовую единицу среди тех, что есть в наличии.

    Если в нужном количестве нет ни у кого — самое дешёвое вообще с предупреждением.
    При равной цене — выше приоритет поставщика, затем больше остаток.

    Принимает: offers — предложения из find_offers (не пустой список); need_base_qty — потребность.
    Возвращает: (выбранное предложение, список предупреждений).
    """
    warnings: list[str] = []
    in_stock = [offer for offer in offers if offer.available(need_base_qty)]
    pool = in_stock or offers
    if not in_stock:
        warnings.append("нет в наличии в нужном количестве ни у одного поставщика")
    best = min(pool, key=lambda o: (o.material.base_price, -o.supplier.priority, -(o.material.stock or 0)))
    if best.match_type == "name":
        warnings.append(f"товар у поставщика {best.supplier.code} сопоставлен по наименованию (оценка {best.match_score}), "
                        "а не по артикулу — проверьте")
    if best.price_outdated():
        warnings.append(f"цена {best.supplier.code} от {best.supplier.price_date:%d.%m.%Y} старше "
                        f"{best.supplier.price_valid_days} дней")
    if best.material.sold_by_pack and best.material.base_unit == best.material.unit:
        warnings.append("состав упаковки в прайсе не указан — количество к покупке в упаковках")
    return best, warnings


def purchase(material: Material, need_base_qty: Decimal) -> tuple[Decimal, Decimal]:
    """Считает, сколько покупать в единицах цены, и сумму.

    Цена за упаковку («компл», «уп») — покупаем целыми упаковками, округляя вверх.
    Цена за штуку — целыми штуками; за метр — сколько нужно.

    Принимает: material — выбранная позиция; need_base_qty — потребность в базовых единицах.
    Возвращает: (количество к покупке в единицах цены, сумма без округления).
    """
    if material.sold_by_pack:
        buy = Decimal(math.ceil(need_base_qty / material.units_per_price_unit))
    elif material.base_unit == "м":
        buy = need_base_qty
    else:
        buy = Decimal(math.ceil(need_base_qty))
    return buy, buy * material.price
