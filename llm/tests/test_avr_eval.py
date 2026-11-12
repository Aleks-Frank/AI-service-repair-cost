"""Тесты расчёта метрик оценки АВР (llm/estimator/avr_eval.py) — на синтетических сметах, без моделей."""

from llm.estimator.avr_eval import aggregate, evaluate_request

GOLD = {
    "work_items": [{"name": "Смена ламп", "any_of": ["2-220"], "related": [], "quantity": 2, "unit": "шт"}],
    "materials": [{"name": "Лампа G13", "match": "ламп.*(g13|t8)", "quantity": 2, "amount": 200.0}],
    "additional_resources": [{"name": "Автовышка", "match": "вышк", "quantity": 1, "amount": 9000.0}],
    "works_total": 417.0, "materials_total": 200.0, "resources_total": 9000.0, "actual_total": 9617.0,
}


def estimate(codes_qty, materials, works_total, total, equipment=()):
    """Смета в формате make_estimate (только нужные оценке поля)."""
    return {"works": [{"work_id": code, "quantity": qty} for code, qty in codes_qty],
            "materials": [{"name": name} for name in materials], "works_total_rub": works_total,
            "total_rub": total, "equipment_requirements": [{"name": name} for name in equipment]}


def trace(*retrieved):
    """Трассировка с кандидатами одного дефекта."""
    return {"findings": {"f1": {"retrieved": [{"work_id": code} for code in retrieved]}}}


def test_exact_work_quantity_and_material():
    """Верная работа, количество и лампа; техника найдена; ранг эталона в поиске учитывается."""
    result = evaluate_request(estimate([("2-220", "2")], ["Лампа LED T8 G13"], "417", "617", ["Автовышка"]),
                              trace("1-282", "2-218", "2-220"), GOLD)
    assert result["retrieval_ranks"] == [2]  # 1-282 не считается кандидатом
    assert result["works"] == [{"gold": "2-220", "outcome": "exact"}]
    assert result["quantities"] == [{"code": "2-220", "predicted": 2.0, "actual": 2}]
    assert result["materials"]["tp"] == 1 and result["materials"]["fp"] == 0
    assert result["resources"] == [{"name": "Автовышка", "detected": True}]


def test_wrong_neighbour_and_extra_material():
    """Соседняя лампа E27: работа ложная, материал лишний, эталонный пропущен."""
    result = evaluate_request(estimate([("2-218", "1")], ["Лампа E27", "Краска фасадная"], "45", "100"),
                              trace("2-218"), GOLD)
    assert result["work_counts"] == {"tp": 0, "fp": 1, "fn": 1, "related": 0}
    assert result["retrieval_ranks"] == [None]
    assert result["materials"]["tp"] == 0 and result["materials"]["fp"] == 2 and result["materials"]["fn"] == 1


def test_aggregate_money_metrics():
    """WAPE, смещение и допуски: n — число заявок с ненулевым фактом; без техники — отдельно."""
    good = evaluate_request(estimate([("2-220", "2")], [], "417", "9000"), trace("2-220"), GOLD)
    bad = evaluate_request(estimate([("2-218", "1")], [], "45", "100"), trace("2-218"), GOLD)
    metrics = aggregate({"a": good, "b": bad})
    assert metrics["total"]["n"] == 2
    assert metrics["total"]["within_10pct"] == 1 and metrics["total"]["within_20pct"] == 1
    assert metrics["works_cost"]["within_10pct"] == 1
    assert metrics["total"]["bias_pct"] < 0 and metrics["total"]["wape_pct"] > 0
    assert metrics["retrieval"]["recall_at_6"] == 0.5
    assert metrics["total_without_resources"]["within_20pct"] == 0  # 9000 против 617 без техники
