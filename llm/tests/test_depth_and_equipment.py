"""Тесты эксперимента 6/10 кандидатов и учёта техники (автовышка) — без баз и моделей."""

import json

import pytest
from test_retrieval import Retriever, observations as one_finding
from test_selection_fallback import (
    FakeLLM, FakeMaterialRetriever, FakeRetriever, LAMP, RATES, finding, make_rate, observations, run, selection, work,
)

from llm.estimator import depth_experiment
from llm.estimator.candidates import UNIVERSAL_WORK_ID, find_candidates
from llm.estimator.equipment import equipment_requirements
from llm.estimator.estimate import estimate_csv
from llm.estimator.pipeline import analyze_case
from llm.estimator.schemas import Observations

# --- Глубина и эксперимент ------------------------------------------------------------------


def test_candidates_7_to_10_only_at_depth_10():
    """1, 3: на одном входе глубина 10 даёт кандидатов с 7–10 мест, глубина 6 — нет."""
    lists = {}
    for depth in (6, 10):
        log: dict = {}
        find_candidates(Retriever({}, generated="лампа"), one_finding(), per_finding=depth, log=log)
        lists[depth] = [item["work_id"] for item in log["f1"]["retrieved"]][:depth]
    assert "лампа-8" in lists[10] and "лампа-8" not in lists[6]
    assert lists[10][:6] == lists[6]


@pytest.mark.parametrize("depth", [6, 10])
def test_universal_excluded_at_any_depth(depth):
    """2: 1-282 не попадает в кандидаты ни при одной глубине."""
    codes = [c["work_id"] for c in find_candidates(Retriever({"лампа": [UNIVERSAL_WORK_ID, "2-218"]}), one_finding(),
                                                   per_finding=depth)]
    assert UNIVERSAL_WORK_ID not in codes


GOLD = [{"name": "Смена ламп", "any_of": ["2-220"], "related": ["2-221"]}]


def test_recall_is_not_accuracy():
    """4: эталон среди показанных кандидатов, но модель отказалась — это не верный выбор."""
    verdict = depth_experiment.classify_finding([UNIVERSAL_WORK_ID], "model_unmatched", ["2-218", "2-220"], GOLD)
    assert verdict["outcome"] == "unjustified_unmatched" and verdict["gold_shown"]


@pytest.mark.parametrize("works,status,shown,expected", [
    (["2-220"], "matched", ["2-220"], "correct"),
    (["2-221"], "matched", ["2-221"], "related"),            # 6: родственная — не точная
    (["2-14"], "matched", ["2-14"], "false"),
    ([UNIVERSAL_WORK_ID], "model_unmatched", ["2-14"], "justified_unmatched"),
    ([], "model_empty", ["2-220"], "error"),                  # 7: ошибки различаются
    ([], "llm_error", [], "error"),
])
def test_finding_outcomes(works, status, shown, expected):
    """6, 7: точная, родственная, ложная работа, отказы и ошибки — разные исходы."""
    assert depth_experiment.classify_finding(works, status, shown, GOLD)["outcome"] == expected


def test_gold_universal_outcomes():
    """Эталон «Ремонтные работы»: 1-282 и конкретная работа — отдельные исходы, не «ложь»."""
    gold = [{"name": "Ремонтные работы", "any_of": [UNIVERSAL_WORK_ID], "universal": True}]
    assert depth_experiment.classify_finding([UNIVERSAL_WORK_ID], "model_unmatched", [], gold)["outcome"] == \
        "gold_universal_fallback"
    assert depth_experiment.classify_finding(["2-159"], "matched", [], gold)["outcome"] == "gold_universal_specific"


def write_run(out, depth, name, findings_count, works):
    """Сохраняет фиктивную пару (глубина, заявка) в формате run."""
    run_dir = out / f"d{depth}" / name
    run_dir.mkdir(parents=True)
    findings = {f"f{i}": {"status": "matched", "prompt_candidates": [{"work_id": "2-220"}],
                          "retrieved": [{"work_id": "2-220"}]} for i in range(1, findings_count + 1)}
    estimate = {"works": [{"work_id": w, "finding_id": "f1"} for w in works], "finding_statuses": {}}
    (run_dir / "estimate.json").write_text(json.dumps(estimate), encoding="utf-8")
    (run_dir / "trace.json").write_text(json.dumps({"findings": findings, "elapsed_s": 1.0}), encoding="utf-8")


def test_ambiguous_mapping_excluded(tmp_path, monkeypatch):
    """5: заявка с несколькими дефектами не входит в метрики по дефектам, а помечается ambiguous."""
    monkeypatch.setattr(depth_experiment, "load_requests", lambda: [{"name": "a01"}, {"name": "a07"}])
    gold = {"a01": GOLD, "a07": GOLD}
    (tmp_path / "gold.json").write_text(json.dumps(gold), encoding="utf-8")
    for depth in (6, 10):
        write_run(tmp_path, depth, "a01", 1, ["2-220"])
        write_run(tmp_path, depth, "a07", 2, ["2-220"])
    report = depth_experiment.report(tmp_path, tmp_path / "gold.json")
    assert report["labeled"]["d6"]["findings"] == 1
    assert report["labeled"]["d6"]["outcomes"] == {"correct": 1}
    assert report["ambiguous"] == [{"request": "a07", "findings": 2, "reason": "ambiguous_finding_mapping"}]


def test_saved_observations_skip_stage_one():
    """8: при готовых наблюдениях этап 1 не вызывается — обе глубины получают одинаковый вход."""
    saved = Observations.model_validate(observations(finding("f1", "лампа")))
    llm = FakeLLM(selection=[selection([work("f1", "2-218")])])  # ответов этапа 1 нет — вызов упал бы
    result, trace = analyze_case([], "заявка", llm, FakeRetriever(LAMP), observations=saved)
    assert [line["work_id"] for line in result["works"]] == ["2-218"]
    assert llm.prompts("observation") == [] and trace["observations"] == saved.model_dump()


# --- Техника --------------------------------------------------------------------------------

LAMP_LINE = {"work_id": "2-221", "name": "Смена ламп в светильниках наружного освещения с приставных лестниц",
             "composition": "", "finding_id": "f1"}


def findings_of(**issues):
    """{finding_id: {"object_name", "visible_issue"}}."""
    return {fid: {"object_name": issue, "visible_issue": issue} for fid, issue in issues.items()}


def test_high_pole_gives_requirement_needing_confirmation():
    """9, 11: высокая опора — требование к автовышке, но с подтверждением; количество не выдумывается."""
    found, warnings = equipment_requirements([LAMP_LINE], findings_of(f1="фонарь"), "Не горит фонарь на столбе у дома")
    assert len(found) == 1
    item = found[0]
    assert item["name"] == "Автовышка" and item["needs_human_confirmation"] is True
    assert item["quantity"] is None and item["unit"] is None and item["pricing_status"] == "not_priced"
    assert any("стоимость и единица тарификации не определены" in w for w in warnings)


def test_explicitly_named_equipment_is_confirmed():
    """9: техника прямо названа в заявке — требование установлено по данным заявки."""
    found, _ = equipment_requirements([LAMP_LINE], findings_of(f1="фонарь"), "Замена прожекторов, нужна автовышка 15-20 м")
    assert found[0]["needs_human_confirmation"] is False


def test_low_indoor_lamp_no_equipment():
    """10: лампа в холле — техники нет и предупреждений о ней нет."""
    line = {**LAMP_LINE, "work_id": "2-220", "name": "Смена ламп люминесцентных"}
    assert equipment_requirements([line], findings_of(f1="лампа"), "Перегорела лампа в лифтовом холле") == ([], [])


def test_outdoor_without_height_only_warns():
    """11: одного слова «фонарь» недостаточно — только предупреждение проверить доступ."""
    found, warnings = equipment_requirements([LAMP_LINE], findings_of(f1="фонарь"), "Во дворе не работает фонарь")
    assert found == [] and any("способ доступа" in w for w in warnings)


def test_included_in_rate_not_paid_twice():
    """13: техника уже в расценке — pricing_status included_in_rate, без предупреждения о второй оплате."""
    line = {**LAMP_LINE, "name": "Смена ламп в светильниках наружного освещения с автовышки"}
    found, warnings = equipment_requirements([line], findings_of(f1="фонарь"), "Нужна автовышка")
    assert found[0]["pricing_status"] == "included_in_rate" and warnings == []


def test_equipment_stays_with_its_finding():
    """15: признак высоты у одного дефекта не переносит технику на соседний."""
    lines = [{**LAMP_LINE, "finding_id": "f1", "work_id": "2-220", "name": "Смена ламп люминесцентных"},
             {**LAMP_LINE, "finding_id": "f2"}]
    found, _ = equipment_requirements(lines, findings_of(f1="лампа в холле", f2="фонарь на высокой опоре"),
                                      "В холле перегорела лампа, во дворе не горит фонарь на столбе")
    assert [item["finding_id"] for item in found] == ["f2"]


def test_equipment_not_in_materials_and_totals_unchanged():
    """12, 14: автовышка не уходит в материалы и подбор товаров, сумма сметы не меняется."""
    RATES.setdefault("2-221", make_rate("2-221", LAMP_LINE["name"], price="750.00"))
    retriever = FakeRetriever({"фонарь": [("2-221", 0.99)]})
    chosen = {**work("f1", "2-221", [{"query": "лампа светодиодная E27", "unit": "шт", "qty_per_work_unit": 1}]),
              "quantity_source": "description", "quantity_evidence": "заменить 1 лампу"}
    llm = FakeLLM(observation=[observations(finding("f1", "фонарь"))], selection=[selection([chosen])])
    result, trace = analyze_case([], "Не горит фонарь на столбе, заменить 1 лампу, нужна автовышка", llm, retriever,
                                 material_retriever=FakeMaterialRetriever())
    assert [item["name"] for item in result["equipment_requirements"]] == ["Автовышка"]
    needs = " ".join(need["query"] for need in trace.get("material_needs", []))
    assert "вышк" not in needs.lower()
    assert all("вышк" not in item["query"].lower() for item in result["materials_not_found"])
    assert result["works_total_rub"] == "750.00"


def test_old_estimate_without_equipment_still_works():
    """16: смета без нового поля (старый JSON) читается: CSV формируется без раздела «Техника»."""
    llm = FakeLLM(observation=[observations(finding("f1", "лампа"))], selection=[selection([work("f1", "2-218")])])
    result, _ = run(llm, FakeRetriever(LAMP))
    result.pop("equipment_requirements", None)
    assert "Техника" not in estimate_csv(result)


def test_equipment_for_universal_work_uses_finding_text():
    """Регрессия a01: работа ушла в 1-282 — признак освещения берётся из дефекта, автовышка не теряется."""
    line = {"work_id": UNIVERSAL_WORK_ID, "name": "Ремонтные работы", "composition": "", "finding_id": "f1"}
    found, warnings = equipment_requirements(
        [line], findings_of(f1="светильники наружного освещения на опорах"),
        "Частично неисправно освещение на 2-х столбах на придомовой территории")
    assert [item["status"] for item in found] == ["probable"] and found[0]["work_id"] == UNIVERSAL_WORK_ID
    assert found[0]["pricing_status"] == "not_priced"


def test_universal_work_without_lighting_no_equipment():
    """Контроль: 1-282 для дефекта не по освещению (экран батареи на высоком этаже) — техники нет."""
    line = {"work_id": UNIVERSAL_WORK_ID, "name": "Ремонтные работы", "composition": "", "finding_id": "f1"}
    assert equipment_requirements([line], findings_of(f1="экран батареи оторван"), "На 16 этаже оторван экран") == ([], [])


def test_equipment_tariff_confirmed_into_total_probable_into_review():
    """B9: названная в заявке автовышка и вероятная на опоре — в итог; прочая вероятная — в предварительный."""
    from decimal import Decimal

    from llm.estimator.equipment import price_equipment

    def estimate(status):
        return {"total_rub": "750.00", "warnings": ["… стоимость и единица тарификации не определены …"],
                "equipment_requirements": [{"name": "Автовышка", "status": status, "pricing_status": "not_priced",
                                            "quantity": None, "unit": None}]}

    named = price_equipment(estimate("confirmed"), Decimal("9000"))
    assert named["total_rub"] == "9750.00" and named["total_with_review_rub"] == "9750.00"
    assert named["equipment_requirements"][0]["amount_rub"] == "9000.00" and not named["warnings"]
    probable = price_equipment(estimate("probable"), Decimal("9000"))
    assert probable["total_rub"] == "750.00" and probable["total_with_review_rub"] == "9750.00"
    assert probable["resources_review_total_rub"] == "9000.00"
    on_pole = estimate("probable")
    on_pole["equipment_requirements"][0]["on_pole"] = True
    on_pole = price_equipment(on_pole, Decimal("9000"))
    assert on_pole["total_rub"] == "9750.00"  # работа на опоре: вероятная автовышка — в итог
    untouched = price_equipment(estimate("confirmed"), None)
    assert untouched["total_rub"] == "750.00" and untouched["equipment_requirements"][0]["pricing_status"] == "not_priced"


def test_equipment_tariff_in_pipeline_does_not_touch_works():
    """B9 в конвейере: работы и их сумма те же, техника — отдельной строкой суммы."""
    RATES.setdefault("2-221", make_rate("2-221", LAMP_LINE["name"], price="750.00"))
    retriever = FakeRetriever({"фонарь": [("2-221", 0.99)]})
    chosen = {**work("f1", "2-221", []), "quantity_source": "description", "quantity_evidence": "заменить 1 лампу"}
    llm = FakeLLM(observation=[observations(finding("f1", "фонарь"))], selection=[selection([chosen])])
    result, _ = analyze_case([], "Не горит фонарь на столбе, заменить 1 лампу, нужна автовышка", llm, retriever,
                             equipment_tariff_rub=9000.0)
    assert result["works_total_rub"] == "750.00" and result["resources_total_rub"] == "9000.00"
    assert result["total_rub"] == "9750.00"
