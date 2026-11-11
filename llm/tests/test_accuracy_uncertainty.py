"""Тесты точности сметы и неопределённости (сценарии A–I задания) — без моделей и баз.

Наименования расценок — как в справочнике ПЗР (2-218, 2-221, 2-231, 2-232 и др.): проверка
совместимости опирается на наименование, поэтому тексты взяты дословно.
"""

from decimal import Decimal

import pytest
from test_selection_fallback import (
    RATES, FakeLLM, FakeMaterialRetriever, FakeRetriever, make_rate, observations, selection,
)

from llm.estimator.candidates import UNIVERSAL_WORK_ID
from llm.estimator.compatibility import material_conflict, number_counts_object, work_conflicts
from llm.estimator.estimate import estimate_csv
from llm.estimator.llm_client import LLMServiceError
from llm.estimator.photos import resolve_photos
from llm.estimator.pipeline import analyze_case
from llm.estimator.schemas import Observations

NAMES = {
    "2-218": "Замена ламп накаливания /светодиодной в открытых светильниках цоколь E27/ E14 / GX53 / GU10/ G9 и подобные.",
    "2-221": "Смена ламп в светильниках наружного освещения с приставных лестниц",
    "2-224": "Смена светильников: с лампами накаливания/ светодиодными",
    "2-231": "Снятие светильников уличного освещения",
    "2-232": "Установка светильников уличного освещения",
}
# Общий словарь RATES других тестов не переопределяется: добавляются только отсутствующие коды.
for code, name in NAMES.items():
    RATES.setdefault(code, make_rate(code, name, price={"2-221": "750.00", "2-232": "2487.00"}.get(code, "100.00")))
RATES.setdefault("1-21", make_rate("1-21", "Заделка трещин в стенах", unit="м"))

STREET = {"освещ": [("2-218", 0.99), ("2-221", 0.95), ("2-231", 0.9), ("2-232", 0.88)]}
POLES = "Частично неисправно освещение на 2-х столбах установленных на придомовой территории."
INSPECTION = "Требуется замена светильников на Прожектора 200 Вт IP 65 - 4 шт ( автовышка 15-20 м )"


def street_finding(operation="", status="unknown", possible=(), environment="outdoor") -> dict:
    """Дефект «наружное освещение на столбах» со структурой ремонта."""
    return {"finding_id": "f1", "object_name": "светильники наружного освещения на опорах",
            "visible_issue": "частично не горит освещение", "image_numbers": [],
            "search_queries": ["освещение наружное"], "operation": operation, "operation_status": status,
            "possible_operations": list(possible), "environment": environment}


def choice(work_id, quantity=None, source="unknown", evidence="", materials=(), **axes) -> dict:
    """Работа в ответе этапа 2 (объём и оси совместимости задаются явно)."""
    return {"finding_id": "f1", "work_id": work_id, "reason": "подходит", "quantity": quantity,
            "quantity_unit": RATES[work_id].unit, "quantity_source": source, "quantity_evidence": evidence,
            "materials": list(materials), **axes}


def lamp_need(query):
    """Потребность в материале: 1 изделие на единицу работы."""
    return {"query": query, "unit": "шт", "qty_per_work_unit": 1}


# --- Сценарий A: первичная заявка без осмотра ---------------------------------------------------


def test_a_unknown_operation_most_probable_work():
    """A (решение заказчика): план строится до осмотра — при неясной операции выбирается наиболее
    вероятная работа как обычная строка, без пометок «предположение»; варианты — в finding_details;
    количество не выводится из числа столбов (минимум 1 шт); автовышка — только «проверить»."""
    obs = observations(street_finding("Замена ламп", "unknown",
                                      ["Замена светильника целиком", "Ремонт проводки"]))
    llm = FakeLLM(observation=[obs], selection=[selection([choice("2-221", 2, "description", "на 2-х столбах")])])
    result, _ = analyze_case([], POLES, llm, FakeRetriever(STREET))
    line, = result["works"]
    assert line["work_id"] == "2-221" and float(line["quantity"]) == 1 and line["amount_rub"] == "750.00"
    assert result["finding_statuses"] == {"f1": "matched"}
    details = result["finding_details"]["f1"]
    assert details["possible_operations"] and any("автовышка" in q for q in details["questions"])
    assert not any("предполож" in w or "не установлена" in w for w in result["warnings"])
    equipment, = result["equipment_requirements"]
    assert equipment["status"] == "probable" and equipment["needs_human_confirmation"] is True


def test_a_poles_number_is_not_fixture_count():
    """A, E: «2-х столбах» — не количество ламп/светильников."""
    assert not number_counts_object("на 2-х столбах", 2, NAMES["2-221"])
    assert number_counts_object("4 шт", 4, NAMES["2-232"])
    assert number_counts_object("заменить 4 лампы", 4, NAMES["2-221"])


# --- Сценарий B: заявка + подтверждённый осмотр ------------------------------------------------


def run_b(*works):
    """Сценарий B: осмотр «замена светильников на прожекторы 200 Вт IP65 — 4 шт, автовышка 15–20 м»."""
    obs = observations(street_finding("Замена светильников целиком на прожекторы 200 Вт IP65", "confirmed"))
    llm = FakeLLM(observation=[obs], selection=[selection(list(works))], defect_materials=[{"materials": []}])
    return analyze_case([], POLES, llm, FakeRetriever(STREET), material_retriever=FakeMaterialRetriever(),
                        inspection=INSPECTION) + (llm,)


def test_b_inspection_reaches_prompts():
    """B: запись осмотра передаётся на этапы 1 и 2."""
    _, trace, llm = run_b(choice("2-232", 4, "inspection", "4 шт"))
    assert "200 Вт IP 65" in llm.prompts("observation")[0] and "200 Вт IP 65" in llm.prompts("selection")[0]
    assert trace["inspection"] == INSPECTION


@pytest.mark.parametrize("lamp_work", ["2-218", "2-221"])
def test_b_lamp_rate_rejected_for_fixture_replacement(lamp_work):
    """B, 4: расценка на смену ламп не принимается как замена светильника; 1-282 — с честной пометкой."""
    result, _, _ = run_b(choice(lamp_work, 4, "inspection", "4 шт"))
    assert lamp_work not in [line["work_id"] for line in result["works"]]
    assert result["finding_statuses"]["f1"] == "compatibility_rejected"
    assert result["finding_details"]["f1"]["resolution"] == "retrieval_unresolved"
    universal = [line for line in result["works"] if line["work_id"] == UNIVERSAL_WORK_ID]
    assert universal and "не подтверждено" in universal[0]["note"]
    assert any(item["work_id"] == lamp_work for item in result["rejected_choices"])


def test_b_quantity_specs_materials_and_equipment():
    """B, 3, 9, 10: 4 шт из осмотра, светильник (не лампа), автовышка 15–20 м без цены."""
    work = choice("2-232", 4, "inspection", "4 шт",
                  materials=[lamp_need("прожектор светодиодный 200 Вт IP65"), lamp_need("лампа светодиодная E27")])
    result, trace, _ = run_b(work)
    line, = [line for line in result["works"] if line["work_id"] == "2-232"]
    assert float(line["quantity"]) == 4 and line["quantity_source"] == "inspection" and line["quantity_status"] == "confirmed"
    assert line["amount_rub"] == "9948.00"
    queries = [item["query"] for item in result["materials_not_found"]]
    assert queries == ["прожектор светодиодный 200 Вт IP65"]  # лампа убрана: операция — замена узла целиком
    assert trace["rejected_materials"][0]["query"] == "лампа светодиодная E27"
    equipment, = result["equipment_requirements"]
    assert equipment["status"] == "confirmed" and equipment["working_height_m"] == "15-20"
    assert equipment["pricing_status"] == "not_priced" and equipment["quantity"] is None and equipment["unit"] is None
    assert result["works_total_rub"] == "9948.00"  # техника не добавлена в сумму нулём


def test_b_model_marked_mismatch_rejected():
    """5.4: позиция, по которой модель сама отметила «no», в смету не попадает."""
    result, _, _ = run_b(choice("2-232", 4, "inspection", "4 шт", scale_match="no"))
    assert "2-232" not in [line["work_id"] for line in result["works"]]


# --- Сценарий C: явная замена ламп ------------------------------------------------------------


def test_c_explicit_lamp_replacement():
    """C: «Заменить 4 лампы» — смена ламп, 4 шт из описания; установка светильников отклоняется."""
    text = "Заменить 4 лампы в уличных светильниках"
    obs = observations(street_finding("Замена ламп", "confirmed"))
    llm = FakeLLM(observation=[obs], selection=[selection([
        choice("2-221", 4, "description", "4 лампы", materials=[lamp_need("лампа светодиодная E40")]),
        choice("2-232", 4, "description", "4 лампы"),
    ])])
    result, _ = analyze_case([], text, llm, FakeRetriever(STREET), material_retriever=FakeMaterialRetriever())
    assert [line["work_id"] for line in result["works"]] == ["2-221"]
    assert float(result["works"][0]["quantity"]) == 4 and result["works"][0]["quantity_status"] == "confirmed"
    assert [item["query"] for item in result["materials_not_found"]] == ["лампа светодиодная E40"]


# --- Сценарий D: похожая тематика, другая операция ---------------------------------------------


@pytest.mark.parametrize("operation,work_name", [
    ("Замена лампы", NAMES["2-224"]),
    ("Замена светильника целиком", NAMES["2-218"]),
    ("Ремонт существующего короба из ГКЛ", "Устройство перегородок из гипсокартона"),
    ("Ремонт люка", "Установка люков"),
    ("Замена прокладки", "Смена кранов водоразборных"),
    ("Замена лампы", NAMES["2-232"]),
])
def test_d_related_but_different_operation_rejected(operation, work_name):
    """D: тематически родственная позиция с другой операцией или масштабом отклоняется кодом."""
    assert work_conflicts(operation, "unknown", work_name)


@pytest.mark.parametrize("operation,work_name", [
    ("Замена лампы", "Смена ламп: люминесцентных / светодиодных цокольТ5/ Т8/ 12, G13"),
    ("Замена ламп в уличных светильниках", NAMES["2-221"]),
    ("Замена светильника", NAMES["2-224"]),
    ("Замена крана", "Смена кранов водоразборных"),
    ("Ремонт штукатурки", "Ремонт штукатурки внутренних стен"),
    ("Очистка фильтров", "Очистка фильтров грубой очистки на системах ХВС, ГВС"),
])
def test_d_matching_operation_not_rejected(operation, work_name):
    """D (контроль): совпадающая операция и масштаб не отклоняются — правила не режут всё подряд."""
    assert work_conflicts(operation, "unknown", work_name) == []


def test_d_filter_vs_sewer_relies_on_model_axis():
    """D: «очистка фильтра» и «прочистка канализации» — одна операция, разный объект. Код этого по
    наименованию не различает; отсев — по оси object_match модели (ограничение правил)."""
    assert work_conflicts("Очистка фильтров", "indoor", "Прочистка канализационной сети") == []
    rates = {"фильтр": [("1-21", 0.9)]}
    obs = observations({**street_finding("Очистка фильтров", "confirmed", environment="indoor"),
                        "search_queries": ["очистка фильтр"]})
    llm = FakeLLM(observation=[obs], selection=[selection([choice("1-21", object_match="no")])])
    result, _ = analyze_case([], "прочистить фильтры", llm, FakeRetriever(rates))
    assert "1-21" not in [line["work_id"] for line in result["works"]]


def test_environment_conflict():
    """5.2: наружный объект — работа «в помещении» отклоняется."""
    assert work_conflicts("Замена светильника", "outdoor", "Установка светильников в подвесные потолки")


# --- Сценарии E, F: количество ----------------------------------------------------------------


def test_e_poles_number_gives_minimum_not_two():
    """E: число светильников не указано — минимум 1 шт, а не «2» по числу столбов."""
    obs = observations(street_finding("Замена светильников целиком", "inferred"))
    llm = FakeLLM(observation=[obs], selection=[selection([choice("2-232", 2, "description", "на 2-х столбах")])])
    result, _ = analyze_case([], POLES, llm, FakeRetriever(STREET))
    line, = result["works"]
    assert float(line["quantity"]) == 1 and line["quantity_status"] == "estimate" and "минимум" in line["note"]


@pytest.mark.parametrize("source", ["estimate", "unknown"])
def test_e_discrete_estimate_or_missing_gives_minimum(source):
    """7 (решение заказчика): штучные изделия «на глаз» не оцениваются — минимум 1 шт с пометкой."""
    obs = observations(street_finding("Замена светильников целиком", "inferred"))
    qty = 3 if source == "estimate" else None
    llm = FakeLLM(observation=[obs], selection=[selection([choice("2-232", qty, source, "видно на фото")])])
    result, _ = analyze_case([], POLES, llm, FakeRetriever(STREET))
    assert float(result["works"][0]["quantity"]) == 1


def test_e_visible_count_on_photo():
    """Решение заказчика: на фото опора с 4 светильниками — 4 шт (visible_count со ссылкой на фото)."""
    obs = observations({**street_finding("Замена светильников целиком", "inferred"), "image_numbers": [1]})
    counted = {**choice("2-232", 4, "visible_count", "на опоре видно 4 светильника"), "image_numbers": [1]}
    llm = FakeLLM(observation=[obs], selection=[selection([counted])])
    result, _ = analyze_case([b"photo"], POLES, llm, FakeRetriever(STREET))
    line, = result["works"]
    assert float(line["quantity"]) == 4 and line["quantity_status"] == "counted"


def test_f_continuous_estimate_allowed_with_mark():
    """F, 8: трещина — оценка длины допускается с типом estimate и обоснованием."""
    obs = observations({"finding_id": "f1", "object_name": "стена", "visible_issue": "трещина",
                        "image_numbers": [], "search_queries": ["трещина"], "operation": "Заделка трещины",
                        "operation_status": "inferred"})
    llm = FakeLLM(observation=[obs], selection=[selection([
        {**choice("1-21", 1.5, "estimate", "видимая длина около 1,5 м"), "quantity_unit": "м"}])])
    result, _ = analyze_case([], "трещина в стене", llm, FakeRetriever({"трещина": [("1-21", 0.9)]}))
    line, = result["works"]
    assert line["quantity"] == "1.5" and line["quantity_status"] == "estimate" and "проверьте" in line["note"]


# --- Сценарии G, I: нет кандидатов и ошибки модели ---------------------------------------------


def test_g_no_candidates_is_retrieval_unresolved():
    """G: RAG пуст — retrieval_unresolved; 1-282 с пометкой, что отсутствие в ПЗР не подтверждено."""
    obs = observations(street_finding("Замена светильника", "confirmed"))
    llm = FakeLLM(observation=[obs])
    result, _ = analyze_case([], "заменить светильник", llm, FakeRetriever({}))
    assert result["finding_details"]["f1"]["resolution"] == "retrieval_unresolved"
    assert "не подтверждено" in result["works"][0]["note"]


def test_i_llm_error_is_not_missing_work():
    """I: невалидный ответ модели — error, без 1-282 и без случайной строки ПЗР."""
    obs = observations(street_finding("Замена светильника", "confirmed"))
    bad = {"works": "не список"}
    llm = FakeLLM(observation=[obs], selection=[bad, bad])
    result, _ = analyze_case([], "заменить светильник", llm, FakeRetriever(STREET))
    assert result["works"] == [] and result["finding_details"]["f1"]["resolution"] == "error"


def test_i_service_outage_raises():
    """I: недоступность LLM-сервиса не превращается в «нет работы» — ошибка заявки."""
    obs = observations(street_finding("Замена светильника", "confirmed"))
    llm = FakeLLM(observation=[obs], selection=[LLMServiceError("503")])
    with pytest.raises(LLMServiceError):
        analyze_case([], "заменить светильник", llm, FakeRetriever(STREET))


# --- Статус операции ---------------------------------------------------------------------------


def test_confirmed_without_named_action_downgraded():
    """1: «confirmed» без осмотра и без слова-действия в заявке понижается до inferred."""
    obs = observations(street_finding("Замена лампы", "confirmed"))
    llm = FakeLLM(observation=[obs], selection=[selection([choice("2-221")])])
    result, _ = analyze_case([], "не горит фонарь во дворе", llm, FakeRetriever(STREET))
    assert result["finding_details"]["f1"]["operation_status"] == "inferred"


# --- Материалы ---------------------------------------------------------------------------------


def test_material_conflicts():
    """9: светильник для смены ламп и лампа при замене светильника целиком — противоречия."""
    assert material_conflict(NAMES["2-221"], "светильник консольный светодиодный")
    assert material_conflict(NAMES["2-232"], "лампа светодиодная", "Замена светильников целиком")
    assert material_conflict(NAMES["2-221"], "лампа ДНаТ 250 Вт") is None
    assert material_conflict(NAMES["2-232"], "светильник консольный", "Замена светильников целиком") is None


# --- Фото --------------------------------------------------------------------------------------


def test_photos_belong_to_request(tmp_path):
    """10.1: фото соседней заявки не попадает в чужую."""
    for name in ("a03_1.png", "a04_1.png"):
        (tmp_path / name).write_bytes(b"x")
    paths, manifest, warnings = resolve_photos("a03", ["a03_1.png", "a04_1.png"], tmp_path)
    assert [p.name for p in paths] == ["a03_1.png"] and manifest == [{"image_number": 1, "file": "a03_1.png",
                                                                        "state": "unknown"}]
    assert any("не принадлежит" in w for w in warnings)


def test_missing_and_after_photos(tmp_path):
    """10.2, 10.3: отсутствующее фото — предупреждение; фото «после» не уходит в модель."""
    (tmp_path / "a01_1.png").write_bytes(b"x")
    (tmp_path / "a01_2.png").write_bytes(b"x")
    paths, manifest, warnings = resolve_photos(
        "a01", [{"file": "a01_1.png", "state": "before"}, {"file": "a01_2.png", "state": "after"}, "a01_3.png"],
        tmp_path)
    assert [p.name for p in paths] == ["a01_1.png"] and manifest[0]["state"] == "before"
    assert any("не найдено" in w for w in warnings) and any("после работ" in w for w in warnings)


def test_empty_photos_text_only(tmp_path):
    """10.4: без фото — пустой список без ошибок; смета строится по тексту."""
    assert resolve_photos("b01", [], tmp_path) == ([], [], [])


# --- Обратная совместимость ---------------------------------------------------------------------


def test_old_observations_and_estimates_still_read():
    """13: старый ответ этапа 1 (без новых полей) читается; CSV старой сметы без finding_details строится."""
    old = Observations.model_validate(observations({"finding_id": "f1", "object_name": "лампа",
                                                    "visible_issue": "не горит", "image_numbers": [],
                                                    "search_queries": ["смена лампы"]}))
    assert old.findings[0].operation_status == "inferred" and old.findings[0].operation == ""
    llm = FakeLLM(selection=[selection([choice("2-218", 1, "description", "1 лампа")])])
    result, _ = analyze_case([], "не горит 1 лампа", llm, FakeRetriever({"лампа": [("2-218", 0.9)]}), observations=old)
    assert result["works"][0]["amount_rub"] == RATES["2-218"].unit_price.quantize(Decimal("0.01")).to_eng_string()
    result.pop("finding_details")
    assert "2-218" in estimate_csv(result)





@pytest.mark.parametrize("operation,work_name", [
    ("Замена замка", "Смена дверного замка врезного"),          # b18: «дверного» — не дверь целиком
    ("Замена лампы", "Смена ламп в светильниках наружного освещения с приставных лестниц"),
    ("Замена ручки", "Замена, установка дверной ручки (дуга, скоба)"),
])
def test_part_named_with_its_whole_is_part(operation, work_name):
    """Регрессия реального прогона: деталь с указанием узла («дверного замка») — работа над деталью."""
    assert work_conflicts(operation, "unknown", work_name) == []


def test_whole_work_still_rejected_for_part():
    """Контроль: «Смена дверных блоков» при замене замка — узел целиком, отклоняется."""
    assert work_conflicts("Замена замка", "unknown", "Смена дверных блоков")


def test_adjusting_whole_covers_part():
    """Регрессия реального прогона (b10): регулировка петли — «Комплексная регулировка двери» не отклоняется."""
    assert work_conflicts("регулировка дверной петли", "indoor",
                          "Комплексная регулировка двери (в стоимость входят смазка и чистка фурнитуры)") == []
