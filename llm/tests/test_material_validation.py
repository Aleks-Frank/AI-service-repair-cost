"""Тесты проверки совместимости материалов и работ (llm/estimator/material_validation.py, этап B2.1).

Синтетические строки материалов и сметы — без моделей и базы.
"""

import copy
from decimal import Decimal

from llm.estimator.estimate import attach_materials
from llm.estimator.material_validation import (
    ALLOW, REJECT, REVIEW, attach_validation, judge, material_class, validate_lines, work_domain,
)

ELECTRIC = "Раздел. 2 Электромонтажные работы"
PLUMBING = "Раздел 1. Внутренние санитарно-технические работы"
PLASTER = "Раздел 8. Штукатурные работы"

LAMP_WORK = {"work_id": "2-219", "section": ELECTRIC, "composition": "1. Снятие плафона.\n2. Снятие лампы.",
             "name": "Замена ламп накаливания /светодиодной в закрытых светильниках, с установкой стекол."}
CLAMP_WORK = {"work_id": "2-149", "section": PLUMBING, "composition": "1. Установка нового хомута крепления.",
              "name": "Установка крепежного хомута"}
PLASTER_WORK = {"work_id": "1-160", "section": PLASTER, "composition": "Нанесение штукатурки.",
                "name": "Ремонт штукатурки стен"}


def line(need_id, name, amount, work=LAMP_WORK, category="", finding="f1"):
    """Строка материала в формате price_materials (только нужные проверке поля)."""
    return {"need_id": need_id, "finding_id": finding, "work_id": work["work_id"], "need": name.lower(),
            "name": name, "category": category, "amount_rub": amount, "warnings": [],
            "shared": [{"finding_id": finding, "work_id": work["work_id"], "need_qty": "1"}]}


def test_allow_lamp_for_lamp_work():
    """ALLOW: лампа к замене ламп."""
    verdict = judge(LAMP_WORK, "Лампа светодиодная E27 8 Вт", "Лампы")
    assert verdict["decision"] == ALLOW


def test_reject_facade_paint_for_lamp_work():
    """REJECT: фасадная краска и грунтовка к замене лампы (утечка a02)."""
    assert judge(LAMP_WORK, "Краска фасадная Марта 0,9 л",
                 "Интерьер и отделка / Лакокрасочные материалы")["decision"] == REJECT
    verdict = judge(LAMP_WORK, "Грунтовка фасадная акриловая 2.5 л")
    assert verdict["decision"] == REJECT and verdict["rule_id"] == "MV-REJECT-electrical-finishing"


def test_review_when_pair_not_in_table():
    """REVIEW: отделочный материал к работам по проёмам-благоустройству описан, а к 1-282 без вида — нет."""
    unknown_work = {"work_id": "1-282", "section": "Раздел 12. Прочие работы", "name": "Ремонтные работы",
                    "finding_text": "что-то сломано"}
    verdict = judge(unknown_work, "Краска фасадная")
    assert verdict["decision"] == REVIEW and verdict["rule_id"] == "MV-UNKNOWN-WORK"


def test_unknown_material_category_is_review_not_reject():
    """Незнакомый материал не запрещается: REVIEW с причиной."""
    verdict = judge(LAMP_WORK, "Изделие XYZ-100")
    assert material_class("Изделие XYZ-100") is None
    assert verdict["decision"] == REVIEW and verdict["rule_id"] == "MV-UNKNOWN-MATERIAL"


def test_correct_consumable_absent_from_work_name():
    """Грунтовки нет в наименовании штукатурной работы, но по таблице она допустима."""
    verdict = judge(PLASTER_WORK, "Грунтовка глубокого проникновения 10 л")
    assert verdict["decision"] == ALLOW and verdict["rule_id"] == "MV-ALLOW-finishing-finishing"


def test_material_named_in_work_composition():
    """Стекло — не электрика, но названо в работе 2-219 «с установкой стекол»: ALLOW."""
    assert judge(LAMP_WORK, "Стекло для светильника")["decision"] == ALLOW


def test_clamp_kept_for_pipe_and_universal_work():
    """a08: хомут сохраняется и для 2-149, и для 1-282 по дефекту трубы ГВС."""
    assert judge(CLAMP_WORK, "Хомут для труб с гайкой 1 3/4\"", "Инженерная сантехника")["decision"] == ALLOW
    universal = {"work_id": "1-282", "section": "Раздел 12. Прочие работы", "name": "Ремонтные работы",
                 "finding_text": "трубопровод ГВС подтекает"}
    assert work_domain("1-282", universal["section"], universal["finding_text"]) == "plumbing"
    assert judge(universal, "Хомут ремонтный для трубы")["decision"] == ALLOW


def test_led_tape_and_power_supply_kept():
    """a10: LED-лента и блок питания к работе по освещению — ALLOW."""
    assert judge(LAMP_WORK, "Лента LED 5м 12В IEK", "Светильники / Светодиодная лента")["decision"] == ALLOW
    assert judge(LAMP_WORK, "Блок питания 60W 12V")["decision"] == ALLOW


def test_words_inside_other_words_do_not_misclassify():
    """«водопровод» — не электрика, «электроды» — не электрика."""
    assert material_class("Труба водопроводная ПП 20") == "plumbing"
    assert material_class("Электроды сварочные МР-3") != "electrical"


def test_several_materials_of_one_work_split_by_decision():
    """Несколько материалов одной работы: каждый решается отдельно; исходный порядок в журнале."""
    lines = [line("m1", "Лампа светодиодная E27", "95.00", category="Лампы"),
             line("m2", "Грунтовка фасадная акриловая", "380.00"),
             line("m3", "Изделие XYZ-100", "50.00")]
    works = {("f1", "2-219"): LAMP_WORK}
    allowed, review, rejected, log = validate_lines(lines, works)
    assert [l["need_id"] for l in allowed] == ["m1"]
    assert [l["need_id"] for l in review] == ["m3"]
    assert [l["need_id"] for l in rejected] == ["m2"]
    assert [e["need_id"] for e in log] == ["m1", "m2", "m3"]
    assert all(e["rule_id"] and e["reason"] for e in log)


def test_shared_purchase_allowed_if_any_work_compatible():
    """Общая покупка двух работ подтверждается, если совместима хотя бы с одной."""
    shared = line("m1", "Хомут для труб", "46.00", work=CLAMP_WORK)
    shared["shared"].append({"finding_id": "f2", "work_id": "2-219", "need_qty": "1"})
    allowed, _, _, _ = validate_lines([shared], {("f1", "2-149"): CLAMP_WORK, ("f2", "2-219"): LAMP_WORK})
    assert len(allowed) == 1


def _estimate():
    """Смета make_estimate с одной работой и техникой (только поля, которые проверяются)."""
    return {"works": [{"finding_id": "f1", "work_id": "2-219", "name": LAMP_WORK["name"], "quantity": "2",
                       "unit": "шт", "labor_hours": "0.5", "amount_rub": "200.00"}],
            "equipment_requirements": [{"name": "Автовышка", "status": "probable"}],
            "works_total_rub": "200.00", "total_rub": "200.00", "warnings": []}


def test_validator_changes_only_materials_and_recomputes_totals():
    """Коды ПЗР, объёмы, трудозатраты, техника не меняются; итог — работы + только ALLOW."""
    result = _estimate()
    before = copy.deepcopy({k: result[k] for k in ("works", "equipment_requirements", "works_total_rub")})
    lines = [line("m1", "Лампа светодиодная E27", "95.00", category="Лампы"),
             line("m2", "Краска фасадная", "291.00"),
             line("m3", "Изделие XYZ-100", "50.00")]
    original = copy.deepcopy(lines)
    allowed, review, rejected, log = validate_lines(lines, {("f1", "2-219"): LAMP_WORK})
    attach_materials(result, allowed, [])
    attach_validation(result, original, review, rejected, log)

    assert {k: result[k] for k in before} == before
    assert result["materials_total_rub"] == "95.00"
    assert Decimal(result["total_rub"]) == Decimal("295.00")  # краска исключена, XYZ — отдельно
    assert result["materials_review_total_rub"] == "50.00"
    assert Decimal(result["total_with_review_rub"]) == Decimal("345.00")
    assert [l["name"] for l in result["materials_rejected"]] == ["Краска фасадная"]
    # Исходный список сохранён целиком.
    assert [m["need_id"] for m in result["material_validation"]["original"]] == ["m1", "m2", "m3"]
    assert result["material_validation"]["counts"] == {ALLOW: 1, REVIEW: 1, REJECT: 1}
    assert any("исключён" in w and "MV-REJECT" in w for w in result["warnings"])
