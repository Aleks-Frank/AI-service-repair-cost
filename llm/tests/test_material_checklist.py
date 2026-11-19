"""Тесты чек-листа обязательных материалов по виду работы (llm/estimator/material_checklist.py, EXP-C6)."""

from llm.estimator.material_checklist import REASON, add_checklist_needs


def line(work_id: str, name: str, quantity: str = "1") -> dict:
    """Строка основной сметы с полями, которые нужны чек-листу."""
    return {"finding_id": "f1", "work_id": work_id, "name": name, "unit": "шт", "quantity": quantity}


def need(work_id: str, query: str) -> dict:
    """Потребность, которую уже запросила модель."""
    return {"need_id": "m1", "finding_id": "f1", "work_id": work_id, "query": query}


def test_clamp_and_gasket_added_for_temporary_leak_repair():
    needs: list[dict] = []
    added = add_checklist_needs(needs, [line("2-152", "Временная заделка свищей и трещин … (хомут)")])
    assert [n["query"] for n in added] == ["хомут ремонтный для трубы", "пластина резиновая техническая"]
    assert all(n["reason"] == REASON for n in added) and needs == added


def test_material_already_requested_is_not_duplicated():
    needs = [need("2-152", "хомут червячный оцинкованный")]
    added = add_checklist_needs(needs, [line("2-152", "Временная заделка свищей и трещин … (хомут)")])
    assert [n["query"] for n in added] == ["пластина резиновая техническая"]


def test_threaded_joints_get_sealing_and_valve_install_gets_valve():
    added = add_checklist_needs([], [line("2-43", "Установка муфтовой арматуры (вентилей, … кранов проходных)", "2")])
    assert {n["query"] for n in added} == {"паста уплотнительная сантехническая", "лён сантехнический",
                                          "кран шаровый латунный муфтовый"}


def test_unrelated_work_gets_nothing():
    assert add_checklist_needs([], [line("2-220", "Смена ламп: люминесцентных / светодиодных цоколь Т8, G13")]) == []
