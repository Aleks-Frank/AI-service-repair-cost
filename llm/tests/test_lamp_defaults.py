"""Тесты правила «T8/G13 по умолчанию в местах общего пользования» (llm/estimator/lamp_defaults.py, B6)."""

from decimal import Decimal
from types import SimpleNamespace

from llm.estimator.lamp_defaults import LINEAR_LAMP_QUERY, apply_linear_lamp_default
from llm.estimator.schemas import Finding, MaterialNeed, Observations, Selection, WorkChoice
from rag.db import Rate

LINEAR = Rate(code="2-220", name="Смена ламп: люминесцентных / светодиодных цоколь Т8, G13", work_composition="",
              unit="шт", unit_price=Decimal("208.50"), section_title="Раздел. 2 Электромонтажные работы",
              norm_reference=None, source_sheet="ПЗР", source_row=1, issues=[])
RETRIEVER = SimpleNamespace(session=SimpleNamespace(scalar=lambda statement: LINEAR))


def case(description, object_name="светильник в лифтовом холле", issue="не горит лампа",
         environment="indoor", work_id="2-218", summary="Потолочный светильник"):
    """Дефект, выбор 2-218 с лампой в материалах и кандидат 2-218."""
    finding = Finding(finding_id="f1", object_name=object_name, visible_issue=issue, image_numbers=[],
                      search_queries=["замена лампы"], operation="замена лампы", environment=environment)
    observations = Observations(summary=summary, image_relation="unknown", findings=[finding], unknowns=[])
    choice = WorkChoice(finding_id="f1", work_id=work_id, reason="лампа",
                        materials=[MaterialNeed(query="лампа светодиодная E27", unit="шт", qty_per_work_unit=1)])
    selection = Selection(works=[choice], alternatives=[], unmatched_works=[], assumptions=[])
    candidates = [{"work_id": work_id, "finding_ids": ["f1"]}]
    return selection, observations, candidates, description


def run(description, **kwargs):
    selection, observations, candidates, text = case(description, **kwargs)
    log = apply_linear_lamp_default(selection, observations, candidates, RETRIEVER, text)
    return selection, candidates, log


def test_common_area_unknown_lamp_becomes_linear():
    """«Перегорела лампа в лифтовом холле» — 2-220, запрос лампы — T8 G13; 2-220 разрешена дефекту."""
    selection, candidates, log = run("Добрый день, перегорела лампа в лифтовом холле 12 этаж")
    choice = selection.works[0]
    assert choice.work_id == "2-220" and choice.materials[0].query == LINEAR_LAMP_QUERY
    assert log and log[0]["from"] == "2-218"
    assert any(c["work_id"] == "2-220" and "f1" in c["finding_ids"] for c in candidates)


def test_known_socket_type_keeps_choice():
    """Тип лампы назван (E27, «лампочка», «подсветка») — правило не применяется."""
    for text in ("В холле на 3 этаже перегорела лампа E27", "в подъезде не горит лампочка",
                 "Около лифта перегорела светодиодная подсветка"):
        selection, _, log = run(text)
        assert selection.works[0].work_id == "2-218" and not log, text


def test_type_seen_on_photo_keeps_choice():
    """Тип виден на фото (в описании снимков модели «лампа-шар E27») — не меняется."""
    selection, _, log = run("не горит лампа на 5 этаже", summary="Светильник с лампой-шаром E27")
    assert selection.works[0].work_id == "2-218" and not log


def test_outdoor_apartment_and_other_works_untouched():
    """Улица, квартира и другие работы (светильник целиком) правилом не затрагиваются."""
    assert run("не горит лампа у подъезда", environment="outdoor")[2] == []
    assert run("в квартире на кухне не горит лампа")[2] == []
    selection, _, log = run("в холле не работает светильник", work_id="2-224")
    assert selection.works[0].work_id == "2-224" and not log


def test_no_common_area_marker_untouched():
    """Нет признака места общего пользования — не меняется."""
    selection, _, log = run("перегорела лампа", object_name="светильник", issue="не горит")
    assert selection.works[0].work_id == "2-218" and not log
