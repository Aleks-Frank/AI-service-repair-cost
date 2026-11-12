"""Тесты проверки «деталь или узел» по выбранному товару (эксперимент B7)."""

from types import SimpleNamespace

from llm.estimator.candidates import find_material_candidates
from llm.estimator.compatibility import item_scale_conflict


def test_lamp_need_rejects_fixture_and_back():
    """Лампа ≠ светильник: в обе стороны; тип товара — по первому слову таблицы в наименовании."""
    assert item_scale_conflict("лампа уличного освещения", "Светильник накладной GX53 под две лампы")
    assert item_scale_conflict("светильник", "Лампа светодиодная E27")
    assert item_scale_conflict("лампа T8", "Светильник под светодиодную лампу 2xLED-T8 G13")
    assert item_scale_conflict("прокладка для крана", "Кран шаровой 1/2")


def test_matching_or_unknown_kinds_pass():
    """Тот же масштаб и товары вне таблицы (лента, гирлянда, драйвер) не отбрасываются."""
    assert item_scale_conflict("лампа уличного освещения", "Лампа светодиодная GU10 11 Вт") is None
    assert item_scale_conflict("светильник", "Светильник ДПО (в комплекте с LED лампами G13)") is None
    assert item_scale_conflict("лампа", "Электрогирлянда уличная") is None
    assert item_scale_conflict("светодиодная лента", "Светильник LED") is None


def test_candidates_filtered_before_model_and_logged():
    """В кандидаты модели светильник на запрос «лампа» не попадает; отброшенное — в журнале."""
    def item(i, name):
        return SimpleNamespace(item=SimpleNamespace(id=i, article=f"A{i}", article_norm=f"a{i}", name=name,
                                                    category="", base_unit="шт", supplier_code="s"),
                               rerank_score=1.0, rrf_score=1.0)
    hits = [item(1, "Светильник уличный настенный E27"), item(2, "Лампа светодиодная GU10"),
            item(3, "Светильник накладной GX53"), item(4, "Лампа светодиодная E27")]
    retriever = SimpleNamespace(search=lambda query, top_k: SimpleNamespace(hits=hits))
    log: dict = {}
    result = find_material_candidates(retriever, [{"need_id": "m1", "query": "лампа уличного освещения"}], 5, log)
    assert [c["name"] for c in result["m1"]] == ["Лампа светодиодная GU10", "Лампа светодиодная E27"]
    assert [x["name"] for x in log["m1"]] == ["Светильник уличный настенный E27", "Светильник накладной GX53"]
