"""Тесты синонимов операции в тексте поиска расценки (rag/parser.py, эксперимент B3) — без базы."""

from decimal import Decimal

from rag.parser import ParsedRate, ParsedSection, action_synonyms


def rate(name: str) -> ParsedRate:
    """Минимальная расценка с заданным наименованием."""
    section = ParsedSection(book_number=2, sheet_name="Лист", title="Раздел. 2 Электромонтажные работы", position=1)
    return ParsedRate(code="2-224", item_number=None, section=section, name=name, work_composition=None,
                      unit="шт", unit_raw="шт", quantity=Decimal("1"), labor_hours=None, labor_rate=None,
                      unit_price=None, norm_reference=None, notes=None, source_sheet="Лист", source_row=1)


def test_smena_gets_zamena_and_back():
    """«Смена» → «замена», «Замена» → «смена»; наименование не меняется."""
    assert action_synonyms("Смена светильников: с лампами накаливания") == ["замена, заменить"]
    assert action_synonyms("Замена ламп накаливания /светодиодной") == ["смена"]
    item = rate("Смена светильников")
    assert "Синонимы: замена, заменить." in item.search_text and item.name == "Смена светильников"


def test_no_synonym_for_other_words():
    """Слова с той же основой, но не операция («сменных», «изменение»), синонимов не дают."""
    assert action_synonyms("Установка сменных фильтров") == []
    assert action_synonyms("Изменение схемы, заменитель") == []
    assert "Синонимы" not in rate("Ремонтные работы").search_text


def test_outdoor_lighting_family_synonyms():
    """«… наружного / уличного (дворового) освещения» получают «уличный фонарь …»; трубы — нет."""
    from rag.parser import family_synonyms

    assert family_synonyms("Смена ламп в светильниках наружного освещения с приставных лестниц")
    assert family_synonyms("Установка светильников уличного освещения")
    assert family_synonyms("Перепрограммирование таймера времени уличного (дворового) освещения")
    assert family_synonyms("Прокладка трубопроводов водоснабжения из полипропиленовых труб, наружным диаметром: 20 мм") == []
    assert family_synonyms("Установка сливных воронок наружных") == []
    text = rate("Смена ламп в светильниках наружного освещения с приставных лестниц").search_text
    # Синонимы операции и семейства — в одной части «Синонимы: …» (её берёт полнотекстовый индекс).
    assert "Синонимы: замена, заменить, уличный фонарь, фонарь на столбе, светильник на опоре." in text


def test_pipe_leak_family_synonyms():
    """«Заделка свищей и трещин …» — «течь трубы …»; с хомутом — ещё «ремонтный хомут»; хомут крепежа — нет."""
    from rag.parser import family_synonyms

    clamp = family_synonyms("Временная заделка свищей и трещин на внутренних трубопроводах: до 75 мм (хомут)")
    assert clamp == ["течь трубы, протечка трубы, устранение течи трубопровода", "ремонтный хомут на трубу"]
    assert family_synonyms("Заделка свищей и трещин на внутренних трубопроводах: до 75 мм (СВАРКА)") == [
        "течь трубы, протечка трубы, устранение течи трубопровода"]
    assert family_synonyms("Установка крепежного хомута") == []
    assert family_synonyms("Смена кранов водоразборных") == []
