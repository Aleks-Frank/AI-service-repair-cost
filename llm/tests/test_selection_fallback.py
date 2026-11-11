"""Тесты выбора работ и fallback на «Ремонтные работы» (1-282) — без моделей и баз.

LLM и поиск подменены: FakeLLM отвечает заранее заданными JSON по этапу (определяется по
системному промпту), FakeRetriever возвращает заданные расценки на запрос. Проверяется логика
конвейера: какие кандидаты видит модель, какое состояние получает каждый дефект, когда ставится
1-282 и когда запрашиваются материалы по дефекту.
"""

import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from llm.estimator import cli
from llm.estimator.candidates import UNIVERSAL_WORK_ID, find_candidates
from llm.estimator.llm_client import LLMServiceClient, LLMServiceError
from llm.estimator.pipeline import analyze_case
from llm.estimator.prompts import DEFECT_MATERIALS_PROMPT, OBSERVATION_PROMPT, SELECTION_PROMPT
from llm.estimator.schemas import Observations
from rag.db import Rate

# --- Подмены --------------------------------------------------------------------------------


def make_rate(code: str, name: str, unit: str = "шт", price: str = "100.00") -> Rate:
    """Расценка ПЗР без базы данных.

    Принимает: code, name, unit, price — код, наименование, единица и цена.
    Возвращает: объект Rate.
    """
    return Rate(code=code, name=name, work_composition="", unit=unit, unit_price=Decimal(price),
                section_title="Раздел", norm_reference=None, source_sheet="ПЗР", source_row=1, issues=[])


RATES = {
    "2-218": make_rate("2-218", "Замена ламп накаливания / светодиодных в открытых светильниках"),
    "2-224": make_rate("2-224", "Смена светильников с лампами накаливания"),
    "2-14": make_rate("2-14", "Смена кранов водоразборных"),
    "1-161": make_rate("1-161", "Ремонт штукатурки внутренних стен", unit="м2"),
    "1-21": make_rate("1-21", "Заделка трещин в стенах", unit="м"),
    UNIVERSAL_WORK_ID: make_rate(UNIVERSAL_WORK_ID, "Ремонтные работы", unit="чел.-ч", price="1500.00"),
}


class FakeRetriever:
    """Поиск по ПЗР: на запрос, содержащий ключ, возвращает заданные коды с оценками."""

    def __init__(self, results: dict[str, list[tuple[str, float]]]):
        """Принимает: results — {подстрока запроса: [(код, оценка), …]}. Возвращает: ничего."""
        self.results = results
        self.session = SimpleNamespace(scalar=lambda statement: RATES[UNIVERSAL_WORK_ID])

    def search(self, query: str, top_k: int = 10, **filters) -> SimpleNamespace:
        """Возвращает SearchResult-подобный объект с hits по первому совпавшему ключу."""
        for key, codes in self.results.items():
            if key in query:
                hits = [SimpleNamespace(item=RATES[code], rerank_score=score, rrf_score=score)
                        for code, score in codes[:top_k]]
                return SimpleNamespace(hits=hits)
        return SimpleNamespace(hits=[])


class FakeMaterialRetriever:
    """Поиск по прайсам, который ничего не находит (материалы проверяются на уровне потребностей)."""

    session = None

    def search(self, query: str, top_k: int = 10, **filters) -> SimpleNamespace:
        """Возвращает пустой результат."""
        return SimpleNamespace(hits=[])


class FakeLLM(LLMServiceClient):
    """LLM по сценарию: очередь ответов для каждого этапа; Exception в очереди — бросается."""

    STAGES = {OBSERVATION_PROMPT: "observation", SELECTION_PROMPT: "selection",
              DEFECT_MATERIALS_PROMPT: "defect_materials"}

    def __init__(self, **answers):
        """Принимает: answers — {этап: [ответ (dict или Exception), …]}. Возвращает: ничего."""
        self.answers = {stage: list(queue) for stage, queue in answers.items()}
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str, images: list[bytes], max_new_tokens: int,
                 schema: dict | None = None) -> str:
        """Возвращает следующий ответ этапа и запоминает вызов (этап, текст промпта)."""
        stage = self.STAGES.get(system_prompt, "material_selection")
        self.calls.append((stage, user_prompt))
        answer = self.answers[stage].pop(0)
        if isinstance(answer, Exception):
            raise answer
        return json.dumps(answer, ensure_ascii=False)

    def prompts(self, stage: str) -> list[str]:
        """Тексты промптов, отправленных на этапе stage."""
        return [prompt for name, prompt in self.calls if name == stage]


def finding(finding_id: str, query: str) -> dict:
    """Дефект этапа 1 с одним поисковым запросом."""
    return {"finding_id": finding_id, "object_name": query, "visible_issue": "неисправность",
            "image_numbers": [], "search_queries": [query]}


def observations(*findings: dict) -> dict:
    """Ответ этапа 1 с заданными дефектами."""
    return {"summary": "заявка", "image_relation": "unknown", "findings": list(findings), "unknowns": []}


def work(finding_id: str, work_id: str, materials: list | None = None) -> dict:
    """Выбранная работа в ответе этапа 2."""
    unit = RATES[work_id].unit if work_id in RATES else "шт"
    return {"finding_id": finding_id, "work_id": work_id, "reason": "подходит", "quantity": 1,
            "quantity_unit": unit, "quantity_source": "estimate", "quantity_evidence": "1 шт",
            "materials": materials or []}


def selection(works=(), unmatched=(), alternatives=()) -> dict:
    """Ответ этапа 2."""
    return {"works": list(works), "alternatives": list(alternatives), "unmatched_works": list(unmatched),
            "assumptions": []}


def run(llm: FakeLLM, retriever: FakeRetriever, materials: bool = False) -> tuple[dict, dict]:
    """Запускает конвейер без фото с описанием «заявка»."""
    return analyze_case([], "заявка", llm, retriever, material_retriever=FakeMaterialRetriever() if materials else None)


def candidates_shown(prompt: str) -> list[str]:
    """work_id кандидатов из промпта выбора работ (то, что реально увидела модель)."""
    payload = json.loads(prompt.split("ДАННЫЕ ЗАЯВКИ:\n", 1)[1].split("\nJSON_SCHEMA:", 1)[0])
    return [item["work_id"] for item in payload["CANDIDATES"]]


LAMP = {"лампа": [("2-218", 0.99), ("2-224", 0.9)]}

# --- Сценарии -------------------------------------------------------------------------------


def test_exact_work_is_matched_without_fallback():
    """3: есть «Замена ламп» — выбрана она, 1-282 нет, материалы по дефекту не запрашиваются."""
    llm = FakeLLM(observation=[observations(finding("f1", "лампа"))],
                  selection=[selection([work("f1", "2-218")])], defect_materials=[])
    result, trace = run(llm, FakeRetriever(LAMP), materials=True)
    assert [line["work_id"] for line in result["works"]] == ["2-218"]
    assert result["finding_statuses"] == {"f1": "matched"}
    assert "fallback_reason" not in result["works"][0]
    assert llm.prompts("defect_materials") == []


def test_no_candidates_gives_fallback_with_defect_materials():
    """1: поиск ничего не нашёл — модель не вызывается, 1-282 с причиной no_candidates и материалы по дефекту."""
    llm = FakeLLM(observation=[observations(finding("f1", "непонятное"))], selection=[],
                  defect_materials=[{"materials": []}])
    result, _ = run(llm, FakeRetriever({}), materials=True)
    assert llm.prompts("selection") == []
    assert [line["work_id"] for line in result["works"]] == [UNIVERSAL_WORK_ID]
    assert result["works"][0]["fallback_reason"] == "no_candidates"
    assert result["works"][0]["note"].startswith("Поиск не нашёл")
    assert len(llm.prompts("defect_materials")) == 1


def test_model_unmatched_gives_fallback():
    """2: модель объяснила отказ — 1-282 с причиной model_unmatched и подсказками ближайших позиций."""
    llm = FakeLLM(observation=[observations(finding("f1", "лампа"))],
                  selection=[selection(unmatched=["нужен ремонт патрона, кандидаты про замену ламп"])])
    result, _ = run(llm, FakeRetriever(LAMP))
    line = result["works"][0]
    assert (line["work_id"], line["fallback_reason"]) == (UNIVERSAL_WORK_ID, "model_unmatched")
    assert "2-218" in line["note"]
    assert result["finding_statuses"] == {"f1": "model_unmatched"}


@pytest.mark.parametrize("query", ["лампа", "лампа и ремонт"])
def test_universal_work_never_reaches_model(query):
    """4, 14: 1-282 в выдаче поиска (в т.ч. по одному из нескольких запросов) — модель её не видит."""
    retriever = FakeRetriever({"ремонт": [(UNIVERSAL_WORK_ID, 0.95), ("1-161", 0.8)],
                               "лампа": [("2-218", 0.99), (UNIVERSAL_WORK_ID, 0.7)]})
    found = finding("f1", query)
    found["search_queries"] = ["лампа", "ремонт"] if query == "лампа и ремонт" else ["лампа"]
    llm = FakeLLM(observation=[observations(found)], selection=[selection([work("f1", "2-218")])])
    result, trace = run(llm, retriever)
    shown = candidates_shown(llm.prompts("selection")[0])
    assert UNIVERSAL_WORK_ID not in shown and "2-218" in shown
    assert UNIVERSAL_WORK_ID in trace["findings"]["f1"]["excluded"]
    assert [item["work_id"] for item in trace["findings"]["f1"]["prompt_candidates"]] == shown


def test_only_universal_found_means_no_candidates():
    """После исключения 1-282 кандидатов нет — no_candidates, без вызова модели."""
    observations_value = Observations.model_validate(observations(finding("f1", "ремонт")))
    retriever = FakeRetriever({"ремонт": [(UNIVERSAL_WORK_ID, 0.95)]})
    assert find_candidates(retriever, observations_value) == []
    llm = FakeLLM(observation=[observations(finding("f1", "ремонт"))], selection=[])
    result, _ = run(llm, retriever)
    assert result["works"][0]["fallback_reason"] == "no_candidates"


def test_empty_answer_is_retried_and_then_accepted():
    """5: пустой ответ — повтор с объяснением ошибки; вторая попытка с работой принимается."""
    llm = FakeLLM(observation=[observations(finding("f1", "лампа"))],
                  selection=[selection(), selection([work("f1", "2-218")])])
    result, trace = run(llm, FakeRetriever(LAMP))
    prompts = llm.prompts("selection")
    assert len(prompts) == 2 and "нет решения" in prompts[1]
    assert result["finding_statuses"] == {"f1": "matched"}
    assert len(trace["findings"]["f1"]["raw_answers"]) == 2


def test_empty_answer_twice_is_error_not_fallback():
    """6: дважды пустой ответ — model_empty, 1-282 не ставится, предупреждение для сметчика."""
    llm = FakeLLM(observation=[observations(finding("f1", "лампа"))], selection=[selection(), selection()])
    result, trace = run(llm, FakeRetriever(LAMP))
    assert result["works"] == []
    assert result["finding_statuses"] == {"f1": "model_empty"}
    assert any("не дала решения" in warning for warning in result["warnings"])
    assert len(trace["findings"]["f1"]["raw_answers"]) == 2


def test_unmatched_after_retry_gives_normal_fallback():
    """Пустой ответ, затем объяснённый отказ — штатный fallback model_unmatched."""
    llm = FakeLLM(observation=[observations(finding("f1", "лампа"))],
                  selection=[selection(), selection(unmatched=["нужна другая работа"])])
    result, _ = run(llm, FakeRetriever(LAMP))
    assert result["works"][0]["fallback_reason"] == "model_unmatched"


@pytest.mark.parametrize("bad", [work("f1", "9-999"), work("f2", "2-218"), work("f1", UNIVERSAL_WORK_ID)])
def test_invalid_selection_is_error_not_fallback(bad):
    """8, 9: неизвестный work_id, чужой finding_id или 1-282 от модели — invalid_selection, без 1-282."""
    llm = FakeLLM(observation=[observations(finding("f1", "лампа"))], selection=[selection([bad]), selection([bad])])
    result, _ = run(llm, FakeRetriever(LAMP))
    assert result["finding_statuses"] == {"f1": "invalid_selection"}
    assert result["works"] == []


def test_work_of_another_finding_is_rejected():
    """9: работа из кандидатов другого дефекта не принимается для текущего."""
    llm = FakeLLM(observation=[observations(finding("f1", "лампа"), finding("f2", "кран"))],
                  selection=[selection([work("f1", "2-14")]), selection([work("f1", "2-14")]),
                             selection([work("f2", "2-14")])])
    result, _ = run(llm, FakeRetriever({**LAMP, "кран": [("2-14", 0.98)]}))
    assert result["finding_statuses"] == {"f1": "invalid_selection", "f2": "matched"}
    assert [line["work_id"] for line in result["works"]] == ["2-14"]


def test_invalid_json_twice_is_llm_error():
    """Дважды ответ не по схеме — llm_error, 1-282 не ставится."""
    llm = FakeLLM(observation=[observations(finding("f1", "лампа"))], selection=[{"bad": 1}, {"bad": 2}])
    result, _ = run(llm, FakeRetriever(LAMP))
    assert result["finding_statuses"] == {"f1": "llm_error"}
    assert result["works"] == []


def test_service_failure_is_not_hidden():
    """7: LLM-сервис недоступен — ошибка пробрасывается (заявка завершается ошибкой), не unmatched."""
    llm = FakeLLM(observation=[observations(finding("f1", "лампа"))],
                  selection=[LLMServiceError("LLM-сервис недоступен")])
    with pytest.raises(LLMServiceError):
        run(llm, FakeRetriever(LAMP))


def test_mixed_request_fallback_only_for_one_finding():
    """10, 11, 12: f1 без работы, f2 и f3 с работами — 1-282 и материалы по дефекту только у f1."""
    retriever = FakeRetriever({**LAMP, "кран": [("2-14", 0.98)], "трещина": [("1-21", 0.9)]})
    tap_material = [{"query": "кран водоразборный", "unit": "шт", "qty_per_work_unit": 1}]
    llm = FakeLLM(
        observation=[observations(finding("f1", "лампа"), finding("f2", "кран"), finding("f3", "трещина"))],
        selection=[selection(unmatched=["нужна другая работа"]),
                   selection([work("f2", "2-14", tap_material)]),
                   selection([work("f3", "1-21")])],
        defect_materials=[{"materials": []}],
    )
    result, trace = run(llm, retriever, materials=True)
    lines = {line["finding_id"]: line["work_id"] for line in result["works"]}
    assert lines == {"f1": UNIVERSAL_WORK_ID, "f2": "2-14", "f3": "1-21"}
    assert result["finding_statuses"] == {"f1": "model_unmatched", "f2": "matched", "f3": "matched"}
    # Материалы по дефекту — один вызов, и только для f1.
    defect_prompts = llm.prompts("defect_materials")
    assert len(defect_prompts) == 1 and '"object":"лампа"' in defect_prompts[0]
    # Потребность из выбора работ для f2 не продублирована и привязана к своей работе.
    missing = result["materials_not_found"]
    assert [(item["query"], item["work_id"]) for item in missing] == [("кран водоразборный", "2-14")]


def test_alternative_is_not_in_total():
    """13: альтернатива не входит в сумму."""
    llm = FakeLLM(observation=[observations(finding("f1", "лампа"))],
                  selection=[selection([{**work("f1", "2-218"), "quantity_source": "description",
                                              "quantity_evidence": "1 лампа"}], alternatives=[work("f1", "2-224")])])
    result, _ = analyze_case([], "не горит 1 лампа", llm, FakeRetriever(LAMP))
    assert [line["work_id"] for line in result["alternatives"]] == ["2-224"]
    assert result["works_total_rub"] == "100.00"


def test_trace_is_saved_and_failure_is_reported(tmp_path, caplog):
    """15: трассировка пишется в trace.json; ошибка записи — предупреждение, без исключения."""
    llm = FakeLLM(observation=[observations(finding("f1", "лампа"))], selection=[selection([work("f1", "2-218")])])
    _, trace = run(llm, FakeRetriever(LAMP))
    path = cli.save_trace(tmp_path, trace)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert [item["work_id"] for item in saved["findings"]["f1"]["prompt_candidates"]] == \
        candidates_shown(llm.prompts("selection")[0])
    assert cli.save_trace(tmp_path / "нет-такой-папки", trace) is None
    assert "Трассировка не сохранена" in caplog.text


def test_refusal_is_not_overridden_by_similar_candidate():
    """Отказ модели не превращается в похожую по тексту работу: дефект получает 1-282."""
    llm = FakeLLM(observation=[observations(finding("f1", "лампа"))],
                  selection=[selection(unmatched=["замена ламп не подходит: " + "текст " * 200])])
    result, _ = run(llm, FakeRetriever(LAMP))
    assert [line["work_id"] for line in result["works"]] == [UNIVERSAL_WORK_ID]
    assert result["works"][0]["fallback_reason"] == "model_unmatched"
