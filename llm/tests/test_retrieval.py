"""Тесты веток поиска (Г2), глубины (Г3), словаря (Г1) и офлайн-оценки — без баз и моделей."""

import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from llm.estimator import retrieval_eval
from llm.estimator.candidates import UNIVERSAL_WORK_ID, description_query, find_candidates
from llm.estimator.schemas import Observations
from rag.db import Rate
from rag.term_dictionary import TermDictionary, build, build_from_export, stems


def rate(code: str, name: str | None = None) -> Rate:
    """Расценка без базы."""
    return Rate(code=code, name=name or f"работа {code}", work_composition="", unit="шт", unit_price=Decimal("1"),
                section_title="Раздел", norm_reference=None, source_sheet="ПЗР", source_row=1, issues=[])


class Retriever:
    """Поиск-подмена: на запрос с ключевым словом — заданные коды, иначе — пусто.

    Если ключа «*» нет, возвращается ровно top_k кодов вида «<ключ>-<n>» (для проверки глубины).
    """

    def __init__(self, results: dict[str, list[str]], generated: str | None = None):
        self.results, self.generated, self.queries = results, generated, []

    def search(self, query: str, top_k: int = 10, **filters):
        self.queries.append((query, top_k))
        codes = next((codes for key, codes in self.results.items() if key in query), [])
        if self.generated and self.generated in query:
            codes = [f"{self.generated}-{n}" for n in range(1, top_k + 1)]
        hits = [SimpleNamespace(item=rate(code), rerank_score=1 - n / 100, rrf_score=0.0)
                for n, code in enumerate(codes[:top_k])]
        return SimpleNamespace(hits=hits)


def observations(query: str = "лампа", object_name: str = "светильник") -> Observations:
    """Один дефект с одним запросом."""
    return Observations.model_validate({"summary": "", "image_relation": "unknown", "unknowns": [], "findings": [
        {"finding_id": "f1", "object_name": object_name, "visible_issue": "не горит", "image_numbers": [],
         "search_queries": [query]}]})


def test_description_is_extra_branch():
    """1, 4: текст заявки — отдельная ветка; запросы модели сохраняются; видно, какая ветка нашла кандидата."""
    retriever = Retriever({"лампа": ["2-218"], "течёт": ["2-14"]})
    log: dict = {}
    codes = {c["work_id"] for c in find_candidates(retriever, observations(), log=log,
                                                   description="Течёт кран на кухне в подъезде")}
    branches = [b["branch"] for b in log["f1"]["query_branches"]]
    assert branches == ["finding_query", "object_issue", "description"]
    assert codes == {"2-218", "2-14"}
    found = {item["work_id"]: [f["branch"] for f in item["found_by"]] for item in log["f1"]["retrieved"]}
    assert found == {"2-218": ["finding_query"], "2-14": ["description"]}


@pytest.mark.parametrize("text", ["Номер обращения 66-2026-1 Дата получения обращения 05.10.2026",
                                  "Прилагаю фотографии. Прошу принять меры.", "1 под. 1 и 16 этаж", "", None])
def test_template_description_gives_no_query(text):
    """2: шаблонный или пустой текст не создаёт запрос."""
    assert description_query(text) is None
    log: dict = {}
    find_candidates(Retriever({}), observations(), log=log, description=text)
    assert "description" not in [b["branch"] for b in log["f1"]["query_branches"]]


def test_duplicates_merged_by_work_id():
    """3: одна работа из двух веток — один кандидат, обе ветки записаны."""
    log: dict = {}
    candidates = find_candidates(Retriever({"лампа": ["2-218"], "светильник": ["2-218"]}), observations(), log=log)
    assert [c["work_id"] for c in candidates] == ["2-218"]
    assert len(log["f1"]["retrieved"][0]["found_by"]) == 2


@pytest.mark.parametrize("depth", [6, 10, 15])
def test_depth_is_kept_and_queries_unchanged(depth):
    """5, 6: глубина сохраняется на выходе (нет скрытых 6); ветки запросов от глубины не зависят."""
    retriever = Retriever({}, generated="лампа")
    log: dict = {}
    candidates = find_candidates(retriever, observations(), per_finding=depth, log=log)
    assert len(candidates) == depth
    assert {top_k for _, top_k in retriever.queries} == {depth}
    assert [b["branch"] for b in log["f1"]["query_branches"]] == ["finding_query", "object_issue"]


def test_universal_excluded_in_every_branch():
    """7: 1-282 не попадает в кандидаты ни из запросов модели, ни из текста заявки, ни из словаря."""
    retriever = Retriever({"лампа": [UNIVERSAL_WORK_ID, "2-218"], "течёт": [UNIVERSAL_WORK_ID],
                           "Ремонтные": [UNIVERSAL_WORK_ID]})
    codes = [c["work_id"] for c in find_candidates(retriever, observations(), description="Течёт кран на кухне сильно",
                                                   extra_queries=["Ремонтные работы"])]
    assert codes == ["2-218"]


def train_requests(n: int, text: str, work: str, start: int = 0) -> list[dict]:
    """n заявок обучающей части с одинаковым текстом и работой."""
    return [{"request_id": f"r{start + i}", "text": text, "work_ids": [work]} for i in range(n)]


WORKS = {"2-152": "Временная заделка свищей хомутом", "2-14": "Смена кранов водоразборных",
         "2-220": "Смена ламп люминесцентных"}


def test_dictionary_links_gap_terms_with_support():
    """9, 10: связь — только при достаточной поддержке; слово может вести к нескольким работам."""
    train = (train_requests(6, "капает вода с трубы", "2-152") + train_requests(5, "капает кран", "2-14")
             + train_requests(2, "мигает свет", "2-220"))
    dictionary = build(train, WORKS, min_support=5, min_confidence=0.3)
    capa = {link["work_id"] for link in dictionary.entries[next(iter(stems("капает")))]}
    assert capa == {"2-152", "2-14"}
    assert not any(link["work_id"] == "2-220" for links in dictionary.entries.values() for link in links)
    # «кран» уже есть в наименовании 2-14 — это не разрыв терминологии, связи нет.
    assert not any(link["work_id"] == "2-14" for link in dictionary.entries.get(next(iter(stems("кран"))), []))


def test_dictionary_uses_only_given_train_and_is_reproducible():
    """8, 12: словарь зависит только от переданной обучающей части; повторная сборка — тот же результат."""
    train = train_requests(6, "капает вода", "2-152")
    test_part = train_requests(6, "мигает свет", "2-220", start=100)
    first, second = build(train, WORKS), build(train, WORKS)
    assert first.entries == second.entries
    assert first.meta["train_ids_sha256"] == second.meta["train_ids_sha256"]
    assert next(iter(stems("мигает"))) not in first.entries
    assert build(train + test_part, WORKS).meta["train_ids_sha256"] != first.meta["train_ids_sha256"]


def test_dictionary_expansions_and_save(tmp_path):
    """Расширение — наименование работы из ПЗР, не больше limit; словарь сохраняется и читается."""
    dictionary = build(train_requests(6, "капает вода", "2-152"), WORKS)
    assert dictionary.expansions("сильно капает на лестнице", WORKS) == ["Временная заделка свищей хомутом"]
    dictionary.save(tmp_path / "d.json")
    assert TermDictionary.load(tmp_path / "d.json").entries == dictionary.entries


def test_no_export_is_blocked():
    """11: без выгрузки АВР — статус blocked_missing_data, а не выдуманный словарь."""
    status = build_from_export(None)
    assert status["status"] == "blocked_missing_data" and status["required_fields"]


def test_exact_hit_does_not_count_related():
    """13: родственная работа (другой диаметр) не засчитывается как точное попадание."""
    work = {"name": "муфтовая 20–32", "any_of": ["2-43"], "related": ["2-41"]}
    assert retrieval_eval.gold_ranks({"f1": ["2-41", "2-14"]}, work) == (None, 1)
    assert retrieval_eval.gold_ranks({"f1": ["2-14", "2-43"], "f2": ["2-43"]}, work) == (1, None)


def test_gold_never_reaches_search(tmp_path):
    """14: эталонные коды и наименования мастера не попадают в поисковые запросы."""
    (tmp_path / "requests.json").write_text(json.dumps([{"n": 1, "description": "Не горит лампа в холле"}]),
                                            encoding="utf-8")
    (tmp_path / "gold.json").write_text(json.dumps({"a01": [{"name": "Смена ламп секретная", "any_of": ["2-220"]}]}),
                                        encoding="utf-8")
    trace_dir = tmp_path / "run" / "a01" / "t"
    trace_dir.mkdir(parents=True)
    (trace_dir / "trace.json").write_text(json.dumps({"observations": observations().model_dump()}), encoding="utf-8")
    cases = retrieval_eval.load_cases(tmp_path / "requests.json", tmp_path / "gold.json", [tmp_path / "run"])
    retriever = Retriever({"лампа": ["2-218", "2-220"]})
    records = retrieval_eval.evaluate(cases, retriever, retrieval_eval.VARIANTS[:2])
    sent = " ".join(query for query, _ in retriever.queries)
    assert "2-220" not in sent and "секретная" not in sent
    assert {r["variant"]: r["exact_rank"] for r in records} == {"B0": 2, "B1": 2}
    summary = {row["variant"]: row for row in retrieval_eval.summarize(records)}
    assert summary["B0"]["recall@6"] == 1.0 and summary["B0"]["mrr"] == 0.5
