"""Стык LLM и RAG: поиск расценок-кандидатов для каждого дефекта, найденного на фото.

LLM на первом этапе для каждого дефекта формулирует 1–3 поисковых запроса
(«смена розетки», «ремонт штукатурки стен»). Здесь по каждому запросу вызывается
гибридный поиск RAG-модуля, результаты объединяются, и для дефекта остаются лучшие
кандидаты. Именно этот список модель увидит на втором этапе — выбрать можно только из него.
"""

from rag.db import Rate
from rag.retriever import HybridRetriever
from llm.estimator.schemas import Observations, Selection, WorkChoice

# Минимальная оценка reranker'а, при которой «работа без расценки» сопоставляется с кандидатом.
# Для верных пар оценка обычно > 0,9, для посторонних расценок — < 0,1.
AUTO_MATCH_THRESHOLD = 0.5


def _candidate(rate: Rate, score: float) -> dict:
    """Превращает расценку из БД в словарь кандидата для промпта и расчёта.

    Принимает: rate — строка таблицы rates; score — оценка релевантности из поиска.
    Возвращает: словарь с кодом, наименованием, составом, единицей, ценой и источником.
    """
    return {
        "work_id": rate.code,
        "name": rate.name,
        "composition": rate.work_composition or "",
        "unit": rate.unit,
        # Цена строкой, чтобы Decimal не терял точность при сериализации в JSON.
        "unit_price_rub": str(rate.unit_price),
        "section": rate.section_title,
        "norm_reference": rate.norm_reference,
        "source_sheet": rate.source_sheet,
        "source_row": rate.source_row,
        # Замечания к данным справочника (найдены при загрузке) — попадут в предупреждения сметы.
        "data_issues": [issue["message"] for issue in rate.issues],
        "retrieval_score": round(score, 4),
        "finding_ids": [],
    }


def find_candidates(retriever: HybridRetriever, observations: Observations, per_finding: int = 6) -> list[dict]:
    """Подбирает расценки-кандидаты для каждого дефекта через RAG-модуль.

    Для каждого дефекта ищем по всем его поисковым запросам и по связке «объект + проблема»,
    для каждой расценки запоминаем лучшую оценку reranker'а и оставляем per_finding лучших.
    Одна расценка может подходить нескольким дефектам — тогда у неё несколько finding_ids.

    Принимает:
        retriever — гибридный поиск RAG-модуля;
        observations — результат анализа фото (дефекты и запросы);
        per_finding — сколько кандидатов оставить на дефект.
    Возвращает: список кандидатов-словарей (без повторов, с finding_ids).
    """
    merged: dict[str, dict] = {}
    for finding in observations.findings:
        best: dict[str, tuple[float, Rate]] = {}
        queries = [*finding.search_queries, f"{finding.object_name} {finding.visible_issue}"]
        for query in queries:
            for hit in retriever.search(query, top_k=per_finding).hits:
                # Основная оценка — reranker; если он не применялся, берём балл RRF.
                score = hit.rerank_score if hit.rerank_score is not None else hit.rrf_score
                code = hit.item.code
                if code not in best or score > best[code][0]:
                    best[code] = (score, hit.item)
        top = sorted(best.values(), key=lambda pair: pair[0], reverse=True)[:per_finding]
        for score, rate in top:
            merged.setdefault(rate.code, _candidate(rate, score))["finding_ids"].append(finding.finding_id)
    return list(merged.values())


def match_unmatched(selection: Selection, candidates: list[dict], retriever: HybridRetriever) -> list[dict]:
    """Сопоставляет «работы без расценки» из ответа модели с кандидатами через reranker.

    Модель иногда верно называет нужную работу текстом («Замена раковины со снятием и установкой»),
    но кладёт её в unmatched_works, хотя подходящий кандидат («Смена раковин») есть. Здесь
    каждая такая работа сравнивается reranker'ом с наименованиями кандидатов; если лучшая оценка
    не ниже порога, работа переносится в основной вариант. Объём при этом не задаётся.

    Принимает:
        selection — выбор модели (изменяется на месте: works дополняется, unmatched_works сокращается);
        candidates — кандидаты из RAG;
        retriever — поиск RAG (нужен reranker).
    Возвращает: список сопоставлений {"text", "work_id", "score"} — для трассировки и предупреждений.
    """
    if not selection.unmatched_works or not candidates:
        return []
    chosen = {choice.work_id for choice in selection.works}
    matches, still_unmatched = [], []
    for text in selection.unmatched_works:
        scores = retriever.rerank_texts(text, [item["name"] for item in candidates])
        score, item = max(zip(scores, candidates), key=lambda pair: pair[0])
        if score < AUTO_MATCH_THRESHOLD or item["work_id"] in chosen:
            still_unmatched.append(text)
            continue
        chosen.add(item["work_id"])
        selection.works.append(WorkChoice(
            finding_id=item["finding_ids"][0],
            work_id=item["work_id"],
            # Обоснование — формулировка работы самой моделью; детали сопоставления — в trace.json.
            reason=text,
        ))
        matches.append({"text": text, "work_id": item["work_id"], "score": round(score, 3)})
    selection.unmatched_works = still_unmatched
    return matches
