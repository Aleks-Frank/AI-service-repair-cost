"""Стык LLM и RAG: поиск расценок-кандидатов для каждого дефекта, найденного на фото.

LLM на первом этапе для каждого дефекта формулирует 1–3 поисковых запроса
(«смена розетки», «ремонт штукатурки стен»). Здесь по каждому запросу вызывается
гибридный поиск RAG-модуля, результаты объединяются, и для дефекта остаются лучшие
кандидаты. Именно этот список модель увидит на втором этапе — выбрать можно только из него.
"""

import re

from sqlalchemy import select

from rag.db import Rate
from rag.prices_db import Material
from rag.retriever import HybridRetriever, MaterialRetriever
from llm.estimator.compatibility import item_scale_conflict
from llm.estimator.schemas import Observations, Selection, WorkChoice

# Универсальная расценка «Ремонтные работы» (чел.-ч): по указанию заказчика ставится, если для дефекта
# в ПЗР нет подходящей работы.
UNIVERSAL_WORK_ID = "1-282"
# Сколько ближайших позиций ПЗР назвать сметчику в примечании к «Ремонтным работам».
UNIVERSAL_HINTS = 3


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


# Шаблонные фразы обращений (ГИС ЖКХ, письма), которые не описывают дефект: при поиске по тексту
# заявки они вырезаются, иначе запрос «Номер обращения … Дата получения …» тянет случайные работы.
_TEMPLATE_PATTERNS = [
    r"номер обращения[:\s]*[\d-]+",
    r"дата получения обращения[:\s]*[\d.:\s]+",
    r"адрес дома/территории",
    # Фрагменты адреса: область, город, улица, дом, квартира, подъезд, этаж.
    r"\b[а-яё-]+ская обл\.?",
    r"\bг\.\s*[а-яё-]+",
    r"\b(ул|пр|пер|б-р|ш)\.\s*[а-яё0-9 -]+?(?=,|\.|$)",
    r"\b(д|кв|корп|стр)\.\s*[\d/а-яё]+",
    r"прошу (вас )?рассмотреть (мо[её] )?обращение[^.]*",
    r"предоставить ответ в установленный срок",
    r"прилагаю фотографи[ия][^.]*",
    r"\b\d{2}-\d{4}-\d+\b",
]
# Сколько символов текста заявки идёт в запрос (длинный текст размывает вектор запроса).
DESCRIPTION_QUERY_LIMIT = 300
# Меньше стольких содержательных слов (от 4 букв) — текст не годится для отдельного поиска.
DESCRIPTION_MIN_WORDS = 3
# Слова просьбы и вежливости: дефект не описывают и в счёт содержательных не идут.
_FILLER_WORDS = {"прошу", "просим", "просьба", "принять", "меры", "срочно", "помочь", "пожалуйста",
                 "добрый", "день", "здравствуйте", "спасибо", "заранее", "года", "сентября", "октября"}


def description_query(description: str | None) -> str | None:
    """Готовит текст заявки к поиску: убирает шаблонные фразы обращений и номера.

    Принимает: description — исходный текст заявки.
    Возвращает: строку запроса или None, если содержательных слов слишком мало.
    """
    text = (description or "").lower()
    for pattern in _TEMPLATE_PATTERNS:
        text = re.sub(pattern, " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .,;:-")
    words = [word for word in re.findall(r"[а-яё]{4,}", text) if word not in _FILLER_WORDS]
    if len(words) < DESCRIPTION_MIN_WORDS:
        return None
    return text[:DESCRIPTION_QUERY_LIMIT]


def find_candidates(
    retriever: HybridRetriever,
    observations: Observations,
    per_finding: int = 6,
    log: dict | None = None,
    description: str | None = None,
    extra_queries: list[str] | None = None,
) -> list[dict]:
    """Подбирает расценки-кандидаты для каждого дефекта через RAG-модуль.

    Для каждого дефекта ищем по нескольким веткам запросов, для каждой расценки запоминаем
    лучшую оценку reranker'а и оставляем per_finding лучших. Ветки:
        finding_query — поисковые запросы модели из этапа 1 (search_queries);
        object_issue — связка «объект + проблема» из этапа 1;
        description — исходный текст заявки без шаблонных фраз (если передан description);
        dictionary — дополнительные запросы (если переданы extra_queries, например расширения словаря).
    Каждый запрос возвращает per_finding результатов, так что глубина выдачи меняется одним параметром.
    Одна расценка может подходить нескольким дефектам — тогда у неё несколько finding_ids.
    Универсальная «Ремонтные работы» (1-282) в кандидаты не попадает ни из одной ветки: её назначает
    код, если модель не нашла конкретной работы (add_universal_works).

    Принимает:
        retriever — гибридный поиск RAG-модуля;
        observations — результат анализа фото (дефекты и запросы);
        per_finding — сколько кандидатов оставить на дефект (и сколько брать с каждого запроса);
        log — словарь для трассировки (дополняется на месте): по finding_id — запросы с ветками,
            все найденные расценки до отбора (с ветками, местом и оценкой) и исключённые универсальные;
        description — исходный текст заявки для ветки description; None — ветка не используется;
        extra_queries — запросы ветки dictionary; None — ветка не используется.
    Возвращает: список кандидатов-словарей (без повторов, с finding_ids).
    """
    shared = []
    if (query := description_query(description)) is not None:
        shared.append(("description", query))
    shared += [("dictionary", query) for query in extra_queries or []]
    merged: dict[str, dict] = {}
    for finding in observations.findings:
        best: dict[str, tuple[float, Rate]] = {}
        found_by: dict[str, list[dict]] = {}
        queries = [*(("finding_query", q) for q in finding.search_queries),
                   ("object_issue", f"{finding.object_name} {finding.visible_issue}"), *shared]
        for branch, query in queries:
            for rank, hit in enumerate(retriever.search(query, top_k=per_finding).hits, start=1):
                # Основная оценка — reranker; если он не применялся, берём балл RRF.
                score = hit.rerank_score if hit.rerank_score is not None else hit.rrf_score
                code = hit.item.code
                found_by.setdefault(code, []).append({"branch": branch, "rank": rank, "score": round(score, 4)})
                if code not in best or score > best[code][0]:
                    best[code] = (score, hit.item)
        ranked = sorted(best.values(), key=lambda pair: pair[0], reverse=True)
        if log is not None:
            log[finding.finding_id] = {
                "queries": [query for _, query in queries],
                "query_branches": [{"branch": branch, "query": query} for branch, query in queries],
                "retrieved": [{"work_id": rate.code, "name": rate.name, "score": round(score, 4),
                               "found_by": found_by[rate.code]} for score, rate in ranked],
                "excluded": [rate.code for _, rate in ranked if rate.code == UNIVERSAL_WORK_ID],
            }
        # Исключение — по коду, до отбора лучших: место 1-282 занимает следующая конкретная работа.
        top = [(score, rate) for score, rate in ranked if rate.code != UNIVERSAL_WORK_ID][:per_finding]
        for score, rate in top:
            merged.setdefault(rate.code, _candidate(rate, score))["finding_ids"].append(finding.finding_id)
    return list(merged.values())


# Состояния дефекта после выбора работ (statuses в add_universal_works и трассировке).
#   matched — выбрана конкретная работа из кандидатов;
#   no_candidates — поиск не нашёл ни одной конкретной расценки;
#   model_unmatched — модель ответила и объяснила, что ни один кандидат не подходит;
#   model_empty — модель дважды не дала решения (ни работ, ни объяснения);
#   invalid_selection — модель дважды назвала work_id не из кандидатов или чужой finding_id;
#   llm_error — модель дважды вернула ответ не по схеме (невалидный JSON);
#   compatibility_rejected — все выбранные моделью позиции отклонены проверкой совместимости
#       (compatibility.py: другая операция, деталь вместо узла, другая среда);
# «Ремонтные работы» — временная строка по бизнес-правилу для no_candidates, model_unmatched и
# compatibility_rejected: операция известна, но подходящей позиции нет в ПРОВЕРЕННОЙ ВЫДАЧЕ поиска.
# Это не доказывает, что расценки нет во всём ПЗР, — так и пишется в примечании.
# Ошибки модели — не повод для 1-282: дефект остаётся без работы с предупреждением.
FALLBACK_STATUSES = {"no_candidates", "model_unmatched", "compatibility_rejected"}
_FALLBACK_NOTES = {
    "no_candidates": "Поиск не нашёл расценок для дефекта",
    "model_unmatched": "Модель не нашла подходящей работы среди кандидатов",
    "compatibility_rejected": "Выбранные моделью позиции не совпали с операцией дефекта и отклонены",
}


def add_universal_works(
    selection: Selection,
    observations: Observations,
    candidates: list[dict],
    retriever: HybridRetriever,
    statuses: dict[str, dict],
) -> dict[tuple[str, str], str]:
    """Добавляет «Ремонтные работы» (1-282) дефектам, для которых подтверждено отсутствие работы.

    Дефект покрыт, если у него есть работа (в works или alternatives) с work_id из кандидатов
    именно этого дефекта — проверка по коду и finding_id. Покрытый дефект получает статус matched.
    Непокрытый — 1-282, только если его статус из
    select_for_finding — no_candidates или model_unmatched. При ошибке модели (model_empty,
    invalid_selection, llm_error) работа не ставится: дефект остаётся без работы с предупреждением.

    Объём не задаётся — при расчёте подставится минимум 1 чел.-ч. В примечании называются ближайшие
    найденные позиции ПЗР (по убыванию оценки поиска): различить «работы нет в справочнике» и
    «модель не выбрала подходящую» по оценке поиска нельзя — reranker оценивает похожесть текста,
    а не пригодность работы (неподходящие позиции получали и 0,99), поэтому решает сметчик.

    Принимает:
        selection — выбор модели (изменяется на месте: works дополняется);
        observations — дефекты этапа 1;
        candidates — кандидаты RAG (изменяется на месте: дефекты добавляются в finding_ids 1-282);
        retriever — поиск по ПЗР (нужна сессия базы);
        statuses — {finding_id: {"status": …}} из select_for_finding (изменяется на месте:
            итоговый status, у 1-282 — fallback_reason).
    Возвращает: {(finding_id, work_id): примечание} для добавленных работ.
    """
    allowed = {item["work_id"]: item for item in candidates}
    covered = {
        choice.finding_id for choice in [*selection.works, *selection.alternatives]
        if choice.work_id != UNIVERSAL_WORK_ID
        and choice.work_id in allowed and choice.finding_id in allowed[choice.work_id]["finding_ids"]
    }
    notes: dict[tuple[str, str], str] = {}
    universal = allowed.get(UNIVERSAL_WORK_ID)
    for finding in observations.findings:
        state = statuses.setdefault(finding.finding_id, {"status": "model_empty"})
        if finding.finding_id in covered:
            state["status"] = "matched"
            continue
        if state["status"] not in FALLBACK_STATUSES:
            continue
        if universal is None:
            rate = retriever.session.scalar(select(Rate).where(Rate.code == UNIVERSAL_WORK_ID))
            if rate is None:
                continue
            universal = _candidate(rate, 0.0)
            candidates.append(universal)
        universal["finding_ids"].append(finding.finding_id)
        state["fallback_reason"] = state["status"]
        selection.works.append(WorkChoice(
            finding_id=finding.finding_id,
            work_id=UNIVERSAL_WORK_ID,
            reason=f"{finding.object_name}: {finding.visible_issue}"[:800],
        ))
        own = sorted((item for item in candidates
                      if finding.finding_id in item["finding_ids"] and item["work_id"] != UNIVERSAL_WORK_ID),
                     key=lambda item: item["retrieval_score"], reverse=True)[:UNIVERSAL_HINTS]
        hints = "; ".join(f"{item['work_id']} {item['name'][:60]}" for item in own)
        notes[(finding.finding_id, UNIVERSAL_WORK_ID)] = (
            f"{_FALLBACK_NOTES[state['status']]} — подставлены «Ремонтные работы», 1 чел.-ч; уточните состав и объём. "
            "Это fallback после автоматического поиска: проверен только верх выдачи, отсутствие расценки "
            "в ПЗР не подтверждено — проверьте справочник."
            + (f" Ближайшие позиции ПЗР: {hints}." if hints else "")
        )
    return notes


def find_material_candidates(retriever: MaterialRetriever, needs: list[dict], per_need: int = 5,
                             log: dict | None = None) -> dict[str, list[dict]]:
    """Подбирает кандидатов типа товара для каждой потребности в материалах.

    Один и тот же товар у разных поставщиков (одинаковый артикул производителя) показывается
    модели один раз: на этом шаге выбирается «какой товар», а «у кого дешевле» решает код позже.

    Принимает:
        retriever — поиск по прайсам;
        needs — потребности из materials.collect_needs;
        per_need — сколько разных товаров оставить на потребность;
        log — если задан, сюда пишутся отброшенные товары {need_id: [{name, reason}]}.
    Товар другого масштаба (светильник на запрос «лампа» и наоборот, compatibility.item_scale_conflict)
    модели не показывается (эксперимент B7).
    Возвращает: {need_id: [кандидат-словарь]} — артикул, наименование, категория, базовая единица,
        оценка поиска и id позиции в базе (для подбора поставщика).
    """
    result: dict[str, list[dict]] = {}
    for need in needs:
        unique: dict[str, dict] = {}
        # Берём с запасом: часть результатов — тот же товар у других поставщиков.
        for hit in retriever.search(need["query"], top_k=per_need * 3).hits:
            material: Material = hit.item  # type: ignore[assignment]
            if material.article_norm in unique or len(unique) >= per_need:
                continue
            reason = item_scale_conflict(need["query"], material.name)
            if reason:
                if log is not None:
                    log.setdefault(need["need_id"], []).append({"name": material.name, "reason": reason})
                continue
            unique[material.article_norm] = {
                "material_id": material.id,
                "article": material.article,
                "name": material.name,
                "category": material.category,
                "unit": material.base_unit,
                "supplier": material.supplier_code,
                "retrieval_score": round(hit.rerank_score if hit.rerank_score is not None else hit.rrf_score, 4),
            }
        result[need["need_id"]] = list(unique.values())
    return result
