"""Конвейер «фото + описание -> черновик сметы» (LLM + RAG).

    фото, описание
        │
        ▼  этап 1 — LLM (llm_service): что видно на фото, дефекты и поисковые запросы
    Observations
        │
        ▼  RAG (ml_service + pzr-db): по запросам каждого дефекта — расценки-кандидаты с ценами
    candidates
        │
        ▼  этап 2 — LLM: выбрать работы ТОЛЬКО из кандидатов, указать объём с доказательством
    Selection
        │
        ▼  код: проверка кодов и объёмов, стоимость = объём × цена из БД
    работы сметы
        │
        ▼  этап «Материалы» (если передан поиск по прайсам): потребности из ответа этапа 2 ->
        │  RAG по прайсам -> LLM выбирает товар -> код подбирает самого дешёвого поставщика
    смета (черновик): работы + материалы
"""

import io
import logging
import math
from decimal import Decimal

from PIL import Image, ImageOps

from rag.retriever import HybridRetriever, MaterialRetriever
from llm.estimator.candidates import (
    FALLBACK_STATUSES, UNIVERSAL_WORK_ID, add_universal_works, find_candidates,
)
from llm.estimator.compatibility import explicit_action, material_conflict, model_mismatches, work_conflicts
from llm.estimator.equipment import _HEIGHT, _OUTDOOR, equipment_requirements, price_equipment
from llm.estimator.estimate import attach_materials, make_estimate
from llm.estimator.llm_client import InputTooLong, LLMServiceClient, LLMServiceError
from llm.estimator.materials import collect_needs, price_materials
from llm.estimator.lamp_defaults import apply_linear_lamp_default
from llm.estimator.material_validation import attach_validation, validate_lines
from llm.estimator.prompts import DEFECT_MATERIALS_PROMPT, OBSERVATION_PROMPT, SELECTION_PROMPT
from llm.estimator.schemas import DefectMaterials, Finding, Observations, Selection

logger = logging.getLogger(__name__)

MAX_DESCRIPTION_CHARS = 4000


class EmptySelection(ValueError):
    """Модель не дала решения по дефекту: нет ни работ, ни объяснения в unmatched_works."""


class InvalidSelection(ValueError):
    """Модель назвала work_id не из кандидатов дефекта или чужой finding_id."""


# Какой ошибке проверки какое состояние дефекта соответствует (см. candidates.FALLBACK_STATUSES).
_ERROR_STATUSES = ((EmptySelection, "model_empty"), (InvalidSelection, "invalid_selection"))
_STATUS_WARNINGS = {
    "model_empty": "модель не дала решения (ни работы, ни причины отказа)",
    "invalid_selection": "модель назвала работы не из кандидатов",
    "llm_error": "модель вернула некорректный ответ",
}
# Смысловая группа состояния дефекта (поле resolution в finding_details):
#   matched — работа выбрана и не противоречит операции (при неясной операции — наиболее вероятный
#       вариант: плановая калькуляция строится до осмотра, расхождение плана и факта — норма процесса);
#   retrieval_unresolved — операция известна, но среди найденных кандидатов подходящей нет. Это НЕ
#       доказывает, что расценки нет во всём ПЗР (поиск видел только верх выдачи);
#   no_catalog_match_confirmed — отсутствие расценки подтвердил сметчик (автоматически не ставится);
#   error — техническая ошибка модели, не решение по существу.
RESOLUTIONS = {
    "matched": "matched",
    "no_candidates": "retrieval_unresolved",
    "model_unmatched": "retrieval_unresolved",
    "compatibility_rejected": "retrieval_unresolved",
    "model_empty": "error",
    "invalid_selection": "error",
    "llm_error": "error",
}
MAX_INSPECTION_CHARS = 2000


def normalize_operation(finding: Finding, own_text: str, inspection: str) -> None:
    """Сверяет статус операции, который дала модель, с текстом заявки и осмотра.

    «confirmed» без осмотра и без слова-действия в тексте дефекта понижается до «inferred»: модель
    не может подтвердить операцию, которую никто не назвал. Для «unknown» без вопросов добавляются
    вопросы по умолчанию (что менять, количество; для наружных/высотных объектов — способ доступа).

    Принимает:
        finding — дефект этапа 1 (изменяется на месте);
        own_text — текст, относящийся к дефекту (описание заявки, если дефект один, + сам дефект);
        inspection — текст осмотра мастера.
    Возвращает: ничего.
    """
    if finding.operation_status == "confirmed" and not inspection.strip() and not explicit_action(own_text):
        finding.operation_status = "inferred"
    if finding.operation_status == "unknown" and not finding.questions:
        questions = ["Какую операцию выполнить (что именно заменить или отремонтировать)?",
                     "Сколько изделий требует работы?"]
        if finding.environment == "outdoor" or _HEIGHT.search(own_text) or _OUTDOOR.search(own_text):
            questions.append("Высота и способ доступа (лестница или автовышка)?")
        finding.questions = questions


def prepare_image(data: bytes, max_pixels: int) -> bytes:
    """Готовит фото к отправке в модель: поворот по EXIF, RGB, уменьшение до max_pixels, JPEG.

    Уменьшение сокращает число токенов изображения и расход видеопамяти; при этом
    очень мелкие дефекты могут стать хуже видны — это осознанный компромисс прототипа.

    Принимает: data — байты исходного файла (JPEG/PNG/WebP); max_pixels — предел пикселей.
    Возвращает: байты JPEG.
    """
    with Image.open(io.BytesIO(data)) as source:
        if source.format not in {"JPEG", "PNG", "WEBP"}:
            raise ValueError("Поддерживаются JPEG, PNG и WebP.")
        # exif_transpose поворачивает снимок телефона так, как его видел пользователь.
        image = ImageOps.exif_transpose(source).convert("RGB")
    # Коэффициент масштаба по площади: sqrt(лимит / текущие пиксели).
    scale = min(1.0, math.sqrt(max_pixels / (image.width * image.height)))
    if scale < 1:
        image = image.resize((max(32, int(image.width * scale)), max(32, int(image.height * scale))), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def analyze_case(
    images: list[bytes],
    description: str,
    llm: LLMServiceClient,
    retriever: HybridRetriever,
    per_finding: int = 6,
    observation_max_tokens: int = 1500,
    selection_max_tokens: int = 3000,
    material_retriever: MaterialRetriever | None = None,
    material_candidates: int = 5,
    material_selection_max_tokens: int = 1500,
    search_by_description: bool = False,
    observations: Observations | None = None,
    inspection: str = "",
    linear_lamp_default: bool = False,
    equipment_tariff_rub: float | None = None,
    equipment_tariff_unit: str = "смена",
) -> tuple[dict, dict]:
    """Составляет черновик сметы по фотографиям одной заявки и необязательному описанию.

    Принимает:
        images — 0–4 подготовленных фото (байты JPEG, см. prepare_image);
        description — описание заявки (обязательно, если фото нет);
        llm — клиент LLM-сервиса;
        retriever — гибридный поиск RAG-модуля;
        per_finding — сколько кандидатов RAG брать на один дефект;
        observation_max_tokens, selection_max_tokens — лимиты длины ответа модели на этапах 1 и 2;
        material_retriever — поиск по прайсам поставщиков; None — смета только по работам;
        material_candidates — сколько товаров-кандидатов показывать модели на потребность;
        material_selection_max_tokens — лимит ответа модели при выборе товаров;
        search_by_description — искать работы и по исходному тексту заявки (ветка description);
        observations — готовый результат этапа 1 (например, из сохранённого trace.json): этап 1
            не выполняется. Нужно для экспериментов, где варианты сравниваются на одинаковом входе.
        inspection — запись осмотра мастера (до ремонта): операция, количество, характеристики, доступ.
        linear_lamp_default — правило «T8/G13 по умолчанию в местах общего пользования» (lamp_defaults);
        equipment_tariff_rub, equipment_tariff_unit — тариф автовышки (None — техника не оценивается).
            Важнее догадки по фото; "" — осмотра нет (предварительный проект по первичной заявке).
    Возвращает: кортеж (смета, трассировка всех этапов — для разбора ошибок).
    """
    description = description.strip()
    inspection = (inspection or "").strip()
    if len(inspection) > MAX_INSPECTION_CHARS:
        raise ValueError(f"Запись осмотра длиннее {MAX_INSPECTION_CHARS} символов.")
    if len(images) > 4:
        raise ValueError("Не больше 4 фотографий одной заявки.")
    if not images and not description:
        raise ValueError("Нужно хотя бы одно фото или описание заявки.")
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise ValueError(f"Описание длиннее {MAX_DESCRIPTION_CHARS} символов.")

    def validate_observations(value: Observations) -> None:
        """Проверяет, что модель ссылается только на существующие номера фото."""
        for finding in value.findings:
            if not images:
                # Без фото модель по привычке пишет [1]; это не ошибка по существу — просто обнуляем.
                finding.image_numbers = []
                continue
            if any(n < 1 or n > len(images) for n in finding.image_numbers):
                raise ValueError("Номера изображений выходят за пределы загруженных фото.")

    # --- Этап 1. Анализ фотографий. ---
    if observations is None:
        logger.info("1/4 · Анализ фотографий…")
        observations, observation_raw = llm.ask_json(
            OBSERVATION_PROMPT, {"description": description or None, "inspection": inspection or None}, images,
            Observations, validate_observations, max_new_tokens=observation_max_tokens,
        )
    else:
        # Готовые наблюдения (эксперимент): копия, чтобы не менять объект вызывающего кода.
        observations, observation_raw = observations.model_copy(deep=True), ["(из сохранённой трассировки)"]
        validate_observations(observations)
    trace: dict = {"observations": observations.model_dump(), "raw_observations": observation_raw,
                   "inspection": inspection or None}
    if not observations.findings and description:
        # Модель иногда возвращает пустой список дефектов, хотя описание называет неисправность
        # («не светит лампа»). Тогда вся заявка — один дефект, поиск идёт по её тексту.
        observations.findings = [Finding(
            finding_id="f1",
            object_name="по описанию заявки",
            visible_issue=description[:700],
            image_numbers=list(range(1, len(images) + 1)),
            search_queries=[description[:200]],
        )]
        trace.update(findings_from_description=True)
    if not observations.findings:
        empty = Selection(works=[], alternatives=[], unmatched_works=[], assumptions=[])
        trace.update(candidates=[], selection=empty.model_dump(), raw_selection=[])
        return make_estimate(empty, observations, [], description, len(images), inspection), trace
    single = len(observations.findings) == 1
    own_texts = {
        # Текст заявки общий: при нескольких дефектах он не переносит признаки на соседний дефект.
        f.finding_id: f"{description if single else ''} {inspection} {f.object_name} {f.visible_issue}"
        for f in observations.findings
    }
    for finding in observations.findings:
        normalize_operation(finding, own_texts[finding.finding_id], inspection)
    trace["observations"] = observations.model_dump()

    # --- Этап 2. Кандидаты из RAG и выбор работ моделью — отдельным вызовом на каждый дефект. ---
    # Короткий вход (один дефект, его кандидаты и его фото) модель 8B обрабатывает заметно
    # надёжнее, чем всю заявку сразу: на длинном входе она часто возвращает пустой список работ.
    logger.info("2/4 · Поиск расценок в RAG (по %s на дефект) и выбор работ…", per_finding)
    retrieval_log: dict[str, dict] = {}
    candidates = find_candidates(retriever, observations, per_finding=per_finding, log=retrieval_log,
                                 description=description if search_by_description else None)
    parts: list[Selection] = []
    selection_raw: list[str] = []
    statuses: dict[str, dict] = {}
    names = {c["work_id"]: c["name"] for c in candidates}
    rejected: list[dict] = []
    for number, finding in enumerate(observations.findings, start=1):
        # Неясная операция (operation_status = unknown) не останавливает выбор: плановая калькуляция
        # делается по заявке и фото до осмотра, поэтому модель берёт наиболее вероятный вариант
        # (operation этапа 1), а остальные варианты и вопросы остаются в finding_details.
        logger.info("2/4 · Выбор работ: дефект %s из %s…", number, len(observations.findings))
        part, raw, state = select_for_finding(finding, observations, candidates, description, images, llm,
                                              selection_max_tokens, inspection)
        # Код проверяет явные противоречия «операция дефекта — расценка» и отметки модели «no».
        dropped = []
        for bucket in (part.works, part.alternatives):
            for choice in list(bucket):
                reasons = work_conflicts(finding.operation, finding.environment, names.get(choice.work_id, ""))
                reasons += [f"модель отметила несоответствие: {axis}" for axis in model_mismatches(choice)]
                if reasons:
                    bucket.remove(choice)
                    dropped.append({**choice.model_dump(), "rejection_reasons": reasons})
        if dropped:
            rejected += dropped
            state["compatibility_rejected"] = [{"work_id": d["work_id"], "reasons": d["rejection_reasons"]}
                                               for d in dropped]
            if state["status"] == "matched" and not part.works and not part.alternatives:
                state["status"] = "compatibility_rejected"
        parts.append(part)
        selection_raw += raw
        statuses[finding.finding_id] = state
    selection = Selection(
        works=[w for part in parts for w in part.works][:15],
        alternatives=[w for part in parts for w in part.alternatives][:10],
        unmatched_works=[t for part in parts for t in part.unmatched_works][:8],
        assumptions=list(dict.fromkeys(t for part in parts for t in part.assumptions))[:10],
    )

    # --- Этап 3. Сопоставление «работ без расценки», проверка и расчёт. ---
    logger.info("3/4 · Проверка кодов, объёмов и расчёт стоимости…")
    trace.update(candidates=candidates, raw_selection=selection_raw, selection_model=selection.model_dump())
    # Отказ модели (unmatched_works) не перекрывается похожей по тексту работой: раньше reranker
    # сопоставлял текст отказа с кандидатами, и в смету попадали работы, которые модель отвергла
    # («Установка полотенцесушителя» для инфостенда). Отказ — это 1-282 с подсказками ниже.
    trace.update(selection=selection.model_dump())
    # Дефектам с подтверждённым отсутствием работы (no_candidates, model_unmatched) — универсальная
    # «Ремонтные работы» с подсказками ближайших позиций ПЗР (требование заказчика).
    # Правило владельца (под флагом): в МОП при неизвестном типе лампы — 2-220 вместо 2-218/2-219.
    if linear_lamp_default:
        trace["lamp_default"] = apply_linear_lamp_default(selection, observations, candidates, retriever,
                                                          description, inspection)
    added_notes = add_universal_works(selection, observations, candidates, retriever, statuses)
    trace.update(added_works={f"{finding_id} {work_id}": note for (finding_id, work_id), note in added_notes.items()})
    # По каждому дефекту: запросы, найденное до отбора, что показали модели, её ответы и итог.
    trace["findings"] = {
        finding_id: {**retrieval_log.get(finding_id, {}), **{k: v for k, v in state.items() if k != "raw"},
                     "raw_answers": state.get("raw", [])}
        for finding_id, state in statuses.items()
    }
    result = make_estimate(selection, observations, candidates, description, len(images), inspection)
    for item in rejected:
        result["rejected_choices"].append(item)
        result["warnings"].append(f"Позиция {item['work_id']} отклонена для {item['finding_id']}: "
                                  f"{'; '.join(item['rejection_reasons'])}.")
    for line in result["works"]:
        if (line["finding_id"], line["work_id"]) in added_notes:
            line["note"] = added_notes[(line["finding_id"], line["work_id"])]
            line["fallback_reason"] = statuses[line["finding_id"]]["fallback_reason"]
    # Итоговое состояние каждого дефекта — машинно-читаемо (необязательное поле сметы).
    result["finding_statuses"] = {finding_id: state["status"] for finding_id, state in statuses.items()}
    # Структура ремонта и уровень достоверности по каждому дефекту (необязательное поле сметы).
    result["finding_details"] = {
        f.finding_id: {
            "status": statuses[f.finding_id]["status"],
            "resolution": RESOLUTIONS.get(statuses[f.finding_id]["status"], "error"),
            "object": f.object_name,
            "symptom": f.visible_issue,
            "operation": f.operation,
            "operation_status": f.operation_status,
            "possible_operations": f.possible_operations,
            "environment": f.environment,
            "questions": f.questions,
        }
        for f in observations.findings
    }
    for finding_id, state in statuses.items():
        if state["status"] in _STATUS_WARNINGS:
            result["warnings"].append(
                f"Для {finding_id} {_STATUS_WARNINGS[state['status']]} — работа не назначена, нужна ручная проверка."
            )
    # Техника (автовышка и т. п.) — отдельный необязательный блок, не материалы; стоимость не считается.
    equipment, equipment_warnings = equipment_requirements(
        result["works"], {f.finding_id: f.model_dump() for f in observations.findings}, description, inspection,
    )
    if equipment:
        result["equipment_requirements"] = equipment
    result["warnings"] += equipment_warnings

    # --- Этап 4. Материалы для принятых работ и самый дешёвый поставщик. ---
    if material_retriever is not None:
        logger.info("4/4 · Подбор материалов и поставщиков…")
        # У «Ремонтных работ» нет состава — материалы модель определяет по самому дефекту.
        # Единственный источник: дефекты, которым 1-282 поставлена как fallback (statuses).
        fallback_findings = {f for f, state in statuses.items() if state.get("fallback_reason") in FALLBACK_STATUSES}
        trace.update(raw_defect_materials=suggest_defect_materials(
            selection, observations, fallback_findings, description, images, llm, material_selection_max_tokens,
        ))
        trace["rejected_materials"] = drop_conflicting_materials(
            selection, observations, names, result["warnings"])
        needs = collect_needs(selection, result["works"])
        lines, not_found, material_trace = price_materials(
            needs, description, images, llm, material_retriever, material_candidates, material_selection_max_tokens,
        )
        trace.update(material_trace)
        # Проверка совместимости «материал — работа» (B2.1): меняет только список материалов.
        # В подтверждённую калькуляцию идут ALLOW; REVIEW — отдельной суммой; REJECT — исключены.
        sections = {c["work_id"]: c for c in candidates}
        findings = {f.finding_id: f for f in observations.findings}
        works_info = {
            (line["finding_id"], line["work_id"]): {
                "work_id": line["work_id"], "name": line["name"],
                "section": sections.get(line["work_id"], {}).get("section", ""),
                "composition": sections.get(line["work_id"], {}).get("composition", ""),
                "finding_text": " ".join(filter(None, (
                    getattr(findings.get(line["finding_id"]), field, "") for field in
                    ("object_name", "visible_issue", "operation")))),
            }
            for line in result["works"]
        }
        allowed, review, rejected, validation_log = validate_lines(lines, works_info)
        trace["material_validation"] = validation_log
        attach_materials(result, allowed, not_found)
        attach_validation(result, lines, review, rejected, validation_log)
    # Техника по тарифу владельца (если задан): подтверждённая — в итог, вероятная — в предварительный.
    if equipment_tariff_rub is not None:
        price_equipment(result, Decimal(str(equipment_tariff_rub)), equipment_tariff_unit)
    return result, trace


def drop_conflicting_materials(selection: Selection, observations: Observations, names: dict[str, str],
                               warnings: list[str]) -> list[dict]:
    """Убирает материалы, противоречащие работе или операции (лампа при замене светильника и наоборот).

    Принимает:
        selection — выбор работ (изменяется на месте: materials у работ);
        observations — дефекты этапа 1 (операция);
        names — {work_id: наименование};
        warnings — предупреждения сметы (дополняются).
    Возвращает: список убранных материалов (для трассировки).
    """
    operations = {f.finding_id: f.operation for f in observations.findings}
    dropped: list[dict] = []
    for choice in selection.works:
        kept = []
        for material in choice.materials:
            reason = material_conflict(names.get(choice.work_id, ""), material.query,
                                       operations.get(choice.finding_id, ""))
            if reason:
                dropped.append({"finding_id": choice.finding_id, "work_id": choice.work_id,
                                "query": material.query, "reason": reason})
                warnings.append(f"Материал «{material.query}» ({choice.work_id}) не подбирается: {reason}.")
            else:
                kept.append(material)
        choice.materials = kept
    return dropped


def _selection_validator(finding_id: str, allowed: set[str]):
    """Проверка ответа выбора работ для одного дефекта (передаётся в ask_json).

    Принимает: finding_id — текущий дефект; allowed — work_id его кандидатов (без 1-282).
    Возвращает: функцию check(selection), которая бросает InvalidSelection или EmptySelection.
    Текст ошибки уходит модели при повторном запросе — поэтому он описывает, как исправить ответ.
    """
    def check(selection: Selection) -> None:
        bad = [f"{choice.work_id} ({choice.finding_id})" for choice in [*selection.works, *selection.alternatives]
               if choice.finding_id != finding_id or choice.work_id not in allowed]
        if bad:
            raise InvalidSelection(
                f"Позиции не из CANDIDATES или не для {finding_id}: {', '.join(bad)}. "
                f"Допустимы только work_id {', '.join(sorted(allowed))} и finding_id {finding_id}."
            )
        if not selection.works and not selection.alternatives and not selection.unmatched_works:
            raise EmptySelection(
                f"Для {finding_id} нет решения. Либо добавь подходящую позицию из CANDIDATES в works, "
                "либо, если ни одна не подходит, напиши в unmatched_works, какая работа нужна и почему "
                "кандидаты не подходят."
            )
    return check


def select_for_finding(
    finding: Finding,
    observations: Observations,
    candidates: list[dict],
    description: str,
    images: list[bytes],
    llm: LLMServiceClient,
    max_new_tokens: int,
    inspection: str = "",
) -> tuple[Selection, list[str], dict]:
    """Выбор работ моделью для одного дефекта: его кандидаты и только его фото.

    Фото дефекта в запросе нумеруются заново с 1; номера в ответе модели переводятся обратно
    в номера фото заявки. Если вход слишком длинный, число кандидатов сокращается до 4, затем до 3.
    Ответ проверяется (_selection_validator): работы — только из кандидатов этого дефекта, и решение
    обязательно — работа или объяснённый отказ. При ошибке модель получает её текст и отвечает
    ещё раз (механизм ask_json); вторая ошибка — состояние дефекта model_empty / invalid_selection /
    llm_error, без работ. Остальные дефекты заявки при этом считаются как обычно.
    Недоступность LLM-сервиса не скрывается: LLMServiceError пробрасывается, заявка завершается ошибкой.

    Принимает:
        finding — дефект из этапа 1;
        observations — весь результат этапа 1 (нужны summary и unknowns);
        candidates — кандидаты RAG по всем дефектам (с finding_ids);
        description — описание заявки;
        images — все фото заявки;
        llm — клиент LLM-сервиса;
        max_new_tokens — лимит длины ответа;
        inspection — запись осмотра мастера ("" — нет).
    Возвращает: кортеж (выбор по этому дефекту, сырые ответы модели, состояние для трассировки:
        {"status", "prompt_candidates", "error"?, "raw"}).
    """
    own = sorted((c for c in candidates if finding.finding_id in c["finding_ids"]),
                 key=lambda c: c["retrieval_score"], reverse=True)
    empty = Selection(works=[], alternatives=[], unmatched_works=[], assumptions=[])
    if not own:
        unmatched = empty.model_copy(update={"unmatched_works": [f"{finding.object_name}: {finding.visible_issue}"]})
        return unmatched, [], {"status": "no_candidates", "prompt_candidates": []}

    # Только фото этого дефекта, с новой нумерацией 1..k.
    numbers = [n for n in finding.image_numbers if 1 <= n <= len(images)]
    local_finding = finding.model_copy(update={"image_numbers": list(range(1, len(numbers) + 1))})
    local_observations = {
        "summary": observations.summary,
        "findings": [local_finding.model_dump()],
        "unknowns": observations.unknowns,
    }
    for limit in sorted({len(own), 4, 3}, reverse=True):
        if limit > len(own):
            continue
        # Цены модели не показываем — их считает код.
        # finding_ids — только этот дефект, чтобы модель не привязывала работу к чужому.
        prompt_candidates = [{**{k: c[k] for k in ("work_id", "name", "composition", "unit")},
                              "finding_ids": [finding.finding_id]} for c in own[:limit]]
        payload = {"description": description or None, "inspection": inspection or None,
                   "observations": local_observations, "CANDIDATES": prompt_candidates}
        validator = _selection_validator(finding.finding_id, {c["work_id"] for c in prompt_candidates})
        state = {"prompt_candidates": [{k: c[k] for k in ("work_id", "name")} for c in prompt_candidates]}
        try:
            part, raw = llm.ask_json(SELECTION_PROMPT, payload, [images[n - 1] for n in numbers], Selection,
                                     validator, max_new_tokens=max_new_tokens)
        except InputTooLong:
            continue
        except LLMServiceError as error:
            cause = error.__cause__
            if not isinstance(cause, (ValueError, TypeError)):
                raise  # сервис недоступен или ответил ошибкой — это не решение модели по дефекту
            status = next((s for kind, s in _ERROR_STATUSES if isinstance(cause, kind)), "llm_error")
            logger.warning("Дефект %s: %s (%s)", finding.finding_id, status, cause)
            return empty, [], {**state, "status": status, "error": str(cause)[:500],
                               "raw": getattr(error, "attempts", [])}
        # Номера фото из ответа — обратно в номера фото заявки.
        for choice in [*part.works, *part.alternatives]:
            choice.image_numbers = [numbers[n - 1] for n in choice.image_numbers if 1 <= n <= len(numbers)]
        status = "matched" if part.works or part.alternatives else "model_unmatched"
        return part, raw, {**state, "status": status, "raw": raw}
    raise LLMServiceError("Дефект слишком объёмный для модели даже с 3 кандидатами. Разделите заявку по участкам.")


def suggest_defect_materials(
    selection: Selection,
    observations: Observations,
    finding_ids: set[str],
    description: str,
    images: list[bytes],
    llm: LLMServiceClient,
    max_new_tokens: int,
) -> list[str]:
    """Определяет материалы для «Ремонтных работ» (1-282) по описанию дефекта и его фото.

    У универсальной работы нет состава, поэтому материалы нельзя вывести из работы, как обычно.
    Для каждого такого дефекта — отдельный короткий вызов модели; найденные потребности
    записываются в materials работы и дальше подбираются по прайсам как обычные.
    Ошибка модели по одному дефекту оставляет его без материалов, но не роняет заявку.

    Принимает:
        selection — выбор работ (изменяется на месте: materials у работ 1-282);
        observations — дефекты этапа 1;
        finding_ids — дефекты, которым подставлены «Ремонтные работы»;
        description — описание заявки;
        images — все фото заявки;
        llm — клиент LLM-сервиса;
        max_new_tokens — лимит длины ответа.
    Возвращает: сырые ответы модели (для трассировки).
    """
    findings = {finding.finding_id: finding for finding in observations.findings}
    raw_answers: list[str] = []
    for choice in selection.works:
        if choice.work_id != UNIVERSAL_WORK_ID or choice.finding_id not in finding_ids:
            continue
        finding = findings[choice.finding_id]
        numbers = [n for n in finding.image_numbers if 1 <= n <= len(images)]
        payload = {
            "description": description or None,
            "defect": {"object": finding.object_name, "issue": finding.visible_issue},
        }
        try:
            answer, raw = llm.ask_json(DEFECT_MATERIALS_PROMPT, payload, [images[n - 1] for n in numbers],
                                       DefectMaterials, max_new_tokens=max_new_tokens)
        except (InputTooLong, LLMServiceError) as error:
            logger.warning("Дефект %s: материалы не определены (%s)", finding.finding_id, error)
            continue
        choice.materials = answer.materials
        raw_answers += raw
    return raw_answers
