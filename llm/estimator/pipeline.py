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

from PIL import Image, ImageOps

from rag.retriever import HybridRetriever, MaterialRetriever
from llm.estimator.candidates import UNIVERSAL_WORK_ID, add_universal_works, find_candidates, match_unmatched
from llm.estimator.estimate import attach_materials, make_estimate
from llm.estimator.llm_client import InputTooLong, LLMServiceClient, LLMServiceError
from llm.estimator.materials import collect_needs, price_materials
from llm.estimator.prompts import DEFECT_MATERIALS_PROMPT, OBSERVATION_PROMPT, SELECTION_PROMPT
from llm.estimator.schemas import DefectMaterials, Finding, Observations, Selection

logger = logging.getLogger(__name__)

MAX_DESCRIPTION_CHARS = 4000


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
        material_selection_max_tokens — лимит ответа модели при выборе товаров.
    Возвращает: кортеж (смета, трассировка всех этапов — для разбора ошибок).
    """
    description = description.strip()
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
    logger.info("1/4 · Анализ фотографий…")
    observations, observation_raw = llm.ask_json(
        OBSERVATION_PROMPT, {"description": description or None}, images,
        Observations, validate_observations, max_new_tokens=observation_max_tokens,
    )
    trace: dict = {"observations": observations.model_dump(), "raw_observations": observation_raw}
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
        return make_estimate(empty, observations, [], description, len(images)), trace

    # --- Этап 2. Кандидаты из RAG и выбор работ моделью — отдельным вызовом на каждый дефект. ---
    # Короткий вход (один дефект, его кандидаты и его фото) модель 8B обрабатывает заметно
    # надёжнее, чем всю заявку сразу: на длинном входе она часто возвращает пустой список работ.
    logger.info("2/4 · Поиск расценок в RAG (по %s на дефект) и выбор работ…", per_finding)
    candidates = find_candidates(retriever, observations, per_finding=per_finding)
    parts: list[Selection] = []
    selection_raw: list[str] = []
    for number, finding in enumerate(observations.findings, start=1):
        logger.info("2/4 · Выбор работ: дефект %s из %s…", number, len(observations.findings))
        part, raw = select_for_finding(finding, observations, candidates, description, images, llm, selection_max_tokens)
        parts.append(part)
        selection_raw += raw
    selection = Selection(
        works=[w for part in parts for w in part.works][:15],
        alternatives=[w for part in parts for w in part.alternatives][:10],
        unmatched_works=[t for part in parts for t in part.unmatched_works][:8],
        assumptions=list(dict.fromkeys(t for part in parts for t in part.assumptions))[:10],
    )

    # --- Этап 3. Сопоставление «работ без расценки», проверка и расчёт. ---
    logger.info("3/4 · Проверка кодов, объёмов и расчёт стоимости…")
    trace.update(candidates=candidates, raw_selection=selection_raw, selection_model=selection.model_dump())
    auto_matched = match_unmatched(selection, candidates, retriever)
    trace.update(selection=selection.model_dump(), auto_matched=auto_matched)
    # Дефектам без единой работы — лучший кандидат при высокой оценке поиска, иначе универсальная
    # «Ремонтные работы» (требование заказчика).
    added_notes = add_universal_works(selection, observations, candidates, retriever)
    trace.update(added_works={f"{finding_id} {work_id}": note for (finding_id, work_id), note in added_notes.items()})
    # Сведения об автоматическом сопоставлении остаются только в трассировке, в смету не выводятся.
    result = make_estimate(selection, observations, candidates, description, len(images))
    for line in result["works"]:
        if (line["finding_id"], line["work_id"]) in added_notes:
            line["note"] = added_notes[(line["finding_id"], line["work_id"])]

    # --- Этап 4. Материалы для принятых работ и самый дешёвый поставщик. ---
    if material_retriever is not None:
        logger.info("4/4 · Подбор материалов и поставщиков…")
        # У «Ремонтных работ» нет состава — материалы модель определяет по самому дефекту.
        trace.update(raw_defect_materials=suggest_defect_materials(
            selection, observations, {f for f, w in added_notes if w == UNIVERSAL_WORK_ID},
            description, images, llm, material_selection_max_tokens,
        ))
        needs = collect_needs(selection, result["works"])
        lines, not_found, material_trace = price_materials(
            needs, description, images, llm, material_retriever, material_candidates, material_selection_max_tokens,
        )
        trace.update(material_trace)
        attach_materials(result, lines, not_found)
    return result, trace


def select_for_finding(
    finding: Finding,
    observations: Observations,
    candidates: list[dict],
    description: str,
    images: list[bytes],
    llm: LLMServiceClient,
    max_new_tokens: int,
) -> tuple[Selection, list[str]]:
    """Выбор работ моделью для одного дефекта: его кандидаты и только его фото.

    Фото дефекта в запросе нумеруются заново с 1; номера в ответе модели переводятся обратно
    в номера фото заявки. Если вход слишком длинный, число кандидатов сокращается до 4, затем до 3.
    Если модель дважды ответила некорректно, дефект уходит в «работы без расценки» —
    остальные дефекты заявки при этом считаются как обычно.

    Принимает:
        finding — дефект из этапа 1;
        observations — весь результат этапа 1 (нужны summary и unknowns);
        candidates — кандидаты RAG по всем дефектам (с finding_ids);
        description — описание заявки;
        images — все фото заявки;
        llm — клиент LLM-сервиса;
        max_new_tokens — лимит длины ответа.
    Возвращает: кортеж (выбор по этому дефекту, сырые ответы модели).
    """
    own = sorted((c for c in candidates if finding.finding_id in c["finding_ids"]),
                 key=lambda c: c["retrieval_score"], reverse=True)
    unmatched = Selection(works=[], alternatives=[], assumptions=[],
                          unmatched_works=[f"{finding.object_name}: {finding.visible_issue}"])
    if not own:
        return unmatched, []

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
        payload = {"description": description or None, "observations": local_observations,
                   "CANDIDATES": prompt_candidates}
        try:
            part, raw = llm.ask_json(SELECTION_PROMPT, payload, [images[n - 1] for n in numbers], Selection,
                                     max_new_tokens=max_new_tokens)
        except InputTooLong:
            continue
        except LLMServiceError as error:
            logger.warning("Дефект %s: модель не дала корректный ответ (%s)", finding.finding_id, error)
            return unmatched, []
        # Номера фото из ответа — обратно в номера фото заявки.
        for choice in [*part.works, *part.alternatives]:
            choice.image_numbers = [numbers[n - 1] for n in choice.image_numbers if 1 <= n <= len(numbers)]
        return part, raw
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
