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
from llm.estimator.candidates import find_candidates, match_unmatched
from llm.estimator.estimate import attach_materials, make_estimate
from llm.estimator.llm_client import InputTooLong, LLMServiceClient, LLMServiceError
from llm.estimator.materials import collect_needs, price_materials
from llm.estimator.prompts import OBSERVATION_PROMPT, SELECTION_PROMPT
from llm.estimator.schemas import Observations, Selection

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
    if not observations.findings:
        empty = Selection(works=[], alternatives=[], unmatched_works=[], assumptions=[])
        trace.update(candidates=[], selection=empty.model_dump(), raw_selection=[])
        return make_estimate(empty, observations, [], description, len(images)), trace

    # --- Этап 2. Кандидаты из RAG и выбор работ моделью. ---
    # Если вход получился слишком длинным для модели — сокращаем число кандидатов на дефект.
    selection, selection_raw, candidates = None, [], []
    for limit in sorted({per_finding, 4, 3}, reverse=True):
        logger.info("2/4 · Поиск расценок в RAG (по %s на дефект) и выбор работ…", limit)
        candidates = find_candidates(retriever, observations, per_finding=limit)
        if not candidates:
            selection = Selection(
                works=[], alternatives=[], assumptions=[],
                unmatched_works=[f"{f.object_name}: {f.visible_issue}" for f in observations.findings],
            )
            break
        # В промпт отдаём только то, что нужно для выбора; цены модели не показываем — их считает код.
        prompt_candidates = [{k: c[k] for k in ("work_id", "name", "composition", "unit", "finding_ids")} for c in candidates]
        try:
            selection, selection_raw = llm.ask_json(
                SELECTION_PROMPT,
                {"description": description or None, "observations": observations.model_dump(), "CANDIDATES": prompt_candidates},
                images, Selection, max_new_tokens=selection_max_tokens,
            )
            break
        except InputTooLong:
            continue
    if selection is None:
        raise LLMServiceError("Заявка слишком объёмная для модели даже с 3 кандидатами на дефект. Разделите её по участкам.")

    # --- Этап 3. Сопоставление «работ без расценки», проверка и расчёт. ---
    logger.info("3/4 · Проверка кодов, объёмов и расчёт стоимости…")
    trace.update(candidates=candidates, raw_selection=selection_raw, selection_model=selection.model_dump())
    auto_matched = match_unmatched(selection, candidates, retriever)
    trace.update(selection=selection.model_dump(), auto_matched=auto_matched)
    # Сведения об автоматическом сопоставлении остаются только в трассировке, в смету не выводятся.
    result = make_estimate(selection, observations, candidates, description, len(images))

    # --- Этап 4. Материалы для принятых работ и самый дешёвый поставщик. ---
    if material_retriever is not None:
        logger.info("4/4 · Подбор материалов и поставщиков…")
        needs = collect_needs(selection, result["works"])
        lines, not_found, material_trace = price_materials(
            needs, description, images, llm, material_retriever, material_candidates, material_selection_max_tokens,
        )
        trace.update(material_trace)
        attach_materials(result, lines, not_found)
    return result, trace
