"""Схемы структурированных ответов LLM и разбор JSON из текста модели.

Модель обязана отвечать JSON строго по этим схемам. Pydantic проверяет ответ:
лишние поля, неверные типы, слишком длинные списки — ошибка, и модель отвечает повторно.
Так «фантазии» модели отсекаются до того, как попадут в смету.
"""

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Базовая схема: запрещает поля, которых нет в описании (extra='forbid')."""

    model_config = ConfigDict(extra="forbid")


# --- Этап 1: что видно на фотографиях -------------------------------------------------


class Finding(StrictModel):
    """Один дефект или участок ремонта, найденный на фото."""

    finding_id: str = Field(min_length=1, max_length=30)
    # Объект («розетка», «стена в подъезде») и что с ним не так.
    object_name: str = Field(min_length=1, max_length=200)
    visible_issue: str = Field(min_length=1, max_length=700)
    # На каких фото виден дефект (нумерация с 1); пустой список — заявка без фото.
    image_numbers: list[int] = Field(max_length=4)
    # Короткие запросы «объект + работа» для поиска в справочнике («смена розетки»).
    search_queries: list[str] = Field(min_length=1, max_length=3)


class Observations(StrictModel):
    """Результат анализа фотографий."""

    summary: str = Field(max_length=1500)
    # Показывают ли фото один участок «до и после» — только при явном указании в описании.
    image_relation: Literal["before_only", "before_after", "unknown"]
    findings: list[Finding] = Field(max_length=5)
    # Чего не видно по фото: размеры, материал, диаметр и т.п.
    unknowns: list[str] = Field(max_length=10)

    @model_validator(mode="after")
    def unique_findings(self) -> "Observations":
        """Проверяет, что идентификаторы дефектов не повторяются.

        Принимает: ничего (проверяет уже заполненную схему).
        Возвращает: эту же схему, если всё в порядке.
        """
        ids = [finding.finding_id for finding in self.findings]
        if len(ids) != len(set(ids)):
            raise ValueError("finding_id должны быть уникальны; один дефект не дублируется по фото.")
        return self


# --- Этап 2: какие работы выбрать из кандидатов --------------------------------------


class MaterialNeed(StrictModel):
    """Материал, нужный для выполнения работы: что искать в прайсах и сколько на единицу работы."""

    # Описание для поиска в прайсах: тип товара и известные характеристики, без артикулов и цен
    # («розетка с заземлением скрытой установки белая», «лампа светодиодная E27»).
    query: str = Field(min_length=3, max_length=200)
    # Сколько материала на одну единицу работы (на 1 шт смены розетки — 1 розетка).
    qty_per_work_unit: float = Field(gt=0, le=1000, allow_inf_nan=False)
    # Базовая единица материала: штуки или метры.
    unit: Literal["шт", "м"]
    reason: str = Field(default="", max_length=300)


class WorkChoice(StrictModel):
    """Выбранная работа: код расценки из кандидатов, обоснование и объём (если его можно установить)."""

    finding_id: str = Field(min_length=1, max_length=30)
    # Код расценки из справочника (например, «2-236»). Должен быть среди кандидатов.
    work_id: str = Field(min_length=1, max_length=30)
    reason: str = Field(min_length=1, max_length=800)
    quantity: float | None = Field(default=None, gt=0, le=1_000_000, allow_inf_nan=False)
    quantity_unit: str = Field(default="", max_length=40)
    # Откуда объём: из описания (нужна цитата), подсчёт предметов на фото или неизвестен.
    quantity_source: Literal["description", "visible_count", "unknown"] = "unknown"
    quantity_evidence: str = Field(default="", max_length=500)
    image_numbers: list[int] = Field(default_factory=list, max_length=4)
    # Основные материалы для работы (норм расхода нет — их предлагает модель).
    materials: list[MaterialNeed] = Field(default_factory=list, max_length=5)


class Selection(StrictModel):
    """Выбор работ: основной вариант, альтернативы, работы без расценки и допущения."""

    works: list[WorkChoice] = Field(max_length=15)
    alternatives: list[WorkChoice] = Field(max_length=10)
    unmatched_works: list[str] = Field(max_length=8)
    assumptions: list[str] = Field(max_length=10)


# --- Этап 3: какой именно товар взять для каждой потребности ----------------------------


class MaterialChoice(StrictModel):
    """Выбор товара для одной потребности: артикул из кандидатов или «подходящего нет»."""

    need_id: str = Field(min_length=1, max_length=20)
    # Артикул одного из кандидатов этой потребности; null — подходящего товара среди кандидатов нет.
    article: str | None = Field(default=None, max_length=128)
    reason: str = Field(default="", max_length=300)


class MaterialSelection(StrictModel):
    """Выбор товаров для всех потребностей сметы."""

    choices: list[MaterialChoice] = Field(max_length=40)


def parse_json_object(text: str) -> dict:
    """Достаёт JSON-объект из ответа модели (текст никогда не исполняется).

    Модель иногда добавляет пояснение перед JSON или блок рассуждений <think>…</think>.
    Сначала пробуем разобрать весь текст, затем — с первой открывающей фигурной скобки.

    Принимает: text — ответ модели.
    Возвращает: словарь из JSON.
    Исключения: ValueError, если полного JSON-объекта в тексте нет.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    # У Thinking-моделей открывающий <think> стоит в самом промпте, и в ответе остаётся
    # только «рассуждение…</think>JSON». Берём всё, что после последнего </think>.
    text = text.rsplit("</think>", 1)[-1].strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start >= 0:
        try:
            # raw_decode разбирает JSON с начала строки и игнорирует «хвост» после него.
            value, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    raise ValueError("Модель не вернула полный JSON-объект.")
