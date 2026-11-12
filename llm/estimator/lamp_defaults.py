"""Правило «лампы T8/G13 по умолчанию в местах общего пользования» (эксперимент B6, под флагом).

Наблюдение по итоговым калькуляциям: в подъездах, холлах и у лифтов мастера меняют линейные
лампы T8/G13 (2-220), а модель выбирает лампы с цоколем E27 (2-218/2-219) — по фото тип лампы
почти не различить. Правило: если для дефекта выбрана 2-218/2-219, дефект в помещении общего
пользования, а тип лампы нигде не назван (ни в заявке, ни в осмотре, ни в описании фото) —
работа заменяется на 2-220, а запрос лампы в материалах — на «лампа светодиодная T8 G13».

Это решение владельца продукта, а не вывод модели: включается настройкой
DEFAULT_LINEAR_LAMPS_COMMON_AREAS=true (по умолчанию выключено). Пометок «предположение»
в смете нет (бизнес-решение: наиболее вероятный вариант без пометок); срабатывание видно в трассировке.
"""

import re

from sqlalchemy import select

from llm.estimator.candidates import _candidate
from llm.estimator.schemas import Observations, Selection
from rag.db import Rate

LINEAR_LAMP_WORK_ID = "2-220"
# Работы «лампа с цоколем E27/E14/GX53…», которые заменяются правилом.
SOCKET_LAMP_WORK_IDS = {"2-218", "2-219"}
LINEAR_LAMP_QUERY = "лампа светодиодная T8 G13 линейная"

# Места общего пользования: подъезд, холлы, лифт, тамбур, коридор, лестница, этаж.
_COMMON_AREA = re.compile(r"подъезд|холл|лифт|тамбур|коридор|лестни|вестибюл|\bмоп\b|этаж", re.IGNORECASE)
# Явно не общее пользование.
_PRIVATE = re.compile(r"в квартире|санузл|ванн|кухн", re.IGNORECASE)
# Тип лампы или светильника назван — правило не применяется.
_LAMP_TYPE_KNOWN = re.compile(
    r"[eе]\s?27|[eе]\s?14|gx\s?53|gu\s?10|\bg9\b|цокол|шар|груш|свеч|лампочк|накаливан|подсветк|лент"
    r"|точечн|спот|\bбра\b|настольн|t8|т8|g13|люминесцент|линейн|трубчат",
    re.IGNORECASE,
)


def _rate_candidate(candidates: list[dict], retriever, finding_id: str) -> dict | None:
    """Кандидат 2-220 для дефекта: из списка кандидатов или из справочника.

    Принимает: candidates — кандидаты RAG (изменяется на месте); retriever — поиск по ПЗР (нужна
        сессия базы); finding_id — дефект, которому разрешается 2-220.
    Возвращает: словарь кандидата или None, если расценки нет в справочнике.
    """
    item = next((c for c in candidates if c["work_id"] == LINEAR_LAMP_WORK_ID), None)
    if item is None:
        rate = retriever.session.scalar(select(Rate).where(Rate.code == LINEAR_LAMP_WORK_ID))
        if rate is None or rate.code != LINEAR_LAMP_WORK_ID:
            return None
        item = _candidate(rate, 0.0)
        candidates.append(item)
    if finding_id not in item["finding_ids"]:
        item["finding_ids"].append(finding_id)
    return item


def apply_linear_lamp_default(selection: Selection, observations: Observations, candidates: list[dict],
                              retriever, description: str, inspection: str = "") -> list[dict]:
    """Заменяет 2-218/2-219 на 2-220 там, где выполняются условия правила (см. описание модуля).

    Принимает:
        selection — выбор работ (изменяется на месте: work_id и запросы материалов);
        observations — дефекты этапа 1;
        candidates — кандидаты RAG (изменяется на месте: 2-220 разрешается дефекту);
        retriever — поиск по ПЗР (сессия базы — чтобы взять расценку 2-220);
        description, inspection — текст заявки и осмотра.
    Возвращает: список срабатываний для трассировки [{finding_id, from, to, reason}].
    """
    findings = {f.finding_id: f for f in observations.findings}
    request_text = f"{description} {inspection}"
    log: list[dict] = []
    for choice in [*selection.works, *selection.alternatives]:
        finding = findings.get(choice.finding_id)
        if choice.work_id not in SOCKET_LAMP_WORK_IDS or finding is None:
            continue
        # Что модель увидела на фото: объект, симптом, операция и общее описание снимков.
        finding_text = " ".join([finding.object_name, finding.visible_issue, finding.operation, observations.summary])
        if finding.environment == "outdoor" or _PRIVATE.search(request_text):
            continue
        if not _COMMON_AREA.search(f"{request_text} {finding_text}"):
            continue
        if _LAMP_TYPE_KNOWN.search(f"{request_text} {finding_text}"):
            continue
        if _rate_candidate(candidates, retriever, choice.finding_id) is None:
            continue
        log.append({"finding_id": choice.finding_id, "from": choice.work_id, "to": LINEAR_LAMP_WORK_ID,
                    "reason": "место общего пользования, тип лампы не назван"})
        choice.work_id = LINEAR_LAMP_WORK_ID
        for material in choice.materials:
            if re.search(r"ламп", material.query, re.IGNORECASE):
                material.query = LINEAR_LAMP_QUERY
    return log
