"""Словарь соответствий «язык заявки → работа ПЗР», построенный по истории АВР (гипотеза Г1).

Идея: жители и диспетчеры пишут «капает», «фитинги», «заглушка», а в справочнике — «свищ»,
«муфтовая арматура», «крышка на коробке». В истории АВР есть пары «текст заявки — код работы,
которую фактически выполнил мастер». Если слово заявки устойчиво ведёт к одной работе, а в тексте
самой работы этого слова нет, поиск такую работу по этому слову не найдёт. Словарь хранит такие
связи и для новой заявки добавляет отдельный поисковый запрос — наименование работы из ПЗР.

Как строится (простой воспроизводимый baseline):
    - тексты заявок обучающей части нормализуются в основы слов (stem);
    - для каждой пары (основа, код работы) считается число независимых заявок (support) и доля
      заявок с этой основой, где была эта работа (confidence);
    - остаются пары с support ≥ min_support и confidence ≥ min_confidence, причём основы нет
      в тексте самой работы (иначе поиск и так её находит — это не разрыв терминологии);
    - одна основа может вести к нескольким работам, если каждая связь проходит пороги.
Пороги подбираются на валидационной части, тестовая для этого не используется.

Без выгрузки АВР словарь не строится: build_from_export возвращает статус blocked_missing_data.
Придумывать словарь вручную или строить его по нескольким десяткам разобранных заявок нельзя:
это подгонка под тестовые примеры.
"""

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Поля выгрузки АВР, без которых словарь не построить.
REQUIRED_EXPORT_FIELDS = (
    "ID смарт-процесса по АВР (или другой ключ связи с задачей Битрикс24)",
    "текст заявки (или ID задачи, по которому текст берётся из выгрузки задач)",
    "коды ПЗР фактически выполненных работ (итоговая калькуляция)",
    "количество по каждой работе",
    "дата АВР (для разделения по времени)",
)
# Окончания, которые срезаются при получении основы слова (грубый стеммер без внешних библиотек).
_ENDINGS = sorted(
    "ами ями ого его ому ему ыми ими ой ей ий ый ая яя ое ее ые ие ую юю ом ем ах ях ов ев ам ям "
    "ы и а я о е у ю ь".split(), key=len, reverse=True,
)
_STOP = {"прошу", "просьба", "подъезд", "подъезде", "этаж", "этаже", "квартира", "квартиры", "дом", "доме",
         "очень", "который", "которая", "также", "заявитель", "заявителя", "слов"}


def stems(text: str) -> set[str]:
    """Основы содержательных слов текста (от 4 букв, без служебных).

    Принимает: text — произвольный текст.
    Возвращает: множество основ («капает» -> «капа», «фитингов» -> «фитинг»).
    """
    result = set()
    for word in re.findall(r"[а-яё]{4,}", (text or "").lower().replace("ё", "е")):
        if word in _STOP:
            continue
        for ending in _ENDINGS:
            if word.endswith(ending) and len(word) - len(ending) >= 4:
                word = word[: -len(ending)]
                break
        result.add(word)
    return result


@dataclass
class TermDictionary:
    """Построенный словарь: основа слова заявки -> работы ПЗР со статистикой."""

    entries: dict[str, list[dict]]
    meta: dict = field(default_factory=dict)

    def expansions(self, text: str, work_names: dict[str, str], limit: int = 2) -> list[str]:
        """Дополнительные поисковые запросы для текста заявки.

        Берутся работы по всем основам текста, у которых есть связи в словаре; при совпадении
        работы у нескольких основ — лучшая уверенность. Запрос — наименование работы из ПЗР.

        Принимает:
            text — текст заявки (только он: коды мастера в поиск не передаются);
            work_names — {код работы: наименование} справочника ПЗР;
            limit — максимум запросов (чтобы не раздувать поиск).
        Возвращает: список запросов, по убыванию уверенности.
        """
        best: dict[str, float] = {}
        for stem in stems(text):
            for link in self.entries.get(stem, []):
                best[link["work_id"]] = max(best.get(link["work_id"], 0.0), link["confidence"])
        ranked = sorted(best, key=lambda code: (-best[code], code))
        return [work_names[code] for code in ranked if code in work_names][:limit]

    def save(self, path: Path) -> None:
        """Сохраняет словарь в JSON (версионируемый артефакт с параметрами и контрольной суммой)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"meta": self.meta, "entries": self.entries}, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TermDictionary":
        """Загружает словарь, сохранённый save()."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(entries=data["entries"], meta=data["meta"])


def build(
    train: list[dict],
    work_texts: dict[str, str],
    min_support: int = 5,
    min_confidence: float = 0.3,
) -> TermDictionary:
    """Строит словарь по обучающей части истории АВР.

    Принимает:
        train — заявки обучающей части: {"request_id", "text", "work_ids": [коды мастера]};
            одна заявка — один голос за каждую пару (основа, работа), сколько бы раз слово ни повторялось;
        work_texts — {код работы: наименование и состав из ПЗР} — чтобы отбросить слова, которые
            уже есть в тексте работы;
        min_support — минимум независимых заявок для связи;
        min_confidence — минимальная доля заявок с основой, где была эта работа.
    Возвращает: TermDictionary с meta (параметры, дата, контрольная сумма набора).
    """
    with_stem: Counter = Counter()
    pair: Counter = Counter()
    for request in train:
        request_stems = stems(request["text"])
        with_stem.update(request_stems)
        for stem in request_stems:
            for work_id in set(request["work_ids"]):
                pair[(stem, work_id)] += 1
    work_stems = {code: stems(text) for code, text in work_texts.items()}
    entries: dict[str, list[dict]] = defaultdict(list)
    for (stem, work_id), support in pair.items():
        confidence = support / with_stem[stem]
        if support < min_support or confidence < min_confidence:
            continue
        if stem in work_stems.get(work_id, set()):
            continue  # слово уже есть в тексте работы — поиск находит её и так
        entries[stem].append({"work_id": work_id, "support": support, "confidence": round(confidence, 3)})
    for links in entries.values():
        links.sort(key=lambda link: (-link["confidence"], link["work_id"]))
    ids = sorted(str(request["request_id"]) for request in train)
    meta = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "train_requests": len(train),
        "train_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "min_support": min_support,
        "min_confidence": min_confidence,
        "entries": sum(len(links) for links in entries.values()),
    }
    return TermDictionary(entries=dict(sorted(entries.items())), meta=meta)


def build_from_export(export_path: Path | None) -> dict:
    """Проверяет наличие выгрузки АВР для построения словаря.

    Принимает: export_path — путь к выгрузке АВР или None.
    Возвращает: {"status": "blocked_missing_data", "required_fields": […]} — если выгрузки нет;
        чтение выгрузки будет добавлено, когда станет известна её структура.
    """
    if export_path is None or not export_path.exists():
        return {"status": "blocked_missing_data", "required_fields": list(REQUIRED_EXPORT_FIELDS)}
    return {"status": "export_format_unknown", "path": str(export_path),
            "required_fields": list(REQUIRED_EXPORT_FIELDS)}
