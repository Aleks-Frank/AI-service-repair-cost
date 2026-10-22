"""Детерминированная проверка выбора модели и расчёт сметы (без LLM, только правила и Decimal).

Здесь модель уже ничего не решает. Код проверяет каждый её выбор:
  * код расценки есть среди кандидатов RAG для этого дефекта — иначе отклоняем;
  * объём подтверждён: цитатой из описания или подсчётом предметов на фото;
  * единица объёма совпадает с единицей расценки;
  * «смена» не дублируется отдельными «демонтажем» и «установкой» того же узла.
Стоимость = объём × цена из БД. Если объём не подтверждён, сумма остаётся пустой (а не нулевой).
Раздел «Материалы» добавляется функцией attach_materials (строки готовит llm/estimator/materials.py).
"""

import csv
import io
import re
from decimal import ROUND_HALF_UP, Decimal

from rag.normalizers import normalize_unit
from llm.estimator.schemas import Observations, Selection, WorkChoice

# Единицы, которые разрешено «считать по фото» — отдельные различимые предметы.
COUNTABLE_UNITS = {"шт", "лампа", "прибор", "счетчик", "узел"}


def normalized_text(text: str) -> str:
    """Приводит текст к виду для сравнения: нижний регистр, «ё» -> «е», одиночные пробелы.

    Принимает: text — исходный текст.
    Возвращает: нормализованную строку.
    """
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


def money(value: Decimal | str) -> str:
    """Округляет сумму до копеек по правилам бухгалтерии (половина — вверх).

    Принимает: value — число.
    Возвращает: строку с двумя знаками после точки.
    """
    return str(Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def checked_quantity(choice: WorkChoice, item: dict, description: str, image_count: int) -> tuple[Decimal | None, str | None]:
    """Проверяет объём, предложенный моделью; неподтверждённый объём не принимается.

    Принимает:
        choice — выбор модели (объём, единица, источник, цитата);
        item — кандидат-расценка (нужна единица измерения);
        description — исходное описание заявки;
        image_count — сколько фото в заявке.
    Возвращает: кортеж (объём или None, пояснение, почему объём не принят, или None).
    """
    if choice.quantity is None or choice.quantity_source == "unknown":
        return None, "Объём не определён по исходным данным."
    quantity = Decimal(str(choice.quantity))
    # Единицы сравниваем после той же нормализации, что и в справочнике («мп» = «м» и т.п.).
    if normalize_unit(choice.quantity_unit) != item["unit"]:
        return None, "Единица объёма не совпадает со справочником; пересчёт не выполнен."

    if choice.quantity_source == "description":
        # Цитата должна дословно встречаться в описании, а число — в цитате.
        # Пунктуацию и кавычки по краям убираем: модель часто цитирует с точкой на конце.
        quote = choice.quantity_evidence.strip().strip(" .,;:!?\"'«»")
        if not quote or normalized_text(quote) not in normalized_text(description):
            return None, "Для объёма нет точной цитаты в исходном описании."
        numbers = [Decimal(s.replace(",", ".")) for s in re.findall(r"(?<!\w)\d+(?:[.,]\d+)?", quote)]
        if quantity not in numbers:
            return None, "Числовой объём не найден в процитированном описании."
    elif choice.quantity_source == "visible_count":
        # По фото без масштаба можно только посчитать предметы, но не измерить метры.
        if item["unit"] not in COUNTABLE_UNITS:
            return None, "По фото разрешён только счёт отдельных предметов; размеры не оцениваются."
        if quantity != quantity.to_integral_value():
            return None, "Видимое количество предметов должно быть целым."
        if not choice.quantity_evidence.strip() or not choice.image_numbers or any(
            n < 1 or n > image_count for n in choice.image_numbers
        ):
            return None, "Нет корректной ссылки на фото для подсчёта предметов."
    return quantity, None


def _action(name: str) -> str:
    """Определяет тип операции по началу наименования расценки.

    Принимает: name — наименование работы.
    Возвращает: "replacement" (смена/замена), "removal" (демонтаж/снятие/разборка),
        "installation" (установка/монтаж) или "other".
    """
    name = normalized_text(name)
    if re.match(r"смена\b|замена\b", name):
        return "replacement"
    if re.match(r"демонтаж\b|снятие\b|разборка\b", name):
        return "removal"
    if re.match(r"установка\b|монтаж\b", name):
        return "installation"
    return "other"


def make_estimate(
    selection: Selection, observations: Observations, candidates: list[dict], description: str, image_count: int
) -> dict:
    """Проверяет выбор модели и собирает черновик сметы.

    Принимает:
        selection — выбор работ моделью (этап 2);
        observations — найденные дефекты (этап 1);
        candidates — кандидаты из RAG (только из них можно выбирать);
        description — описание заявки; image_count — число фото.
    Возвращает: словарь сметы: основные работы, альтернативы, суммы, предупреждения.
    """
    allowed = {item["work_id"]: item for item in candidates}
    known_findings = {finding.finding_id for finding in observations.findings}
    warnings: list[str] = []
    main: list[dict] = []
    alternatives: list[dict] = []
    rejected: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for bucket, choices in (("main", selection.works), ("alternative", selection.alternatives)):
        target = main if bucket == "main" else alternatives
        for choice in choices:
            item = allowed.get(choice.work_id)
            # Защита от «выдуманных» кодов: расценка должна быть кандидатом именно для этого дефекта.
            if item is None or choice.finding_id not in known_findings or choice.finding_id not in item["finding_ids"]:
                rejected.append(choice.model_dump())
                warnings.append(f"Позиция {choice.work_id} отклонена: нет в кандидатах для указанного дефекта.")
                continue
            key = (choice.finding_id, choice.work_id)
            if key in seen:
                warnings.append(f"Повтор {choice.work_id} для {choice.finding_id} исключён.")
                continue
            seen.add(key)

            quantity, note = checked_quantity(choice, item, description, image_count)
            if quantity is not None and choice.quantity_source == "visible_count":
                note = "Объём по видимому количеству на фото; фактическое выполнение не подтверждено."
            if item["data_issues"]:
                warnings.append(f"{item['work_id']}: замечания к данным справочника — {'; '.join(item['data_issues'])}")
            target.append({
                **{k: item[k] for k in ("work_id", "name", "composition", "unit", "unit_price_rub",
                                        "norm_reference", "source_sheet", "source_row")},
                "finding_id": choice.finding_id,
                "reason": choice.reason,
                "quantity": str(quantity) if quantity is not None else None,
                "quantity_source": choice.quantity_source if quantity is not None else "unknown",
                "quantity_evidence": choice.quantity_evidence,
                "image_numbers": choice.image_numbers,
                # Стоимость = объём × цена из БД; цена от модели никогда не берётся.
                "amount_rub": money(quantity * Decimal(item["unit_price_rub"])) if quantity is not None else None,
                "note": note,
            })

    # Модель иногда кладёт единственный подходящий вариант в альтернативы, а основной оставляет
    # пустым. Тогда первая альтернатива этого дефекта становится основной — с пометкой для сметчика.
    for finding_id in known_findings - {line["finding_id"] for line in main}:
        promoted = next((line for line in alternatives if line["finding_id"] == finding_id), None)
        if promoted:
            alternatives.remove(promoted)
            promoted["note"] = "Модель предложила как альтернативу; основной вариант не был выбран — проверьте."
            main.append(promoted)
            warnings.append(f"{promoted['work_id']} перенесена из альтернатив в основной вариант для {finding_id}.")

    # «Смена» обычно уже включает демонтаж и установку. Если для того же дефекта выбраны
    # и смена, и отдельные демонтаж/установка — последние выносим в альтернативы.
    replacements = {line["finding_id"]: line for line in main if _action(line["name"]) == "replacement"}
    filtered = []
    for line in main:
        replacement = replacements.get(line["finding_id"])
        kind = _action(line["name"])
        composition = normalized_text(replacement["composition"]) if replacement else ""
        overlap = (kind == "removal" and re.search(r"сняти|демонтаж|отсоедин|разбор", composition)) or (
            kind == "installation" and re.search(r"установ|монтаж|присоедин", composition)
        )
        if replacement and overlap:
            line["note"] = f"Возможное пересечение с {replacement['work_id']}; исключено из основной суммы."
            alternatives.append(line)
            warnings.append(f"{line['work_id']} вынесена в альтернативы: проверьте состав {replacement['work_id']}.")
        else:
            filtered.append(line)
    main = filtered

    # Неполная замена: для дефекта выбран только демонтаж, без установки и без «смены».
    # Маленькая модель так ошибается часто (снимает старое, но «забывает» поставить новое).
    # Работы не добавляем сами — подсказываем сметчику подходящие коды из кандидатов.
    for finding_id in known_findings:
        lines = [line for line in main if line["finding_id"] == finding_id]
        kinds = {_action(line["name"]) for line in lines}
        if "removal" in kinds and not kinds & {"installation", "replacement"}:
            # Предлагаем только работы с тем же объектом: общий «корень» (первые 5 букв)
            # у слов наименования — «Демонтаж раковин» -> «Смена раковин», но не «Установка дверей».
            stems = {word[:5] for line in lines if _action(line["name"]) == "removal"
                     for word in re.findall(r"[а-я]{5,}", normalized_text(line["name"]))[1:]}
            options = [f"{item['work_id']} «{item['name']}»" for item in candidates
                       if finding_id in item["finding_ids"]
                       and _action(item["name"]) in {"installation", "replacement"}
                       and stems & {word[:5] for word in re.findall(r"[а-я]{5,}", normalized_text(item["name"]))}]
            if options:
                warnings.append(f"Для {finding_id} выбран только демонтаж. Если требуется замена, добавьте: {', '.join(options)}.")

    priced =[line for line in main if line["amount_rub"] is not None]
    subtotal = money(sum((Decimal(line["amount_rub"]) for line in priced), Decimal("0"))) if priced else None
    uncovered = known_findings - {line["finding_id"] for line in main}
    for finding_id in sorted(uncovered):
        warnings.append(f"Для {finding_id} основной вариант работ не выбран; полный расчёт отсутствует.")
    complete = bool(main) and len(priced) == len(main) and not selection.unmatched_works and not rejected and not uncovered

    return {
        "status": "draft",
        "summary": observations.summary,
        "image_relation": observations.image_relation,
        "catalog": "pzr_pricing.rates",
        "works": main,
        "alternatives": alternatives,
        "unmatched_works": selection.unmatched_works,
        "unknowns": observations.unknowns,
        "assumptions": selection.assumptions,
        "warnings": list(dict.fromkeys(warnings)),  # без повторов, с сохранением порядка
        "rejected_choices": rejected,
        "priced_lines": len(priced),
        "subtotal_priced_lines_rub": subtotal,
        "selected_works_total_rub": subtotal if complete else None,
        "calculation_complete_for_selected_works": complete,
        # Раздел «Материалы» заполняет attach_materials; без него смета — только работы.
        "materials_priced": False,
        "materials": [],
        "materials_not_found": [],
        "works_total_rub": subtotal,
        "materials_total_rub": None,
        "total_rub": subtotal,
    }


def attach_materials(result: dict, lines: list[dict], not_found: list[dict]) -> dict:
    """Добавляет в смету раздел «Материалы», итоги и предупреждения по материалам.

    Итог считается только по строкам, где известны и количество, и цена; строки без суммы
    видны в смете, но в итог не входят (как и работы без подтверждённого объёма).

    Принимает:
        result — смета из make_estimate (изменяется на месте);
        lines — строки материалов из materials.price_materials;
        not_found — потребности, для которых товар не подобран.
    Возвращает: ту же смету.
    """
    priced = [line for line in lines if line["amount_rub"] is not None]
    materials_total = money(sum((Decimal(line["amount_rub"]) for line in priced), Decimal("0"))) if priced else None
    works_total = result["works_total_rub"]
    parts = [Decimal(value) for value in (works_total, materials_total) if value is not None]
    warnings = list(result["warnings"])
    for line in lines:
        warnings += [f"Материал «{line['need']}» ({line['work_id']}): {text}." for text in line["warnings"]]
    for item in not_found:
        warnings.append(f"Материал «{item['query']}» ({item['work_id']}) не подобран: {item['reason']}.")
    result.update(
        materials_priced=True,
        materials=lines,
        materials_not_found=not_found,
        materials_total_rub=materials_total,
        total_rub=money(sum(parts, Decimal("0"))) if parts else None,
        warnings=list(dict.fromkeys(warnings)),
    )
    return result


# --- Экспорт -------------------------------------------------------------------------


def estimate_csv(result: dict) -> str:
    """Формирует CSV для Excel (разделитель «;», BOM для корректной кириллицы).

    Принимает: result — смета из make_estimate.
    Возвращает: текст CSV.
    """
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter=";")
    writer.writerow(["Вариант", "Код", "Работа", "Ед.", "Объём", "Цена, руб.", "Сумма, руб.", "Основание", "Примечание", "Лист", "Строка"])

    def safe(value: object) -> str:
        """Защита от «формул» в ячейках Excel: значения, начинающиеся с = + - @, экранируются."""
        text = "" if value is None else str(value)
        return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text

    for label, lines in (("Основной", result["works"]), ("Альтернатива", result["alternatives"])):
        for line in lines:
            writer.writerow([safe(v) for v in (label, line["work_id"], line["name"], line["unit"], line["quantity"],
                                               line["unit_price_rub"], line["amount_rub"], line["reason"], line["note"],
                                               line["source_sheet"], line["source_row"])])
    writer.writerow([])
    writer.writerow(["По рассчитанным основным позициям, руб.", result["subtotal_priced_lines_rub"] or "Не рассчитано"])
    writer.writerow(["Полный расчёт выбранных работ", "Да" if result["calculation_complete_for_selected_works"] else "Нет"])
    if result["materials"] or result["materials_not_found"]:
        writer.writerow([])
        writer.writerow(["Материалы", "Работа", "Наименование", "Артикул", "Поставщик", "Нужно", "Ед.",
                         "Купить", "Ед. цены", "Цена, руб.", "Сумма, руб.", "Дата цены", "Потребность"])
        for line in result["materials"]:
            writer.writerow([safe(v) for v in ("", line["work_id"], line["name"], line["article"], line["supplier"],
                                               line["need_qty"], line["base_unit"], line["buy_qty"], line["price_unit"],
                                               line["price_rub"], line["amount_rub"], line["price_date"], line["need"])])
        for item in result["materials_not_found"]:
            writer.writerow([safe(v) for v in ("Не подобран", item["work_id"], item["query"], "", "", "", item["unit"],
                                               "", "", "", "", "", item["reason"])])
        writer.writerow(["Материалы итого, руб.", result["materials_total_rub"] or "Не рассчитано"])
        writer.writerow(["Всего (работы + материалы), руб.", result["total_rub"] or "Не рассчитано"])
    return "﻿" + output.getvalue()
