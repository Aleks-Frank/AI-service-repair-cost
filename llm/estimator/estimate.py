"""Детерминированная проверка выбора модели и расчёт сметы (без LLM, только правила и Decimal).

Здесь модель уже ничего не решает. Код проверяет каждый её выбор:
  * код расценки есть среди кандидатов RAG для этого дефекта — иначе отклоняем;
  * объём подтверждён: цитатой из описания или подсчётом предметов на фото;
  * единица объёма совпадает с единицей расценки;
  * «смена» не дублируется отдельными «демонтажем» и «установкой» того же узла.
Стоимость = объём × цена из БД. Если объём не подтверждён, сумма остаётся пустой (а не нулевой).
"""

import csv
import html
import io
import re
from decimal import ROUND_HALF_UP, Decimal

from rag.normalizers import normalize_unit
from estimator.schemas import Observations, Selection, WorkChoice

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
    if normalize_unit(choice.quantity_unit)[0] != item["unit"]:
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
    }


# --- Экспорт -------------------------------------------------------------------------


def _ru_number(value: str | None) -> str:
    """Форматирует сумму по-русски: «1 312,50»; пустое значение — «—».

    Принимает: value — число строкой или None.
    Возвращает: отформатированную строку.
    """
    if value is None:
        return "—"
    return f"{Decimal(value):,.2f}".replace(",", " ").replace(".", ",")


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
    return "﻿" + output.getvalue()


def estimate_html(result: dict) -> str:
    """Формирует HTML-страницу сметы (открывается в браузере, печатается в PDF).

    Принимает: result — смета из make_estimate.
    Возвращает: HTML-документ строкой.
    """
    esc = lambda value: html.escape(str(value if value is not None else "—"))  # noqa: E731

    def table(lines: list[dict]) -> str:
        """Таблица работ: наименование с основанием, единица, объём, цена, сумма."""
        rows = []
        for line in lines:
            source = f"{line['source_sheet'].strip()}, строка {line['source_row']}"
            qty = esc(line["quantity"]) if line["quantity"] is not None else '<span class="missing">Не определён</span>'
            details = f"<details><summary>Основание и состав</summary><p>{esc(line['reason'])}</p><p>{esc(line['composition'])}</p>"
            if line["quantity_evidence"]:
                details += f"<p>Объём: {esc(line['quantity_evidence'])}</p>"
            details += "</details>"
            note = f'<p class="note">{esc(line["note"])}</p>' if line["note"] else ""
            rows.append(
                f"<tr><td><strong>{esc(line['name'])}</strong><small>{esc(line['work_id'])} · {esc(source)}</small>{details}{note}</td>"
                f"<td>{esc(line['unit'])}</td><td>{qty}</td><td>{_ru_number(line['unit_price_rub'])}</td><td>{_ru_number(line['amount_rub'])}</td></tr>"
            )
        return ('<div class="scroll"><table><thead><tr><th>Работа</th><th>Ед.</th><th>Объём</th><th>Цена, ₽</th>'
                "<th>Сумма, ₽</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")

    main = table(result["works"]) if result["works"] else "<p>Подходящие позиции не выбраны. Автоматический расчёт не выполнен.</p>"
    total_label = "По выбранным работам" if result["calculation_complete_for_selected_works"] else "По позициям с известным объёмом"
    notices = "".join(f"<li>{esc(t)}</li>" for t in result["unknowns"] + result["assumptions"] + result["warnings"])
    alt = "<h2>Альтернативы — не входят в сумму</h2>" + table(result["alternatives"]) if result["alternatives"] else ""
    unmatched = ("<h2>Без расценки</h2><ul>" + "".join(f"<li>{esc(t)}</li>" for t in result["unmatched_works"]) + "</ul>"
                 if result["unmatched_works"] else "")
    return f"""<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Проект сметы ПЗР</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;color:#182b43;background:#edf2f8;margin:0;padding:28px}}main{{max-width:1120px;margin:auto;background:white;padding:32px;border-radius:16px}}h1{{font-size:30px;margin:6px 0 16px}}h2{{font-size:20px;margin-top:30px}}.eyebrow{{color:#155bc8;font-weight:700;font-size:14px}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th{{text-align:left;background:#eaf0f8;padding:12px}}td{{padding:14px 12px;vertical-align:top;border-bottom:1px solid #dce4ee}}td:first-child{{min-width:260px}}td:nth-child(n+3){{text-align:right;white-space:nowrap}}small{{display:block;color:#62748c;margin-top:5px}}details{{font-size:13px;margin-top:8px}}summary{{cursor:pointer;color:#155bc8}}.missing,.note{{color:#8c5c16}}.note{{font-size:13px}}.total{{padding:20px;background:#edf4ff;border-left:4px solid #155bc8;margin:24px 0;font-size:20px}}.muted{{color:#62748c}}@media print{{body{{background:white;padding:0}}main{{padding:0}}details{{display:none}}}}</style>
<main><div class="eyebrow">ПЗР · АВТОМАТИЧЕСКИЙ ЧЕРНОВИК</div><h1>Проект сметы</h1><p>{esc(result['summary'])}</p>
{main}<div class="total">{total_label}: <strong>{_ru_number(result['subtotal_priced_lines_rub'])} ₽</strong></div>
<p class="muted">Материалы не рассчитаны. Фото не подтверждают фактическое выполнение всех операций.</p>
{alt}{unmatched}{'<h2>Неопределённости и допущения</h2><ul>' + notices + '</ul>' if notices else ''}
<p class="muted">Источник расценок: справочник ПЗР в базе pzr_pricing.</p></main></html>"""
