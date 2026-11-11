"""Детерминированная проверка совместимости материалов и работ (этап B2.1).

Место в конвейере: после подбора товаров по прайсам (materials.price_materials) и до итогов сметы
(estimate.attach_materials). Валидатор меняет ТОЛЬКО список материалов: работы, их коды и объёмы,
трудозатраты, техника и цены не трогаются.

Как принимается решение:
    вид работы    — по разделу ПЗР (section_title), а у «Ремонтных работ» 1-282 — по тексту дефекта;
    вид материала — по наименованию и категории товара в прайсе (описания у позиций в базе нет);
    решение       — по таблице сочетаний COMPATIBILITY «вид работы → вид материала».

Три исхода:
    ALLOW  — материал совместим с работой: остаётся в смете;
    REJECT — явно несовместим (фасадная краска к замене лампы): исключается, пишется предупреждение;
    REVIEW — совместимость не установлена (вид работы или материала не определён, сочетание
             не описано таблицей): материал НЕ удаляется, но и не считается подтверждённым —
             считается отдельно и даёт вторую, предварительную сумму. Человеку ничего не отправляется,
             обработка заявки не останавливается; статус машинно-читаемый для анализа качества.

Незнакомый материал не запрещается (не REJECT), а уходит в REVIEW.
"""

import re
from decimal import Decimal

# --- Виды работ по разделу ПЗР ----------------------------------------------------------------
# Проверяются по порядку: первый совпавший шаблон раздела даёт вид работы.
WORK_DOMAINS_BY_SECTION: list[tuple[str, str]] = [
    (r"электромонтаж", "electrical"),
    (r"санитарно-техн|инженерных систем|гвс|хвс|отоплен|водоотвед", "plumbing"),
    (r"штукатур|маляр|облицов", "finishing"),
    (r"проем|окна|двер", "openings"),
    (r"фундамент|стены|перегородк|полы|кровл|лестниц|крыльц", "construction"),
    (r"благоустр", "landscaping"),
]

# --- Виды материалов по наименованию и категории товара -----------------------------------------
# Порядок важен: «хомут для труб» — сантехника, просто «хомут»/«стяжка» — общий крепёж.
MATERIAL_CLASSES: list[tuple[str, str]] = [
    # Электрика и освещение.
    ("electrical", r"ламп|светильник|прожектор|светодиод|\bled\b|драйвер|блок питания|кабел|\bпровод"
                   r"|розетк|выключател|автомат|узо\b|клемм|патрон|датчик|стартер|дроссель|изолент"
                   r"|коробк.*(распаеч|распред|монтаж)|распаечн|электро(?!д)|щит"),
    # Сантехника и инженерные системы.
    ("plumbing", r"сантехн|труб|муфт|фитинг|кран|вентил|фильтр|смесител|сифон|унитаз|раковин"
                 r"|радиатор|отвод|тройник|сгон|прокладк|л[её]н\b|паста уплотн|фум|задвижк|манометр"
                 r"|водосчетчик|счетчик воды|полипропилен|металлопласт"),
    # Отделка: смеси, краски, грунтовки, плитка.
    ("finishing", r"краск|грунт|шпакл|штукатур|смесь|эмаль|\bлак\b|затирк|плитк|плиточн|обои"
                  r"|колер|растворител|очиститель|лакокрас|отделк|сухие смеси|раствор"),
    # Проёмы: двери, окна и их фурнитура.
    ("openings", r"замок|замк|петл|доводчик|ручк|двер|окон|окн|стекл|фурнитур|уплотнител"),
    # Общестроительные материалы.
    ("construction", r"кирпич|бетон|цемент|песок|щебен|доск|брус|фанер|гипсокартон|профил|утеплит"
                     r"|рубероид|кровел|блок"),
    # Общий крепёж и универсальные расходники — допустимы для любой работы.
    ("general", r"крепеж|крепёж|дюбел|саморез|болт|гайк|шуруп|анкер|хомут|стяжк|герметик|пена монтаж"
                r"|скотч|лента клейк|заглушк|шайб|винт"),
]

# --- Таблица сочетаний «вид работы → вид материала» -------------------------------------------
# allow — совместимо; reject — явно несовместимо; всё остальное — REVIEW.
COMPATIBILITY: dict[str, dict[str, set[str]]] = {
    "electrical": {"allow": {"electrical", "general"}, "reject": {"finishing", "plumbing", "openings"}},
    "plumbing": {"allow": {"plumbing", "general"}, "reject": {"electrical", "finishing", "openings"}},
    "finishing": {"allow": {"finishing", "general"}, "reject": {"electrical", "plumbing"}},
    "openings": {"allow": {"openings", "construction", "finishing", "general"}, "reject": {"electrical", "plumbing"}},
    "construction": {"allow": {"construction", "finishing", "general"}, "reject": {"electrical", "plumbing"}},
    "landscaping": {"allow": {"construction", "finishing", "general"}, "reject": set()},
}

UNIVERSAL_WORK_ID = "1-282"
ALLOW, REJECT, REVIEW = "ALLOW", "REJECT", "REVIEW"


def _norm(text: str) -> str:
    """Нижний регистр и «ё» → «е» для сравнения шаблонов."""
    return (text or "").lower().replace("ё", "е")


def material_class(name: str, category: str = "") -> str | None:
    """Вид материала по наименованию товара, а если не распознан — по категории прайса.

    Наименование важнее категории: «Хомут для труб» в категории «Инженерная сантехника» —
    сантехника и по названию, а «Грунтовка» без категории — отделка по названию.

    Принимает: name — наименование товара; category — путь групп прайса (может быть пустым).
    Возвращает: вид («electrical», «plumbing», …) или None, если не распознан.
    """
    for text in (_norm(name), _norm(category)):
        if not text:
            continue
        # «Хомут для труб» — сантехника, хотя слово «хомут» есть и в общем крепеже.
        for cls, pattern in MATERIAL_CLASSES:
            if re.search(pattern, text):
                return cls
    return None


def work_domain(work_id: str, section: str, finding_text: str = "") -> str | None:
    """Вид работы по разделу ПЗР; для «Ремонтных работ» 1-282 — по тексту дефекта.

    У 1-282 раздел «Прочие работы» ничего не говорит о виде ремонта, поэтому вид берётся
    из объекта и операции дефекта («фонарь не горит» — электрика, «течь трубы» — сантехника).

    Принимает: work_id — код ПЗР; section — раздел ПЗР; finding_text — объект, симптом, операция дефекта.
    Возвращает: вид работы или None, если не определён.
    """
    if work_id != UNIVERSAL_WORK_ID:
        text = _norm(section)
        for pattern, domain in WORK_DOMAINS_BY_SECTION:
            if re.search(pattern, text):
                return domain
    cls = material_class(_finding_hints(finding_text)) if finding_text else None
    return cls if cls in COMPATIBILITY else None


def _finding_hints(text: str) -> str:
    """Добавляет к тексту дефекта синонимы, которых нет в словаре материалов (фонарь, течь, ГВС)."""
    text = _norm(text)
    extra = []
    if re.search(r"фонар|освещен|не горит|искрит|электр", text):
        extra.append("электро")
    if re.search(r"теч|протека|подтек|гвс|хвс|отоплен|канализ|вод[аыу]", text):
        extra.append("сантехн")
    return " ".join([text, *extra])


def judge(work: dict, material_name: str, category: str = "") -> dict:
    """Решение по одной паре «работа — материал».

    Принимает:
        work — {"work_id", "name", "section", "composition", "finding_text"};
        material_name, category — наименование и категория товара.
    Возвращает: {"decision", "rule_id", "reason", "work_domain", "material_class"}.
    """
    domain = work_domain(work["work_id"], work.get("section", ""), work.get("finding_text", ""))
    cls = material_class(material_name, category)
    info = {"work_domain": domain, "material_class": cls}
    if cls is None:
        return {"decision": REVIEW, "rule_id": "MV-UNKNOWN-MATERIAL",
                "reason": "вид материала по наименованию и категории не определён", **info}
    if cls == "general":
        return {"decision": ALLOW, "rule_id": "MV-GENERAL",
                "reason": "общий крепёж или универсальный расходник", **info}
    # Материал прямо назван в наименовании или составе работы («с установкой стекол»).
    # Сравнивается основа первого слова наименования («Стекло» — «стек», «Заглушка» — «заглуш»).
    work_text = _norm(f"{work.get('name', '')} {work.get('composition', '')}")
    first = re.findall(r"[а-я]+", _norm(material_name))[:1]
    if first and len(first[0]) >= 5 and first[0][:max(4, len(first[0]) - 2)] in work_text:
        return {"decision": ALLOW, "rule_id": "MV-NAMED-IN-WORK",
                "reason": "материал назван в наименовании или составе работы", **info}
    if domain is None:
        return {"decision": REVIEW, "rule_id": "MV-UNKNOWN-WORK",
                "reason": "вид работы не определён", **info}
    rules = COMPATIBILITY[domain]
    if cls in rules["allow"]:
        return {"decision": ALLOW, "rule_id": f"MV-ALLOW-{domain}-{cls}",
                "reason": f"материал вида «{cls}» допустим для работ вида «{domain}»", **info}
    if cls in rules["reject"]:
        return {"decision": REJECT, "rule_id": f"MV-REJECT-{domain}-{cls}",
                "reason": f"материал вида «{cls}» несовместим с работой вида «{domain}»", **info}
    return {"decision": REVIEW, "rule_id": f"MV-REVIEW-{domain}-{cls}",
            "reason": f"сочетание «{domain}» — «{cls}» таблицей не описано", **info}


def validate_lines(lines: list[dict], works: dict[tuple[str, str], dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Делит строки материалов на подтверждённые, на проверку и исключённые.

    Строка-покупка может делиться между несколькими работами (поле shared): она подтверждена,
    если совместима хотя бы с одной из них; исключается, если явно несовместима со всеми.

    Принимает:
        lines — строки материалов из price_materials (с полем category);
        works — {(finding_id, work_id): {"work_id", "name", "section", "composition", "finding_text"}}.
    Возвращает: (allowed, review, rejected, log) — log: решение по каждой строке исходного списка.
    """
    allowed, review, rejected, log = [], [], [], []
    for line in lines:
        members = line.get("shared") or [{"finding_id": line["finding_id"], "work_id": line["work_id"]}]
        verdicts = []
        for member in members:
            work = works.get((member["finding_id"], member["work_id"]),
                             {"work_id": member["work_id"], "name": "", "section": ""})
            verdicts.append({"work_id": member["work_id"], **judge(work, line["name"], line.get("category", ""))})
        decisions = {v["decision"] for v in verdicts}
        decision = ALLOW if ALLOW in decisions else (REJECT if decisions == {REJECT} else REVIEW)
        chosen = next(v for v in verdicts if v["decision"] == decision)
        entry = {"need_id": line["need_id"], "need": line["need"], "name": line["name"],
                 "category": line.get("category", ""), "work_id": line["work_id"],
                 "amount_rub": line["amount_rub"], "decision": decision, "rule_id": chosen["rule_id"],
                 "reason": chosen["reason"], "work_domain": chosen["work_domain"],
                 "material_class": chosen["material_class"], "per_work": verdicts}
        log.append(entry)
        line["validation"] = {k: entry[k] for k in ("decision", "rule_id", "reason")}
        {ALLOW: allowed, REVIEW: review, REJECT: rejected}[decision].append(line)
    return allowed, review, rejected, log


def lines_total(lines: list[dict]) -> Decimal | None:
    """Сумма строк с известной суммой; None, если таких строк нет."""
    priced = [Decimal(line["amount_rub"]) for line in lines if line["amount_rub"] is not None]
    return sum(priced, Decimal("0")) if priced else None


def attach_validation(result: dict, original: list[dict], review: list[dict], rejected: list[dict],
                      log: list[dict]) -> dict:
    """Дописывает в смету результат проверки материалов и предварительную сумму с REVIEW.

    Вызывается ПОСЛЕ attach_materials, которому переданы только подтверждённые (ALLOW) строки:
    materials, materials_total_rub и total_rub — это подтверждённая калькуляция.

    Принимает:
        result — смета (изменяется на месте);
        original — исходный список строк материалов (до проверки);
        review, rejected — строки REVIEW и REJECT;
        log — решения validate_lines.
    Возвращает: ту же смету с полями materials_review, materials_rejected, material_validation,
    materials_review_total_rub и total_with_review_rub.
    """
    from llm.estimator.estimate import money

    review_total = lines_total(review)
    confirmed = Decimal(result["total_rub"]) if result.get("total_rub") is not None else None
    parts = [value for value in (confirmed, review_total) if value is not None]
    for entry in log:
        if entry["decision"] == REJECT:
            result["warnings"].append(
                f"Материал «{entry['name']}» ({entry['work_id']}) исключён: {entry['reason']} [{entry['rule_id']}].")
    result.update(
        materials_review=review,
        materials_rejected=rejected,
        materials_review_total_rub=money(review_total) if review_total is not None else None,
        # Предварительная калькуляция: подтверждённая плюс материалы с неустановленной совместимостью.
        total_with_review_rub=money(sum(parts, Decimal("0"))) if parts else None,
        material_validation={
            "original": [{k: line.get(k) for k in ("need_id", "need", "name", "category", "work_id", "amount_rub")}
                         for line in original],
            "decisions": log,
            "counts": {d: sum(e["decision"] == d for e in log) for d in (ALLOW, REVIEW, REJECT)},
        },
    )
    return result
