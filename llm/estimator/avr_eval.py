"""Оценка плановой калькуляции на примерах АВР по регламенту AVR_AI_AGENT_IMPROVEMENT_RUNBOOK.md.

    python -m llm.estimator.avr_eval run --out results/experiments/EXP-20261105-baseline
    python -m llm.estimator.avr_eval report --run results/experiments/EXP-20261105-baseline
    python -m llm.estimator.avr_eval report --run <кандидат> --baseline <baseline> --id EXP-…

run — прогон 10 заявок АВР (заявка + фото, без комментария мастера — основной режим плана) с подбором
материалов. Флаг --diagnostic-comment добавляет записи мастера до и после работ: это верхняя граница
(утечка ответа), её метрики нельзя смешивать с основными. Рядом с результатами сохраняется config.json:
commit, хэш незакоммиченных изменений, хэш промптов, глубина поиска, модель, число товаров у поставщиков.

report — метрики по компонентам (раздел 3 регламента) по полному эталону data/avr_examples/gold_full.json:
    поиск: Recall@6 и MRR@6 по эталонным работам (без 1-282 — она не бывает кандидатом);
    работы: точность, полнота и F1 по кодам ПЗР (множества, микроусреднение), родственные — отдельно;
    количество: точное совпадение и MAE по точно угаданным работам; часы 1-282 — отдельно;
    материалы: точность, полнота и F1 (материал плана засчитывается по шаблону match эталона);
    техника: найдена ли там, где она была у мастера (стоимость техники не рассчитывается);
    суммы: WAPE, смещение (signed bias), доля заявок в ±10 % и ±20 % — по полной сумме, по работам
    и по полной сумме без техники (тарифа нет, поэтому её стоимость в план не попадает).
Пишет metrics.json, per_request.json и report.md (при --baseline — сравнение и изменения по заявкам).
"""

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

GOLD_PATH = Path("data/avr_examples/gold_full.json")
UNIVERSAL = "1-282"
TOP_K = 6
TOLERANCES = (10, 20)


# --- Конфигурация прогона -------------------------------------------------------------------


def _git(*args: str) -> str:
    """Вывод команды git (пустая строка, если git недоступен)."""
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""


def run_config(depth: int, diagnostic_comment: bool) -> dict:
    """Всё, что нужно для воспроизведения прогона (раздел 2.4 регламента).

    Принимает: depth — кандидатов на дефект; diagnostic_comment — режим с комментарием мастера.
    Возвращает: словарь конфигурации.
    """
    from llm.estimator import prompts
    from llm.estimator.config import get_estimator_settings

    settings = get_estimator_settings()
    prompt_text = "".join(getattr(prompts, name) for name in sorted(dir(prompts)) if name.endswith("_PROMPT"))
    # Изменения не закоммичены (коммитит пользователь) — фиксируем хэш диффа и список новых файлов.
    dirty = _git("diff", "HEAD") + _git("status", "--porcelain")
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "HEAD").strip(),
        "working_tree_hash": hashlib.sha256(dirty.encode()).hexdigest()[:16] if dirty else "clean",
        "prompts_hash": hashlib.sha256(prompt_text.encode()).hexdigest()[:16],
        "llm_provider": settings.llm_provider,
        "llm_model": settings.gigachat_model if settings.llm_provider == "gigachat" else "Qwen3-VL-8B (llm_service, do_sample=False)",
        "retrieval_depth": depth,
        "search_by_description": settings.search_by_description,
        "default_linear_lamps_common_areas": settings.default_linear_lamps_common_areas,
        "aerial_platform_tariff_rub": settings.aerial_platform_tariff_rub,
        "material_candidates": settings.material_candidates,
        "mode": "diagnostic_master_comment" if diagnostic_comment else "primary",
        "dataset": str(GOLD_PATH),
    }


def _supplier_counts() -> dict:
    """Число товаров у каждого поставщика в базе прайсов (версия прайсов для конфигурации)."""
    from sqlalchemy import func, select

    from rag.prices_db import Material, get_prices_session_factory

    with get_prices_session_factory()() as session:
        return dict(session.execute(select(Material.supplier_code, func.count()).group_by(Material.supplier_code)).all())


def master_comment(item: dict) -> str:
    """Записи мастера до и после работ — только для диагностического режима (утечка ответа)."""
    parts = []
    for key, label in (("before_note", "До работ"), ("after_note", "После работ")):
        text = (item.get(key) or "").strip()
        if text and text.lower() != "отсутствует":
            parts.append(f"{label}: {text}")
    return "\n".join(parts)


def run(out: Path, diagnostic_comment: bool = False) -> None:
    """Прогоняет 10 заявок АВР полным конвейером и сохраняет сметы, трассировки и config.json.

    Принимает: out — новая папка эксперимента; diagnostic_comment — добавить записи мастера.
    Возвращает: ничего. Уже посчитанные заявки пропускаются (можно перезапустить после сбоя).
    """
    import time

    from llm.estimator.cli import save_trace
    from llm.estimator.config import build_llm_client, get_estimator_settings
    from llm.estimator.depth_experiment import load_requests
    from llm.estimator.pipeline import analyze_case, prepare_image
    from rag.config import get_settings
    from rag.db import get_session_factory
    from rag.ml_client import build_ml_client
    from rag.prices_db import get_prices_session_factory
    from rag.retriever import HybridRetriever, MaterialRetriever

    settings, rag_settings = get_estimator_settings(), get_settings()
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    if not config_path.exists():
        config = run_config(settings.candidates_per_finding, diagnostic_comment)
        config["suppliers"] = _supplier_counts()
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    raw = {f"a{item['n']:02d}": item for item in
           json.loads(Path("data/avr_examples/requests.json").read_text(encoding="utf-8"))}
    llm, ml = build_llm_client(settings), build_ml_client(rag_settings)
    with get_session_factory()() as session, get_prices_session_factory()() as prices:
        retriever, materials = HybridRetriever(session, ml, rag_settings), MaterialRetriever(prices, ml, rag_settings)
        for request in [r for r in load_requests() if r["name"] in raw]:
            run_dir = out / request["name"]
            if (run_dir / "estimate.json").exists():
                continue
            images = [prepare_image(path.read_bytes(), settings.max_image_pixels) for path in request["photos"]]
            comment = master_comment(raw[request["name"]]) if diagnostic_comment else ""
            start = time.time()
            result, trace = analyze_case(
                images, request["description"], llm, retriever, settings.candidates_per_finding,
                settings.observation_max_tokens, settings.selection_max_tokens, materials,
                settings.material_candidates, settings.material_selection_max_tokens,
                settings.search_by_description, inspection=comment,
                linear_lamp_default=settings.default_linear_lamps_common_areas,
                equipment_tariff_rub=settings.aerial_platform_tariff_rub,
                equipment_tariff_unit=settings.aerial_platform_tariff_unit,
            )
            trace.update(elapsed_s=round(time.time() - start, 1), photo_manifest=request["photo_manifest"])
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "estimate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            save_trace(run_dir, trace)
            print(f"{request['name']}: {trace['elapsed_s']} с, итого {result['total_rub']}", flush=True)


# --- Метрики --------------------------------------------------------------------------------


def _number(value) -> float:
    """Сумма или количество из сметы (строка или None) как число; None — 0."""
    return float(value) if value not in (None, "") else 0.0


def _ratio(numerator: float, denominator: float) -> float | None:
    """Деление с None при нулевом знаменателе (без придуманных значений)."""
    return round(numerator / denominator, 3) if denominator else None


def _f1(tp: int, fp: int, fn: int) -> dict:
    """Точность, полнота и F1 по счётчикам."""
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = round(2 * precision * recall / (precision + recall), 3) if precision and recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_request(estimate: dict, trace: dict, gold: dict) -> dict:
    """Сравнивает смету одной заявки с эталоном.

    Принимает: estimate, trace — результат analyze_case; gold — запись gold_full.json.
    Возвращает: словарь показателей заявки (поиск, работы, количество, материалы, техника, суммы).
    """
    lines = estimate["works"]
    codes = [line["work_id"] for line in lines]
    # Поиск: лучшее место эталонной работы среди кандидатов любого дефекта заявки.
    retrieved = [[item["work_id"] for item in state.get("retrieved", []) if item["work_id"] != UNIVERSAL][:TOP_K]
                 for state in trace.get("findings", {}).values()]
    ranks, works, quantities, hours = [], [], [], []
    used: set[int] = set()
    for item in gold["work_items"]:
        if UNIVERSAL not in item["any_of"]:
            places = [found.index(code) + 1 for found in retrieved for code in item["any_of"] if code in found]
            ranks.append(min(places) if places else None)
        exact = [i for i, code in enumerate(codes) if code in item["any_of"] and i not in used]
        related = [i for i, code in enumerate(codes) if code in item["related"] and i not in used]
        outcome = "exact" if exact else "related" if related else "missed"
        if exact:
            used.add(exact[0])
            predicted = _number(lines[exact[0]]["quantity"]) if lines[exact[0]]["quantity"] is not None else None
            target = hours if UNIVERSAL in item["any_of"] else quantities
            target.append({"code": item["any_of"][0], "predicted": predicted, "actual": item["quantity"]})
        elif related:
            used.add(related[0])
        works.append({"gold": item["any_of"][0], "outcome": outcome})
    exact_count = sum(w["outcome"] == "exact" for w in works)
    # Материалы: материал плана засчитывается первому эталонному материалу, чей шаблон совпал.
    plan_materials = [line["name"].lower() for line in estimate.get("materials", [])]
    matched_gold, false_materials = set(), []
    for name in plan_materials:
        hit = next((i for i, m in enumerate(gold["materials"]) if re.search(m["match"], name)), None)
        if hit is None:
            false_materials.append(name)
        else:
            matched_gold.add(hit)
    equipment = " ".join(item.get("name", "") for item in estimate.get("equipment_requirements", [])).lower()
    resources = [{"name": r["name"], "detected": bool(re.search(r["match"], equipment))} for r in gold["additional_resources"]]
    works_total, total = _number(estimate.get("works_total_rub")), _number(estimate.get("total_rub"))
    return {
        "retrieval_ranks": ranks,
        "works": works,
        "work_counts": {"tp": exact_count, "fp": len(codes) - exact_count - sum(w["outcome"] == "related" for w in works),
                        "fn": len(gold["work_items"]) - exact_count,
                        "related": sum(w["outcome"] == "related" for w in works)},
        "predicted_codes": codes,
        "quantities": quantities,
        "hours": hours,
        "materials": {"tp": len(matched_gold), "fp": len(false_materials), "fn": len(gold["materials"]) - len(matched_gold),
                      "false": false_materials,
                      "missed": [m["name"] for i, m in enumerate(gold["materials"]) if i not in matched_gold]},
        "resources": resources,
        "totals": {
            "plan_works": works_total, "actual_works": gold["works_total"],
            "plan_total": total, "actual_total": gold["actual_total"],
            # Без техники: её тарифа нет, план её стоимость не включает — сравнение «как есть».
            "actual_total_without_resources": round(gold["actual_total"] - gold["resources_total"], 2),
        },
    }


def _deviation(plan: float, actual: float) -> float | None:
    """Отклонение плана от факта, % (None при нулевом факте — относительная метрика не считается)."""
    return round((plan - actual) / actual * 100, 1) if actual else None


def aggregate(per_request: dict[str, dict]) -> dict:
    """Сводные метрики по всем заявкам (раздел 3.1 регламента); n — размер знаменателя.

    Принимает: per_request — {заявка: результат evaluate_request}.
    Возвращает: словарь метрик.
    """
    ranks = [rank for result in per_request.values() for rank in result["retrieval_ranks"]]
    work = {key: sum(r["work_counts"][key] for r in per_request.values()) for key in ("tp", "fp", "fn", "related")}
    quantities = [q for r in per_request.values() for q in r["quantities"]]
    hours = [h for r in per_request.values() for h in r["hours"]]
    material = {key: sum(r["materials"][key] for r in per_request.values()) for key in ("tp", "fp", "fn")}
    resources = [res for r in per_request.values() for res in r["resources"]]

    def money(plan_key: str, actual_key: str) -> dict:
        """WAPE, смещение и доля заявок в допусках для пары «план — факт»."""
        pairs = [(r["totals"][plan_key], r["totals"][actual_key]) for r in per_request.values() if r["totals"][actual_key]]
        actual_sum = sum(actual for _, actual in pairs)
        result = {
            "n": len(pairs),
            "plan_sum": round(sum(plan for plan, _ in pairs), 2),
            "actual_sum": round(actual_sum, 2),
            "wape_pct": round(sum(abs(plan - actual) for plan, actual in pairs) / actual_sum * 100, 1) if actual_sum else None,
            "bias_pct": round(sum(plan - actual for plan, actual in pairs) / actual_sum * 100, 1) if actual_sum else None,
        }
        for tolerance in TOLERANCES:
            result[f"within_{tolerance}pct"] = sum(abs(plan / actual - 1) <= tolerance / 100 for plan, actual in pairs)
        return result

    def errors(items: list[dict]) -> dict:
        """Точное совпадение и MAE количества (n — число точно угаданных работ)."""
        known = [i for i in items if i["predicted"] is not None]
        return {"n": len(items), "exact": sum(i["predicted"] == i["actual"] for i in known),
                "mae": round(sum(abs(i["predicted"] - i["actual"]) for i in known) / len(known), 2) if known else None,
                "off_by_more_than_1": sum(abs(i["predicted"] - i["actual"]) > 1 for i in known)}

    found = [rank for rank in ranks if rank is not None and rank <= TOP_K]
    return {
        "requests": len(per_request),
        "retrieval": {"n": len(ranks), "recall_at_6": _ratio(len(found), len(ranks)),
                      "mrr_at_6": round(sum(1 / rank for rank in found) / len(ranks), 3) if ranks else None},
        "works": {**_f1(work["tp"], work["fp"], work["fn"]), "related": work["related"]},
        "quantity": errors(quantities),
        "hours_1_282": errors(hours),
        "materials": _f1(material["tp"], material["fp"], material["fn"]),
        "resources": {"n": len(resources), "detected": sum(r["detected"] for r in resources)},
        "total": money("plan_total", "actual_total"),
        "works_cost": money("plan_works", "actual_works"),
        "total_without_resources": money("plan_total", "actual_total_without_resources"),
    }


def evaluate_run(run_dir: Path, gold_path: Path = GOLD_PATH) -> tuple[dict, dict]:
    """Метрики прогона: (сводные, по заявкам). Отсутствующая заявка в метрики не входит."""
    gold = {key: value for key, value in json.loads(gold_path.read_text(encoding="utf-8")).items() if not key.startswith("_")}
    per_request = {}
    for name, item in gold.items():
        estimate_path = run_dir / name / "estimate.json"
        if not estimate_path.exists():
            continue
        estimate = json.loads(estimate_path.read_text(encoding="utf-8"))
        trace = json.loads((run_dir / name / "trace.json").read_text(encoding="utf-8"))
        per_request[name] = evaluate_request(estimate, trace, item)
    return aggregate(per_request), per_request


# --- Отчёт ----------------------------------------------------------------------------------


_ROWS = (
    ("Recall@6", ("retrieval", "recall_at_6")),
    ("MRR@6", ("retrieval", "mrr_at_6")),
    ("Работы: точность", ("works", "precision")),
    ("Работы: полнота", ("works", "recall")),
    ("Работы: F1", ("works", "f1")),
    ("Работы: родственные", ("works", "related")),
    ("Количество: точно (из n)", ("quantity", "exact")),
    ("Количество: MAE", ("quantity", "mae")),
    ("Часы 1-282: MAE", ("hours_1_282", "mae")),
    ("Материалы: точность", ("materials", "precision")),
    ("Материалы: полнота", ("materials", "recall")),
    ("Материалы: F1", ("materials", "f1")),
    ("Техника найдена (из n)", ("resources", "detected")),
    ("WAPE полной суммы, %", ("total", "wape_pct")),
    ("Смещение полной суммы, %", ("total", "bias_pct")),
    ("Полная сумма в ±10 %", ("total", "within_10pct")),
    ("Полная сумма в ±20 %", ("total", "within_20pct")),
    ("Работы в ±10 %", ("works_cost", "within_10pct")),
    ("Работы в ±20 %", ("works_cost", "within_20pct")),
    ("WAPE работ, %", ("works_cost", "wape_pct")),
    ("Сумма без техники в ±20 %", ("total_without_resources", "within_20pct")),
)


def _get(metrics: dict, path: tuple[str, str]):
    """Значение метрики по пути (раздел, ключ)."""
    return metrics.get(path[0], {}).get(path[1])


def _per_request_line(name: str, result: dict) -> str:
    """Строка таблицы по заявке: коды, суммы, отклонения."""
    totals = result["totals"]
    works = ", ".join(f"{w['gold']}:{w['outcome']}" for w in result["works"])
    return (f"| {name} | {', '.join(result['predicted_codes']) or '—'} | {works} | "
            f"{totals['plan_total']:.0f} / {totals['actual_total']:.0f} ({_deviation(totals['plan_total'], totals['actual_total'])} %) | "
            f"{result['materials']['tp']}/{result['materials']['tp'] + result['materials']['fn']}, лишних {result['materials']['fp']} |")


def report_md(exp_id: str, metrics: dict, per_request: dict, config: dict,
              baseline: tuple[dict, dict, dict] | None = None) -> str:
    """Отчёт эксперимента в формате раздела 9 регламента (решение заполняет человек или агент).

    Принимает: exp_id; metrics, per_request, config — кандидат; baseline — (metrics, per_request, config) или None.
    Возвращает: текст Markdown.
    """
    lines = [f"# {exp_id}", "", "## Решение", "_заполнить: ACCEPT / REJECT / NEEDS_DATA / NEEDS_PRODUCT_DECISION / INCONCLUSIVE_", "",
             "## Конфигурация", *[f"- {key}: {value}" for key, value in config.items()], "",
             f"## Результаты (n заявок = {metrics['requests']}; поиск n = {metrics['retrieval']['n']}, "
             f"количество n = {metrics['quantity']['n']}, техника n = {metrics['resources']['n']})", ""]
    if baseline:
        lines += ["| Метрика | Baseline | Candidate |", "|---|---:|---:|"]
        lines += [f"| {label} | {_get(baseline[0], path)} | {_get(metrics, path)} |" for label, path in _ROWS]
    else:
        lines += ["| Метрика | Значение |", "|---|---:|"] + [f"| {label} | {_get(metrics, path)} |" for label, path in _ROWS]
    lines += ["", "## По заявкам", "", "| Заявка | Коды плана | Эталонные работы | План / факт, ₽ | Материалы найдено/нужно |",
              "|---|---|---|---|---|"] + [_per_request_line(name, result) for name, result in per_request.items()]
    if baseline:
        lines += ["", "## Изменения относительно baseline", ""]
        for name, result in per_request.items():
            before = baseline[1].get(name)
            if before is None:
                continue
            old = [w["outcome"] for w in before["works"]]
            new = [w["outcome"] for w in result["works"]]
            delta_old = _deviation(before["totals"]["plan_total"], before["totals"]["actual_total"])
            delta_new = _deviation(result["totals"]["plan_total"], result["totals"]["actual_total"])
            if old != new or before["predicted_codes"] != result["predicted_codes"] or delta_old != delta_new:
                lines.append(f"- {name}: работы {old} → {new}; коды {before['predicted_codes']} → {result['predicted_codes']}; "
                             f"отклонение суммы {delta_old} % → {delta_new} %")
    lines += ["", "## Ограничения", "- 10 заявок АВР — данные разработки, не отложенная выборка: изменения выбора модели "
              "на них не могут получить ACCEPT (раздел 8 регламента).",
              "- Эталон a01 помечен needs_review: запись мастера и код ПЗР не согласованы.",
              "- Тарифа автовышки нет: её стоимость в план не входит; см. «Сумма без техники»."]
    return "\n".join(lines) + "\n"


def report(run_dir: Path, baseline_dir: Path | None, exp_id: str | None) -> dict:
    """Считает метрики прогона (и baseline), пишет metrics.json, per_request.json, report.md в run_dir."""
    metrics, per_request = evaluate_run(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8")) if (run_dir / "config.json").exists() else {}
    baseline = None
    if baseline_dir:
        base_metrics, base_per_request = evaluate_run(baseline_dir)
        base_config = json.loads((baseline_dir / "config.json").read_text(encoding="utf-8")) \
            if (baseline_dir / "config.json").exists() else {}
        baseline = (base_metrics, base_per_request, base_config)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "per_request.json").write_text(json.dumps(per_request, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "report.md").write_text(report_md(exp_id or run_dir.name, metrics, per_request, config, baseline),
                                       encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:
    """Точка входа: run или report."""
    parser = argparse.ArgumentParser(prog="python -m llm.estimator.avr_eval")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="прогнать 10 заявок АВР")
    p.add_argument("--out", type=Path, required=True, help="новая папка эксперимента")
    p.add_argument("--diagnostic-comment", action="store_true", help="добавить записи мастера (верхняя граница)")
    p = sub.add_parser("report", help="посчитать метрики")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--id")
    args = parser.parse_args(argv)
    if args.command == "run":
        import logging

        logging.basicConfig(level=logging.WARNING, format="%(message)s")
        run(args.out, args.diagnostic_comment)
    else:
        metrics = report(args.run, args.baseline, args.id)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"\nОтчёт: {args.run / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
