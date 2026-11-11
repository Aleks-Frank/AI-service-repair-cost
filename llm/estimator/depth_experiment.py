"""Парный эксперимент: выбор работ Qwen при 6 и при 10 кандидатах на дефект.

Обе конфигурации получают одинаковый вход: те же заявки, фото и дефекты этапа 1 (из сохранённых
trace.json), те же промпты, модель, поиск, fallback и проверки. Различается только глубина
(per_finding). Материалы не подбираются — к вопросу о глубине они не относятся.

    python -m llm.estimator.depth_experiment run --observations results/fix_50 --out results/depth_experiment
    python -m llm.estimator.depth_experiment report --out results/depth_experiment

Команда run пропускает уже посчитанные пары (можно перезапустить после сбоя).

Оценка (report) по правилам задания:
    - эталон — только подтверждённые работы итоговой калькуляции (data/avr_examples/gold.json);
    - уровень дефекта — только если в заявке один дефект: тогда все работы мастера относятся к нему;
      заявки с несколькими дефектами помечаются ambiguous_finding_mapping и в метрики дефектов не входят;
    - родственная работа (другой диаметр) не засчитывается как точная;
    - заявки без итоговой калькуляции — только сравнение 6 и 10 между собой, без «верно/неверно».
"""

import argparse
import ast
import json
import statistics
import time
from pathlib import Path

DEPTHS = (6, 10)
UNIVERSAL = "1-282"


def load_requests() -> list[dict]:
    """Все 50 заявок: имя, текст и фото.

    Возвращает: список {"name", "description", "photos": [пути]} — b01–b30, a01–a10, r02–r11.
    """
    from llm.estimator.photos import resolve_photos

    requests = []
    batch = Path("data/batch_requests")
    for item in json.loads((batch / "requests.json").read_text(encoding="utf-8")):
        requests.append(_with_photos({"name": item["name"], "description": item["description"]},
                                     item["photos"], batch, resolve_photos))
    avr = Path("data/avr_examples")
    for item in json.loads((avr / "requests.json").read_text(encoding="utf-8")):
        photos = item["photos"]
        if isinstance(photos, str):  # в выгрузке АВР список записан строкой «['photos/a01_1.png', …]»
            photos = ast.literal_eval(photos)
        # before_note — запись мастера ДО работ (осмотр); after_note (после работ) осмотром не считается.
        note = (item.get("before_note") or "").strip()
        requests.append(_with_photos({"name": f"a{item['n']:02d}", "description": item["description"].strip(),
                                      "inspection": "" if note.lower() in ("", "отсутствует") else note},
                                     photos, avr, resolve_photos))
    real = Path("data/real_requests")
    for item in json.loads((real / "requests.json").read_text(encoding="utf-8")):
        files = [f"photos/{path.name}" for path in sorted((real / "photos").glob(f"{item['name']}_*"))]
        requests.append(_with_photos({"name": item["name"], "description": item["description"]},
                                     files, real, resolve_photos))
    return requests


def _with_photos(request: dict, entries: list, base: Path, resolve) -> dict:
    """Добавляет к заявке проверенные фото, их manifest и предупреждения (см. photos.resolve_photos)."""
    paths, manifest, warnings = resolve(request["name"], entries, base)
    return {**request, "photos": paths, "photo_manifest": manifest, "photo_warnings": warnings}


def run(observations_dir: Path, out: Path) -> None:
    """Прогоняет каждую заявку при глубине 6 и 10 и сохраняет смету и трассировку.

    Принимает: observations_dir — папка прогона с trace.json (вход этапа 1); out — папка результатов.
    Возвращает: ничего; результаты — out/d<глубина>/<заявка>/estimate.json и trace.json.
    """
    from llm.estimator.cli import save_trace
    from llm.estimator.config import build_llm_client, get_estimator_settings
    from llm.estimator.pipeline import analyze_case, prepare_image
    from llm.estimator.schemas import Observations
    from rag.config import get_settings
    from rag.db import get_session_factory
    from rag.ml_client import build_ml_client
    from rag.retriever import HybridRetriever

    settings, rag_settings = get_estimator_settings(), get_settings()
    llm = build_llm_client(settings)
    with get_session_factory()() as session:
        retriever = HybridRetriever(session, build_ml_client(rag_settings), rag_settings)
        for request in load_requests():
            traces = sorted((observations_dir / request["name"]).glob("*/trace.json"))
            if not traces:
                print(f"{request['name']}: нет trace.json — пропуск", flush=True)
                continue
            observations = Observations.model_validate(json.loads(traces[-1].read_text(encoding="utf-8"))["observations"])
            images = [prepare_image(path.read_bytes(), settings.max_image_pixels) for path in request["photos"]]
            for depth in DEPTHS:
                run_dir = out / f"d{depth}" / request["name"]
                if (run_dir / "estimate.json").exists():
                    continue
                start = time.time()
                result, trace = analyze_case(
                    images, request["description"], llm, retriever, depth,
                    settings.observation_max_tokens, settings.selection_max_tokens, observations=observations,
                )
                trace["photo_manifest"] = request["photo_manifest"]
                trace["elapsed_s"] = round(time.time() - start, 1)
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "estimate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                save_trace(run_dir, trace)
                print(f"{request['name']} d{depth}: {trace['elapsed_s']} с", flush=True)


def classify_finding(works: list[str], status: str, shown: list[str], gold: list[dict]) -> dict:
    """Оценка выбора по одному дефекту с подтверждённым эталоном.

    Принимает:
        works — коды основных работ сметы для дефекта (с 1-282, если она подставлена);
        status — итоговое состояние дефекта (matched, model_unmatched, …);
        shown — коды кандидатов, показанных модели;
        gold — эталонные работы дефекта ({"any_of", "related", "universal"?}).
    Возвращает: {"outcome", "false_works", "gold_shown"}, где outcome — correct / related / false /
        justified_unmatched / unjustified_unmatched / error / gold_universal_*.
    """
    exact = {code for work in gold for code in work["any_of"]}
    related = {code for work in gold for code in work.get("related", [])}
    specific = [code for code in works if code != UNIVERSAL]
    gold_shown = bool(exact & set(shown))
    false_works = [code for code in specific if code not in exact | related]
    if any(work.get("universal") for work in gold):
        outcome = "gold_universal_fallback" if not specific else "gold_universal_specific"
    elif status in ("model_empty", "invalid_selection", "llm_error"):
        outcome = "error"
    elif exact & set(specific):
        outcome = "correct"
    elif related & set(specific):
        outcome = "related"
    elif specific:
        outcome = "false"
    else:
        outcome = "unjustified_unmatched" if gold_shown else "justified_unmatched"
    return {"outcome": outcome, "false_works": false_works, "gold_shown": gold_shown}


def load_run(out: Path, depth: int, name: str) -> tuple[dict, dict] | None:
    """Смета и трассировка одной пары (глубина, заявка) или None, если её нет."""
    run_dir = out / f"d{depth}" / name
    if not (run_dir / "estimate.json").exists():
        return None
    return (json.loads((run_dir / "estimate.json").read_text(encoding="utf-8")),
            json.loads((run_dir / "trace.json").read_text(encoding="utf-8")))


def report(out: Path, gold_path: Path) -> dict:
    """Считает метрики поиска и выбора для обеих глубин и сохраняет report.json.

    Принимает: out — папка результатов run; gold_path — gold.json.
    Возвращает: отчёт (также записывается в out/report.json).
    """
    gold = {key: value for key, value in json.loads(gold_path.read_text(encoding="utf-8")).items()
            if not key.startswith("_")}
    names = [request["name"] for request in load_requests()]
    result: dict = {"labeled": {}, "unlabeled": {}, "ambiguous": [], "per_request": {}}
    for depth in DEPTHS:
        outcomes, ranks, retrieval_items, statuses, fallback_lines, reduced, times = [], [], 0, [], 0, 0, []
        for name in names:
            loaded = load_run(out, depth, name)
            if loaded is None:
                continue
            estimate, trace = loaded
            times.append(trace.get("elapsed_s"))
            findings = trace["findings"]
            statuses += [state["status"] for state in findings.values()]
            fallback_lines += sum(line["work_id"] == UNIVERSAL for line in estimate["works"])
            reduced += sum(len(state.get("prompt_candidates", [])) < min(depth, len(state.get("retrieved", [])))
                           and state.get("status") != "no_candidates" for state in findings.values())
            entry = result["per_request"].setdefault(name, {})
            entry[f"d{depth}"] = {"works": [line["work_id"] for line in estimate["works"]],
                                  "statuses": estimate.get("finding_statuses", {})}
            if name not in gold:
                continue
            if len(findings) != 1:
                if depth == DEPTHS[0]:
                    result["ambiguous"].append({"request": name, "findings": len(findings),
                                                "reason": "ambiguous_finding_mapping"})
                continue
            (finding_id, state), = findings.items()
            shown = [item["work_id"] for item in state.get("prompt_candidates", [])]
            works = [line["work_id"] for line in estimate["works"] if line["finding_id"] == finding_id]
            verdict = classify_finding(works, state["status"], shown, gold[name])
            verdict.update(request=name, works=works, status=state["status"])
            outcomes.append(verdict)
            entry[f"d{depth}"]["verdict"] = verdict["outcome"]
            # Поиск: место точной эталонной работы среди кандидатов дефекта (до обрезки — retrieved).
            listed = [item["work_id"] for item in state.get("retrieved", []) if item["work_id"] != UNIVERSAL][:depth]
            for work in gold[name]:
                if work.get("universal"):
                    continue
                retrieval_items += 1
                places = [listed.index(code) + 1 for code in work["any_of"] if code in listed]
                ranks.append(min(places) if places else None)
        found = [rank for rank in ranks if rank is not None]
        counts: dict[str, int] = {}
        for verdict in outcomes:
            counts[verdict["outcome"]] = counts.get(verdict["outcome"], 0) + 1
        result["labeled"][f"d{depth}"] = {
            "findings": len(outcomes),
            "outcomes": counts,
            "false_work_lines": sum(len(verdict["false_works"]) for verdict in outcomes),
            "retrieval_items": retrieval_items,
            "recall": round(len(found) / retrieval_items, 3) if retrieval_items else None,
            "mrr": round(sum(1 / rank for rank in found) / retrieval_items, 3) if retrieval_items else None,
            "median_rank": statistics.median(found) if found else None,
            "details": outcomes,
        }
        status_counts: dict[str, int] = {}
        for status in statuses:
            status_counts[status] = status_counts.get(status, 0) + 1
        clean = [t for t in times if t is not None]
        result["unlabeled" if False else "all"] = result.get("all", {})
        result["all"][f"d{depth}"] = {
            "requests": len(clean), "statuses": status_counts, "fallback_lines": fallback_lines,
            "reduced_by_input_limit": reduced,
            "time_p50_s": statistics.median(clean) if clean else None,
            "time_p95_s": sorted(clean)[max(0, round(0.95 * len(clean)) - 1)] if clean else None,
        }
    changed = [name for name, entry in result["per_request"].items()
               if name not in gold and len(entry) == 2 and sorted(entry["d6"]["works"]) != sorted(entry["d10"]["works"])]
    result["unlabeled"] = {"requests": sum(1 for name in result["per_request"] if name not in gold),
                           "changed": changed}
    result.pop("unlabeled" if False else "_", None)
    (out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    """Точка входа: run или report."""
    parser = argparse.ArgumentParser(prog="python -m llm.estimator.depth_experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="прогнать 50 заявок при глубине 6 и 10")
    p.add_argument("--observations", type=Path, default=Path("results/fix_50"))
    p.add_argument("--out", type=Path, default=Path("results/depth_experiment"))
    p = sub.add_parser("report", help="посчитать метрики")
    p.add_argument("--out", type=Path, default=Path("results/depth_experiment"))
    p.add_argument("--gold", type=Path, default=Path("data/avr_examples/gold.json"))
    args = parser.parse_args(argv)
    if args.command == "run":
        import logging

        logging.basicConfig(level=logging.WARNING, format="%(message)s")
        run(args.observations, args.out)
    else:
        result = report(args.out, args.gold)
        for depth in DEPTHS:
            labeled, overall = result["labeled"][f"d{depth}"], result["all"][f"d{depth}"]
            print(f"d{depth}: эталон {labeled['findings']} деф. {labeled['outcomes']} ложных строк "
                  f"{labeled['false_work_lines']} | recall {labeled['recall']} mrr {labeled['mrr']} | "
                  f"все: {overall}")
        print("неоднозначные:", result["ambiguous"], "| без эталона:", result["unlabeled"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
