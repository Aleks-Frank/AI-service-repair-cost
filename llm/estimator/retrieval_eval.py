"""Офлайн-оценка поиска работ ПЗР: попадает ли работа мастера в кандидаты до вызова модели.

Модель выбора здесь не вызывается. Вход этапа 1 (дефекты и поисковые запросы) берётся из
сохранённых trace.json, поэтому все варианты поиска сравниваются на одинаковом входе. Для
каждого варианта (глубина выдачи, ветка по тексту заявки, словарь) считается, на каком месте
среди кандидатов дефекта стоит каждая эталонная работа из итоговой калькуляции АВР.

Метрики:
    Recall@K — доля эталонных работ, найденных на местах 1..K (точное попадание по коду);
    MRR — среднее 1/место точного попадания (0, если не найдено);
    median_rank — медианное место среди найденных;
    no_hit — доля эталонных работ, не найденных вовсе;
    related — доля, где найдена только родственная работа (другой диаметр), а не точная;
    candidates — среднее число кандидатов на дефект; latency — время поиска на заявку.
Эталонный код мастера используется только для подсчёта метрик и в поиск не передаётся.

Запуск (нужны базы и ml_service):
    python -m llm.estimator.retrieval_eval --traces results/qwen_avr_noauto results/giga_max_avr
Результат: results/retrieval_eval/<время>/report.json и records.jsonl.
"""

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from llm.estimator.candidates import UNIVERSAL_WORK_ID, find_candidates
from llm.estimator.schemas import Observations
from rag.term_dictionary import TermDictionary, build_from_export

# Варианты сравнения (раздел 9 задания): глубина, ветка по тексту заявки, словарь.
VARIANTS = [
    {"name": "B0", "depth": 6, "description": False, "dictionary": False, "checks": "baseline"},
    {"name": "B1", "depth": 6, "description": True, "dictionary": False, "checks": "Г2"},
    {"name": "B2", "depth": 10, "description": False, "dictionary": False, "checks": "Г3"},
    {"name": "B3", "depth": 15, "description": False, "dictionary": False, "checks": "Г3"},
    {"name": "B1@10", "depth": 10, "description": True, "dictionary": False, "checks": "Г2+Г3"},
    {"name": "B1@15", "depth": 15, "description": True, "dictionary": False, "checks": "Г2+Г3"},
    {"name": "B4", "depth": 6, "description": False, "dictionary": True, "checks": "Г1"},
    {"name": "B5", "depth": 6, "description": True, "dictionary": True, "checks": "Г1+Г2"},
]
RECALL_AT = (6, 10, 15)


def load_cases(requests_path: Path, gold_path: Path, trace_dirs: list[Path]) -> list[dict]:
    """Собирает размеченные случаи: текст заявки, дефекты этапа 1 из трассировки и эталон.

    Принимает:
        requests_path — requests.json заявок АВР (n, description);
        gold_path — gold.json: {имя заявки: [{"name", "any_of", "related", "universal"?}]};
        trace_dirs — папки прогонов (<папка>/<имя заявки>/<время>/trace.json); каждая — отдельный
            источник дефектов и запросов этапа 1.
    Возвращает: список {"request_id", "source", "description", "observations", "gold"}; работы
        «Ремонтные работы» (universal) в эталон не входят — в кандидаты они не попадают по правилам.
    """
    gold = {key: value for key, value in json.loads(gold_path.read_text(encoding="utf-8")).items()
            if not key.startswith("_")}
    descriptions = {f"a{item['n']:02d}": item["description"].strip()
                    for item in json.loads(requests_path.read_text(encoding="utf-8"))}
    cases = []
    for trace_dir in trace_dirs:
        for request_id, works in sorted(gold.items()):
            works = [work for work in works if not work.get("universal")]
            # Прогоны CLI: <папка>/<заявка>/<время>/trace.json; прогоны avr_eval: <папка>/<заявка>/trace.json.
            traces = sorted((trace_dir / request_id).glob("*/trace.json")) or sorted(
                (trace_dir / request_id).glob("trace.json"))
            if not works or not traces:
                continue
            trace = json.loads(traces[-1].read_text(encoding="utf-8"))
            cases.append({"request_id": request_id, "source": trace_dir.name, "description": descriptions[request_id],
                          "observations": trace["observations"], "gold": works})
    return cases


def gold_ranks(candidate_lists: dict[str, list[str]], work: dict) -> tuple[int | None, int | None]:
    """Лучшее место эталонной работы среди кандидатов дефектов заявки.

    Какой дефект соответствует работе мастера, неизвестно, поэтому берётся лучшее место по всем дефектам.

    Принимает: candidate_lists — {finding_id: [коды кандидатов по порядку]}; work — эталонная работа.
    Возвращает: (место точного попадания или None, место родственной работы или None).
    """
    def best(codes: set[str]) -> int | None:
        places = [codes_list.index(code) + 1 for codes_list in candidate_lists.values()
                  for code in codes if code in codes_list]
        return min(places) if places else None
    return best(set(work["any_of"])), best(set(work.get("related", [])))


def evaluate(cases: list[dict], retriever, variants: list[dict], dictionary: TermDictionary | None = None,
             work_names: dict[str, str] | None = None) -> list[dict]:
    """Прогоняет поиск по каждому случаю и варианту и записывает места эталонных работ.

    Принимает:
        cases — из load_cases; retriever — HybridRetriever;
        variants — варианты из VARIANTS; варианты со словарём пропускаются, если словаря нет;
        dictionary — словарь Г1 или None; work_names — {код: наименование} для расширений словаря.
    Возвращает: записи по каждой паре (случай, вариант) — кандидаты, ветки, места эталона, время.
    """
    records = []
    for variant in variants:
        if variant["dictionary"] and dictionary is None:
            continue
        for case in cases:
            observations = Observations.model_validate(case["observations"])
            extra = dictionary.expansions(case["description"], work_names or {}) if variant["dictionary"] else None
            log: dict = {}
            start = time.perf_counter()
            find_candidates(retriever, observations, per_finding=variant["depth"], log=log,
                            description=case["description"] if variant["description"] else None,
                            extra_queries=extra)
            elapsed = time.perf_counter() - start
            # Кандидаты дефекта — ровно то, что получила бы модель: без 1-282, первые depth.
            lists = {finding_id: [item["work_id"] for item in entry["retrieved"]
                                  if item["work_id"] != UNIVERSAL_WORK_ID][:variant["depth"]]
                     for finding_id, entry in log.items()}
            for work in case["gold"]:
                exact, related = gold_ranks(lists, work)
                records.append({
                    "variant": variant["name"], "request_id": case["request_id"], "source": case["source"],
                    "gold": work["name"], "gold_codes": work["any_of"], "exact_rank": exact, "related_rank": related,
                    "candidates_per_finding": [len(codes) for codes in lists.values()],
                    "duplicates": sum(max(0, len(item["found_by"]) - 1) for entry in log.values()
                                      for item in entry["retrieved"]),
                    "latency_s": round(elapsed, 3), "queries": {f: e["query_branches"] for f, e in log.items()},
                    "candidates": lists,
                })
    return records


def summarize(records: list[dict]) -> list[dict]:
    """Метрики по вариантам.

    Принимает: records — из evaluate.
    Возвращает: по варианту — items, recall@K, mrr, median_rank, no_hit, related_only, candidates, latency.
    """
    rows = []
    for name in dict.fromkeys(record["variant"] for record in records):
        part = [record for record in records if record["variant"] == name]
        ranks = [record["exact_rank"] for record in part]
        found = [rank for rank in ranks if rank is not None]
        # Время — по заявке (запись повторяется для каждой эталонной работы заявки).
        latency = list({(r["request_id"], r["source"]): r["latency_s"] for r in part}.values())
        counts = [count for record in part for count in record["candidates_per_finding"]]
        rows.append({
            "variant": name,
            "items": len(part),
            **{f"recall@{k}": round(sum(rank is not None and rank <= k for rank in ranks) / len(part), 3)
               for k in RECALL_AT},
            "mrr": round(sum(1 / rank for rank in found) / len(part), 3),
            "median_rank": statistics.median(found) if found else None,
            "no_hit": round(sum(rank is None for rank in ranks) / len(part), 3),
            "related_only": round(sum(r["exact_rank"] is None and r["related_rank"] is not None for r in part)
                                  / len(part), 3),
            "candidates_mean": round(statistics.mean(counts), 1) if counts else 0,
            "latency_p50_s": round(statistics.median(latency), 2),
            "latency_p95_s": round(sorted(latency)[max(0, round(0.95 * len(latency)) - 1)], 2),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    """Запускает оценку и сохраняет отчёт.

    Принимает: argv — аргументы командной строки.
    Возвращает: код выхода (0 — успех).
    """
    parser = argparse.ArgumentParser(prog="python -m llm.estimator.retrieval_eval", description=__doc__.split("\n")[0])
    parser.add_argument("--requests", type=Path, default=Path("data/avr_examples/requests.json"))
    parser.add_argument("--gold", type=Path, default=Path("data/avr_examples/gold.json"))
    parser.add_argument("--traces", type=Path, nargs="+", required=True, help="папки прогонов с trace.json")
    parser.add_argument("--dictionary", type=Path, help="словарь Г1 (rag.term_dictionary); без него B4/B5 пропускаются")
    parser.add_argument("--avr-export", type=Path, help="выгрузка АВР для построения словаря Г1")
    parser.add_argument("--out", type=Path, default=Path("results/retrieval_eval"))
    parser.add_argument("--variants", nargs="+", help="только эти варианты (например B0 B2); по умолчанию все")
    args = parser.parse_args(argv)
    variants = [v for v in VARIANTS if not args.variants or v["name"] in args.variants]

    from sqlalchemy import select

    from rag.config import get_settings
    from rag.db import Rate, get_session_factory
    from rag.ml_client import build_ml_client
    from rag.retriever import HybridRetriever

    cases = load_cases(args.requests, args.gold, args.traces)
    dictionary = TermDictionary.load(args.dictionary) if args.dictionary else None
    g1_status = dictionary.meta if dictionary else build_from_export(args.avr_export)
    settings = get_settings()
    with get_session_factory()() as session:
        retriever = HybridRetriever(session, build_ml_client(settings), settings)
        work_names = {rate.code: rate.name for rate in session.scalars(select(Rate))}
        records = evaluate(cases, retriever, variants, dictionary, work_names)
    summary = summarize(records)

    run_dir = args.out / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {"cases": len(cases), "sources": [str(path) for path in args.traces], "g1": g1_status,
              "settings": {"rerank_top_n": settings.rerank_top_n, "per_branch": settings.search_candidates_per_branch,
                           "rrf_k": settings.rrf_k},
              "variants": variants, "summary": summary}
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "records.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Случаев: {len(cases)} (заявки × источники этапа 1), эталонных работ: {len(records) // max(1, len(summary))}")
    print(f"Г1: {g1_status.get('status', 'словарь загружен')}")
    header = ["variant", "recall@6", "recall@10", "recall@15", "mrr", "median_rank", "no_hit", "related_only",
              "candidates_mean", "latency_p50_s"]
    print(" | ".join(header))
    for row in summary:
        print(" | ".join(str(row[key]) for key in header))
    print(f"\nОтчёт: {run_dir}/report.json, records.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
