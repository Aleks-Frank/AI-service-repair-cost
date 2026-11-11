"""Проверка поиска по прайсам: находит ли он тип товара, который фактически использовал мастер.

Запросы — как их формулирует модель на этапе материалов («светодиодная лента», «паста уплотнительная»);
ожидаемый тип — регулярное выражение по наименованию товара. Набор составлен по материалам итоговых
калькуляций 10 примеров АВР (data/avr_examples/gold_full.json) — это данные разработки, не отложенная выборка.

    python -m rag.material_search_eval --out results/experiments/<ID>/material_search.json

Метрики: Recall@5 (нужный тип среди первых 5 после переранжирования — столько видит модель)
и место первого подходящего товара. Запуск только читает базу.
"""

import argparse
import json
import re
from pathlib import Path

# (запрос, шаблон ожидаемого типа товара по наименованию в нижнем регистре)
CASES = [
    ("светодиодная лента", r"лента.*(светодиод|led)|(светодиод|led).*лента"),
    ("блок питания для светодиодной ленты", r"блок питания|драйвер"),
    ("лампа светодиодная T8 G13", r"ламп.*(t8|т8|g13)"),
    ("лампа светодиодная линейная", r"ламп.*(t8|т8|g13|линейн)"),
    ("светильник консольный светодиодный уличный", r"светильник.*(консол|дку)"),
    ("прожектор светодиодный 200 Вт IP65", r"прожектор"),
    ("выключатель накладной IP54", r"выключател"),
    ("коробка распаечная накладная", r"коробк"),
    ("хомут нейлоновый", r"хомут.*(нейлон|полиамид|кабел)|стяжк"),
    ("паста уплотнительная", r"паста.*уплотн"),
    ("лён сантехнический", r"(^|\W)л[её]н(\W|$)"),
    ("кран шаровой латунный 1 дюйм", r"кран.*шаров"),
    ("хомут ремонтный для трубы", r"хомут"),
    ("пластина резиновая техническая", r"пластин|резин"),
]
TOP = 5


def evaluate(retriever, cases=CASES, top: int = TOP) -> dict:
    """Ищет каждый запрос и находит место первого товара ожидаемого типа.

    Принимает: retriever — MaterialRetriever; cases — [(запрос, шаблон)]; top — сколько видит модель.
    Возвращает: {"recall_at_5", "mrr_at_5", "n", "cases": [{query, rank, top_names}]}.
    """
    results = []
    for query, pattern in cases:
        hits = retriever.search(query, top_k=15).hits
        names = [hit.item.name for hit in hits]
        rank = next((i + 1 for i, name in enumerate(names) if re.search(pattern, name.lower())), None)
        results.append({"query": query, "rank": rank, "top_names": names[:top]})
    found = [r["rank"] for r in results if r["rank"] is not None and r["rank"] <= top]
    return {"n": len(results), "recall_at_5": round(len(found) / len(results), 3),
            "mrr_at_5": round(sum(1 / rank for rank in found) / len(results), 3), "cases": results}


def main(argv: list[str] | None = None) -> int:
    """Точка входа: печатает метрики и сохраняет результат."""
    from rag.config import get_settings
    from rag.ml_client import build_ml_client
    from rag.prices_db import get_prices_session_factory
    from rag.retriever import MaterialRetriever

    parser = argparse.ArgumentParser(prog="python -m rag.material_search_eval")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    settings = get_settings()
    with get_prices_session_factory()() as session:
        result = evaluate(MaterialRetriever(session, build_ml_client(settings), settings))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for case in result["cases"]:
        print(f"{case['rank'] or '—':>3}  {case['query']}  |  {case['top_names'][0][:60] if case['top_names'] else ''}")
    print(f"Recall@5 = {result['recall_at_5']}, MRR@5 = {result['mrr_at_5']} (n = {result['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
