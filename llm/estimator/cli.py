"""Командная строка: черновик сметы по фотографиям заявки.

Пример:
    python -m llm.estimator.cli фото1.jpg фото2.jpg -d "Трещина в стене подъезда, длина 3 м"

Результат сохраняется в results/<время>/: estimate.csv (для Excel) и estimate.json (расчёт сметы).
Нужны запущенные контейнеры pzr-db, prices-db, ml_service и llm_service, загруженный справочник
(python -m rag.cli load …) и прайсы поставщиков (python -m rag.cli load-prices …).
Флаг --no-materials — смета только по работам, без подбора материалов.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from llm.estimator.config import get_estimator_settings
from llm.estimator.estimate import estimate_csv
from llm.estimator.llm_client import LLMServiceClient
from llm.estimator.pipeline import analyze_case, prepare_image
from rag.config import get_settings
from rag.db import get_session_factory
from rag.ml_client import build_ml_client
from rag.prices_db import get_prices_session_factory
from rag.retriever import HybridRetriever, MaterialRetriever


def main(argv: list[str] | None = None) -> int:
    """Читает фото, запускает конвейер, печатает итог и сохраняет файлы сметы.

    Принимает: argv — аргументы командной строки (по умолчанию из sys.argv).
    Возвращает: код выхода (0 — успех).
    """
    parser = argparse.ArgumentParser(prog="python -m llm.estimator.cli", description="Черновик сметы ПЗР по фото")
    parser.add_argument("photos", nargs="*", type=Path, help="0–4 фото одной заявки (JPEG/PNG/WebP)")
    parser.add_argument("-d", "--description", default="", help="описание заявки (необязательно)")
    parser.add_argument("--out", type=Path, default=Path("results"), help="папка для результатов")
    parser.add_argument("--no-materials", action="store_true", help="не подбирать материалы (только работы)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = get_estimator_settings()
    if len(args.photos) > settings.max_images:
        parser.error(f"не больше {settings.max_images} фото")
    if not args.photos and not args.description.strip():
        parser.error("нужно хотя бы одно фото или описание (-d)")
    images = [prepare_image(path.read_bytes(), settings.max_image_pixels) for path in args.photos]

    rag_settings = get_settings()
    llm = LLMServiceClient(settings.llm_service_url, settings.llm_timeout_s)
    ml = build_ml_client(rag_settings)
    # Две базы — две сессии: справочник ПЗР (работы) и прайсы поставщиков (материалы).
    with get_session_factory()() as session, get_prices_session_factory()() as prices_session:
        retriever = HybridRetriever(session, ml, rag_settings)
        material_retriever = None if args.no_materials else MaterialRetriever(prices_session, ml, rag_settings)
        result, _trace = analyze_case(
            images, args.description, llm, retriever, settings.candidates_per_finding,
            settings.observation_max_tokens, settings.selection_max_tokens,
            material_retriever, settings.material_candidates, settings.material_selection_max_tokens,
        )

    # --- Сохранение результатов. ---
    run_dir = args.out / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "estimate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "estimate.csv").write_text(estimate_csv(result), encoding="utf-8", newline="")

    # --- Краткий итог в терминал. ---
    print(f"\n{result['summary']}\n")
    for line in result["works"]:
        qty = line["quantity"] or "?"
        amount = line["amount_rub"] or "—"
        print(f"  [{line['work_id']:<7}] {line['name'][:70]:<70} {qty:>6} {line['unit']:<5} × {line['unit_price_rub']:>9} = {amount}")
    if result["alternatives"]:
        print(f"  Альтернатив (не в сумме): {len(result['alternatives'])}")
    print(f"\nРаботы, итого по позициям с известным объёмом: {result['subtotal_priced_lines_rub'] or '—'} руб.")
    if result["materials_priced"]:
        print("\nМатериалы:")
        for line in result["materials"]:
            need = f"{line['need_qty']} {line['base_unit']}" if line["need_qty"] else "?"
            buy = f"{line['buy_qty']} {line['price_unit']}" if line["buy_qty"] else "?"
            print(f"  [{line['supplier']:<12}] {line['name'][:58]:<58} нужно {need:>8} купить {buy:>9} "
                  f"× {line['price_rub']:>9} = {line['amount_rub'] or '—'}")
        for item in result["materials_not_found"]:
            print(f"  [не подобран ] {item['query'][:58]:<58} — {item['reason']}")
        print(f"Материалы итого: {result['materials_total_rub'] or '—'} руб. · Всего: {result['total_rub'] or '—'} руб.")
    for warning in result["warnings"]:
        print(f"  ! {warning}")
    print(f"\nФайлы: {run_dir}/estimate.csv, estimate.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
