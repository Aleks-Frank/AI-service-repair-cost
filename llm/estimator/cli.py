"""Командная строка: черновик сметы по фотографиям заявки.

Пример:
    python -m llm.estimator.cli фото1.jpg фото2.jpg -d "Трещина в стене подъезда, длина 3 м"

Результат сохраняется в results/<время>/: estimate.csv (для Excel), estimate.json (расчёт сметы)
и trace.json (трассировка: кандидаты, показанные модели, её ответы и состояние каждого дефекта).
Нужны запущенные контейнеры pzr-db, prices-db, ml_service и llm_service, загруженный справочник
(python -m rag.cli load …) и прайсы поставщиков (python -m rag.cli load-prices …).
Флаг --no-materials — смета только по работам, без подбора материалов.
Флаг --inspection «текст» — запись осмотра мастера до ремонта (операция, количество, характеристики,
доступ); без него смета — предварительный проект по первичной заявке.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from llm.estimator.config import build_llm_client, get_estimator_settings
from llm.estimator.estimate import estimate_csv
from llm.estimator.pipeline import analyze_case, prepare_image
from rag.config import get_settings
from rag.db import get_session_factory
from rag.ml_client import build_ml_client
from rag.prices_db import get_prices_session_factory
from rag.retriever import HybridRetriever, MaterialRetriever


logger = logging.getLogger(__name__)


def save_trace(run_dir: Path, trace: dict) -> Path | None:
    """Сохраняет трассировку конвейера рядом со сметой.

    В трассировке нет фото и ключей — только тексты: описание заявки, ответы модели, кандидаты.
    Ошибка записи не роняет расчёт (смета уже сохранена), но выводится предупреждением.

    Принимает: run_dir — папка результатов прогона; trace — трассировка из analyze_case.
    Возвращает: путь к trace.json или None, если записать не удалось.
    """
    path = run_dir / "trace.json"
    try:
        # default=str — Decimal и прочие нестандартные значения записываются строкой.
        path.write_text(json.dumps(trace, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except (OSError, TypeError, ValueError) as error:
        logger.warning("Трассировка не сохранена (%s): %s", path, error)
        return None
    return path


def main(argv: list[str] | None = None) -> int:
    """Читает фото, запускает конвейер, печатает итог и сохраняет файлы сметы.

    Принимает: argv — аргументы командной строки (по умолчанию из sys.argv).
    Возвращает: код выхода (0 — успех).
    """
    parser = argparse.ArgumentParser(prog="python -m llm.estimator.cli", description="Черновик сметы ПЗР по фото")
    parser.add_argument("photos", nargs="*", type=Path, help="0–4 фото одной заявки (JPEG/PNG/WebP)")
    parser.add_argument("-d", "--description", default="", help="описание заявки (необязательно)")
    parser.add_argument("--inspection", default="", help="запись осмотра мастера до ремонта (необязательно)")
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
    llm = build_llm_client(settings)
    ml = build_ml_client(rag_settings)
    # Две базы — две сессии: справочник ПЗР (работы) и прайсы поставщиков (материалы).
    with get_session_factory()() as session, get_prices_session_factory()() as prices_session:
        retriever = HybridRetriever(session, ml, rag_settings)
        material_retriever = None if args.no_materials else MaterialRetriever(prices_session, ml, rag_settings)
        result, trace = analyze_case(
            images, args.description, llm, retriever, settings.candidates_per_finding,
            settings.observation_max_tokens, settings.selection_max_tokens,
            material_retriever, settings.material_candidates, settings.material_selection_max_tokens,
            settings.search_by_description, inspection=args.inspection,
            linear_lamp_default=settings.default_linear_lamps_common_areas,
            equipment_tariff_rub=settings.aerial_platform_tariff_rub,
            equipment_tariff_unit=settings.aerial_platform_tariff_unit,
        )
    # Какой файл стал «изображением N» промпта — для аудита соответствия фото заявке.
    trace["photo_manifest"] = [{"image_number": n, "file": str(path)} for n, path in enumerate(args.photos, start=1)]

    # --- Сохранение результатов. ---
    run_dir = args.out / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "estimate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "estimate.csv").write_text(estimate_csv(result), encoding="utf-8", newline="")
    save_trace(run_dir, trace)

    # --- Краткий итог в терминал. ---
    print(f"\n{result['summary']}\n")
    for line in result["works"]:
        # «≈» — объём оценён моделью (в заявке не было чисел), его нужно проверить.
        qty = (("≈" if line["quantity_source"] == "estimate" else "") + line["quantity"]) if line["quantity"] else "?"
        amount = line["amount_rub"] or "—"
        print(f"  [{line['work_id']:<7}] {line['name'][:70]:<70} {qty:>6} {line['unit']:<5} × {line['unit_price_rub']:>9} = {amount}")
    if result["alternatives"]:
        print(f"  Альтернатив (не в сумме): {len(result['alternatives'])}")
    print(f"\nРаботы, итого по позициям с известным объёмом: {result['subtotal_priced_lines_rub'] or '—'} руб.")
    if result["materials_priced"]:
        print("\nМатериалы:")
        for line in result["materials"]:
            # «≈» — расходник: количество упаковок оценено моделью.
            need = (("≈" if line["consumable"] else "") + f"{line['need_qty']} {line['base_unit']}") if line["need_qty"] else "?"
            buy = f"{line['buy_qty']} {line['price_unit']}" if line["buy_qty"] else "?"
            print(f"  [{line['supplier']:<12}] {line['name'][:58]:<58} нужно {need:>8} купить {buy:>9} "
                  f"× {line['price_rub']:>9} = {line['amount_rub'] or '—'}")
        for item in result["materials_not_found"]:
            print(f"  [не подобран ] {item['query'][:58]:<58} — {item['reason']}")
        print(f"Материалы итого: {result['materials_total_rub'] or '—'} руб. · Всего: {result['total_rub'] or '—'} руб.")
    for item in result.get("equipment_requirements", []):
        print(f"\nТехника: {item['name']} ({item['technical_name']}) для {item['finding_id']} — "
              f"{'стоимость в расценке' if item['pricing_status'] == 'included_in_rate' else 'стоимость не рассчитана'}"
              f"{', требует подтверждения' if item['needs_human_confirmation'] else ''}.")
    for warning in result["warnings"]:
        print(f"  ! {warning}")
    print(f"\nФайлы: {run_dir}/estimate.csv, estimate.json, trace.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
