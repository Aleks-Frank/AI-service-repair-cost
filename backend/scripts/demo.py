import argparse
import base64
import json
import time
from pathlib import Path

import httpx

from app.core.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("photos", nargs="*", type=Path)
    parser.add_argument("--description", default="Заменить 1 неисправную розетку в подъезде.")
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    if len(args.photos) > 4 or args.timeout <= 0:
        parser.error("Нужно не больше 4 фото и положительное время ожидания")
    settings = Settings()
    headers = {"X-API-Key": settings.api_key.get_secret_value()}
    with httpx.Client(
        base_url=f"http://127.0.0.1:{settings.port}",
        timeout=15,
        trust_env=False,
        headers=headers,
    ) as client:
        response = client.post(
            "/api/v1/requests",
            json={
                "external_id": "demo-42",
                "data": {
                    "description": args.description,
                    "images": [
                        base64.b64encode(path.read_bytes()).decode() for path in args.photos
                    ],
                },
            },
        )
        if response.is_error:
            raise SystemExit(f"Ошибка приёма заявки: {response.status_code} {response.text}")
        request_id = response.json()["request_id"]
        print(f"Заявка: {request_id}", flush=True)
        deadline = time.monotonic() + args.timeout
        previous = None
        while time.monotonic() < deadline:
            saved = client.get(f"/api/v1/requests/{request_id}")
            saved.raise_for_status()
            record = saved.json()
            if record["status"] != previous:
                previous = record["status"]
                print(f"Состояние: {previous}", flush=True)
            if record["result"] is not None:
                print(json.dumps(record["result"], ensure_ascii=False, indent=2))
                return
            if previous == "processing_failed":
                raise SystemExit(f"Ошибка обработки: {record['last_error']}")
            if previous == "received":
                raise SystemExit("Заявка сохранена. Для обработки включите AI_ENABLED.")
            time.sleep(2)
        raise SystemExit(f"Время ожидания истекло. Заявка {request_id} остаётся на сервере.")


if __name__ == "__main__":
    main()
