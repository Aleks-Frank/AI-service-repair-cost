import time

import httpx

from app.core.config import Settings


def main() -> None:
    settings = Settings()
    headers = {"X-API-Key": settings.api_key.get_secret_value()}
    with httpx.Client(
        base_url=f"http://127.0.0.1:{settings.port}", timeout=10, trust_env=False
    ) as client:
        health = client.get("/health")
        health.raise_for_status()
        event = client.post(
            "/api/v1/webhooks/bitrix",
            data={
                "event": "ONCRMDEALADD",
                "data[FIELDS][ID]": "demo-42",
                "data[description]": "Демонстрационная заявка: протечка трубы",
                "ts": str(time.time_ns()),
                "auth[application_token]": settings.bitrix_outbound_token.get_secret_value(),
            },
        )
        event.raise_for_status()
        request_id = event.json()["request_id"]
        print(f"Accepted: {request_id}")
        saved = client.get(f"/api/v1/requests/{request_id}", headers=headers)
        saved.raise_for_status()
        print(f"Saved status: {saved.json()['status']}")
        print("Request saved. AI is not connected; result remains empty.")
        print("No request was sent to a real Bitrix portal.")


if __name__ == "__main__":
    main()
