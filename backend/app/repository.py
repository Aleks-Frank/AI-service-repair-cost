import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from pydantic import ConfigDict, Field, JsonValue, TypeAdapter

from app.core.errors import ServiceError
from app.models import RequestRecord, RequestStatus

result_adapter = TypeAdapter(
    Annotated[dict[str, JsonValue], Field(min_length=1)],
    config=ConfigDict(allow_inf_nan=False),
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


class RequestRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                """CREATE TABLE IF NOT EXISTS requests (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    external_id TEXT,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    dedup_key TEXT UNIQUE,
                    result TEXT,
                    delivery_attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            db.execute("CREATE INDEX IF NOT EXISTS requests_status ON requests(status, created_at)")
            db.execute(
                """UPDATE requests SET status = 'processing_failed', last_error = ?, updated_at = ?
                   WHERE status = 'processing'""",
                ("processing_interrupted", timestamp()),
            )
            db.execute(
                """UPDATE requests SET status = 'delivery_failed', last_error = ?, updated_at = ?
                   WHERE status = 'sending'""",
                ("delivery_interrupted_check_bitrix_before_retry", timestamp()),
            )

    def check_health(self) -> None:
        with self.connection() as db:
            db.execute("SELECT id FROM requests LIMIT 1").fetchone()

    @staticmethod
    def record(row: sqlite3.Row | None) -> RequestRecord:
        if row is None:
            raise ServiceError(404, "request_not_found", "Request not found")
        data = dict(row)
        data["payload"] = json.loads(data["payload"])
        data["result"] = json.loads(data["result"]) if data["result"] is not None else None
        return RequestRecord.model_validate(data)

    def create(
        self,
        source: str,
        external_id: str | None,
        payload: dict,
        dedup_key: str | None,
        *,
        queue_limit: int | None = None,
    ) -> tuple[RequestRecord, bool]:
        request_id, now = str(uuid4()), timestamp()
        payload_hash = fingerprint({"external_id": external_id, "payload": payload})
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if dedup_key:
                existing = db.execute(
                    "SELECT * FROM requests WHERE dedup_key = ?", (dedup_key,)
                ).fetchone()
                if existing:
                    if existing["payload_hash"] != payload_hash:
                        raise ServiceError(
                            409,
                            "idempotency_conflict",
                            "Idempotency key reused with different data",
                        )
                    return self.record(existing), True
            if queue_limit is not None:
                self.check_queue_capacity(db, queue_limit)
            db.execute(
                """INSERT INTO requests
                   (id, source, external_id, status, payload, payload_hash, dedup_key,
                    created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_id,
                    source,
                    external_id,
                    "queued" if queue_limit is not None else "received",
                    canonical_json(payload),
                    payload_hash,
                    dedup_key,
                    now,
                    now,
                ),
            )
            return self.record(
                db.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
            ), False

    @staticmethod
    def check_queue_capacity(db: sqlite3.Connection, limit: int) -> None:
        count = db.execute(
            "SELECT COUNT(*) FROM requests WHERE status IN ('queued', 'processing')"
        ).fetchone()[0]
        if count >= limit:
            raise ServiceError(503, "queue_full", "Processing queue is full; retry later")

    def claim_processing(self) -> RequestRecord | None:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM requests WHERE status = 'queued' ORDER BY created_at, id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "UPDATE requests SET status = 'processing', updated_at = ? WHERE id = ?",
                (timestamp(), row["id"]),
            )
            return self.record(
                db.execute("SELECT * FROM requests WHERE id = ?", (row["id"],)).fetchone()
            )

    def fail_processing(self, request_id: str, code: str) -> None:
        with self.connection() as db:
            db.execute(
                """UPDATE requests SET status = 'processing_failed', last_error = ?,
                   updated_at = ? WHERE id = ? AND status = 'processing'""",
                (code, timestamp(), request_id),
            )

    def retry_processing(self, request_id: str, queue_limit: int) -> RequestRecord:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            current = self.record(
                db.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
            )
            if current.status not in {RequestStatus.RECEIVED, RequestStatus.PROCESSING_FAILED}:
                raise ServiceError(409, "processing_not_retryable", "Request cannot be queued")
            self.check_queue_capacity(db, queue_limit)
            db.execute(
                """UPDATE requests SET status = 'queued', last_error = NULL, updated_at = ?
                   WHERE id = ?""",
                (timestamp(), request_id),
            )
            return self.record(
                db.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
            )

    def get(self, request_id: str) -> RequestRecord:
        with self.connection() as db:
            return self.record(
                db.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
            )

    def list(self, limit: int, offset: int, status: RequestStatus | None) -> list[RequestRecord]:
        with self.connection() as db:
            if status is None:
                rows = db.execute(
                    "SELECT * FROM requests ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            else:
                rows = db.execute(
                    """SELECT * FROM requests WHERE status = ?
                       ORDER BY created_at DESC, id LIMIT ? OFFSET ?""",
                    (status.value, limit, offset),
                ).fetchall()
            return [self.record(row) for row in rows]

    def save_result(self, request_id: str, result: dict) -> RequestRecord:
        result = result_adapter.validate_python(result)
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            current = self.record(
                db.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
            )
            if current.result == result:
                return current
            if current.status in {RequestStatus.SENDING, RequestStatus.DELIVERED}:
                raise ServiceError(
                    409, "result_locked", "Cannot replace a sending or delivered result"
                )
            db.execute(
                """UPDATE requests SET result = ?, status = 'result_ready', last_error = NULL,
                   updated_at = ? WHERE id = ?""",
                (canonical_json(result), timestamp(), request_id),
            )
            return self.record(
                db.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
            )

    def claim_delivery(self, request_id: str) -> tuple[RequestRecord, bool]:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            current = self.record(
                db.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
            )
            if current.status == RequestStatus.DELIVERED:
                return current, False
            if current.status == RequestStatus.SENDING:
                raise ServiceError(409, "delivery_in_progress", "Delivery is already in progress")
            if current.result is None:
                raise ServiceError(409, "result_missing", "Processing result is not available yet")
            if not current.external_id:
                raise ServiceError(409, "external_id_missing", "Bitrix object ID is unknown")
            db.execute(
                """UPDATE requests SET status = 'sending',
                   delivery_attempts = delivery_attempts + 1,
                   last_error = NULL, updated_at = ? WHERE id = ?""",
                (timestamp(), request_id),
            )
            return self.record(
                db.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
            ), True

    def finish_delivery(self, request_id: str, error: str | None = None) -> RequestRecord:
        with self.connection() as db:
            db.execute(
                """UPDATE requests SET status = ?, last_error = ?, updated_at = ?
                   WHERE id = ? AND status = 'sending'""",
                ("delivery_failed" if error else "delivered", error, timestamp(), request_id),
            )
            return self.record(
                db.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
            )
