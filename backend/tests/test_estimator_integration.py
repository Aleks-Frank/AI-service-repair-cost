import base64
import io
import json
from decimal import Decimal

import httpx
import pytest
from llm.estimator.llm_client import LLMServiceClient
from PIL import Image
from rag.db import Rate
from rag.ml_client import MLServiceClient
from sqlalchemy.exc import OperationalError

from app.core.config import Settings
from app.integrations.agent import EstimatorAgent
from tests.test_processing import wait_for_status


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    observations = {
        "summary": "Неисправная розетка",
        "image_relation": "before_only",
        "findings": [
            {
                "finding_id": "f1",
                "object_name": "Розетка",
                "visible_issue": "Не работает",
                "image_numbers": [1],
                "search_queries": ["Смена розетки"],
            }
        ],
        "unknowns": [],
    }
    selection = {
        "works": [
            {
                "finding_id": "f1",
                "work_id": "2-1",
                "reason": "Розетка неисправна",
                "quantity": 1,
                "quantity_unit": "шт",
                "quantity_source": "description",
                "quantity_evidence": "1 розетку",
                "image_numbers": [],
            }
        ],
        "alternatives": [],
        "unmatched_works": [],
        "assumptions": [],
    }
    rate = Rate(
        id=1,
        code="2-1",
        book_number=2,
        section_title="Электрика",
        name="Смена розеток",
        work_composition="Снятие и установка",
        unit="шт",
        unit_raw="шт.",
        quantity=Decimal("1"),
        unit_price=Decimal("125.50"),
        norm_reference="ПЗР 2-1",
        issues=[],
        source_sheet="Электрика",
        source_row=7,
        search_text="Смена розеток",
    )
    state = {"calls": [], "clients": [], "sessions": [], "failure": None}

    def handle(request):
        body = json.loads(request.content)
        state["calls"].append((request.url.path, body))
        if request.url.path == "/generate":
            if state["failure"] == "llm":
                raise httpx.ReadTimeout("secret detail", request=request)
            if state["failure"] == "invalid_llm":
                return httpx.Response(200, json={"text": "invalid answer"})
            prompt = body["user_prompt"]
            answer = selection if '"CANDIDATES"' in prompt else observations
            return httpx.Response(200, json={"text": json.dumps(answer, ensure_ascii=False)})
        if state["failure"] == "ml":
            return httpx.Response(503, json={"detail": "secret detail"})
        if request.url.path == "/embed":
            return httpx.Response(200, json={"embeddings": [[0.01] * 1024]})
        if request.url.path == "/rerank":
            return httpx.Response(200, json={"scores": [0.9] * len(body["documents"])})
        raise AssertionError(f"Unexpected request {request.url.path}")

    def client_init(client, base_url, timeout_s=900, embed_batch=64):
        client._http = httpx.Client(base_url=base_url, transport=httpx.MockTransport(handle))
        client._embed_batch = embed_batch
        state["clients"].append(client._http)

    monkeypatch.setattr(LLMServiceClient, "__init__", client_init)
    monkeypatch.setattr(MLServiceClient, "__init__", client_init)

    class Session:
        def __init__(self):
            self.closed = False
            state["sessions"].append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def execute(self, statement, params=None):
            if state["failure"] == "db":
                raise OperationalError("query", {}, RuntimeError("secret detail"))
            return self

        def mappings(self):
            return self

        def all(self):
            return [
                {
                    "id": 1,
                    "rrf_score": 0.03,
                    "dense_rank": 1,
                    "dense_similarity": 0.9,
                    "lexical_rank": 1,
                    "lexical_score": 0.5,
                }
            ]

        def scalars(self, statement):
            return [rate]

        def commit(self):
            pass

    settings = Settings(_env_file=None, ai_env_file=tmp_path / "absent.env")
    agent = EstimatorAgent(settings)
    agent.sessions = Session
    state["agent"] = agent
    return state


def test_api_runs_real_pipeline_retriever_clients_and_calculation(
    client_factory,
    headers,
    pipeline,
):
    image = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(image, format="PNG")
    client = client_factory(agent=pipeline["agent"], ai_enabled=True, ai_poll_seconds=0.01)
    response = client.post(
        "/api/v1/requests",
        headers=headers,
        json={
            "data": {
                "description": "Заменить 1 розетку.",
                "images": [base64.b64encode(image.getvalue()).decode()],
            }
        },
    )
    assert response.status_code == 202, response.text
    record = wait_for_status(client, headers, response.json()["request_id"], "result_ready")
    result = record["result"]
    assert result["works"][0]["work_id"] == "2-1"
    assert result["works"][0]["source_row"] == 7
    assert result["selected_works_total_rub"] == "125.50"
    assert result["calculation_complete_for_selected_works"] is True
    paths = [path for path, _ in pipeline["calls"]]
    assert paths.count("/generate") == 2
    assert "/embed" in paths and "/rerank" in paths
    generated = [body for path, body in pipeline["calls"] if path == "/generate"]
    assert base64.b64decode(generated[0]["images"][0]).startswith(b"\xff\xd8")
    assert all(client.is_closed for client in pipeline["clients"])
    assert all(session.closed for session in pipeline["sessions"])


@pytest.mark.parametrize(
    "failure,code",
    [
        ("llm", "llm_failed"),
        ("invalid_llm", "llm_failed"),
        ("ml", "ml_failed"),
        ("db", "rag_database_failed"),
    ],
)
def test_real_adapter_translates_dependency_errors(
    client_factory,
    headers,
    pipeline,
    failure,
    code,
    caplog,
):
    pipeline["failure"] = failure
    client = client_factory(agent=pipeline["agent"], ai_enabled=True, ai_poll_seconds=0.01)
    response = client.post(
        "/api/v1/requests", headers=headers, json={"data": {"description": "Заменить 1 розетку."}}
    )
    record = wait_for_status(client, headers, response.json()["request_id"], "processing_failed")
    assert record["last_error"] == code
    assert record["result"] is None
    assert all(client.is_closed for client in pipeline["clients"])
    assert all(session.closed for session in pipeline["sessions"])
    assert "secret detail" not in caplog.text
