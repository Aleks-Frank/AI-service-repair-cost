import base64
import io
import json
from datetime import date
from decimal import Decimal

import httpx
import pytest
from llm.estimator.llm_client import LLMServiceClient
from PIL import Image
from rag.db import Rate
from rag.ml_client import MLServiceClient
from rag.prices_db import Material, Supplier
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
    state = {"calls": [], "clients": [], "sessions": [], "failure": None, "materials": False}

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
            if state["materials"]:
                selection["works"][0]["materials"] = [
                    {
                        "query": "Розетка с заземлением",
                        "qty_per_work_unit": 3,
                        "unit": "шт",
                        "reason": "Тестовая потребность",
                    }
                ]
            if '"NEEDS"' in prompt:
                answer = {"choices": [{"need_id": "m1", "article": "ABC", "reason": "Подходит"}]}
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
    suppliers = [
        Supplier(
            code=code,
            name=code,
            active=True,
            article_is_manufacturer=True,
            priority=0,
            price_valid_days=30,
            price_date=date.today(),
        )
        for code in ("first", "second")
    ]
    materials = [
        Material(
            id=index,
            supplier_code=supplier,
            article="ABC",
            article_norm="abc",
            name="Розетка",
            category="Электрика",
            unit=unit,
            unit_raw=unit,
            price=Decimal(price),
            base_unit="шт",
            base_price=Decimal(base_price),
            units_per_price_unit=Decimal(pack),
            sold_by_pack=packed,
            stock=Decimal("100"),
            params={},
            search_text="Розетка",
            source_row=9,
        )
        for index, supplier, unit, price, base_price, pack, packed in (
            (7, "first", "шт", "10", "10", "1", False),
            (8, "second", "уп", "7.50", "1.50", "5", True),
        )
    ]

    class PricesSession(Session):
        def execute(self, statement, params=None):
            if state["failure"] == "prices_db":
                raise OperationalError("query", {}, RuntimeError("secret detail"))
            return self

        def all(self):
            return [{**super().all()[0], "id": 7}]

        def scalars(self, statement):
            if statement.column_descriptions[0]["entity"] is Supplier:
                return suppliers
            if "materials.supplier_code !=" in str(statement):
                return [materials[1]]
            return [materials[0]]

        def get(self, model, primary_key):
            return next(item for item in materials if item.id == primary_key)

    agent = EstimatorAgent(settings)
    agent.sessions = Session
    agent.prices_sessions = PricesSession
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


def test_api_prices_materials_selects_supplier_and_rounds_packages(
    client_factory, headers, pipeline
):
    pipeline["materials"] = True
    client = client_factory(agent=pipeline["agent"], ai_enabled=True, ai_poll_seconds=0.01)
    response = client.post(
        "/api/v1/requests", headers=headers, json={"data": {"description": "Заменить 1 розетку."}}
    )
    record = wait_for_status(client, headers, response.json()["request_id"], "result_ready")
    result = record["result"]
    assert result["materials_priced"] is True
    assert result["materials_not_found"] == []
    material = result["materials"][0]
    assert material["supplier"] == "second"
    assert material["need_qty"] == "3"
    assert material["buy_qty"] == "1"
    assert material["amount_rub"] == "7.50"
    assert result["works_total_rub"] == "125.50"
    assert result["materials_total_rub"] == "7.50"
    assert result["total_rub"] == "133.00"
    assert sum(path == "/generate" for path, _ in pipeline["calls"]) == 3
    assert all(client.is_closed for client in pipeline["clients"])
    assert all(session.closed for session in pipeline["sessions"])


def test_material_database_failure_is_visible(client_factory, headers, pipeline):
    pipeline.update(materials=True, failure="prices_db")
    client = client_factory(agent=pipeline["agent"], ai_enabled=True, ai_poll_seconds=0.01)
    response = client.post(
        "/api/v1/requests", headers=headers, json={"data": {"description": "Заменить 1 розетку."}}
    )
    record = wait_for_status(client, headers, response.json()["request_id"], "processing_failed")
    assert record["last_error"] == "rag_database_failed"
    assert record["result"] is None
