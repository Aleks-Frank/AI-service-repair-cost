import base64
import binascii
import io
import json
import warnings
from contextlib import ExitStack
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from app.core.config import Settings
from app.core.errors import ServiceError


class AgentInput(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, allow_inf_nan=False)

    description: str = Field(default="", max_length=4000)
    object: dict[str, JsonValue] = Field(default_factory=dict)
    images: list[str] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def check_content(self) -> "AgentInput":
        if not self.description.strip() and not self.images:
            raise ValueError("Expected a description or photographs")
        if len(self.description_with_context()) > 4000:
            raise ValueError("Description and object context exceed 4000 characters")
        return self

    def description_with_context(self) -> str:
        description = self.description.strip()
        if self.object:
            context = json.dumps(self.object, ensure_ascii=False, allow_nan=False)
            description += f"\nХарактеристики объекта: {context}"
        return description

    def image_bytes(self) -> list[bytes]:
        from PIL import Image

        images = []
        for encoded in self.images:
            try:
                data = base64.b64decode(encoded, validate=True)
                if not data or len(data) > 5_242_880:
                    raise ValueError("Invalid image size")
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(io.BytesIO(data)) as image:
                        if image.format not in {"JPEG", "PNG", "WEBP"}:
                            raise ValueError("Unsupported image format")
                        if image.width * image.height > 16_000_000:
                            raise ValueError("Too many pixels")
                        image.verify()
            except (
                ValueError,
                binascii.Error,
                OSError,
                Image.DecompressionBombError,
                Image.DecompressionBombWarning,
            ):
                raise ServiceError(
                    422, "invalid_image", "Use base64 JPEG, PNG or WebP, up to 5 MiB and 16 MP"
                ) from None
            images.append(data)
        return images


def agent_input(payload: dict, source: str, settings: Settings) -> AgentInput:
    value = payload
    if source == "bitrix":
        for part in settings.bitrix_input_path.split("."):
            value = value.get(part) if isinstance(value, dict) else None
    try:
        parsed = AgentInput.model_validate(value)
    except ValidationError:
        raise ServiceError(
            422,
            "invalid_agent_input",
            "Expected description, optional object and up to 4 base64 images; text limit is 4000",
        ) from None
    parsed.image_bytes()
    return parsed


class Agent(Protocol):
    def __call__(self, data: AgentInput) -> dict: ...

    def close(self) -> None: ...


class EstimatorAgent:
    def __init__(self, settings: Settings) -> None:
        from llm.estimator.config import EstimatorSettings
        from rag.config import Settings as RAGSettings
        from sqlalchemy import URL, create_engine
        from sqlalchemy.orm import sessionmaker

        self.estimator = EstimatorSettings(_env_file=settings.ai_env_file)
        self.rag = RAGSettings(_env_file=settings.ai_env_file)
        self.engine = create_engine(
            URL.create(
                "postgresql+psycopg",
                username=self.rag.postgres_user,
                password=self.rag.postgres_password,
                host=self.rag.postgres_host,
                port=self.rag.postgres_port,
                database=self.rag.postgres_db,
            ),
            pool_pre_ping=True,
            connect_args={"connect_timeout": 10, "options": "-c statement_timeout=60000"},
        )
        self.sessions = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.prices_engine = create_engine(
            URL.create(
                "postgresql+psycopg",
                username=self.rag.postgres_user,
                password=self.rag.postgres_password,
                host=self.rag.prices_host,
                port=self.rag.prices_port,
                database=self.rag.prices_db,
            ),
            pool_pre_ping=True,
            connect_args={"connect_timeout": 10, "options": "-c statement_timeout=60000"},
        )
        self.prices_sessions = sessionmaker(bind=self.prices_engine, expire_on_commit=False)

    def __call__(self, data: AgentInput) -> dict:
        from llm.estimator.llm_client import InputTooLong, LLMServiceClient, LLMServiceError
        from llm.estimator.pipeline import analyze_case, prepare_image
        from rag.ml_client import MLServiceError, build_ml_client
        from rag.retriever import HybridRetriever, MaterialRetriever
        from sqlalchemy.exc import SQLAlchemyError

        try:
            if len(data.images) > self.estimator.max_images:
                raise ServiceError(422, "too_many_images", "Estimator image limit exceeded")
            images = [
                prepare_image(image, self.estimator.max_image_pixels)
                for image in data.image_bytes()
            ]
            with ExitStack() as stack:
                llm = LLMServiceClient(self.estimator.llm_service_url, self.estimator.llm_timeout_s)
                stack.callback(llm.close)
                ml = build_ml_client(self.rag)
                stack.callback(ml.close)
                session = stack.enter_context(self.sessions())
                prices_session = stack.enter_context(self.prices_sessions())
                result, _ = analyze_case(
                    images,
                    data.description_with_context(),
                    llm,
                    HybridRetriever(session, ml, self.rag),
                    self.estimator.candidates_per_finding,
                    self.estimator.observation_max_tokens,
                    self.estimator.selection_max_tokens,
                    material_retriever=MaterialRetriever(prices_session, ml, self.rag),
                    material_candidates=self.estimator.material_candidates,
                    material_selection_max_tokens=self.estimator.material_selection_max_tokens,
                )
                return result
        except InputTooLong:
            raise ServiceError(422, "agent_input_too_long", "Model input is too long") from None
        except LLMServiceError:
            raise ServiceError(502, "llm_failed", "LLM request failed") from None
        except MLServiceError:
            raise ServiceError(502, "ml_failed", "Embedding or reranking request failed") from None
        except SQLAlchemyError:
            raise ServiceError(503, "rag_database_failed", "RAG database request failed") from None

    def close(self) -> None:
        self.engine.dispose()
        self.prices_engine.dispose()
