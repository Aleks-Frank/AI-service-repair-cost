"""ML-сервис: модель эмбеддингов и модель переранжирования за HTTP-API.

Эндпоинты:
    POST /embed   — тексты -> векторы (BAAI/bge-m3, размерность 1024);
    POST /rerank  — запрос + список текстов -> оценки релевантности (BAAI/bge-reranker-v2-m3);
    GET  /health  — состояние сервиса и загруженные модели.

Модели загружаются один раз при старте контейнера и дальше живут в памяти видеокарты.
Поэтому RAG-модулю не нужны torch и CUDA: он просто отправляет HTTP-запросы.

Запуск без Docker (для отладки):
    uvicorn ml_service.main:app --port 8001
"""

import logging
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
# Каждый HTTP-запрос к Hugging Face (проверка версий моделей) логируется на уровне INFO — это шум.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("ml_service")

# Настройки берём из переменных окружения (их задаёт docker-compose.yml).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
DEVICE = os.getenv("DEVICE", "auto")
# Расценки — короткие тексты, 512 токенов хватает с запасом и экономит память.
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "512"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
# Ограничение размера одного запроса, чтобы случайно не отправить в GPU миллион текстов.
MAX_TEXTS_PER_REQUEST = int(os.getenv("MAX_TEXTS_PER_REQUEST", "256"))


# ---------------------------------------------------------------------------
# Схемы запросов и ответов
# ---------------------------------------------------------------------------


class EmbedRequest(BaseModel):
    """Запрос на расчёт векторов."""

    texts: list[str] = Field(min_length=1, max_length=MAX_TEXTS_PER_REQUEST)


class EmbedResponse(BaseModel):
    """Векторы в том же порядке, что и тексты в запросе."""

    model: str
    dim: int
    embeddings: list[list[float]]


class RerankRequest(BaseModel):
    """Запрос на переранжирование: один запрос и список документов-кандидатов."""

    query: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1, max_length=MAX_TEXTS_PER_REQUEST)


class RerankResponse(BaseModel):
    """Оценки релевантности (0..1) в том же порядке, что и documents в запросе."""

    model: str
    scores: list[float]


class HealthResponse(BaseModel):
    """Состояние сервиса."""

    status: str
    device: str
    embedding_model: str
    embedding_dim: int
    reranker_model: str


# ---------------------------------------------------------------------------
# Загрузка моделей
# ---------------------------------------------------------------------------


def resolve_device(device: str) -> str:
    """Выбирает устройство для вычислений.

    Принимает: device — "auto", "cuda" или "cpu".
    Возвращает: "cuda", если запрошено auto и есть видеокарта, иначе запрошенное значение или "cpu".
    """
    if device != "auto":
        return device
    import torch  # тяжёлый импорт — только когда действительно нужен

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_models() -> dict[str, Any]:
    """Загружает модель эмбеддингов и reranker.

    Принимает: ничего (параметры берутся из переменных окружения).
    Возвращает: словарь {"device", "embedder", "reranker", "dim"}.
    """
    from sentence_transformers import CrossEncoder, SentenceTransformer

    device = resolve_device(DEVICE)
    logger.info("Загрузка %s и %s на %s", EMBEDDING_MODEL, RERANKER_MODEL, device)

    embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
    embedder.max_seq_length = MAX_SEQ_LENGTH
    reranker = CrossEncoder(RERANKER_MODEL, device=device, max_length=MAX_SEQ_LENGTH)
    if device == "cuda":
        # Половинная точность на GPU: вдвое меньше памяти и быстрее, качество практически то же.
        embedder.half()
        reranker.model.half()

    # В sentence-transformers >= 6 метод переименован; поддерживаем обе версии.
    get_dim = getattr(embedder, "get_embedding_dimension", None) or embedder.get_sentence_embedding_dimension
    return {"device": device, "embedder": embedder, "reranker": reranker, "dim": get_dim()}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Загружает модели при старте сервиса.

    Если модели уже положены в app.state, повторно не загружаем.

    Принимает: app — экземпляр FastAPI.
    Возвращает: асинхронный контекст (код до yield — старт, после — остановка).
    """
    if not getattr(app.state, "models", None):
        app.state.models = load_models()
        # Прогрев: первый вызов на GPU заметно медленнее (инициализация CUDA-ядер).
        app.state.models["embedder"].encode(["прогрев"])
        app.state.models["reranker"].predict([("прогрев", "прогрев")], show_progress_bar=False)
        logger.info("Модели готовы")
    yield


app = FastAPI(title="ML-сервис AI-сметчика", description="Эмбеддинги и переранжирование", lifespan=lifespan)

# Одна видеокарта на все запросы: выполняем вычисления по очереди,
# чтобы параллельные запросы не боролись за память GPU.
_gpu_lock = threading.Lock()


def _models(request: Request) -> dict[str, Any]:
    """Достаёт загруженные модели из состояния приложения.

    Принимает: request — текущий HTTP-запрос.
    Возвращает: словарь моделей (см. load_models).
    Исключения: 503, если модели ещё не загружены.
    """
    models = getattr(request.app.state, "models", None)
    if not models:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Модели ещё загружаются")
    return models


# ---------------------------------------------------------------------------
# Эндпоинты. Объявлены через def (не async): FastAPI выполнит их в пуле потоков,
# и долгий расчёт на GPU не заблокирует обработку других запросов (например, /health).
# ---------------------------------------------------------------------------


@app.post("/embed", response_model=EmbedResponse)
def embed(body: EmbedRequest, request: Request) -> EmbedResponse:
    """Считает нормированные векторы для списка текстов.

    Принимает: body.texts — список текстов.
    Возвращает: EmbedResponse со списком векторов длины dim.
    """
    models = _models(request)
    with _gpu_lock:
        # normalize_embeddings=True: векторы единичной длины, тогда косинусная
        # близость равна скалярному произведению — так их и сравнивает pgvector.
        vectors = models["embedder"].encode(
            body.texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    return EmbedResponse(model=EMBEDDING_MODEL, dim=models["dim"], embeddings=vectors.astype("float32").tolist())


@app.post("/rerank", response_model=RerankResponse)
def rerank(body: RerankRequest, request: Request) -> RerankResponse:
    """Оценивает, насколько каждый документ подходит к запросу.

    Cross-encoder получает пару «запрос + документ» целиком, поэтому оценивает точнее,
    чем сравнение векторов, но медленнее — его применяют только к короткому списку кандидатов.

    Принимает: body.query — запрос; body.documents — тексты кандидатов.
    Возвращает: RerankResponse с оценками 0..1 в порядке documents.
    """
    models = _models(request)
    pairs = [(body.query, document) for document in body.documents]
    with _gpu_lock:
        scores = models["reranker"].predict(pairs, batch_size=BATCH_SIZE, convert_to_numpy=True, show_progress_bar=False)
    return RerankResponse(model=RERANKER_MODEL, scores=[float(s) for s in scores])


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Сообщает, что сервис жив и какие модели загружены.

    Принимает: request — текущий HTTP-запрос.
    Возвращает: HealthResponse.
    """
    models = _models(request)
    return HealthResponse(
        status="ok",
        device=models["device"],
        embedding_model=EMBEDDING_MODEL,
        embedding_dim=models["dim"],
        reranker_model=RERANKER_MODEL,
    )
