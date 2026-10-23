"""HTTP-клиент к ML-сервису (контейнер ml_service): эмбеддинги и переранжирование.

RAG-модуль не загружает модели сам — он отправляет тексты в ML-сервис
и получает обратно векторы (/embed) или оценки релевантности (/rerank).
"""

import httpx

from rag.config import Settings


class MLServiceError(RuntimeError):
    """ML-сервис недоступен или ответил ошибкой."""


class MLServiceClient:
    """Клиент ML-сервиса поверх httpx."""

    def __init__(self, base_url: str, timeout_s: float = 120.0, embed_batch: int = 64):
        """Создаёт HTTP-клиент.

        Принимает:
            base_url — адрес ML-сервиса, например http://localhost:8001;
            timeout_s — таймаут одного запроса в секундах;
            embed_batch — сколько текстов отправлять в /embed за один запрос.
        Возвращает: ничего.
        """
        # Один httpx.Client переиспользует TCP-соединение между запросами — это быстрее.
        self._http = httpx.Client(base_url=base_url, timeout=timeout_s, trust_env=False)
        self._embed_batch = embed_batch

    def close(self) -> None:
        self._http.close()

    def _post(self, path: str, payload: dict) -> dict:
        """Отправляет POST-запрос и превращает сетевые ошибки в MLServiceError.

        Принимает: path — путь эндпоинта; payload — тело запроса (JSON).
        Возвращает: JSON-ответ в виде словаря.
        """
        try:
            response = self._http.post(path, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise MLServiceError(
                f"ML-сервис {self._http.base_url} недоступен или вернул ошибку: {error}. "
                "Проверьте, что контейнер запущен: docker compose up -d ml_service"
            ) from error
        return response.json()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Считает векторы для текстов, разбивая их на пачки по embed_batch.

        Принимает: texts — список текстов (может быть длинным, например весь справочник).
        Возвращает: список векторов в том же порядке.
        """
        vectors: list[list[float]] = []
        # Пачками: у сервиса есть лимит на число текстов в одном запросе.
        for start in range(0, len(texts), self._embed_batch):
            batch = texts[start : start + self._embed_batch]
            vectors.extend(self._post("/embed", {"texts": batch})["embeddings"])
        return vectors

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Получает оценки релевантности документов запросу.

        Принимает: query — запрос; documents — тексты кандидатов.
        Возвращает: список оценок 0..1 в порядке documents.
        """
        if not documents:
            return []
        return self._post("/rerank", {"query": query, "documents": documents})["scores"]

    def health(self) -> dict:
        """Проверяет состояние ML-сервиса.

        Принимает: ничего.
        Возвращает: ответ /health (модели, устройство, размерность).
        """
        try:
            return self._http.get("/health").raise_for_status().json()
        except httpx.HTTPError as error:
            raise MLServiceError(f"ML-сервис {self._http.base_url} недоступен: {error}") from error


def build_ml_client(settings: Settings) -> MLServiceClient:
    """Создаёт клиент ML-сервиса по настройкам.

    Принимает: settings — настройки RAG-модуля.
    Возвращает: MLServiceClient.
    """
    return MLServiceClient(settings.ml_service_url, settings.ml_timeout_s, settings.ml_embed_batch)
