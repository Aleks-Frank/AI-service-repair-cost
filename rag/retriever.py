"""Гибридный поиск расценок: векторный + полнотекстовый + RRF + переранжирование.

    запрос ──► /embed ──► векторная ветка: top-N по косинусной близости (pgvector) ─┐
          └─────────────► полнотекстовая ветка: top-N по ts_rank (русский стемминг) ─┴► RRF ─► /rerank ─► top-k

Зачем две ветки:
  * векторная находит по смыслу: «отвалилась плитка» ≈ «облицовка стен плиткой»;
  * полнотекстовая ловит точные термины и числа, которые эмбеддинги «размывают»:
    «до 50 мм» и «до 80 мм», «ГВЛ», «ПВХ», «сгон».

Reciprocal Rank Fusion (Cormack et al., 2009) складывает не сами оценки
(у веток они в разных шкалах), а обратные места в списках: score = Σ 1 / (k + rank).
"""

import time
from dataclasses import dataclass, field

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from rag.config import Settings
from rag.db import Rate
from rag.ml_client import MLServiceClient


@dataclass
class SearchHit:
    """Найденная расценка и оценки каждого этапа (по ним видно, почему она в выдаче)."""

    rate: Rate
    rrf_score: float
    # Место в векторной ветке и косинусная близость (None — не попала в top-N ветки).
    dense_rank: int | None = None
    dense_similarity: float | None = None
    # Место в полнотекстовой ветке и балл ts_rank.
    lexical_rank: int | None = None
    lexical_score: float | None = None
    # Оценка cross-encoder'а (если применялось переранжирование).
    rerank_score: float | None = None


@dataclass
class SearchResult:
    """Результат поиска и время этапов в миллисекундах."""

    hits: list[SearchHit]
    reranked: bool
    timings_ms: dict[str, float] = field(default_factory=dict)


# Фильтры записаны как «(параметр IS NULL OR условие)»: если фильтр не передан,
# условие всегда истинно. CAST(... AS int[]) нужен, чтобы Postgres знал тип NULL.
_FILTERS_SQL = """
    (CAST(:book_numbers AS int[]) IS NULL OR r.book_number = ANY(CAST(:book_numbers AS int[])))
    AND (CAST(:units AS text[]) IS NULL OR r.unit = ANY(CAST(:units AS text[])))
    AND (NOT :only_priced OR r.unit_price IS NOT NULL)
"""

# Один SQL-запрос выполняет обе ветки и сливает их через RRF.
_HYBRID_SQL = f"""
WITH q AS (
    SELECT
        CAST(:query_vector AS vector) AS qvec,
        -- plainto_tsquery даёт запрос с логическим И («отвал & плитк & подъезд»):
        -- длинное описание дефекта почти никогда не совпадёт целиком. Меняем И на ИЛИ,
        -- а ts_rank сам поднимет документы, где совпало больше слов.
        CAST(replace(CAST(plainto_tsquery('russian', :query_text) AS text), ' & ', ' | ') AS tsquery) AS tsq
),
dense_top AS (
    -- Ветка 1: ближайшие векторы. <=> — косинусное расстояние (0 — полное совпадение).
    SELECT r.id, r.embedding <=> q.qvec AS distance
    FROM rates r, q
    WHERE {_FILTERS_SQL}
    ORDER BY r.embedding <=> q.qvec
    LIMIT :per_branch
),
dense AS (
    -- Нумеруем отобранные строки: место 1 — самая близкая.
    SELECT id, 1 - distance AS similarity, row_number() OVER (ORDER BY distance, id) AS rnk
    FROM dense_top
),
lexical_top AS (
    -- Ветка 2: полнотекстовый поиск. @@ — «документ соответствует запросу».
    SELECT r.id, ts_rank(r.tsv, q.tsq) AS score
    FROM rates r, q
    WHERE {_FILTERS_SQL} AND r.tsv @@ q.tsq
    ORDER BY score DESC, r.id
    LIMIT :per_branch
),
lexical AS (
    SELECT id, score, row_number() OVER (ORDER BY score DESC, id) AS rnk
    FROM lexical_top
)
SELECT
    COALESCE(d.id, l.id) AS id,
    d.rnk AS dense_rank,
    d.similarity AS dense_similarity,
    l.rnk AS lexical_rank,
    l.score AS lexical_score,
    -- Формула RRF: каждая ветка добавляет 1 / (k + место); нет документа в ветке — 0.
    COALESCE(1.0 / (:rrf_k + d.rnk), 0) + COALESCE(1.0 / (:rrf_k + l.rnk), 0) AS rrf_score
FROM dense d
FULL OUTER JOIN lexical l ON l.id = d.id
ORDER BY rrf_score DESC, id
LIMIT :limit
"""


def _vector_literal(vector: list[float]) -> str:
    """Переводит вектор в текстовый литерал pgvector.

    Принимает: vector — список чисел.
    Возвращает: строку «[0.1,0.2,…]», которую Postgres приводит к типу vector.
    """
    return "[" + ",".join(f"{value:.7g}" for value in vector) + "]"


def _elapsed_ms(start: float) -> float:
    """Считает, сколько миллисекунд прошло с момента start.

    Принимает: start — значение time.perf_counter() в начале этапа.
    Возвращает: миллисекунды, округлённые до 0.1.
    """
    return round((time.perf_counter() - start) * 1000, 1)


class HybridRetriever:
    """Поиск расценок по текстовому описанию работы — публичный интерфейс RAG-модуля."""

    def __init__(self, session: Session, models: MLServiceClient, settings: Settings):
        """Сохраняет зависимости.

        Принимает:
            session — сессия БД pzr_pricing;
            models — ML-сервис (методы embed и rerank);
            settings — настройки (размер веток, k для RRF и т.п.).
        Возвращает: ничего.
        """
        self._session = session
        self._models = models
        self._settings = settings

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_reranker: bool = True,
        book_numbers: list[int] | None = None,
        units: list[str] | None = None,
        only_priced: bool = True,
    ) -> SearchResult:
        """Ищет расценки, наиболее подходящие к описанию работы.

        Принимает:
            query — текст запроса («заделка трещин в стене»);
            top_k — сколько результатов вернуть;
            use_reranker — переранжировать ли результаты через ML-сервис;
            book_numbers — фильтр по справочнику (1 — строительные, 2 — инженерные работы);
            units — фильтр по единицам измерения («м2», «шт», …);
            only_priced — не возвращать расценки без цены.
        Возвращает: SearchResult с расценками по убыванию релевантности.
        """
        timings: dict[str, float] = {}
        # С переранжированием берём из SQL больше кандидатов, чем нужно в ответе:
        # reranker может поднять наверх расценку, которую RRF поставил, скажем, на 20-е место.
        sql_limit = max(top_k, self._settings.rerank_top_n) if use_reranker else top_k

        # --- 1. Вектор запроса через ML-сервис. ---
        start = time.perf_counter()
        query_vector = self._models.embed([query])[0]
        timings["embed"] = _elapsed_ms(start)

        # --- 2. SQL: обе ветки + RRF. ---
        start = time.perf_counter()
        # Параметры HNSW действуют только в текущей транзакции (SET LOCAL):
        #  ef_search — ширина поиска по графу; iterative_scan — если фильтры отсеяли
        #  часть найденного, индекс продолжит поиск, а не вернёт меньше строк.
        self._session.execute(text("SET LOCAL hnsw.ef_search = 100"))
        self._session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
        rows = self._session.execute(
            text(_HYBRID_SQL),
            {
                "query_vector": _vector_literal(query_vector),
                "query_text": query,
                "book_numbers": book_numbers,
                "units": units,
                "only_priced": only_priced,
                "per_branch": self._settings.search_candidates_per_branch,
                "rrf_k": self._settings.rrf_k,
                "limit": sql_limit,
            },
        ).mappings().all()
        timings["sql"] = _elapsed_ms(start)

        # --- 3. Полные данные найденных расценок одним запросом. ---
        ids = [row["id"] for row in rows]
        rates_by_id = {rate.id: rate for rate in self._session.scalars(select(Rate).where(Rate.id.in_(ids)))}
        hits = [
            SearchHit(
                rate=rates_by_id[row["id"]],
                rrf_score=float(row["rrf_score"]),
                dense_rank=row["dense_rank"],
                dense_similarity=None if row["dense_similarity"] is None else float(row["dense_similarity"]),
                lexical_rank=row["lexical_rank"],
                lexical_score=None if row["lexical_score"] is None else float(row["lexical_score"]),
            )
            for row in rows
        ]

        # --- 4. Переранжирование кандидатов cross-encoder'ом в ML-сервисе. ---
        if use_reranker and hits:
            start = time.perf_counter()
            scores = self._models.rerank(query, [hit.rate.search_text for hit in hits])
            for hit, score in zip(hits, scores, strict=True):
                hit.rerank_score = score
            hits.sort(key=lambda hit: hit.rerank_score, reverse=True)  # type: ignore[arg-type, return-value]
            timings["rerank"] = _elapsed_ms(start)

        # Закрываем транзакцию, чтобы SET LOCAL не влиял на следующие запросы в этой сессии.
        # Используем commit, а не rollback: транзакция только читала данные, а после rollback
        # SQLAlchemy пометила бы загруженные расценки устаревшими и перечитывала бы каждую из БД.
        self._session.commit()
        timings["total"] = round(sum(timings.values()), 1)
        return SearchResult(hits[:top_k], use_reranker and bool(hits), timings)
