"""High-level retrieval pipeline: embed → search → fuse → filter → rerank.

The single public function :func:`retrieve` orchestrates the full flow:

1. Embed the question with OpenAI text-embedding-3-large.
2. Run dense (cosine) and sparse (BM25) searches concurrently.
3. Merge results with Reciprocal Rank Fusion.
4. Filter out chunks below ``settings.score_threshold``.
5. Rerank surviving chunks with CrossEncoder, returning the top-6.
"""

from __future__ import annotations

import asyncio

from qdrant_client import AsyncQdrantClient

from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.embedder import embed_chunks
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import rerank
from app.retrieval.searcher import ScoredChunk, dense_search, sparse_search

logger = get_logger(__name__)


async def retrieve(
    question: str,
    doc_id: str,
    qdrant_client: AsyncQdrantClient,
) -> list[ScoredChunk]:
    """Retrieve the most relevant chunks for *question* from *doc_id*.

    Parameters
    ----------
    question:
        The user's question (or an expanded query variant).
    doc_id:
        Restrict retrieval to chunks belonging to this document.
    qdrant_client:
        Async Qdrant client injected by the caller (FastAPI dependency or
        Celery task).

    Returns
    -------
    list[ScoredChunk]
        Up to ``settings.top_k_reranked`` chunks sorted by descending
        cross-encoder score.  Chunks whose RRF score falls below
        ``settings.score_threshold`` are filtered out before reranking.
    """
    logger.info(
        "Retrieval pipeline started",
        doc_id=doc_id,
        question_len=len(question),
    )

    # 1. Embed the question
    embeddings = await embed_chunks([question])
    query_vector: list[float] = embeddings[0]

    # 2. Dense and sparse searches in parallel
    dense_results, sparse_results = await asyncio.gather(
        dense_search(qdrant_client, query_vector, doc_id, settings.top_k_initial),
        sparse_search(qdrant_client, question, doc_id, settings.top_k_initial),
    )

    logger.debug(
        "Search complete",
        doc_id=doc_id,
        dense_hits=len(dense_results),
        sparse_hits=len(sparse_results),
    )

    # 3. RRF fusion
    fused = reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        k=settings.rrf_k,
    )

    # 4. Filter by score threshold
    filtered = [chunk for chunk in fused if chunk.score >= settings.score_threshold]

    logger.debug(
        "Score threshold filtering applied",
        doc_id=doc_id,
        before=len(fused),
        after=len(filtered),
        threshold=settings.score_threshold,
    )

    if not filtered:
        logger.info(
            "No chunks survived score threshold filter",
            doc_id=doc_id,
            threshold=settings.score_threshold,
        )
        return []

    # 5. Rerank and return top-k
    final_chunks = await rerank(question, filtered, top_k=settings.top_k_reranked)

    logger.info(
        "Retrieval pipeline complete",
        doc_id=doc_id,
        returned=len(final_chunks),
    )
    return final_chunks
