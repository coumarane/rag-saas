"""Reciprocal Rank Fusion (RRF) for merging dense and sparse retrieval results.

RRF score for a chunk across N ranked lists:
    score = sum(1 / (k + rank_i))   for each list i that contains the chunk

The standard value for *k* is 60 (Cormack et al., 2009), configurable via
``settings.rrf_k``.
"""

from __future__ import annotations

from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.searcher import ScoredChunk

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    dense_results: list[ScoredChunk],
    sparse_results: list[ScoredChunk],
    k: int = 60,
) -> list[ScoredChunk]:
    """Merge dense and sparse result lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    dense_results:
        Chunks from dense (cosine) search, ordered by descending score.
    sparse_results:
        Chunks from sparse (BM25) search, ordered by descending score.
    k:
        RRF smoothing constant (default 60).  Lower values give higher weight
        to top-ranked items.

    Returns
    -------
    list[ScoredChunk]
        Deduplicated chunks sorted by descending RRF score.  Each chunk's
        ``score`` field is replaced with its RRF score.  At most
        ``settings.top_k_initial`` chunks are returned.
    """
    # Map chunk_id → accumulated RRF score and the best ScoredChunk instance
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, ScoredChunk] = {}

    for result_list in (dense_results, sparse_results):
        for rank, chunk in enumerate(result_list, start=1):
            rrf_contribution = 1.0 / (k + rank)
            if chunk.chunk_id in rrf_scores:
                rrf_scores[chunk.chunk_id] += rrf_contribution
            else:
                rrf_scores[chunk.chunk_id] = rrf_contribution
                # Store the first (dense) version of the chunk; metadata is identical
                chunk_map[chunk.chunk_id] = chunk

    # Sort by descending RRF score
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    # Build output, replacing score with RRF score, capped at top_k_initial
    fused: list[ScoredChunk] = []
    for cid in sorted_ids[: settings.top_k_initial]:
        original = chunk_map[cid]
        fused.append(
            ScoredChunk(
                chunk_id=original.chunk_id,
                score=rrf_scores[cid],
                text=original.text,
                page_number=original.page_number,
                section_title=original.section_title,
                doc_id=original.doc_id,
                chunk_index=original.chunk_index,
            )
        )

    logger.debug(
        "RRF fusion complete",
        dense_count=len(dense_results),
        sparse_count=len(sparse_results),
        merged_count=len(fused),
        k=k,
    )
    return fused
