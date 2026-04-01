"""CrossEncoder reranker using sentence-transformers.

The model is loaded once (lazily) and cached as a module-level singleton so
the expensive ``from_pretrained`` download only happens on the first call.
The synchronous ``CrossEncoder.predict`` is offloaded to a thread-pool
executor so it does not block the asyncio event loop.

The ``sentence_transformers`` import is also deferred to first use so that
importing this module in test environments without PyTorch/CUDA does not fail
at collection time.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.searcher import ScoredChunk

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_reranker: CrossEncoder | None = None

# Single-thread executor for the blocking CrossEncoder.predict call.
# Using one thread prevents multiple concurrent rerank calls from thrashing
# the CPU / GPU with competing model inference.
_executor = ThreadPoolExecutor(max_workers=1)


def _get_reranker() -> "CrossEncoder":
    """Return the cached CrossEncoder, loading it on first call.

    The ``sentence_transformers`` import is deferred here so that importing
    this module does not trigger the heavy torch/transformers dependency chain
    at collection time in environments where only the CPU-less test extras are
    installed.
    """
    global _reranker  # noqa: PLW0603
    if _reranker is None:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        logger.info("Loading CrossEncoder reranker model", model=settings.reranker_model)
        _reranker = CrossEncoder(settings.reranker_model)
        logger.info("CrossEncoder reranker model loaded", model=settings.reranker_model)
    return _reranker


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def rerank(
    query: str,
    chunks: list[ScoredChunk],
    top_k: int = 6,
) -> list[ScoredChunk]:
    """Rerank *chunks* against *query* using a cross-encoder model.

    The model runs synchronously in a thread executor so the event loop stays
    free during inference.  Each chunk's ``score`` is replaced with the
    cross-encoder logit score.

    Parameters
    ----------
    query:
        The user's original (or expanded) query string.
    chunks:
        Candidate chunks to rerank (typically the RRF-fused top-20).
    top_k:
        Number of highest-scored chunks to return.

    Returns
    -------
    list[ScoredChunk]
        Top *top_k* chunks sorted by descending cross-encoder score.
    """
    if not chunks:
        return []

    model = _get_reranker()
    pairs = [(query, chunk.text) for chunk in chunks]

    loop = asyncio.get_running_loop()
    scores: list[float] = await loop.run_in_executor(
        _executor,
        model.predict,
        pairs,
    )

    # Attach the reranker score and sort descending
    scored = sorted(
        zip(scores, chunks),
        key=lambda pair: pair[0],
        reverse=True,
    )

    results: list[ScoredChunk] = []
    for score, chunk in scored[:top_k]:
        results.append(
            ScoredChunk(
                chunk_id=chunk.chunk_id,
                score=float(score),
                text=chunk.text,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                doc_id=chunk.doc_id,
                chunk_index=chunk.chunk_index,
            )
        )

    logger.debug(
        "Reranking complete",
        query_len=len(query),
        input_chunks=len(chunks),
        returned=len(results),
        top_score=results[0].score if results else None,
    )
    return results
