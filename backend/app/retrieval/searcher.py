"""Dense and sparse search wrappers returning typed ScoredChunk objects.

Both functions delegate to ``app.core.vector_store`` and convert the raw
``list[dict]`` results into strongly-typed :class:`ScoredChunk` dataclasses.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient

from app.core import vector_store
from app.core.logging import get_logger

logger = get_logger(__name__)

# Hash-based sparse vocabulary size — must match workers/tasks.py
_VOCAB_SIZE = 65536


@dataclass
class ScoredChunk:
    """A single retrieved chunk with its retrieval score and metadata."""

    chunk_id: str
    score: float
    text: str
    page_number: int | None
    section_title: str | None
    doc_id: str
    chunk_index: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _payload_to_chunk(hit: dict) -> ScoredChunk:
    """Convert a raw Qdrant hit dict to a :class:`ScoredChunk`."""
    payload = hit["payload"] or {}
    return ScoredChunk(
        chunk_id=hit["id"],
        score=hit["score"],
        text=payload.get("text", ""),
        page_number=payload.get("page_number"),
        section_title=payload.get("section_title"),
        doc_id=payload.get("doc_id", ""),
        chunk_index=payload.get("chunk_index", 0),
    )


def _build_sparse_vector(query_text: str) -> dict:
    """Build a TF-based sparse vector from *query_text*.

    Mirrors the ``_compute_sparse_vector`` implementation in
    ``workers/tasks.py``: each unique normalised token is hashed to an index
    in ``[0, 65535]`` and the value is its term frequency (count / total).
    Hash collisions are resolved by summing TF contributions.

    Returns a dict with keys ``indices`` (list[int]) and ``values``
    (list[float]).
    """
    tokens = query_text.lower().split()
    if not tokens:
        return {"indices": [], "values": []}

    counts = Counter(tokens)
    total = len(tokens)

    index_value: dict[int, float] = {}
    for term, count in counts.items():
        idx = hash(term) % _VOCAB_SIZE
        tf = count / total
        index_value[idx] = index_value.get(idx, 0.0) + tf

    sorted_items = sorted(index_value.items())
    return {
        "indices": [item[0] for item in sorted_items],
        "values": [item[1] for item in sorted_items],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def dense_search(
    client: AsyncQdrantClient,
    query_vector: list[float],
    doc_id: str,
    top_k: int = 20,
) -> list[ScoredChunk]:
    """Perform a dense (cosine) vector search for *doc_id* and return typed results.

    Parameters
    ----------
    client:
        Async Qdrant client.
    query_vector:
        Embedded query vector (1536-dimensional for text-embedding-3-large).
    doc_id:
        Only return chunks belonging to this document.
    top_k:
        Maximum number of results to return.

    Returns
    -------
    list[ScoredChunk]
        Hits sorted by descending cosine similarity score.
    """
    hits = await vector_store.search_dense(client, query_vector, doc_id, top_k)
    chunks = [_payload_to_chunk(h) for h in hits]
    logger.debug(
        "Dense search results converted",
        doc_id=doc_id,
        top_k=top_k,
        returned=len(chunks),
    )
    return chunks


async def sparse_search(
    client: AsyncQdrantClient,
    query_text: str,
    doc_id: str,
    top_k: int = 20,
) -> list[ScoredChunk]:
    """Perform a sparse (BM25-style) search for *doc_id* and return typed results.

    Builds the sparse vector from *query_text* using the same hash-based TF
    approach as the ingestion pipeline, then delegates to
    :func:`~app.core.vector_store.search_sparse`.

    Parameters
    ----------
    client:
        Async Qdrant client.
    query_text:
        Raw query string; tokenised and hashed internally.
    doc_id:
        Only return chunks belonging to this document.
    top_k:
        Maximum number of results to return.

    Returns
    -------
    list[ScoredChunk]
        Hits sorted by descending sparse similarity score.
    """
    sparse_vector = _build_sparse_vector(query_text)
    hits = await vector_store.search_sparse(client, sparse_vector, doc_id, top_k)
    chunks = [_payload_to_chunk(h) for h in hits]
    logger.debug(
        "Sparse search results converted",
        doc_id=doc_id,
        top_k=top_k,
        returned=len(chunks),
    )
    return chunks
