"""Async embedding utilities using OpenAI text-embedding-3-large.

Batches texts in groups of 100, retries transient API errors with
exponential back-off (up to 3 attempts), and returns a flat list of
vectors sized to ``settings.embedding_dims``.
"""

from __future__ import annotations

import asyncio

import structlog
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings

logger = structlog.get_logger(__name__)

BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Low-level batch call (retried)
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def _embed_batch(client: AsyncOpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a single batch of texts.  Retried up to 3 times with exponential
    back-off on any exception.

    Returns a list of float vectors in the same order as *texts*.
    """
    logger.debug("Embedding batch", batch_size=len(texts), model=settings.embedding_model)
    response = await client.embeddings.create(
        input=texts,
        model=settings.embedding_model,
        dimensions=settings.embedding_dims,
    )
    # Sort by index to guarantee order (OpenAI preserves order but be safe).
    vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
    return vectors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def embed_chunks(texts: list[str]) -> list[list[float]]:
    """Embed *texts* in batches of :data:`BATCH_SIZE`.

    Creates a single :class:`~openai.AsyncOpenAI` client for the call, splits
    the input into batches, embeds each batch concurrently, and returns a
    flattened list of vectors sized to ``settings.embedding_dims``.

    Raises whatever the OpenAI SDK raises after exhausting retries.
    """
    if not texts:
        return []

    client = AsyncOpenAI(api_key=settings.openai_api_key)

    batches = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    logger.info(
        "Starting embedding",
        total_texts=len(texts),
        total_batches=len(batches),
        model=settings.embedding_model,
    )

    results = await asyncio.gather(*[_embed_batch(client, batch) for batch in batches])

    # Flatten: results is a list-of-list-of-vectors
    vectors: list[list[float]] = [vec for batch_result in results for vec in batch_result]

    logger.info("Embedding complete", total_vectors=len(vectors))
    return vectors
