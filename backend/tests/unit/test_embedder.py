"""Unit tests for app.ingestion.embedder."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ingestion.embedder import BATCH_SIZE, _embed_batch, embed_chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_embedding(dim: int = 1536) -> list[float]:
    return [0.01] * dim


def _make_openai_response(texts: list[str]) -> MagicMock:
    """Build a mock OpenAI Embeddings response for *texts*."""
    response = MagicMock()
    data = []
    for i, _ in enumerate(texts):
        item = MagicMock()
        item.index = i
        item.embedding = _fake_embedding()
        data.append(item)
    response.data = data
    return response


# ---------------------------------------------------------------------------
# _embed_batch
# ---------------------------------------------------------------------------


class TestEmbedBatch:
    @pytest.mark.asyncio
    async def test_returns_vectors_for_each_text(self):
        texts = ["hello", "world", "foo"]
        client = AsyncMock()
        client.embeddings.create = AsyncMock(return_value=_make_openai_response(texts))

        vectors = await _embed_batch(client, texts)

        assert len(vectors) == len(texts)
        assert all(len(v) == 1536 for v in vectors)

    @pytest.mark.asyncio
    async def test_calls_openai_with_correct_model(self):
        from app.core.config import settings  # noqa: PLC0415

        texts = ["test"]
        client = AsyncMock()
        client.embeddings.create = AsyncMock(return_value=_make_openai_response(texts))

        await _embed_batch(client, texts)

        client.embeddings.create.assert_called_once()
        call_kwargs = client.embeddings.create.call_args.kwargs
        assert call_kwargs.get("model") == settings.embedding_model
        assert call_kwargs.get("dimensions") == settings.embedding_dims
        assert call_kwargs.get("input") == texts

    @pytest.mark.asyncio
    async def test_retries_on_transient_failure_then_succeeds(self):
        """First call raises, second succeeds — tenacity should retry and return vectors."""
        texts = ["retry test"]
        client = AsyncMock()
        success_response = _make_openai_response(texts)

        call_count = 0

        async def _flaky_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("transient error")
            return success_response

        client.embeddings.create = _flaky_create

        # Suppress the real sleep introduced by tenacity's exponential backoff.
        with patch("asyncio.sleep", new_callable=AsyncMock):
            vectors = await _embed_batch(client, texts)

        assert call_count == 2
        assert len(vectors) == 1
        assert len(vectors[0]) == 1536

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        """After 3 failures the exception should propagate."""
        client = AsyncMock()
        client.embeddings.create = AsyncMock(side_effect=Exception("always fails"))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(Exception, match="always fails"):
                await _embed_batch(client, ["text"])


# ---------------------------------------------------------------------------
# embed_chunks
# ---------------------------------------------------------------------------


class TestEmbedChunks:
    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_list(self):
        result = await embed_chunks([])
        assert result == []

    @pytest.mark.asyncio
    async def test_batches_into_groups_of_batch_size(self):
        """With BATCH_SIZE texts per batch, two batches should be created for
        BATCH_SIZE + 1 texts."""
        total = BATCH_SIZE + 1
        texts = [f"text {i}" for i in range(total)]

        batch_calls: list[list[str]] = []

        async def _fake_embed_batch(client, batch: list[str]) -> list[list[float]]:
            batch_calls.append(batch)
            return [_fake_embedding() for _ in batch]

        with patch("app.ingestion.embedder._embed_batch", side_effect=_fake_embed_batch):
            vectors = await embed_chunks(texts)

        assert len(batch_calls) == 2
        assert len(batch_calls[0]) == BATCH_SIZE
        assert len(batch_calls[1]) == 1
        assert len(vectors) == total

    @pytest.mark.asyncio
    async def test_returns_correct_number_of_vectors(self):
        texts = ["a", "b", "c"]

        async def _fake_embed_batch(client, batch: list[str]) -> list[list[float]]:
            return [_fake_embedding() for _ in batch]

        with patch("app.ingestion.embedder._embed_batch", side_effect=_fake_embed_batch):
            vectors = await embed_chunks(texts)

        assert len(vectors) == len(texts)

    @pytest.mark.asyncio
    async def test_each_vector_has_correct_dimension(self):
        texts = ["hello", "world"]

        async def _fake_embed_batch(client, batch: list[str]) -> list[list[float]]:
            return [_fake_embedding(1536) for _ in batch]

        with patch("app.ingestion.embedder._embed_batch", side_effect=_fake_embed_batch):
            vectors = await embed_chunks(texts)

        assert all(len(v) == 1536 for v in vectors)

    @pytest.mark.asyncio
    async def test_single_batch_when_below_batch_size(self):
        texts = [f"text {i}" for i in range(BATCH_SIZE - 1)]
        batch_calls: list[list[str]] = []

        async def _fake_embed_batch(client, batch: list[str]) -> list[list[float]]:
            batch_calls.append(batch)
            return [_fake_embedding() for _ in batch]

        with patch("app.ingestion.embedder._embed_batch", side_effect=_fake_embed_batch):
            await embed_chunks(texts)

        assert len(batch_calls) == 1
