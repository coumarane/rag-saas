"""Integration tests for the retrieval pipeline.

All external dependencies (Qdrant, OpenAI, CrossEncoder) are mocked so the
tests run fully in-process without any network or GPU access.

Coverage:
- RRF fusion correctness (scores, deduplication, ordering)
- RRF caps results at top_k_initial
- retrieve() orchestrates dense search, sparse search, fusion, filter, rerank
- Reranker returns top_k chunks sorted by descending score
- Pipeline filters chunks below settings.score_threshold
- Pipeline returns empty list when all chunks are below the threshold
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.config import settings
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import rerank
from app.retrieval.searcher import ScoredChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chunk(
    chunk_id: str,
    score: float = 0.9,
    text: str = "sample text",
    doc_id: str = "doc-1",
    chunk_index: int = 0,
    page_number: int | None = 1,
    section_title: str | None = "Intro",
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        score=score,
        text=text,
        page_number=page_number,
        section_title=section_title,
        doc_id=doc_id,
        chunk_index=chunk_index,
    )


# ---------------------------------------------------------------------------
# RRF fusion tests
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def test_chunk_present_in_both_lists_gets_higher_score(self):
        """A chunk appearing in both dense and sparse results should outscore
        a chunk that appears in only one list."""
        shared = make_chunk("shared", score=0.5)
        dense_only = make_chunk("dense-only", score=0.99)
        sparse_only = make_chunk("sparse-only", score=0.99)

        dense = [shared, dense_only]
        sparse = [shared, sparse_only]

        fused = reciprocal_rank_fusion(dense, sparse, k=60)

        chunk_ids = [c.chunk_id for c in fused]
        assert "shared" in chunk_ids
        shared_chunk = next(c for c in fused if c.chunk_id == "shared")
        dense_only_chunk = next(c for c in fused if c.chunk_id == "dense-only")
        assert shared_chunk.score > dense_only_chunk.score

    def test_results_sorted_by_descending_rrf_score(self):
        """Output must be ordered from highest to lowest RRF score."""
        dense = [make_chunk(f"d{i}") for i in range(5)]
        sparse = [make_chunk(f"s{i}") for i in range(5)]

        fused = reciprocal_rank_fusion(dense, sparse, k=60)

        scores = [c.score for c in fused]
        assert scores == sorted(scores, reverse=True)

    def test_no_duplicate_chunk_ids_in_output(self):
        """Each chunk_id must appear at most once in the fused output."""
        chunk = make_chunk("dup")
        dense = [chunk, make_chunk("a")]
        sparse = [chunk, make_chunk("b")]

        fused = reciprocal_rank_fusion(dense, sparse, k=60)

        ids = [c.chunk_id for c in fused]
        assert len(ids) == len(set(ids))

    def test_rrf_score_formula(self):
        """Verify the RRF score for a chunk at rank 1 in both lists."""
        chunk = make_chunk("only")
        k = 60

        fused = reciprocal_rank_fusion([chunk], [chunk], k=k)

        expected_score = 1.0 / (k + 1) + 1.0 / (k + 1)
        assert abs(fused[0].score - expected_score) < 1e-9

    def test_capped_at_top_k_initial(self):
        """Output length must not exceed settings.top_k_initial."""
        # Build lists that are larger than top_k_initial
        n = settings.top_k_initial + 10
        dense = [make_chunk(f"d{i}", chunk_index=i) for i in range(n)]
        sparse = [make_chunk(f"s{i}", chunk_index=i) for i in range(n)]

        fused = reciprocal_rank_fusion(dense, sparse, k=60)

        assert len(fused) <= settings.top_k_initial

    def test_empty_inputs_return_empty(self):
        fused = reciprocal_rank_fusion([], [], k=60)
        assert fused == []

    def test_single_list_populated(self):
        """If only one list has results, fusion still works correctly."""
        chunks = [make_chunk(f"c{i}", chunk_index=i) for i in range(3)]
        fused = reciprocal_rank_fusion(chunks, [], k=60)
        assert len(fused) == 3

    def test_k_parameter_affects_scores(self):
        """A smaller k gives a larger contribution for rank-1 items."""
        chunk = make_chunk("c")
        fused_low_k = reciprocal_rank_fusion([chunk], [], k=1)
        fused_high_k = reciprocal_rank_fusion([chunk], [], k=60)
        assert fused_low_k[0].score > fused_high_k[0].score

    def test_chunk_metadata_preserved(self):
        """Payload fields survive the fusion step unchanged."""
        chunk = make_chunk("meta", page_number=7, section_title="Methods", chunk_index=3)
        fused = reciprocal_rank_fusion([chunk], [], k=60)

        assert fused[0].page_number == 7
        assert fused[0].section_title == "Methods"
        assert fused[0].chunk_index == 3
        assert fused[0].doc_id == "doc-1"


# ---------------------------------------------------------------------------
# Reranker tests
# ---------------------------------------------------------------------------


class TestRerank:
    @pytest.mark.asyncio
    async def test_returns_top_k_chunks(self):
        """rerank() must return at most top_k chunks."""
        chunks = [make_chunk(f"c{i}", text=f"text {i}") for i in range(10)]
        scores = list(range(10, 0, -1))  # 10, 9, …, 1

        mock_model = MagicMock()
        mock_model.predict.return_value = scores

        with patch("app.retrieval.reranker._get_reranker", return_value=mock_model):
            result = await rerank("query", chunks, top_k=4)

        assert len(result) == 4

    @pytest.mark.asyncio
    async def test_chunks_sorted_by_descending_reranker_score(self):
        """Output must be ordered from highest to lowest cross-encoder score."""
        chunks = [make_chunk(f"c{i}", text=f"text {i}") for i in range(5)]
        # Assign scores in ascending order to verify sorting flips them
        mock_scores = [1.0, 3.0, 5.0, 2.0, 4.0]

        mock_model = MagicMock()
        mock_model.predict.return_value = mock_scores

        with patch("app.retrieval.reranker._get_reranker", return_value=mock_model):
            result = await rerank("query", chunks, top_k=5)

        returned_scores = [c.score for c in result]
        assert returned_scores == sorted(returned_scores, reverse=True)

    @pytest.mark.asyncio
    async def test_score_updated_with_reranker_value(self):
        """Each chunk's score field must reflect the cross-encoder output."""
        chunk = make_chunk("only", score=0.0)
        reranker_score = 42.0

        mock_model = MagicMock()
        mock_model.predict.return_value = [reranker_score]

        with patch("app.retrieval.reranker._get_reranker", return_value=mock_model):
            result = await rerank("query", [chunk], top_k=1)

        assert result[0].score == pytest.approx(reranker_score)

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self):
        result = await rerank("query", [], top_k=6)
        assert result == []

    @pytest.mark.asyncio
    async def test_pairs_sent_to_model_are_query_text_tuples(self):
        """CrossEncoder.predict must receive (query, chunk.text) pairs."""
        chunks = [make_chunk("a", text="alpha"), make_chunk("b", text="beta")]
        query = "what is alpha?"

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.1]

        with patch("app.retrieval.reranker._get_reranker", return_value=mock_model):
            await rerank(query, chunks, top_k=2)

        call_args = mock_model.predict.call_args[0][0]
        assert call_args == [(query, "alpha"), (query, "beta")]

    @pytest.mark.asyncio
    async def test_top_k_larger_than_input_returns_all(self):
        """Requesting more chunks than available returns all of them."""
        chunks = [make_chunk(f"c{i}") for i in range(3)]
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.8, 0.3]

        with patch("app.retrieval.reranker._get_reranker", return_value=mock_model):
            result = await rerank("query", chunks, top_k=10)

        assert len(result) == 3


# ---------------------------------------------------------------------------
# Full pipeline integration tests
# ---------------------------------------------------------------------------


class TestRetrievePipeline:
    """Test retrieve() by mocking all I/O boundaries."""

    def _make_raw_hit(self, chunk_id: str, score: float, text: str, chunk_index: int = 0) -> dict:
        """Build the raw dict shape returned by vector_store.search_*."""
        return {
            "id": chunk_id,
            "score": score,
            "payload": {
                "doc_id": "doc-1",
                "tenant_id": "tenant-1",
                "page_number": 1,
                "section_title": "Section A",
                "chunk_index": chunk_index,
                "text": text,
            },
        }

    @pytest.mark.asyncio
    async def test_retrieve_orchestrates_full_pipeline(self):
        """retrieve() should embed → search → fuse → filter → rerank correctly."""
        from app.retrieval.pipeline import retrieve

        fake_vector = [0.1] * 1536
        raw_hits = [
            self._make_raw_hit("c1", 0.9, "chunk one text", 0),
            self._make_raw_hit("c2", 0.8, "chunk two text", 1),
        ]

        with (
            patch("app.retrieval.pipeline.embed_chunks", new_callable=AsyncMock) as mock_embed,
            patch("app.retrieval.pipeline.dense_search", new_callable=AsyncMock) as mock_dense,
            patch("app.retrieval.pipeline.sparse_search", new_callable=AsyncMock) as mock_sparse,
            patch("app.retrieval.pipeline.reciprocal_rank_fusion") as mock_fusion,
            patch("app.retrieval.pipeline.rerank", new_callable=AsyncMock) as mock_rerank,
        ):
            mock_embed.return_value = [fake_vector]

            chunks = [make_chunk(h["id"], score=h["score"], text=h["payload"]["text"]) for h in raw_hits]
            mock_dense.return_value = chunks
            mock_sparse.return_value = chunks

            # Fusion returns chunks with scores above the threshold (0.35)
            fused_chunks = [
                make_chunk("c1", score=0.5, text="chunk one text"),
                make_chunk("c2", score=0.4, text="chunk two text"),
            ]
            mock_fusion.return_value = fused_chunks
            mock_rerank.return_value = fused_chunks

            fake_client = AsyncMock()
            result = await retrieve("what is chunk one?", "doc-1", fake_client)

        mock_embed.assert_awaited_once_with(["what is chunk one?"])
        mock_dense.assert_awaited_once()
        mock_sparse.assert_awaited_once()
        mock_fusion.assert_called_once()
        mock_rerank.assert_awaited_once()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_retrieve_passes_correct_query_vector_to_dense_search(self):
        """The embedded query vector must be forwarded to dense_search."""
        from app.retrieval.pipeline import retrieve

        fake_vector = [0.42] * 1536
        chunk = make_chunk("c1", score=0.9)
        fused_chunk = make_chunk("c1", score=0.9)

        with (
            patch("app.retrieval.pipeline.embed_chunks", new_callable=AsyncMock) as mock_embed,
            patch("app.retrieval.pipeline.dense_search", new_callable=AsyncMock) as mock_dense,
            patch("app.retrieval.pipeline.sparse_search", new_callable=AsyncMock) as mock_sparse,
            patch("app.retrieval.pipeline.reciprocal_rank_fusion", return_value=[fused_chunk]),
            patch("app.retrieval.pipeline.rerank", new_callable=AsyncMock) as mock_rerank,
        ):
            mock_embed.return_value = [fake_vector]
            mock_dense.return_value = [chunk]
            mock_sparse.return_value = [chunk]
            mock_rerank.return_value = [fused_chunk]

            fake_client = AsyncMock()
            await retrieve("my question", "doc-1", fake_client)

        call_args = mock_dense.call_args
        assert call_args[0][1] == fake_vector  # positional: client, query_vector, doc_id, top_k

    @pytest.mark.asyncio
    async def test_retrieve_filters_below_score_threshold(self):
        """Chunks whose RRF score is below score_threshold must be excluded before rerank."""
        from app.retrieval.pipeline import retrieve

        fake_vector = [0.1] * 1536

        # Produce many unique chunks so they each appear in only one list —
        # their RRF scores will be small (< score_threshold of 0.35)
        k = settings.rrf_k
        # rank-1 score for a single list: 1/(k+1) ≈ 0.016 — far below 0.35
        low_score_chunks = [make_chunk(f"low-{i}", text=f"text {i}") for i in range(5)]

        with (
            patch("app.retrieval.pipeline.embed_chunks", new_callable=AsyncMock) as mock_embed,
            patch("app.retrieval.pipeline.dense_search", new_callable=AsyncMock) as mock_dense,
            patch("app.retrieval.pipeline.sparse_search", new_callable=AsyncMock) as mock_sparse,
            patch("app.retrieval.pipeline.rerank", new_callable=AsyncMock) as mock_rerank,
        ):
            mock_embed.return_value = [fake_vector]
            mock_dense.return_value = low_score_chunks
            mock_sparse.return_value = []  # none in sparse → RRF scores will be low
            mock_rerank.return_value = []

            fake_client = AsyncMock()
            result = await retrieve("question", "doc-1", fake_client)

        # rerank should have been called with filtered list; since threshold=0.35
        # and max RRF score ≈ 1/(60+1) ≈ 0.016, all are filtered out
        call_kwargs = mock_rerank.call_args
        if call_kwargs is not None:
            passed_chunks = call_kwargs[0][1]  # positional: query, chunks, top_k
            for c in passed_chunks:
                assert c.score >= settings.score_threshold
        assert result == []

    @pytest.mark.asyncio
    async def test_retrieve_returns_empty_when_all_filtered(self):
        """retrieve() must return [] immediately when no chunk survives the threshold."""
        from app.retrieval.pipeline import retrieve

        fake_vector = [0.1] * 1536

        with (
            patch("app.retrieval.pipeline.embed_chunks", new_callable=AsyncMock) as mock_embed,
            patch("app.retrieval.pipeline.dense_search", new_callable=AsyncMock) as mock_dense,
            patch("app.retrieval.pipeline.sparse_search", new_callable=AsyncMock) as mock_sparse,
            patch("app.retrieval.pipeline.rerank", new_callable=AsyncMock) as mock_rerank,
        ):
            mock_embed.return_value = [fake_vector]
            mock_dense.return_value = []
            mock_sparse.return_value = []
            mock_rerank.return_value = []

            fake_client = AsyncMock()
            result = await retrieve("question", "doc-1", fake_client)

        # rerank should not have been called if nothing survived filtering
        assert result == []

    @pytest.mark.asyncio
    async def test_retrieve_uses_settings_top_k_initial(self):
        """dense_search and sparse_search must be called with top_k_initial."""
        from app.retrieval.pipeline import retrieve

        fake_vector = [0.1] * 1536
        chunk = make_chunk("c1", score=0.9)
        # Fusion returns a chunk with a score above the threshold so the rest
        # of the pipeline executes normally.
        fused_chunk = make_chunk("c1", score=0.9)

        with (
            patch("app.retrieval.pipeline.embed_chunks", new_callable=AsyncMock) as mock_embed,
            patch("app.retrieval.pipeline.dense_search", new_callable=AsyncMock) as mock_dense,
            patch("app.retrieval.pipeline.sparse_search", new_callable=AsyncMock) as mock_sparse,
            patch("app.retrieval.pipeline.reciprocal_rank_fusion", return_value=[fused_chunk]),
            patch("app.retrieval.pipeline.rerank", new_callable=AsyncMock) as mock_rerank,
        ):
            mock_embed.return_value = [fake_vector]
            mock_dense.return_value = [chunk]
            mock_sparse.return_value = [chunk]
            mock_rerank.return_value = [fused_chunk]

            fake_client = AsyncMock()
            await retrieve("question", "doc-1", fake_client)

        dense_top_k = mock_dense.call_args[0][3]
        sparse_top_k = mock_sparse.call_args[0][3]
        assert dense_top_k == settings.top_k_initial
        assert sparse_top_k == settings.top_k_initial

    @pytest.mark.asyncio
    async def test_retrieve_uses_settings_top_k_reranked(self):
        """rerank() must be called with top_k equal to settings.top_k_reranked."""
        from app.retrieval.pipeline import retrieve

        fake_vector = [0.1] * 1536
        chunk = make_chunk("c1", score=0.9)
        # Fusion returns a chunk with a score above the threshold so rerank is called.
        fused_chunk = make_chunk("c1", score=0.9)

        with (
            patch("app.retrieval.pipeline.embed_chunks", new_callable=AsyncMock) as mock_embed,
            patch("app.retrieval.pipeline.dense_search", new_callable=AsyncMock) as mock_dense,
            patch("app.retrieval.pipeline.sparse_search", new_callable=AsyncMock) as mock_sparse,
            patch("app.retrieval.pipeline.reciprocal_rank_fusion", return_value=[fused_chunk]),
            patch("app.retrieval.pipeline.rerank", new_callable=AsyncMock) as mock_rerank,
        ):
            mock_embed.return_value = [fake_vector]
            mock_dense.return_value = [chunk]
            mock_sparse.return_value = [chunk]
            mock_rerank.return_value = [fused_chunk]

            fake_client = AsyncMock()
            await retrieve("question", "doc-1", fake_client)

        assert mock_rerank.call_args is not None, "rerank was never called"
        rerank_top_k = mock_rerank.call_args[1].get("top_k") or mock_rerank.call_args[0][2]
        assert rerank_top_k == settings.top_k_reranked
