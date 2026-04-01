"""Unit tests for app.ingestion.chunker."""

from __future__ import annotations

import pytest

from app.ingestion.chunker import ChunkData, _TARGET_TOKENS, chunk_document
from app.ingestion.parser import ParsedPage


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_pages(texts: list[str]) -> list[ParsedPage]:
    return [
        ParsedPage(text=t, page_number=i + 1, section_title=f"Section {i + 1}")
        for i, t in enumerate(texts)
    ]


def _long_text(num_words: int) -> str:
    """Produce a deterministic text of roughly *num_words* words."""
    # Use a paragraph-like sentence repeated to ensure the recursive splitter
    # has natural break points (". ").
    sentence = "The quick brown fox jumps over the lazy dog. "
    words_per_sentence = len(sentence.split())
    repeats = max(1, num_words // words_per_sentence + 1)
    return sentence * repeats


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChunkDocument:
    def test_returns_list_of_chunk_data(self):
        pages = _make_pages(["Hello world. This is a test sentence."])
        chunks = chunk_document(pages)
        assert isinstance(chunks, list)
        assert all(isinstance(c, ChunkData) for c in chunks)

    def test_empty_pages_returns_empty_list(self):
        assert chunk_document([]) == []

    def test_blank_text_pages_returns_empty_list(self):
        pages = _make_pages(["   ", "\n\n", ""])
        assert chunk_document(pages) == []

    def test_chunk_index_is_sequential(self):
        pages = _make_pages([_long_text(400), _long_text(400)])
        chunks = chunk_document(pages)
        assert len(chunks) > 0
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunks_respect_token_limit(self):
        """Each chunk should not significantly exceed the target token count.

        We allow up to 15 % headroom because the overlap window is prepended
        to the *next* chunk (which can momentarily push it over the hard limit
        before the recursive splitter processes it).
        """
        pages = _make_pages([_long_text(2000)])
        chunks = chunk_document(pages)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.token_count <= int(_TARGET_TOKENS * 1.15), (
                f"Chunk {chunk.chunk_index} has {chunk.token_count} tokens "
                f"(limit {_TARGET_TOKENS})"
            )

    def test_overlap_exists_between_consecutive_chunks(self):
        """Consecutive chunks should share some tokens (overlap window)."""
        # Use a text long enough to produce at least 3 chunks
        pages = _make_pages([_long_text(1500)])
        chunks = chunk_document(pages)
        assert len(chunks) >= 2, "Need at least 2 chunks to test overlap"

        # Check that the end of chunk[i] appears at the start of chunk[i+1]
        overlaps_found = 0
        for i in range(len(chunks) - 1):
            prev_words = chunks[i].text.split()
            next_words = chunks[i + 1].text.split()
            # The last few words of chunks[i] should appear at the start of chunks[i+1]
            tail = prev_words[-10:]  # last 10 words of prev chunk
            head = next_words[:20]   # first 20 words of next chunk
            if any(w in head for w in tail):
                overlaps_found += 1

        assert overlaps_found > 0, "Expected at least one overlapping pair of chunks"

    def test_page_number_is_carried_from_source_page(self):
        pages = _make_pages(["Short page one text.", "Short page two text."])
        chunks = chunk_document(pages)
        assert len(chunks) >= 2
        page_numbers = {c.page_number for c in chunks}
        assert 1 in page_numbers
        assert 2 in page_numbers

    def test_section_title_is_preserved(self):
        pages = [
            ParsedPage(text=_long_text(600), page_number=1, section_title="My Section"),
        ]
        chunks = chunk_document(pages)
        assert all(c.section_title == "My Section" for c in chunks)

    def test_token_count_is_positive(self):
        pages = _make_pages(["Some text that should produce tokens."])
        chunks = chunk_document(pages)
        assert all(c.token_count > 0 for c in chunks)

    def test_multi_page_document_produces_chunks(self):
        pages = _make_pages([_long_text(600), _long_text(600), _long_text(600)])
        chunks = chunk_document(pages)
        assert len(chunks) >= 3

    def test_single_short_text_produces_one_chunk(self):
        pages = _make_pages(["A single short sentence."])
        chunks = chunk_document(pages)
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
