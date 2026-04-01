"""Recursive text chunker with token-aware splitting and overlap.

Chunking rules (from CLAUDE.md):
- Target: 512 tokens per chunk
- Overlap: 64 tokens
- Separators (priority order): ["\\n\\n", "\\n", ". ", " "]
- Never split mid-sentence if avoidable
- Each chunk carries: page_number, section_title, chunk_index, token_count
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
import tiktoken

from app.ingestion.parser import ParsedPage

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_ENCODING_NAME = "cl100k_base"
_TARGET_TOKENS = 512
_OVERLAP_TOKENS = 64
_SEPARATORS = ["\n\n", "\n", ". ", " "]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ChunkData:
    text: str
    page_number: int
    section_title: str | None
    chunk_index: int
    token_count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(_ENCODING_NAME)


def _token_count(text: str, encoder: tiktoken.Encoding) -> int:
    return len(encoder.encode(text))


def _split_by_separator(text: str, separator: str) -> list[str]:
    """Split *text* on *separator*, keeping the separator at the end of each
    chunk except the last (so sentences aren't fragmented)."""
    if separator == " ":
        return text.split(separator)
    parts = text.split(separator)
    # Re-attach separator to each part except the last.
    result: list[str] = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            result.append(part + separator)
        else:
            if part:
                result.append(part)
    return [p for p in result if p]


def _recursive_split(
    text: str,
    encoder: tiktoken.Encoding,
    separators: list[str],
    target_tokens: int,
) -> list[str]:
    """Recursively split *text* until every piece is ≤ *target_tokens* tokens.

    Tries each separator in order, falling back to the next when chunks are
    still too large.  As a last resort, hard-splits on token boundaries.
    """
    if _token_count(text, encoder) <= target_tokens:
        return [text]

    # Try separators in priority order.
    for sep in separators:
        if sep not in text:
            continue

        parts = _split_by_separator(text, sep)
        if len(parts) <= 1:
            continue

        result: list[str] = []
        for part in parts:
            if _token_count(part, encoder) <= target_tokens:
                result.append(part)
            else:
                # Recurse with remaining separators.
                remaining_seps = separators[separators.index(sep) + 1 :]
                result.extend(
                    _recursive_split(part, encoder, remaining_seps or [" "], target_tokens)
                )
        return result

    # Hard split: encode → slice → decode
    token_ids = encoder.encode(text)
    hard_chunks: list[str] = []
    for i in range(0, len(token_ids), target_tokens):
        hard_chunks.append(encoder.decode(token_ids[i : i + target_tokens]))
    return hard_chunks


def _merge_small_chunks(
    pieces: list[str],
    encoder: tiktoken.Encoding,
    target_tokens: int,
) -> list[str]:
    """Greedily merge consecutive small pieces into chunks up to *target_tokens*."""
    merged: list[str] = []
    current = ""
    current_tokens = 0

    for piece in pieces:
        piece_tokens = _token_count(piece, encoder)
        if current_tokens + piece_tokens <= target_tokens:
            current = current + piece if current else piece
            current_tokens += piece_tokens
        else:
            if current:
                merged.append(current)
            current = piece
            current_tokens = piece_tokens

    if current:
        merged.append(current)
    return merged


def _add_overlap(
    chunks: list[str],
    encoder: tiktoken.Encoding,
    overlap_tokens: int,
) -> list[str]:
    """Prepend a token-level overlap window from the previous chunk."""
    if len(chunks) <= 1:
        return chunks

    result: list[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_ids = encoder.encode(chunks[i - 1])
        overlap_ids = prev_ids[-overlap_tokens:] if len(prev_ids) > overlap_tokens else prev_ids
        overlap_text = encoder.decode(overlap_ids)
        result.append(overlap_text + chunks[i])
    return result


# ---------------------------------------------------------------------------
# Page-tracking helpers
# ---------------------------------------------------------------------------


@dataclass
class _Segment:
    """A contiguous piece of text associated with a specific page and section."""

    text: str
    page_number: int
    section_title: str | None


def _build_segments(pages: list[ParsedPage]) -> list[_Segment]:
    return [
        _Segment(
            text=page.text,
            page_number=page.page_number,
            section_title=page.section_title,
        )
        for page in pages
        if page.text.strip()
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_document(pages: list[ParsedPage]) -> list[ChunkData]:
    """Chunk a list of parsed pages into token-bounded :class:`ChunkData` objects.

    Algorithm:
    1. For each page, recursively split text into pieces ≤ 512 tokens using
       the configured separators.
    2. Greedily merge tiny pieces so chunks are as close to 512 tokens as
       possible without exceeding the limit.
    3. Add a 64-token overlap window from the previous chunk.
    4. Assign sequential ``chunk_index`` values and carry ``page_number`` /
       ``section_title`` from the originating page.

    Returns an empty list when *pages* is empty.
    """
    encoder = _get_encoder()
    segments = _build_segments(pages)

    if not segments:
        logger.warning("chunk_document called with no text content")
        return []

    all_chunks: list[ChunkData] = []
    chunk_index = 0

    for segment in segments:
        # 1. Recursive split
        pieces = _recursive_split(segment.text, encoder, list(_SEPARATORS), _TARGET_TOKENS)

        # 2. Merge small pieces
        pieces = _merge_small_chunks(pieces, encoder, _TARGET_TOKENS)

        # 3. Overlap
        pieces = _add_overlap(pieces, encoder, _OVERLAP_TOKENS)

        # 4. Wrap into ChunkData
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            tc = _token_count(piece, encoder)
            all_chunks.append(
                ChunkData(
                    text=piece,
                    page_number=segment.page_number,
                    section_title=segment.section_title,
                    chunk_index=chunk_index,
                    token_count=tc,
                )
            )
            chunk_index += 1

    logger.info(
        "Document chunked",
        total_pages=len(pages),
        total_chunks=len(all_chunks),
        avg_tokens=(
            sum(c.token_count for c in all_chunks) // len(all_chunks) if all_chunks else 0
        ),
    )
    return all_chunks
