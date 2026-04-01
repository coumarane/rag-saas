"""LLM streaming and response parsing for the RAG generation layer."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import AsyncGenerator

from anthropic import APIError, AsyncAnthropic

from app.core.config import settings
from app.core.exceptions import LLMError


@dataclass
class Citation:
    source_n: int
    chunk_id: str
    doc_name: str
    page: int | None
    excerpt: str


@dataclass
class CitationResponse:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    out_of_context: bool = False


def _get_client() -> AsyncAnthropic:
    return AsyncAnthropic(api_key=settings.anthropic_api_key)


async def stream_answer(
    system_prompt: str,
    messages: list[dict],
) -> AsyncGenerator[str, None]:
    """Stream text tokens from the LLM for the given messages.

    Yields individual text delta strings as they arrive from the API.
    Raises :class:`~app.core.exceptions.LLMError` on API errors.
    """
    client = _get_client()
    try:
        async with client.messages.stream(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            system=system_prompt,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text
    except APIError as exc:
        raise LLMError(f"Anthropic API error: {exc}") from exc


def parse_citations(raw_response: str) -> CitationResponse:
    """Parse the LLM's JSON response into a :class:`CitationResponse`.

    Handles responses wrapped in markdown code fences (```json ... ```).
    Returns a best-effort :class:`CitationResponse` on parse failure.
    """
    text = raw_response.strip()

    # Strip markdown code fences if present
    code_fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_fence_match:
        text = code_fence_match.group(1).strip()

    try:
        data = json.loads(text)
        citations = [
            Citation(
                source_n=c.get("source_n", 0),
                chunk_id=c.get("chunk_id", ""),
                doc_name=c.get("doc_name", ""),
                page=c.get("page"),
                excerpt=c.get("excerpt", ""),
            )
            for c in data.get("citations", [])
        ]
        return CitationResponse(
            answer=data.get("answer", raw_response),
            citations=citations,
            out_of_context=bool(data.get("out_of_context", False)),
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        return CitationResponse(answer=raw_response, citations=[], out_of_context=False)
