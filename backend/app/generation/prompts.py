"""Prompt builders for the RAG generation layer."""

from __future__ import annotations

from app.retrieval.searcher import ScoredChunk


def build_system_prompt() -> str:
    """Return the system prompt instructing the LLM to answer only from context."""
    return (
        "You are a precise document assistant. Answer questions ONLY using the provided context chunks.\n"
        "Rules:\n"
        "- Always cite your sources using [Source N] format\n"
        "- If the answer is not in the context, say: "
        '"I couldn\'t find this information in the provided documents."\n'
        "- Be concise and factual\n"
        "- Never hallucinate or add information not present in the context"
    )


def build_context_block(chunks: list[ScoredChunk], file_name: str) -> str:
    """Build the CONTEXT section string from a list of retrieved chunks.

    Each chunk is prefixed with its source header:
        [Source N | file_name | Page page_number | Section: section_title]
    followed by the chunk text.
    """
    lines: list[str] = []
    for n, chunk in enumerate(chunks, start=1):
        page = chunk.page_number if chunk.page_number is not None else "?"
        section = chunk.section_title if chunk.section_title else "N/A"
        header = f"[Source {n} | {file_name} | Page {page} | Section: {section}]"
        lines.append(header)
        lines.append(chunk.text)
        lines.append("")  # blank line between chunks
    return "\n".join(lines).strip()


def build_full_prompt(
    question: str,
    chunks: list[ScoredChunk],
    file_name: str,
    history: list[dict],  # [{"role": "user"|"assistant", "content": "..."}]
) -> list[dict]:
    """Build the Anthropic messages list for a RAG query.

    Returns a list of message dicts in Anthropic format
    (``[{"role": ..., "content": ...}]``).

    The structure is:
    - Previous conversation turns (history, up to last 6 turns)
    - A final user message containing:
        CONTEXT block
        CONVERSATION HISTORY (if any)
        USER QUESTION
    """
    context_block = build_context_block(chunks, file_name)

    # Build the final user message body
    parts: list[str] = []

    parts.append("CONTEXT:")
    parts.append(context_block)

    if history:
        parts.append("\nCONVERSATION HISTORY:")
        for turn in history:
            role_label = "User" if turn["role"] == "user" else "Assistant"
            parts.append(f"{role_label}: {turn['content']}")

    parts.append("\nUSER QUESTION:")
    parts.append(question)

    parts.append(
        "\nRespond in this JSON format:\n"
        "{\n"
        '  "answer": "your answer with [Source N] inline citations",\n'
        '  "citations": [{"source_n": 1, "chunk_id": "...", "doc_name": "...", "page": N, "excerpt": "..."}],\n'
        '  "out_of_context": false\n'
        "}"
    )

    final_user_message = "\n".join(parts)

    # Combine previous history turns with the final user turn
    messages: list[dict] = list(history) + [{"role": "user", "content": final_user_message}]
    return messages
