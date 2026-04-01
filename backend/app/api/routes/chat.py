"""Chat endpoints for multi-turn RAG conversations.

Endpoints:
    POST   /api/chat/stream                              Stream an answer as SSE.
    POST   /api/chat/conversations                       Create a new conversation.
    GET    /api/chat/conversations/{conversation_id}/messages  List messages.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import AsyncGenerator

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db
from app.core.exceptions import ConversationNotFoundError
from app.generation.llm import Citation, parse_citations, stream_answer
from app.generation.memory import conversation_memory
from app.generation.prompts import build_full_prompt, build_system_prompt
from app.models.conversation import Conversation, Message
from app.models.document import Document
from app.retrieval.pipeline import retrieve

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str
    doc_id: uuid.UUID
    conversation_id: uuid.UUID | None = None


class ConversationResponse(BaseModel):
    id: uuid.UUID
    doc_id: uuid.UUID | None
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    citations: list | None
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sse(data: dict) -> str:
    """Format a dict as a Server-Sent Events data line."""
    return f"data: {json.dumps(data)}\n\n"


async def _stream_chat(
    request: ChatRequest,
    db: AsyncSession,
    qdrant_client,
) -> AsyncGenerator[str, None]:
    """Core generator: orchestrates retrieval, LLM streaming, and persistence."""

    # 1. Resolve or create conversation_id
    conversation_id = request.conversation_id or uuid.uuid4()
    conversation_id_str = str(conversation_id)

    # 2. Fetch document for file_name
    doc = await db.get(Document, request.doc_id)
    if doc is None:
        yield _sse({"type": "error", "detail": "Document not found"})
        return
    file_name: str = doc.file_name  # type: ignore[assignment]

    # 3. Get conversation history from Redis
    history = await conversation_memory.get_history(
        conversation_id_str,
        last_n=6,
    )

    # 4. Retrieve relevant chunks
    chunks = await retrieve(request.question, str(request.doc_id), qdrant_client)

    # 5. Build prompt messages
    messages = build_full_prompt(request.question, chunks, file_name, history)
    system_prompt = build_system_prompt()

    # 6. Stream tokens from LLM, collecting full response
    full_response_parts: list[str] = []
    async for token in stream_answer(system_prompt, messages):
        full_response_parts.append(token)
        yield _sse({"type": "token", "content": token})

    full_response = "".join(full_response_parts)

    # 7. Parse citations from collected response
    citation_response = parse_citations(full_response)
    citations_payload = [
        {
            "source_n": c.source_n,
            "chunk_id": c.chunk_id,
            "doc_name": c.doc_name,
            "page": c.page,
            "excerpt": c.excerpt,
        }
        for c in citation_response.citations
    ]
    yield _sse({"type": "citations", "citations": citations_payload})

    # 8. Persist messages in Redis memory
    await conversation_memory.add_message(conversation_id_str, "user", request.question)
    await conversation_memory.add_message(
        conversation_id_str, "assistant", citation_response.answer
    )

    # 9. Persist to DB — ensure conversation row exists first
    existing_conv = await db.get(Conversation, conversation_id)
    if existing_conv is None:
        conv = Conversation(
            id=conversation_id,
            tenant_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
            doc_id=request.doc_id,
        )
        db.add(conv)
        await db.flush()

    user_msg = Message(
        id=uuid.uuid4(),
        conversation_id=conversation_id,
        role="user",
        content=request.question,
        citations=None,
    )
    db.add(user_msg)

    assistant_msg = Message(
        id=uuid.uuid4(),
        conversation_id=conversation_id,
        role="assistant",
        content=citation_response.answer,
        citations=citations_payload if citations_payload else None,
    )
    db.add(assistant_msg)
    await db.commit()

    logger.info(
        "Chat turn completed",
        conversation_id=conversation_id_str,
        doc_id=str(request.doc_id),
        chunks_retrieved=len(chunks),
        out_of_context=citation_response.out_of_context,
    )

    yield _sse({"type": "done"})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream a RAG answer for *body.question* over Server-Sent Events.

    SSE event types:
    - ``{"type": "token", "content": "..."}``   — individual streamed tokens
    - ``{"type": "citations", "citations": [...]}`` — after streaming completes
    - ``{"type": "done"}``                       — signals end of stream
    """
    qdrant_client = request.app.state.qdrant_client

    return StreamingResponse(
        _stream_chat(body, db, qdrant_client),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/conversations", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    doc_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> ConversationResponse:
    """Create a new Conversation record linked to *doc_id*."""
    conv = Conversation(
        id=uuid.uuid4(),
        tenant_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        doc_id=doc_id,
    )
    db.add(conv)
    await db.commit()
    await db.refresh(conv)

    return ConversationResponse(
        id=conv.id,  # type: ignore[arg-type]
        doc_id=conv.doc_id,  # type: ignore[arg-type]
        created_at=conv.created_at,  # type: ignore[arg-type]
    )


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[MessageResponse],
)
async def list_messages(
    conversation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> list[MessageResponse]:
    """Return all messages for *conversation_id*, ordered by creation time."""
    conv = await db.get(Conversation, conversation_id)
    if conv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    messages = result.scalars().all()

    return [
        MessageResponse(
            id=m.id,  # type: ignore[arg-type]
            role=m.role,  # type: ignore[arg-type]
            content=m.content,  # type: ignore[arg-type]
            citations=m.citations,
            created_at=m.created_at,  # type: ignore[arg-type]
        )
        for m in messages
    ]
