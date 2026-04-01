"""Integration tests for the /api/chat endpoints.

All external dependencies are mocked:
- retrieve()              — retrieval pipeline
- stream_answer()         — LLM streaming
- ConversationMemory      — Redis-backed history
- AsyncSession / get_db   — database session

Coverage:
- POST /api/chat/stream returns SSE with token events and a citation event
- POST /api/chat/conversations creates a new Conversation and returns it
- GET  /api/chat/conversations/{id}/messages returns messages for that conversation
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app
from app.retrieval.searcher import ScoredChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chunk(chunk_id: str = "c1") -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        score=0.9,
        text="Sample chunk text about the topic.",
        page_number=1,
        section_title="Introduction",
        doc_id="doc-1",
        chunk_index=0,
    )


def _parse_sse_events(content: bytes) -> list[dict]:
    """Parse raw SSE bytes into a list of decoded event dicts."""
    events = []
    for line in content.decode().splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


DOC_ID = uuid.uuid4()
CONV_ID = uuid.uuid4()

_NOW = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)


def _make_mock_doc():
    doc = MagicMock()
    doc.id = DOC_ID
    doc.file_name = "sample.pdf"
    doc.file_type = "pdf"
    doc.status = "ready"
    doc.chunk_count = 10
    doc.token_count = 1000
    doc.created_at = _NOW
    doc.updated_at = _NOW
    return doc


def _make_mock_conversation():
    conv = MagicMock()
    conv.id = CONV_ID
    conv.doc_id = DOC_ID
    conv.created_at = _NOW
    return conv


def _make_mock_message(role: str, content: str, msg_id: uuid.UUID | None = None):
    msg = MagicMock()
    msg.id = msg_id or uuid.uuid4()
    msg.role = role
    msg.content = content
    msg.citations = None
    msg.created_at = _NOW
    return msg


# ---------------------------------------------------------------------------
# POST /api/chat/stream
# ---------------------------------------------------------------------------


class TestChatStream:
    """Tests for the SSE streaming endpoint."""

    @pytest.mark.asyncio
    async def test_stream_returns_token_and_citation_events(self):
        """The SSE response must contain token events, a citations event, and done."""

        fake_doc = _make_mock_doc()
        fake_conv = _make_mock_conversation()

        async def fake_stream(*args, **kwargs):
            for token in ["Hello", " world", "!"]:
                yield token

        llm_json = json.dumps({
            "answer": "Hello world! [Source 1]",
            "citations": [
                {
                    "source_n": 1,
                    "chunk_id": "c1",
                    "doc_name": "sample.pdf",
                    "page": 1,
                    "excerpt": "Sample chunk text",
                }
            ],
            "out_of_context": False,
        })

        mock_db = AsyncMock()
        mock_db.get = AsyncMock(side_effect=[fake_doc, None])  # doc lookup, then conv lookup
        mock_db.flush = AsyncMock()
        mock_db.commit = AsyncMock()

        async def override_get_db():
            yield mock_db

        app.dependency_overrides[
            __import__("app.core.dependencies", fromlist=["get_db"]).get_db
        ] = override_get_db

        # Patch app.state.qdrant_client
        app.state.qdrant_client = AsyncMock()

        try:
            with (
                patch("app.api.routes.chat.retrieve", new_callable=AsyncMock) as mock_retrieve,
                patch("app.api.routes.chat.stream_answer", side_effect=lambda *a, **kw: fake_stream(*a, **kw)),
                patch("app.api.routes.chat.conversation_memory") as mock_memory,
            ):
                mock_retrieve.return_value = [make_chunk()]
                mock_memory.get_history = AsyncMock(return_value=[])
                mock_memory.add_message = AsyncMock()

                # patch parse_citations to return a structured response
                from app.generation.llm import CitationResponse, Citation
                fake_citation_response = CitationResponse(
                    answer="Hello world! [Source 1]",
                    citations=[
                        Citation(
                            source_n=1,
                            chunk_id="c1",
                            doc_name="sample.pdf",
                            page=1,
                            excerpt="Sample chunk text",
                        )
                    ],
                    out_of_context=False,
                )

                with patch("app.api.routes.chat.parse_citations", return_value=fake_citation_response):
                    async with AsyncClient(app=app, base_url="http://test") as client:
                        response = await client.post(
                            "/api/chat/stream",
                            json={
                                "question": "What is this about?",
                                "doc_id": str(DOC_ID),
                            },
                        )

                assert response.status_code == 200
                assert "text/event-stream" in response.headers["content-type"]

                events = _parse_sse_events(response.content)
                event_types = [e["type"] for e in events]

                assert "token" in event_types, "Expected at least one token event"
                assert "citations" in event_types, "Expected a citations event"
                assert "done" in event_types, "Expected a done event"

                token_events = [e for e in events if e["type"] == "token"]
                full_text = "".join(e["content"] for e in token_events)
                assert full_text == "Hello world!"

                citation_event = next(e for e in events if e["type"] == "citations")
                assert len(citation_event["citations"]) == 1
                assert citation_event["citations"][0]["source_n"] == 1
                assert citation_event["citations"][0]["chunk_id"] == "c1"

        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_stream_returns_error_when_doc_not_found(self):
        """When the document is not found, the SSE stream should emit an error event."""
        mock_db = AsyncMock()
        mock_db.get = AsyncMock(return_value=None)  # doc not found

        async def override_get_db():
            yield mock_db

        from app.core.dependencies import get_db
        app.dependency_overrides[get_db] = override_get_db
        app.state.qdrant_client = AsyncMock()

        try:
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/chat/stream",
                    json={
                        "question": "What is this?",
                        "doc_id": str(uuid.uuid4()),
                    },
                )

            assert response.status_code == 200
            events = _parse_sse_events(response.content)
            assert any(e.get("type") == "error" for e in events)
        finally:
            app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# POST /api/chat/conversations
# ---------------------------------------------------------------------------


class TestCreateConversation:
    @pytest.mark.asyncio
    async def test_create_conversation_returns_201(self):
        """POST /api/chat/conversations should create a record and return 201."""
        fake_conv = _make_mock_conversation()

        mock_db = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock(side_effect=lambda obj: None)

        # After add + commit + refresh the object should have its fields set.
        # We intercept db.add to capture the Conversation and patch its attrs.
        added_objects: list = []

        def _add(obj):
            added_objects.append(obj)
            # Simulate server-side defaults
            if not hasattr(obj, "created_at") or obj.created_at is None:
                object.__setattr__(obj, "created_at", _NOW)

        mock_db.add = MagicMock(side_effect=_add)

        async def override_get_db():
            yield mock_db

        from app.core.dependencies import get_db
        app.dependency_overrides[get_db] = override_get_db

        try:
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/chat/conversations",
                    params={"doc_id": str(DOC_ID)},
                )

            assert response.status_code == 201
            data = response.json()
            assert "id" in data
            assert data["doc_id"] == str(DOC_ID)
        finally:
            app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GET /api/chat/conversations/{id}/messages
# ---------------------------------------------------------------------------


class TestListMessages:
    @pytest.mark.asyncio
    async def test_list_messages_returns_messages_in_order(self):
        """GET /api/chat/conversations/{id}/messages returns ordered messages."""
        msg_id_1 = uuid.uuid4()
        msg_id_2 = uuid.uuid4()
        fake_conv = _make_mock_conversation()
        fake_user_msg = _make_mock_message("user", "What is this?", msg_id_1)
        fake_asst_msg = _make_mock_message("assistant", "It is a document.", msg_id_2)

        mock_db = AsyncMock()
        mock_db.get = AsyncMock(return_value=fake_conv)

        # Mock execute to return a result with the two messages
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [fake_user_msg, fake_asst_msg]
        mock_db.execute = AsyncMock(return_value=mock_result)

        async def override_get_db():
            yield mock_db

        from app.core.dependencies import get_db
        app.dependency_overrides[get_db] = override_get_db

        try:
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get(
                    f"/api/chat/conversations/{CONV_ID}/messages"
                )

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["role"] == "user"
            assert data[0]["content"] == "What is this?"
            assert data[1]["role"] == "assistant"
            assert data[1]["content"] == "It is a document."
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_list_messages_returns_404_for_unknown_conversation(self):
        """GET returns 404 when the conversation does not exist."""
        mock_db = AsyncMock()
        mock_db.get = AsyncMock(return_value=None)

        async def override_get_db():
            yield mock_db

        from app.core.dependencies import get_db
        app.dependency_overrides[get_db] = override_get_db

        try:
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get(
                    f"/api/chat/conversations/{uuid.uuid4()}/messages"
                )

            assert response.status_code == 404
        finally:
            app.dependency_overrides.clear()
