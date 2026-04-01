"""Celery worker tasks for the RAG ingestion pipeline.

The single public task ``process_document`` orchestrates the full flow:
  parse → chunk → embed → upsert Qdrant → save chunks DB → update doc status.

Because Celery workers run in a non-async context, all async operations are
executed via ``asyncio.run()``.
"""

from __future__ import annotations

import asyncio
import math
import uuid
from collections import Counter

import structlog
from celery import Celery
from qdrant_client import AsyncQdrantClient

from app.core.config import settings
from app.core.logging import configure_logging
from app.core.vector_store import upsert_chunks
from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import embed_chunks
from app.ingestion.parser import parse_document
from app.models.chunk import Chunk
from app.models.database import AsyncSessionLocal
from app.models.document import Document, DocumentStatus

configure_logging()
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Celery application
# ---------------------------------------------------------------------------

celery_app = Celery(
    "rag_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
_VOCAB_SIZE = 65536  # modulus for term hash → sparse index

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_sparse_vector(text: str) -> dict:
    """Compute a TF-IDF–style sparse vector for *text*.

    Each unique normalised token is hashed to an index in [0, 65535].
    The value is the term frequency (count / total_terms).

    Returns a dict with keys ``indices`` (list[int]) and ``values`` (list[float]).
    """
    tokens = text.lower().split()
    if not tokens:
        return {"indices": [], "values": []}

    counts = Counter(tokens)
    total = len(tokens)

    # Accumulate TF values per hashed index (handle hash collisions by summing).
    index_value: dict[int, float] = {}
    for term, count in counts.items():
        idx = hash(term) % _VOCAB_SIZE
        tf = count / total
        index_value[idx] = index_value.get(idx, 0.0) + tf

    # Sort by index for deterministic output.
    sorted_items = sorted(index_value.items())
    indices = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    return {"indices": indices, "values": values}


async def _get_document(doc_id: uuid.UUID) -> Document:
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, doc_id)
    if doc is None:
        raise ValueError(f"Document {doc_id} not found in database")
    return doc


async def _update_document_status(
    doc_id: uuid.UUID,
    status: DocumentStatus,
    *,
    error_message: str | None = None,
    chunk_count: int | None = None,
    token_count: int | None = None,
) -> None:
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, doc_id)
        if doc is None:
            return
        doc.status = status.value
        if error_message is not None:
            doc.error_message = error_message
        if chunk_count is not None:
            doc.chunk_count = chunk_count
        if token_count is not None:
            doc.token_count = token_count
        await session.commit()


async def _download_file(s3_key: str) -> bytes:
    """Download a file from MinIO/S3 and return raw bytes."""
    import io  # noqa: PLC0415

    import boto3  # noqa: PLC0415
    from botocore.config import Config  # noqa: PLC0415

    loop = asyncio.get_running_loop()

    def _fetch() -> bytes:
        client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            config=Config(signature_version="s3v4"),
        )
        obj = client.get_object(Bucket=settings.s3_bucket, Key=s3_key)
        return obj["Body"].read()

    return await loop.run_in_executor(None, _fetch)


async def _save_chunks(
    doc_id: uuid.UUID,
    tenant_id: uuid.UUID,
    chunks,  # list[ChunkData]
) -> None:
    async with AsyncSessionLocal() as session:
        orm_chunks = [
            Chunk(
                id=uuid.uuid4(),
                doc_id=doc_id,
                tenant_id=tenant_id,
                text=chunk.text,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_count,
            )
            for chunk in chunks
        ]
        session.add_all(orm_chunks)
        await session.commit()


async def _run_pipeline(doc_id: uuid.UUID) -> dict:
    """Full async ingestion pipeline.  Returns a summary dict."""

    # 1. Fetch document record
    doc = await _get_document(doc_id)
    tenant_id: uuid.UUID = doc.tenant_id  # type: ignore[assignment]
    s3_key: str = doc.s3_key  # type: ignore[assignment]
    file_type: str = doc.file_type  # type: ignore[assignment]

    logger.info("Ingestion pipeline started", doc_id=str(doc_id), file_type=file_type)

    # 2. Mark as PROCESSING
    await _update_document_status(doc_id, DocumentStatus.PROCESSING)

    # 3. Download file
    file_bytes = await _download_file(s3_key)
    logger.debug("File downloaded", doc_id=str(doc_id), size_bytes=len(file_bytes))

    # 4. Parse
    pages = parse_document(file_bytes, file_type)
    logger.info("Document parsed", doc_id=str(doc_id), pages=len(pages))

    # 5. Chunk
    chunks = chunk_document(pages)
    logger.info("Document chunked", doc_id=str(doc_id), chunks=len(chunks))

    # 6. Embed
    embeddings = await embed_chunks([c.text for c in chunks])
    logger.info("Chunks embedded", doc_id=str(doc_id), embeddings=len(embeddings))

    # 7. Build Qdrant points
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        sparse = _compute_sparse_vector(chunk.text)
        points.append(
            {
                "id": str(uuid.uuid4()),
                "dense_vector": embedding,
                "sparse_vector": sparse,
                "payload": {
                    "doc_id": str(doc_id),
                    "tenant_id": str(tenant_id),
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                },
            }
        )

    # 8. Upsert to Qdrant (create a fresh client — no app.state in Celery)
    qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        await upsert_chunks(qdrant_client, points)
    finally:
        await qdrant_client.close()

    # 9. Save Chunk records to PostgreSQL
    await _save_chunks(doc_id, tenant_id, chunks)

    # 10. Update document status to READY
    total_tokens = sum(c.token_count for c in chunks)
    await _update_document_status(
        doc_id,
        DocumentStatus.READY,
        chunk_count=len(chunks),
        token_count=total_tokens,
    )

    logger.info(
        "Ingestion pipeline complete",
        doc_id=str(doc_id),
        chunks=len(chunks),
        total_tokens=total_tokens,
    )
    return {"doc_id": str(doc_id), "chunks": len(chunks), "tokens": total_tokens}


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------


@celery_app.task(name="process_document", bind=True, max_retries=3)
def process_document(self, doc_id: str) -> dict:
    """Celery task that runs the full ingestion pipeline for a document.

    Parameters
    ----------
    doc_id:
        String representation of the document UUID.

    Returns
    -------
    dict
        ``{"doc_id": str, "chunks": int, "tokens": int}`` on success.

    On failure the document status is set to FAILED, the error message is
    saved, and the exception is re-raised (triggering Celery retry logic).
    """
    parsed_id = uuid.UUID(doc_id)
    log = logger.bind(doc_id=doc_id, task_id=self.request.id)
    log.info("process_document task started")

    try:
        result = asyncio.run(_run_pipeline(parsed_id))
        log.info("process_document task succeeded", result=result)
        return result
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        log.error("process_document task failed", error=error_msg, exc_info=exc)

        # Persist FAILED status
        try:
            asyncio.run(
                _update_document_status(
                    parsed_id,
                    DocumentStatus.FAILED,
                    error_message=error_msg,
                )
            )
        except Exception as db_exc:  # noqa: BLE001
            log.error("Failed to update document status to FAILED", exc_info=db_exc)

        # Retry with exponential back-off, then re-raise on final attempt.
        try:
            raise self.retry(exc=exc, countdown=int(math.pow(2, self.request.retries)))
        except self.MaxRetriesExceededError:
            raise exc from exc
