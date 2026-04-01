"""Document management routes.

Endpoints:
    POST   /api/documents/upload   Upload a PDF or DOCX file, kick off ingestion.
    GET    /api/documents          List all documents.
    GET    /api/documents/{doc_id} Get a single document.
    DELETE /api/documents/{doc_id} Delete document (storage, vectors, DB).
"""

from __future__ import annotations

import uuid
from datetime import datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db
from app.core.exceptions import FileTooLargeError, UnsupportedFileTypeError
from app.core.storage import delete_file, upload_file
from app.core.vector_store import delete_document_vectors
from app.ingestion.parser import detect_file_type
from app.models.document import Document, DocumentStatus

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["documents"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
_ALLOWED_TYPES = {"pdf", "docx"}
_CONTENT_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class DocumentResponse(BaseModel):
    id: uuid.UUID
    file_name: str
    file_type: str
    status: str
    chunk_count: int | None
    token_count: int | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc_to_response(doc: Document) -> DocumentResponse:
    return DocumentResponse(
        id=doc.id,  # type: ignore[arg-type]
        file_name=doc.file_name,  # type: ignore[arg-type]
        file_type=doc.file_type,  # type: ignore[arg-type]
        status=doc.status,  # type: ignore[arg-type]
        chunk_count=doc.chunk_count,
        token_count=doc.token_count,
        created_at=doc.created_at,  # type: ignore[arg-type]
        updated_at=doc.updated_at,  # type: ignore[arg-type]
    )


async def _get_doc_or_404(doc_id: uuid.UUID, db: AsyncSession) -> Document:
    doc = await db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/documents/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Upload a PDF or DOCX document and queue it for ingestion."""

    # Read file contents (size check included)
    file_bytes = await file.read()

    if len(file_bytes) > _MAX_FILE_SIZE:
        raise FileTooLargeError(
            f"File size {len(file_bytes)} bytes exceeds the 50 MB limit"
        )

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    # Detect file type via magic bytes (raises UnsupportedFileTypeError if unknown)
    try:
        file_type = detect_file_type(file_bytes)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=exc.message,
        ) from exc

    # Build document record
    doc_id = uuid.uuid4()
    safe_filename = file.filename or f"{doc_id}.{file_type}"
    s3_key = f"documents/{doc_id}/{safe_filename}"
    content_type = _CONTENT_TYPES.get(file_type, "application/octet-stream")

    # Upload to MinIO
    await upload_file(file_bytes, s3_key, content_type)
    logger.info("File uploaded", doc_id=str(doc_id), s3_key=s3_key, file_type=file_type)

    # Persist Document record
    doc = Document(
        id=doc_id,
        file_name=safe_filename,
        file_type=file_type,
        s3_key=s3_key,
        status=DocumentStatus.PENDING.value,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    # Trigger Celery ingestion task (import here to avoid circular imports at
    # module load time when Celery is not yet configured)
    from workers.tasks import process_document  # noqa: PLC0415

    process_document.delay(str(doc_id))
    logger.info("Ingestion task queued", doc_id=str(doc_id))

    return _doc_to_response(doc)


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    db: AsyncSession = Depends(get_db),
) -> list[DocumentResponse]:
    """Return all documents for the default tenant."""
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    docs = result.scalars().all()
    return [_doc_to_response(d) for d in docs]


@router.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Return a single document or 404."""
    doc = await _get_doc_or_404(doc_id, db)
    return _doc_to_response(doc)


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    doc_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a document: removes S3 file, Qdrant vectors, and the DB record."""
    doc = await _get_doc_or_404(doc_id, db)

    # Delete from MinIO (best-effort; log errors but don't block deletion)
    try:
        await delete_file(doc.s3_key)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to delete S3 file", doc_id=str(doc_id), exc_info=exc)

    # Delete vectors from Qdrant
    try:
        qdrant_client = request.app.state.qdrant_client
        await delete_document_vectors(qdrant_client, str(doc_id))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to delete Qdrant vectors", doc_id=str(doc_id), exc_info=exc)

    # Delete DB record (cascades to chunks via FK)
    await db.delete(doc)
    await db.commit()
    logger.info("Document deleted", doc_id=str(doc_id))
