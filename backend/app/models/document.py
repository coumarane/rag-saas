import enum
import uuid

import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.dialects.postgresql import UUID

from app.models.database import Base


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Document(Base):
    __tablename__ = "documents"

    id = sa.Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=sa.text("gen_random_uuid()"),
    )
    tenant_id = sa.Column(
        UUID(as_uuid=True),
        nullable=False,
        server_default=sa.text("'00000000-0000-0000-0000-000000000001'::uuid"),
    )
    file_name = sa.Column(sa.Text, nullable=False)
    file_type = sa.Column(sa.Text, nullable=False)  # 'pdf' | 'docx'
    s3_key = sa.Column(sa.Text, nullable=False)
    status = sa.Column(
        sa.Text,
        nullable=False,
        default=DocumentStatus.PENDING.value,
        server_default=sa.text("'pending'"),
    )
    error_message = sa.Column(sa.Text, nullable=True)
    chunk_count = sa.Column(sa.Integer, nullable=True)
    token_count = sa.Column(sa.Integer, nullable=True)
    created_at = sa.Column(
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )
    updated_at = sa.Column(
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )


@event.listens_for(Document, "before_update")
def _set_updated_at(mapper, connection, target: Document) -> None:  # noqa: ARG001
    target.updated_at = sa.func.now()
