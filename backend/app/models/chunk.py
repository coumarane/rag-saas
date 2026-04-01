import uuid

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from app.models.database import Base


class Chunk(Base):
    __tablename__ = "chunks"

    id = sa.Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=sa.text("gen_random_uuid()"),
    )
    doc_id = sa.Column(
        UUID(as_uuid=True),
        sa.ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=True,
    )
    tenant_id = sa.Column(UUID(as_uuid=True), nullable=False)
    text = sa.Column(sa.Text, nullable=False)
    page_number = sa.Column(sa.Integer, nullable=True)
    section_title = sa.Column(sa.Text, nullable=True)
    chunk_index = sa.Column(sa.Integer, nullable=False)
    token_count = sa.Column(sa.Integer, nullable=True)
    created_at = sa.Column(
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )
