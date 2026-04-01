import uuid

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.models.database import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id = sa.Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=sa.text("gen_random_uuid()"),
    )
    tenant_id = sa.Column(UUID(as_uuid=True), nullable=False)
    doc_id = sa.Column(
        UUID(as_uuid=True),
        sa.ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at = sa.Column(
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )


class Message(Base):
    __tablename__ = "messages"

    id = sa.Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=sa.text("gen_random_uuid()"),
    )
    conversation_id = sa.Column(
        UUID(as_uuid=True),
        sa.ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=True,
    )
    role = sa.Column(sa.Text, nullable=False)  # 'user' | 'assistant'
    content = sa.Column(sa.Text, nullable=False)
    citations = sa.Column(JSONB, nullable=True)
    created_at = sa.Column(
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )
