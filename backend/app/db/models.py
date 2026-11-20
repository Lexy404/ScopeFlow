"""Persistence model (see docs/ARCHITECTURE.md, section 4).

Scope tables (`scopes`, `node_scope_links`) are added in Phase 3.
Node ids live inside `flows.graph`, so `node_id` columns are plain strings.
"""

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, created_at_column


def _empty_graph() -> dict[str, Any]:
    return {"nodes": [], "edges": []}


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def _str_enum(enum_cls: type[StrEnum], name: str) -> Enum:
    # Stored as VARCHAR + CHECK (no native PG enum: easier to evolve).
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        values_callable=lambda e: [m.value for m in e],
    )


class Flow(Base):
    __tablename__ = "flows"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200))
    graph: Mapped[dict[str, Any]] = mapped_column(JSONB, default=_empty_graph)
    created_at: Mapped[datetime] = created_at_column()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    sessions: Mapped[list["ChatSession"]] = relationship(
        back_populates="flow", cascade="all, delete-orphan", passive_deletes=True
    )


class ChatSession(Base):
    """One chat on a flow; "New chat" starts a new session (empty InChat window)."""

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    flow_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("flows.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = created_at_column()

    flow: Mapped[Flow] = relationship(back_populates="sessions")


class Message(Base):
    """Per-node conversation history; source of the InChat sliding window."""

    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_window", "session_id", "node_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"))
    node_id: Mapped[str] = mapped_column(String(100))
    role: Mapped[MessageRole] = mapped_column(_str_enum(MessageRole, "message_role"))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = created_at_column()


class Run(Base):
    """One execution of a flow, triggered by a user message."""

    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[RunStatus] = mapped_column(
        _str_enum(RunStatus, "run_status"), default=RunStatus.RUNNING
    )
    started_at: Mapped[datetime] = created_at_column()
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)

    node_runs: Mapped[list["NodeRun"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="NodeRun.position",
    )


class NodeRun(Base):
    """Trace of one node inside a run (input/output, memory, timings, tokens)."""

    __tablename__ = "node_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    node_id: Mapped[str] = mapped_column(String(100))
    position: Mapped[int] = mapped_column(Integer)  # order in the chain
    input: Mapped[str] = mapped_column(Text)
    output: Mapped[str | None] = mapped_column(Text)
    retrieved: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)
    stored: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)
    retrieve_ms: Mapped[int | None] = mapped_column(Integer)
    store_ms: Mapped[int | None] = mapped_column(Integer)
    llm_ms: Mapped[int | None] = mapped_column(Integer)
    tokens: Mapped[dict[str, int] | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = created_at_column()
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    run: Mapped[Run] = relationship(back_populates="node_runs")
