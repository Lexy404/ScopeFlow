import pytest
from alembic import command
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import ChatSession, Flow, Message, MessageRole, NodeRun, Run, RunStatus

from .conftest import alembic_config


def test_migrations_roundtrip_and_no_drift(test_db_url: str) -> None:
    cfg = alembic_config(test_db_url)
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    command.check(cfg)  # raises if models and migrations disagree


async def _make_flow_with_run(db: AsyncSession) -> Flow:
    flow = Flow(name="demo")
    session = ChatSession(flow=flow)
    db.add_all([flow, session])
    await db.flush()  # assigns session.id
    run = Run(
        session_id=session.id,
        # inserted out of order on purpose: the relationship must sort by position
        node_runs=[
            NodeRun(node_id="agent_2", position=2, input="hello", output="bye"),
            NodeRun(node_id="agent_1", position=1, input="hi", output="hello"),
        ],
    )
    db.add_all(
        [
            run,
            Message(session_id=session.id, node_id="agent_1", role=MessageRole.USER, content="hi"),
            Message(
                session_id=session.id,
                node_id="agent_1",
                role=MessageRole.ASSISTANT,
                content="hello",
            ),
        ]
    )
    await db.commit()
    return flow


async def test_defaults(db: AsyncSession) -> None:
    flow = await _make_flow_with_run(db)
    await db.refresh(flow)
    assert flow.graph == {"nodes": [], "edges": []}
    assert flow.created_at is not None
    run = (await db.execute(select(Run))).scalar_one()
    assert run.status is RunStatus.RUNNING


async def test_node_runs_ordered_by_position(db: AsyncSession) -> None:
    await _make_flow_with_run(db)
    db.expunge_all()
    run = (await db.execute(select(Run))).scalar_one()
    await db.refresh(run, ["node_runs"])
    assert [nr.position for nr in run.node_runs] == [1, 2]


async def test_deleting_flow_cascades(db: AsyncSession) -> None:
    flow = await _make_flow_with_run(db)
    await db.delete(flow)
    await db.commit()
    for model in (ChatSession, Message, Run, NodeRun):
        count = (await db.execute(select(func.count()).select_from(model))).scalar_one()
        assert count == 0, model.__tablename__


async def test_invalid_enum_value_rejected_by_db(db: AsyncSession) -> None:
    await _make_flow_with_run(db)
    session_id = (await db.execute(select(ChatSession.id))).scalar_one()
    with pytest.raises(IntegrityError):
        await db.execute(
            text(
                "INSERT INTO messages (id, session_id, node_id, role, content) "
                "VALUES (gen_random_uuid(), :sid, 'n', 'robot', 'x')"
            ),
            {"sid": session_id},
        )
    await db.rollback()
