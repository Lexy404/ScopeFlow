"""Shared fixtures.

DB tests run against a separate database (`<dbname>_test`) that is created on demand,
so the dev database is never touched. If Postgres is unreachable, DB tests are skipped,
unless SCOPEFLOW_REQUIRE_DB=1 (set in CI), in which case they fail.
"""

import asyncio
import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import asyncpg
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import get_settings

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _test_db_url() -> str:
    url = make_url(get_settings().database_url)
    return url.set(database=f"{url.database}_test").render_as_string(hide_password=False)


async def _ensure_database(url: str) -> None:
    u = make_url(url)
    conn = await asyncpg.connect(
        user=u.username, password=u.password, host=u.host, port=u.port, database="postgres"
    )
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", u.database)
        if not exists:
            await conn.execute(f'CREATE DATABASE "{u.database}"')
    finally:
        await conn.close()


def alembic_config(url: str) -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


@pytest.fixture(scope="session")
def test_db_url() -> Iterator[str]:
    url = _test_db_url()
    try:
        asyncio.run(_ensure_database(url))
    except (OSError, asyncpg.PostgresError) as e:
        if os.environ.get("SCOPEFLOW_REQUIRE_DB") == "1":
            raise
        pytest.skip(f"Postgres not available: {e}")
    cfg = alembic_config(url)
    command.downgrade(cfg, "base")  # start from a clean schema every test session
    command.upgrade(cfg, "head")
    yield url


@pytest.fixture
async def db(test_db_url: str) -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(test_db_url, poolclass=NullPool)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        yield session
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE flows CASCADE"))
    await engine.dispose()
