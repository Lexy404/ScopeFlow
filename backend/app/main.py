from typing import Literal

import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings

app = FastAPI(title="ScopeFlow", version="0.1.0")

Status = Literal["ok", "down"]


class Health(BaseModel):
    status: Status
    postgres: Status
    ollama: Status


async def _check_postgres(url: str) -> Status:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return "ok"
    except Exception:
        return "down"
    finally:
        await engine.dispose()


async def _check_ollama(url: str) -> Status:
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get(f"{url}/api/version")
        return "ok" if r.status_code == 200 else "down"
    except httpx.HTTPError:
        return "down"


@app.get("/health")
async def health() -> Health:
    s = get_settings()
    pg = await _check_postgres(s.database_url)
    ol = await _check_ollama(s.ollama_url)
    return Health(status="ok" if pg == ol == "ok" else "down", postgres=pg, ollama=ol)
