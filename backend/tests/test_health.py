import httpx
import pytest

from app import main


@pytest.fixture
def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test")


async def test_health_ok(client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def ok(_: str) -> str:
        return "ok"

    monkeypatch.setattr(main, "_check_postgres", ok)
    monkeypatch.setattr(main, "_check_ollama", ok)
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "postgres": "ok", "ollama": "ok"}


async def test_health_reports_down(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def ok(_: str) -> str:
        return "ok"

    async def down(_: str) -> str:
        return "down"

    monkeypatch.setattr(main, "_check_postgres", ok)
    monkeypatch.setattr(main, "_check_ollama", down)
    r = await client.get("/health")
    assert r.json()["status"] == "down"
    assert r.json()["ollama"] == "down"
