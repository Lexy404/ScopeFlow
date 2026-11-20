import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest
from ddgs.exceptions import DDGSException

from app.llm import ToolSpec
from app.tools import ToolError, ToolRegistry, default_registry
from app.tools.calculator import Calculator, evaluate
from app.tools.current_time import CurrentTime
from app.tools.web_search import WebSearch

# --- calculator -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("17 * 23", 391),
        ("(1 + 2) * 3 - 4 / 2", 7.0),
        ("2 ^ 10", 1024),
        ("-3 + +5", 2),
        ("7 // 2 + 7 % 2", 4),
        ("sqrt(16) + abs(-2)", 6.0),
        ("round(pi, 2)", 3.14),
    ],
)
def test_calculator_evaluates(expr: str, expected: float) -> None:
    assert evaluate(expr) == pytest.approx(expected)


async def test_calculator_formats_integers_without_decimals() -> None:
    assert await Calculator().run({"expression": "10 / 2"}) == "5"
    assert await Calculator().run({"expression": "1 / 3"}) == "0.3333333333"


@pytest.mark.parametrize(
    ("expr", "message"),
    [
        ("__import__('os').system('ls')", "unsupported"),  # code injection
        ("open('/etc/passwd')", "unsupported"),
        ("(1).__class__", "unsupported"),  # attribute access
        ("[1, 2]", "unsupported"),
        ("True + 1", "booleans"),
        ("9 ** 9 ** 9", "exponent"),  # would hang without the limit
        ("10 ** 999 * 10 ** 999", "too large"),
        ("1 / 0", "division by zero"),
        ("sqrt(-1)", "math error"),
        ("2 +", "invalid expression"),
        ("1+" * 150 + "1", "longer than"),
    ],
)
def test_calculator_rejects(expr: str, message: str) -> None:
    with pytest.raises(ToolError, match=message):
        evaluate(expr)


# --- current_time -----------------------------------------------------------------------

FIXED = datetime(2026, 11, 20, 10, 30, 0, tzinfo=UTC)


async def test_current_time_default_utc() -> None:
    out = await CurrentTime(now=lambda: FIXED).run({})
    assert out == "Friday, 2026-11-20 10:30:00 (UTC, UTC+0000)"


async def test_current_time_in_timezone() -> None:
    out = await CurrentTime(now=lambda: FIXED).run({"timezone": "Asia/Jerusalem"})
    assert out == "Friday, 2026-11-20 12:30:00 (Asia/Jerusalem, UTC+0200)"


async def test_current_time_unknown_timezone() -> None:
    with pytest.raises(ToolError, match="unknown timezone"):
        await CurrentTime().run({"timezone": "Mars/Olympus"})


# --- web_search -------------------------------------------------------------------------


async def test_web_search_formats_results() -> None:
    def fake(query: str, n: int) -> list[dict[str, Any]]:
        assert (query, n) == ("scopeflow", 5)
        return [{"title": " A ", "href": "https://a.example", "body": "x" * 500}]

    out = await WebSearch(search=fake).run({"query": " scopeflow "})
    title, url, snippet = out.split("\n")
    assert (title, url) == ("1. A", "https://a.example")
    assert len(snippet) == 300


async def test_web_search_no_results_and_errors() -> None:
    assert await WebSearch(search=lambda q, n: []).run({"query": "q"}) == "No results."

    def failing(q: str, n: int) -> list[dict[str, Any]]:
        raise DDGSException("ratelimit")

    with pytest.raises(ToolError, match="search failed"):
        await WebSearch(search=failing).run({"query": "q"})
    with pytest.raises(ToolError, match="empty"):
        await WebSearch(search=lambda q, n: []).run({"query": "  "})


def _flaky(*outcomes: Exception | list[dict[str, Any]]) -> Any:
    """Search stub returning/raising the given outcomes in order; records call count."""
    queue = list(outcomes)

    def search(q: str, n: int) -> list[dict[str, Any]]:
        search.calls += 1  # type: ignore[attr-defined]
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    search.calls = 0  # type: ignore[attr-defined]
    return search


async def test_web_search_retries_flaky_empty_result() -> None:
    hit = [{"title": "T", "href": "https://t.example", "body": "b"}]
    search = _flaky(DDGSException("No results found."), hit)
    out = await WebSearch(search=search).run({"query": "q"})
    assert out.startswith("1. T") and search.calls == 2


async def test_web_search_no_results_after_retries_is_not_an_error() -> None:
    search = _flaky(DDGSException("No results found."), DDGSException("No results found."))
    assert await WebSearch(search=search).run({"query": "q"}) == "No results."
    assert search.calls == 2


async def test_web_search_real_failure_after_retries_is_an_error() -> None:
    search = _flaky(DDGSException("ratelimit"), DDGSException("ratelimit"))
    with pytest.raises(ToolError, match="search failed: ratelimit"):
        await WebSearch(search=search).run({"query": "q"})


# --- registry ---------------------------------------------------------------------------


class _Echo:
    spec = ToolSpec("echo", "echo", {"type": "object", "properties": {"x": {}}, "required": ["x"]})

    async def run(self, args: dict[str, Any]) -> str:
        return str(args["x"])


class _Slow:
    spec = ToolSpec("slow", "sleeps")

    async def run(self, args: dict[str, Any]) -> str:
        await asyncio.sleep(10)
        return "done"


class _Buggy:
    spec = ToolSpec("buggy", "raises")

    async def run(self, args: dict[str, Any]) -> str:
        raise RuntimeError("secret internal detail")


def test_default_registry_has_the_three_tools() -> None:
    reg = default_registry()
    assert reg.names == ["calculator", "current_time", "web_search"]
    assert [s.name for s in reg.specs(["calculator"])] == ["calculator"]
    with pytest.raises(KeyError):
        reg.specs(["nope"])


async def test_execute_success() -> None:
    out = await ToolRegistry([_Echo()]).execute("echo", {"x": 42})
    assert (out.ok, out.content, out.name, out.arguments) == (True, "42", "echo", {"x": 42})
    assert out.duration_ms >= 0


@pytest.mark.parametrize(
    ("tool", "args", "message"),
    [
        ("missing", {}, "unknown tool 'missing'"),
        ("echo", {}, "missing argument(s) ['x']"),
        ("slow", {}, "timed out"),
        ("buggy", {}, "failed unexpectedly (RuntimeError)"),
    ],
)
async def test_execute_failures_become_text(tool: str, args: dict[str, Any], message: str) -> None:
    reg = ToolRegistry([_Echo(), _Slow(), _Buggy()], timeout_s=0.05)
    out = await reg.execute(tool, args)
    assert not out.ok
    assert out.content.startswith("Error:") and message in out.content
    assert "secret" not in out.content  # internal exception text is not leaked to the model


async def test_execute_truncates_long_output() -> None:
    out = await ToolRegistry([_Echo()]).execute("echo", {"x": "y" * 10_000})
    assert out.content.endswith("[truncated]") and len(out.content) < 4100


async def test_tool_error_from_real_tool_is_reported() -> None:
    out = await default_registry().execute("calculator", {"expression": "1/0"})
    assert (out.ok, out.content) == (False, "Error: division by zero")


# --- live web search (skipped when offline / rate limited) -------------------------------


@pytest.mark.network
async def test_live_web_search() -> None:
    try:
        out = await WebSearch().run({"query": "Python programming language"})
    except ToolError as e:
        pytest.skip(f"web search unavailable: {e}")
    assert "1. " in out and "http" in out
