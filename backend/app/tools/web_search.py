import asyncio
from collections.abc import Callable
from typing import Any

from ddgs import DDGS
from ddgs.exceptions import DDGSException

from app.llm import ToolSpec
from app.tools.base import ToolError

SearchFn = Callable[[str, int], list[dict[str, Any]]]

MAX_RESULTS = 5
SNIPPET_CHARS = 300
ATTEMPTS = 2
NO_RESULTS = "No results found"  # message ddgs raises with on an empty result


def _ddgs_search(query: str, max_results: int) -> list[dict[str, Any]]:
    return DDGS().text(query, max_results=max_results)


class WebSearch:
    """DuckDuckGo search (no API key). The client is synchronous, so it runs in a thread."""

    spec = ToolSpec(
        name="web_search",
        description=(
            "Search the web (DuckDuckGo). Returns titles, URLs and snippets. "
            "Use for current events or facts you are unsure about."
        ),
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "search query"}},
            "required": ["query"],
        },
    )

    def __init__(self, search: SearchFn = _ddgs_search, attempts: int = ATTEMPTS) -> None:
        self._search = search  # injectable for tests
        self._attempts = attempts

    async def _search_with_retry(self, query: str) -> list[dict[str, Any]]:
        """ddgs is flaky: the same query may return results or raise "No results found".
        An empty result is not an error, so it never becomes a ToolError."""
        error: DDGSException | None = None
        for _ in range(self._attempts):
            try:
                results = await asyncio.to_thread(self._search, query, MAX_RESULTS)
            except DDGSException as e:
                error = e
                continue
            if results:
                return results
            error = None
        if error is not None and NO_RESULTS not in str(error):
            raise ToolError(f"search failed: {error}") from error
        return []

    async def run(self, args: dict[str, Any]) -> str:
        query = str(args["query"]).strip()
        if not query:
            raise ToolError("query is empty")
        results = await self._search_with_retry(query)
        if not results:
            return "No results."
        return "\n\n".join(
            f"{i}. {r.get('title', '').strip()}\n{r.get('href', '')}\n"
            f"{str(r.get('body', '')).strip()[:SNIPPET_CHARS]}"
            for i, r in enumerate(results, 1)
        )
