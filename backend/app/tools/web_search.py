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

    def __init__(self, search: SearchFn = _ddgs_search) -> None:
        self._search = search  # injectable for tests

    async def run(self, args: dict[str, Any]) -> str:
        query = str(args["query"]).strip()
        if not query:
            raise ToolError("query is empty")
        try:
            results = await asyncio.to_thread(self._search, query, MAX_RESULTS)
        except DDGSException as e:
            raise ToolError(f"search failed: {e}") from e
        if not results:
            return "No results."
        return "\n\n".join(
            f"{i}. {r.get('title', '').strip()}\n{r.get('href', '')}\n"
            f"{str(r.get('body', '')).strip()[:SNIPPET_CHARS]}"
            for i, r in enumerate(results, 1)
        )
