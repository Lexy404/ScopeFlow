"""Tool registry and the safe execution boundary between the model and tool code.

`execute` never raises: every failure becomes a text result with ok=False, so the agent
loop can hand it back to the model (which may retry or answer anyway) and the trace can
show it.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from app.llm import ToolSpec
from app.tools.base import Tool, ToolError
from app.tools.calculator import Calculator
from app.tools.current_time import CurrentTime
from app.tools.web_search import WebSearch

DEFAULT_TIMEOUT_S = 15.0
MAX_OUTPUT_CHARS = 4000  # keep tool results from flooding a 16k context


@dataclass(frozen=True)
class ToolOutcome:
    name: str
    arguments: dict[str, Any]
    content: str
    ok: bool
    duration_ms: int


class ToolRegistry:
    def __init__(self, tools: list[Tool], timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self._tools = {t.spec.name: t for t in tools}
        self._timeout_s = timeout_s

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self, names: list[str] | None = None) -> list[ToolSpec]:
        """Specs for the given tool names (all when None). Unknown names raise KeyError."""
        selected = self.names if names is None else names
        return [self._tools[n].spec for n in selected]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        start = time.perf_counter()

        def outcome(content: str, ok: bool) -> ToolOutcome:
            if len(content) > MAX_OUTPUT_CHARS:
                content = content[:MAX_OUTPUT_CHARS] + "\n[truncated]"
            ms = int((time.perf_counter() - start) * 1000)
            return ToolOutcome(name, arguments, content, ok, ms)

        tool = self._tools.get(name)
        if tool is None:
            return outcome(
                f"Error: unknown tool '{name}'. Available: {', '.join(self.names)}", False
            )

        missing = [p for p in tool.spec.parameters.get("required", []) if p not in arguments]
        if missing:
            expected = json.dumps(tool.spec.parameters.get("properties", {}))
            return outcome(f"Error: missing argument(s) {missing}. Expected: {expected}", False)

        try:
            async with asyncio.timeout(self._timeout_s):
                return outcome(await tool.run(arguments), True)
        except TimeoutError:
            return outcome(f"Error: tool '{name}' timed out after {self._timeout_s:g}s", False)
        except ToolError as e:
            return outcome(f"Error: {e}", False)
        except Exception as e:  # a bug in tool code must not kill the run
            return outcome(f"Error: tool '{name}' failed unexpectedly ({type(e).__name__})", False)


def default_registry() -> ToolRegistry:
    return ToolRegistry([Calculator(), CurrentTime(), WebSearch()])
