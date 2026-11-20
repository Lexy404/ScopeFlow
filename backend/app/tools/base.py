from typing import Any, Protocol

from app.llm import ToolSpec


class ToolError(Exception):
    """Expected tool failure (bad input, upstream error). Its message is shown to the model."""


class Tool(Protocol):
    @property
    def spec(self) -> ToolSpec: ...

    async def run(self, args: dict[str, Any]) -> str: ...
