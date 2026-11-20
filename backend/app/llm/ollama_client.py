import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import httpx
import ollama

from app.llm.base import (
    ChatMessage,
    LLMError,
    LLMResponse,
    LLMUnavailableError,
    ModelNotFoundError,
    Role,
    TokenUsage,
    ToolCall,
    ToolSpec,
)

T = TypeVar("T")


def _to_ollama_message(m: ChatMessage) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": m.role.value, "content": m.content}
    if m.tool_calls:
        msg["tool_calls"] = [
            {"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls
        ]
    if m.tool_name:
        msg["tool_name"] = m.tool_name
    return msg


def _to_ollama_tool(t: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
    }


def _from_ollama_response(r: ollama.ChatResponse, duration_ms: int) -> LLMResponse:
    calls = tuple(
        ToolCall(
            id=f"call_{uuid.uuid4().hex[:12]}",
            name=c.function.name,
            arguments=dict(c.function.arguments or {}),
        )
        for c in (r.message.tool_calls or [])
    )
    return LLMResponse(
        message=ChatMessage(Role.ASSISTANT, r.message.content or "", calls),
        usage=TokenUsage(prompt=r.prompt_eval_count or 0, completion=r.eval_count or 0),
        duration_ms=duration_ms,
    )


class OllamaClient:
    """`LLMClient` backed by a local Ollama server."""

    def __init__(self, host: str, client: ollama.AsyncClient | None = None) -> None:
        self._client = client or ollama.AsyncClient(host=host)

    async def _call(self, model: str, fn: Callable[[], Awaitable[T]]) -> T:
        try:
            return await fn()
        except ollama.ResponseError as e:
            if e.status_code == 404:
                raise ModelNotFoundError(model) from e
            raise LLMError(f"Ollama error ({e.status_code}): {e.error}") from e
        except (ConnectionError, httpx.ConnectError, httpx.TimeoutException) as e:
            raise LLMUnavailableError("Ollama is not reachable. Is it running?") from e

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        options = {"temperature": temperature} if temperature is not None else None
        start = time.perf_counter()
        r = await self._call(
            model,
            lambda: self._client.chat(
                model=model,
                messages=[_to_ollama_message(m) for m in messages],
                tools=[_to_ollama_tool(t) for t in tools] if tools else None,
                options=options,
            ),
        )
        return _from_ollama_response(r, int((time.perf_counter() - start) * 1000))

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        r = await self._call(model, lambda: self._client.embed(model=model, input=texts))
        return [list(v) for v in r.embeddings]

    async def list_models(self) -> list[str]:
        r = await self._call("", self._client.list)
        return sorted(m.model for m in r.models if m.model)
