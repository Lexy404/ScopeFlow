"""Deterministic `LLMClient` for tests: replays scripted responses and records calls."""

from collections.abc import Callable
from dataclasses import dataclass

from app.llm.base import ChatMessage, LLMResponse, TokenUsage, ToolCall, ToolSpec

Responder = Callable[[list[ChatMessage], list[ToolSpec] | None], LLMResponse]


@dataclass
class FakeCall:
    model: str
    messages: list[ChatMessage]
    tools: list[ToolSpec] | None


def reply(text: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> LLMResponse:
    return LLMResponse(ChatMessage.assistant(text), TokenUsage(prompt_tokens, completion_tokens))


def call_tool(name: str, **arguments: object) -> LLMResponse:
    call = ToolCall(id=f"call_{name}", name=name, arguments=dict(arguments))
    return LLMResponse(ChatMessage.assistant("", (call,)), TokenUsage(10, 5))


class FakeLLMClient:
    """Each `chat` call consumes the next scripted item (a response or a responder function)."""

    def __init__(
        self,
        script: list[LLMResponse | Responder] | None = None,
        models: list[str] | None = None,
    ) -> None:
        self._script = list(script or [])
        self._models = models or ["fake-model"]
        self.calls: list[FakeCall] = []

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls.append(FakeCall(model, list(messages), tools))
        if not self._script:
            raise AssertionError("FakeLLMClient: script exhausted")
        item = self._script.pop(0)
        return item if isinstance(item, LLMResponse) else item(messages, tools)

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        # Cheap deterministic vectors: good enough for wiring tests, not for similarity.
        return [[float(len(t)), float(sum(map(ord, t)) % 997)] for t in texts]

    async def list_models(self) -> list[str]:
        return list(self._models)
