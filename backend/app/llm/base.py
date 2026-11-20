"""Provider-agnostic LLM types and the `LLMClient` protocol.

The engine depends only on this module; Ollama (or any future provider) lives behind it.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class ToolCall:
    id: str  # assigned by us (Ollama has no call ids); used to match tool results in traces
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()  # assistant messages only
    tool_name: str | None = None  # tool messages only: which tool produced this result

    @staticmethod
    def system(content: str) -> "ChatMessage":
        return ChatMessage(Role.SYSTEM, content)

    @staticmethod
    def user(content: str) -> "ChatMessage":
        return ChatMessage(Role.USER, content)

    @staticmethod
    def assistant(content: str, tool_calls: tuple[ToolCall, ...] = ()) -> "ChatMessage":
        return ChatMessage(Role.ASSISTANT, content, tool_calls)

    @staticmethod
    def tool(name: str, content: str) -> "ChatMessage":
        return ChatMessage(Role.TOOL, content, tool_name=name)


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})


@dataclass(frozen=True)
class TokenUsage:
    prompt: int = 0
    completion: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(self.prompt + other.prompt, self.completion + other.completion)


@dataclass(frozen=True)
class LLMResponse:
    message: ChatMessage  # role=assistant; may carry tool_calls
    usage: TokenUsage = TokenUsage()
    duration_ms: int = 0


class LLMError(Exception):
    """Base class for LLM failures the UI can show as a toast."""


class LLMUnavailableError(LLMError):
    """The provider could not be reached (e.g. Ollama is not running)."""


class ModelNotFoundError(LLMError):
    def __init__(self, model: str) -> None:
        super().__init__(f"Model '{model}' is not available. Pull or create it in Ollama first.")
        self.model = model


class LLMClient(Protocol):
    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse: ...

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]: ...

    async def list_models(self) -> list[str]: ...
