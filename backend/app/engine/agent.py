"""Agent node: prompt assembly and the model <-> tools loop.

The node is stateless: InChat history and long-term memories are passed in by the
caller (chain runner), and everything that happened is returned for the trace.
LLM errors propagate; the chain runner marks the node as failed.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from app.llm import ChatMessage, LLMClient, TokenUsage, ToolCall
from app.tools import ToolOutcome, ToolRegistry

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."
FINAL_ANSWER_NUDGE = (
    "Tool budget exhausted. Answer now using the information you already have; do not call tools."
)


class AgentConfig(BaseModel):
    """Per-node settings (stored inside the flow graph)."""

    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    model: str | None = None  # None -> app default model
    tools: list[str] = Field(default_factory=list)
    temperature: float | None = None
    window_size: int = Field(default=6, ge=0, le=50)  # InChat: last N exchanges
    max_tool_iterations: int = Field(default=4, ge=0, le=10)


class AgentConfigError(ValueError):
    pass


class EventType(StrEnum):
    LLM_STARTED = "llm.started"
    LLM_FINISHED = "llm.finished"
    TOOL_STARTED = "tool.started"
    TOOL_FINISHED = "tool.finished"


@dataclass(frozen=True)
class AgentEvent:
    type: EventType
    iteration: int
    tool_call: ToolCall | None = None
    outcome: ToolOutcome | None = None
    usage: TokenUsage | None = None


EventHandler = Callable[[AgentEvent], Awaitable[None]]


@dataclass(frozen=True)
class LLMStep:
    iteration: int
    content: str
    tool_calls: tuple[ToolCall, ...]
    usage: TokenUsage
    duration_ms: int


@dataclass
class AgentResult:
    output: str
    stop_reason: Literal["final", "max_iterations"]
    llm_steps: list[LLMStep] = field(default_factory=list)
    tool_outcomes: list[ToolOutcome] = field(default_factory=list)

    @property
    def usage(self) -> TokenUsage:
        return sum((s.usage for s in self.llm_steps), TokenUsage())

    @property
    def llm_ms(self) -> int:
        return sum(s.duration_ms for s in self.llm_steps)


def build_system_prompt(base: str, memories: list[str]) -> str:
    if not memories:
        return base
    lines = "\n".join(f"- {m}" for m in memories)
    return (
        f"{base}\n\n"
        "Relevant memories from long-term memory (may be incomplete or outdated; "
        f"use them only if relevant):\n{lines}"
    )


async def _noop(_: AgentEvent) -> None:
    return None


class AgentNode:
    def __init__(
        self,
        config: AgentConfig,
        llm: LLMClient,
        tools: ToolRegistry,
        default_model: str,
    ) -> None:
        unknown = sorted(set(config.tools) - set(tools.names))
        if unknown:
            raise AgentConfigError(f"unknown tool(s) {unknown}; available: {tools.names}")
        self.config = config
        self._llm = llm
        self._tools = tools
        self._model = config.model or default_model

    async def run(
        self,
        input: str,
        history: list[ChatMessage] | None = None,
        memories: list[str] | None = None,
        on_event: EventHandler = _noop,
    ) -> AgentResult:
        cfg = self.config
        messages = [
            ChatMessage.system(build_system_prompt(cfg.system_prompt, memories or [])),
            *(history or []),
            ChatMessage.user(input),
        ]
        specs = self._tools.specs(cfg.tools) if cfg.tools else None
        result = AgentResult(output="", stop_reason="final")

        for iteration in range(cfg.max_tool_iterations + 1):
            last = iteration == cfg.max_tool_iterations
            if last and specs and result.tool_outcomes:
                messages.append(ChatMessage.user(FINAL_ANSWER_NUDGE))

            await on_event(AgentEvent(EventType.LLM_STARTED, iteration))
            response = await self._llm.chat(
                self._model,
                messages,
                tools=None if last else specs,  # no tools on the last turn -> forces an answer
                temperature=cfg.temperature,
            )
            msg = response.message
            result.llm_steps.append(
                LLMStep(
                    iteration, msg.content, msg.tool_calls, response.usage, response.duration_ms
                )
            )
            await on_event(AgentEvent(EventType.LLM_FINISHED, iteration, usage=response.usage))
            messages.append(msg)

            # On the last turn any (hallucinated) tool calls are ignored.
            if not msg.tool_calls or last:
                result.output = msg.content.strip()
                result.stop_reason = "max_iterations" if last and result.tool_outcomes else "final"
                return result

            for call in msg.tool_calls:
                await on_event(AgentEvent(EventType.TOOL_STARTED, iteration, tool_call=call))
                outcome = await self._tools.execute(call.name, call.arguments)
                result.tool_outcomes.append(outcome)
                messages.append(ChatMessage.tool(call.name, outcome.content))
                await on_event(
                    AgentEvent(EventType.TOOL_FINISHED, iteration, tool_call=call, outcome=outcome)
                )

        # Unreachable: the last iteration is called without tools, so it always returns.
        raise AssertionError("agent loop exited without a final answer")
