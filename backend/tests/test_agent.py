from typing import Any

import pytest

from app.config import get_settings
from app.engine import AgentConfig, AgentConfigError, AgentEvent, AgentNode, EventType
from app.llm import ChatMessage, LLMResponse, ModelNotFoundError, OllamaClient, Role, ToolSpec
from app.llm.fake import FakeLLMClient, call_tool, reply
from app.tools import default_registry


def agent(
    script: list[Any], tools: list[str] | None = None, **cfg: Any
) -> tuple[AgentNode, FakeLLMClient]:
    fake = FakeLLMClient(script)
    config = AgentConfig(tools=tools or [], **cfg)
    return AgentNode(config, fake, default_registry(), default_model="default-m"), fake


async def test_plain_answer_without_tools() -> None:
    node, fake = agent([reply("  Hello!  ")], system_prompt="Be nice.")
    result = await node.run("hi")

    assert (result.output, result.stop_reason) == ("Hello!", "final")
    [call] = fake.calls
    assert call.model == "default-m"
    assert call.tools is None  # no tools configured -> none sent
    assert [m.role for m in call.messages] == [Role.SYSTEM, Role.USER]
    assert call.messages[0].content == "Be nice."


async def test_model_override() -> None:
    node, fake = agent([reply("x")], model="custom")
    await node.run("hi")
    assert fake.calls[0].model == "custom"


async def test_tool_roundtrip() -> None:
    node, fake = agent(
        [call_tool("calculator", expression="1234 * 5678"), reply("It is 7006652.")],
        tools=["calculator"],
    )
    result = await node.run("What is 1234*5678?")

    assert result.output == "It is 7006652."
    [outcome] = result.tool_outcomes
    assert (outcome.name, outcome.ok, outcome.content) == ("calculator", True, "7006652")
    # second LLM call sees: system, user, assistant(tool call), tool(result)
    second = fake.calls[1].messages
    assert [m.role for m in second] == [Role.SYSTEM, Role.USER, Role.ASSISTANT, Role.TOOL]
    assert (second[3].tool_name, second[3].content) == ("calculator", "7006652")
    assert [s.name for s in fake.calls[0].tools or []] == ["calculator"]


async def test_tool_error_is_given_back_to_the_model() -> None:
    node, fake = agent(
        [call_tool("calculator", expression="1/0"), reply("Cannot divide by zero.")],
        tools=["calculator"],
    )
    result = await node.run("1/0?")
    assert result.tool_outcomes[0].ok is False
    assert fake.calls[1].messages[-1].content == "Error: division by zero"
    assert result.output == "Cannot divide by zero."


async def test_max_iterations_forces_final_answer_without_tools() -> None:
    loop = [call_tool("calculator", expression="1+1")] * 2
    node, fake = agent([*loop, reply("Final.")], tools=["calculator"], max_tool_iterations=2)
    result = await node.run("loop forever")

    assert (result.output, result.stop_reason) == ("Final.", "max_iterations")
    assert len(fake.calls) == 3
    assert fake.calls[0].tools and fake.calls[1].tools
    assert fake.calls[2].tools is None  # last turn: tools withheld
    assert "Tool budget exhausted" in fake.calls[2].messages[-1].content


async def test_tool_calls_on_last_turn_are_ignored() -> None:
    hallucinated = LLMResponse(
        ChatMessage.assistant(
            "partial answer", call_tool("calculator", expression="1").message.tool_calls
        )
    )
    node, _ = agent([hallucinated], tools=["calculator"], max_tool_iterations=0)
    result = await node.run("x")
    assert result.output == "partial answer" and not result.tool_outcomes


async def test_history_and_memories_are_placed_correctly() -> None:
    node, fake = agent([reply("ok")], system_prompt="Base.")
    history = [ChatMessage.user("earlier q"), ChatMessage.assistant("earlier a")]
    await node.run("now", history=history, memories=["User lives in Jerusalem"])

    msgs = fake.calls[0].messages
    assert [m.content for m in msgs[1:]] == ["earlier q", "earlier a", "now"]
    assert msgs[0].content.startswith("Base.\n\nRelevant memories")
    assert "- User lives in Jerusalem" in msgs[0].content


async def test_usage_timing_and_events() -> None:
    node, _ = agent([call_tool("calculator", expression="2*3"), reply("6")], tools=["calculator"])
    events: list[AgentEvent] = []

    async def on_event(e: AgentEvent) -> None:
        events.append(e)

    result = await node.run("2*3", on_event=on_event)

    assert result.usage.total == 30  # 2 fake calls x (10 prompt + 5 completion)
    assert [e.type for e in events] == [
        EventType.LLM_STARTED,
        EventType.LLM_FINISHED,
        EventType.TOOL_STARTED,
        EventType.TOOL_FINISHED,
        EventType.LLM_STARTED,
        EventType.LLM_FINISHED,
    ]
    assert events[3].outcome is not None and events[3].outcome.content == "6"


def test_unknown_tool_in_config_is_rejected() -> None:
    with pytest.raises(AgentConfigError, match="nope"):
        AgentNode(AgentConfig(tools=["nope"]), FakeLLMClient(), default_registry(), "m")


async def test_llm_errors_propagate() -> None:
    def boom(messages: list[ChatMessage], tools: list[ToolSpec] | None) -> LLMResponse:
        raise ModelNotFoundError("ghost")

    node, _ = agent([boom])
    with pytest.raises(ModelNotFoundError):
        await node.run("hi")


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        AgentConfig(max_tool_iterations=99)
    with pytest.raises(ValueError):
        AgentConfig(window_size=-1)


# --- live: real model + real tools -------------------------------------------------------


@pytest.mark.ollama
async def test_live_agent_uses_calculator(live_ollama: OllamaClient) -> None:
    node = AgentNode(
        AgentConfig(tools=["calculator"], temperature=0),
        live_ollama,
        default_registry(),
        get_settings().default_model,
    )
    result = await node.run("What is 1234 * 5678? Use the calculator.")
    assert any(o.name == "calculator" and o.ok for o in result.tool_outcomes)
    assert "7006652" in result.output.replace(",", "").replace(" ", "")
