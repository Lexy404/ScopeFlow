from typing import Any

import httpx
import ollama
import pytest
from ollama import ChatResponse, EmbedResponse, ListResponse, Message

from app.config import get_settings
from app.llm import (
    ChatMessage,
    LLMError,
    LLMUnavailableError,
    ModelNotFoundError,
    OllamaClient,
    Role,
    ToolCall,
    ToolSpec,
)
from app.llm.fake import FakeLLMClient, call_tool, reply

CALC = ToolSpec(
    "calculator",
    "Evaluate an arithmetic expression",
    {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
)


class StubOllama:
    """Stands in for ollama.AsyncClient; records the payload and returns a canned result."""

    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result, self.error = result, error
        self.sent: dict[str, Any] = {}

    async def _respond(self, **kwargs: Any) -> Any:
        self.sent = kwargs
        if self.error:
            raise self.error
        return self.result

    async def chat(self, **kwargs: Any) -> Any:
        return await self._respond(**kwargs)

    async def embed(self, **kwargs: Any) -> Any:
        return await self._respond(**kwargs)

    async def list(self) -> Any:
        return await self._respond()


def client_with(stub: StubOllama) -> OllamaClient:
    return OllamaClient(host="unused", client=stub)  # type: ignore[arg-type]


# --- OllamaClient: request/response mapping -------------------------------------------


async def test_chat_maps_messages_tools_and_options() -> None:
    stub = StubOllama(ChatResponse(message=Message(role="assistant", content="hi")))
    history = [
        ChatMessage.system("be brief"),
        ChatMessage.user("2+2?"),
        ChatMessage.assistant("", (ToolCall("c1", "calculator", {"expression": "2+2"}),)),
        ChatMessage.tool("calculator", "4"),
    ]
    await client_with(stub).chat("m", history, tools=[CALC], temperature=0)

    assert stub.sent["model"] == "m"
    assert stub.sent["options"] == {"temperature": 0}
    assert stub.sent["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "2+2?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "calculator", "arguments": {"expression": "2+2"}}}
            ],
        },
        {"role": "tool", "content": "4", "tool_name": "calculator"},
    ]
    assert stub.sent["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": CALC.description,
                "parameters": CALC.parameters,
            },
        }
    ]


async def test_chat_without_tools_sends_none() -> None:
    stub = StubOllama(ChatResponse(message=Message(role="assistant", content="ok")))
    await client_with(stub).chat("m", [ChatMessage.user("x")])
    assert stub.sent["tools"] is None
    assert stub.sent["options"] is None


async def test_chat_parses_tool_calls_and_usage() -> None:
    fn = Message.ToolCall.Function(name="calculator", arguments={"expression": "6*7"})
    stub = StubOllama(
        ChatResponse(
            message=Message(
                role="assistant", content="", tool_calls=[Message.ToolCall(function=fn)]
            ),
            prompt_eval_count=120,
            eval_count=15,
        )
    )
    r = await client_with(stub).chat("m", [ChatMessage.user("6*7?")], tools=[CALC])

    assert r.message.role is Role.ASSISTANT
    [call] = r.message.tool_calls
    assert (call.name, call.arguments) == ("calculator", {"expression": "6*7"})
    assert call.id.startswith("call_")
    assert (r.usage.prompt, r.usage.completion, r.usage.total) == (120, 15, 135)
    assert r.duration_ms >= 0


async def test_embed_and_list_models() -> None:
    emb = await client_with(StubOllama(EmbedResponse(embeddings=[[0.1, 0.2]]))).embed("e", ["a"])
    assert emb == [[0.1, 0.2]]

    models = ListResponse(models=[ListResponse.Model(model="b"), ListResponse.Model(model="a")])
    assert await client_with(StubOllama(models)).list_models() == ["a", "b"]


# --- OllamaClient: error mapping ------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ollama.ResponseError("model 'x' not found", 404), ModelNotFoundError),
        (ollama.ResponseError("boom", 500), LLMError),
        (ConnectionError("Failed to connect to Ollama"), LLMUnavailableError),
        (httpx.ConnectError("refused"), LLMUnavailableError),
    ],
)
async def test_errors_are_mapped(error: Exception, expected: type[Exception]) -> None:
    with pytest.raises(expected):
        await client_with(StubOllama(error=error)).chat("x", [ChatMessage.user("hi")])


async def test_model_not_found_message_names_the_model() -> None:
    stub = StubOllama(error=ollama.ResponseError("not found", 404))
    with pytest.raises(ModelNotFoundError, match="'missing-model'"):
        await client_with(stub).chat("missing-model", [ChatMessage.user("hi")])


# --- FakeLLMClient --------------------------------------------------------------------


async def test_fake_client_replays_script_and_records_calls() -> None:
    fake = FakeLLMClient([call_tool("calculator", expression="1+1"), reply("2")])
    first = await fake.chat("m", [ChatMessage.user("1+1?")], tools=[CALC])
    second = await fake.chat("m", [ChatMessage.user("1+1?")])

    assert first.message.tool_calls[0].arguments == {"expression": "1+1"}
    assert second.message.content == "2"
    assert len(fake.calls) == 2 and fake.calls[0].tools == [CALC]
    with pytest.raises(AssertionError, match="exhausted"):
        await fake.chat("m", [])


# --- Live Ollama (skipped when Ollama is not running, e.g. in CI) -----------------------


@pytest.fixture
async def live() -> OllamaClient:
    s = get_settings()
    try:
        async with httpx.AsyncClient(timeout=2) as c:
            await c.get(f"{s.ollama_url}/api/version")
    except httpx.HTTPError:
        pytest.skip("Ollama not running")
    client = OllamaClient(s.ollama_url)
    if s.default_model not in {m.split(":")[0] for m in await client.list_models()}:
        pytest.skip(f"model {s.default_model} not installed")
    return client


@pytest.mark.ollama
async def test_live_tool_call_and_embedding(live: OllamaClient) -> None:
    s = get_settings()
    r = await live.chat(
        s.default_model,
        [ChatMessage.user("What is 17 * 23? Use the calculator tool.")],
        tools=[CALC],
        temperature=0,
    )
    assert r.message.tool_calls, f"expected a tool call, got {r.message.content!r}"
    assert r.message.tool_calls[0].name == "calculator"
    assert r.usage.prompt > 0 and r.usage.completion > 0

    [vec] = await live.embed(s.embedding_model, ["hello"])
    assert len(vec) == 768
