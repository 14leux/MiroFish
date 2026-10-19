import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "scripts")
)

import anthropic_camel_model as acm  # noqa: E402

CACHE = {"type": "ephemeral"}


def test_system_and_last_message_carry_cache_breakpoints():
    system, messages = acm.convert_messages(
        [
            {"role": "system", "content": "persona"},
            {"role": "user", "content": "round 1"},
            {"role": "assistant", "content": "ok  \n"},
            {"role": "user", "content": "round 2"},
        ]
    )

    assert system == [{"type": "text", "text": "persona", "cache_control": CACHE}]
    assert messages[0] == {"role": "user", "content": [{"type": "text", "text": "round 1"}]}
    assert messages[1]["content"] == [{"type": "text", "text": "ok"}]
    assert messages[-1]["content"] == [
        {"type": "text", "text": "round 2", "cache_control": CACHE}
    ]


def test_tool_round_trip_merges_results_into_one_user_message():
    _, messages = acm.convert_messages(
        [
            {"role": "user", "content": "act"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "t1", "type": "function",
                     "function": {"name": "like", "arguments": '{"post_id": 3}'}},
                    {"id": "t2", "type": "function",
                     "function": {"name": "follow", "arguments": '{"user_id": 9}'}},
                ],
            },
            {"role": "tool", "tool_call_id": "t1", "content": "liked"},
            {"role": "tool", "tool_call_id": "t2", "content": ""},
        ]
    )

    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"] == [
        {"type": "tool_use", "id": "t1", "name": "like", "input": {"post_id": 3}},
        {"type": "tool_use", "id": "t2", "name": "follow", "input": {"user_id": 9}},
    ]
    assert messages[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "liked"},
        {"type": "tool_result", "tool_use_id": "t2", "content": "(no output)",
         "cache_control": CACHE},
    ]


def test_no_system_prompt_omits_system_block():
    system, messages = acm.convert_messages([{"role": "user", "content": "hi"}])
    assert system == []
    assert messages[0]["content"][-1]["cache_control"] == CACHE


def test_convert_tools_maps_openai_function_schema():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "like_post",
                "description": "Like a post",
                "parameters": {"type": "object", "properties": {"post_id": {"type": "integer"}}},
            },
        }
    ]
    assert acm.convert_tools(tools) == [
        {
            "name": "like_post",
            "description": "Like a post",
            "input_schema": {"type": "object", "properties": {"post_id": {"type": "integer"}}},
        }
    ]
    assert acm.convert_tools(None) == []


def anthropic_message(content, stop_reason="end_turn", **usage):
    return SimpleNamespace(
        id="msg_1",
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=usage.get("input", 10),
            output_tokens=usage.get("output", 5),
            cache_creation_input_tokens=usage.get("write", 0),
            cache_read_input_tokens=usage.get("read", 0),
        ),
    )


def test_convert_response_maps_text_tool_calls_and_usage():
    response = anthropic_message(
        [
            SimpleNamespace(type="text", text="thinking aloud"),
            SimpleNamespace(type="tool_use", id="t1", name="like", input={"post_id": 3}),
        ],
        stop_reason="tool_use",
        input=10,
        write=100,
        read=900,
        output=7,
    )

    completion = acm.convert_response(response, "claude-sonnet-5-5")

    choice = completion.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content == "thinking aloud"
    assert choice.message.tool_calls[0].id == "t1"
    assert choice.message.tool_calls[0].function.name == "like"
    assert choice.message.tool_calls[0].function.arguments == '{"post_id": 3}'
    assert completion.usage.prompt_tokens == 1010
    assert completion.usage.total_tokens == 1017


@pytest.mark.parametrize(
    "stop_reason,expected",
    [("end_turn", "stop"), ("max_tokens", "length"), ("stop_sequence", "stop")],
)
def test_finish_reason_mapping(stop_reason, expected):
    response = anthropic_message([SimpleNamespace(type="text", text="x")], stop_reason)
    assert acm.convert_response(response, "m").choices[0].finish_reason == expected


class FakeAsyncMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def make_model(monkeypatch, responses):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    model = acm.NativeAnthropicModel(
        model_type="claude-sonnet-5-5",
        api_key="test-key",
        url="https://api.anthropic.com/v1/",
    )
    fake = FakeAsyncMessages(responses)
    model._native_client = SimpleNamespace(messages=fake)
    model._native_async_client = SimpleNamespace(messages=fake)
    return model, fake


def test_async_request_shape_and_cache_stats(monkeypatch):
    model, fake = make_model(
        monkeypatch,
        [anthropic_message([SimpleNamespace(type="text", text="done")], read=800, write=50)],
    )

    completion = asyncio.run(
        model._arequest_chat_completion(
            [
                {"role": "system", "content": "persona"},
                {"role": "user", "content": "observe"},
            ],
            tools=None,
        )
    )

    request = fake.calls[0]
    assert request["model"] == "claude-sonnet-5-5"
    assert request["max_tokens"] == acm.DEFAULT_MAX_TOKENS
    assert request["system"][0]["cache_control"] == CACHE
    assert "temperature" not in request and "tools" not in request
    assert completion.choices[0].message.content == "done"
    assert model.cache_stats == {"calls": 1, "input": 10, "cache_write": 50, "cache_read": 800}


def test_native_base_url_drops_openai_v1_suffix():
    assert acm._native_base_url("https://api.anthropic.com/v1/") == "https://api.anthropic.com"
    assert acm._native_base_url("https://api.anthropic.com") == "https://api.anthropic.com"


def test_factory_picks_backend_by_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    native = acm.create_simulation_model(
        model_name="claude-sonnet-5-5",
        api_key="k",
        base_url="https://api.anthropic.com/v1/",
    )
    stock = acm.create_simulation_model(
        model_name="gpt-4o-mini", api_key="k", base_url="https://api.openai.com/v1"
    )

    assert isinstance(native, acm.NativeAnthropicModel)
    assert not isinstance(stock, acm.NativeAnthropicModel)


def test_camel_chat_agent_runs_a_tool_loop_through_the_backend(monkeypatch):
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage
    from camel.toolkits import FunctionTool

    def add(a: int, b: int) -> int:
        """Add two integers.

        Args:
            a (int): first
            b (int): second
        """
        return a + b

    model, fake = make_model(
        monkeypatch,
        [
            anthropic_message(
                [SimpleNamespace(type="tool_use", id="t1", name="add", input={"a": 2, "b": 3})],
                stop_reason="tool_use",
            ),
            anthropic_message([SimpleNamespace(type="text", text="The sum is 5")]),
        ],
    )
    agent = ChatAgent(
        system_message=BaseMessage.make_assistant_message("Bot", "You add numbers."),
        model=model,
        tools=[FunctionTool(add)],
    )

    response = asyncio.run(agent.astep("add 2 and 3"))

    assert response.msgs[0].content == "The sum is 5"
    first, second = fake.calls
    assert first["tools"][0]["name"] == "add"
    result_message = second["messages"][-1]
    assert result_message["role"] == "user"
    assert result_message["content"][0]["type"] == "tool_result"
    assert result_message["content"][0]["content"] == "5"
