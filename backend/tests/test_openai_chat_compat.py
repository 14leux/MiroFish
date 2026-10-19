from types import SimpleNamespace

import pytest

from app.utils.openai_chat_compat import (
    create_chat_completion,
    extract_chat_completion_text,
    is_gpt5_family,
)


class CompletionRecorder:
    def __init__(self, result=None, error=None):
        self.result = result or object()
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def client_for(recorder):
    return SimpleNamespace(chat=SimpleNamespace(completions=recorder))


def test_gpt5_uses_completion_token_limit_without_temperature():
    recorder = CompletionRecorder()
    messages = [{"role": "user", "content": "hello"}]

    result = create_chat_completion(
        client_for(recorder),
        model="gpt-5-2025-08-07",
        messages=messages,
        temperature=0.2,
        max_tokens=123,
        response_format={"type": "json_object"},
    )

    assert result is recorder.result
    assert recorder.calls == [
        {
            "model": "gpt-5-2025-08-07",
            "messages": messages,
            "max_completion_tokens": 123,
            "response_format": {"type": "json_object"},
        }
    ]


def test_legacy_model_preserves_original_request_shape():
    recorder = CompletionRecorder()
    messages = [{"role": "user", "content": "hello"}]

    create_chat_completion(
        client_for(recorder),
        model="third-party-chat-model",
        messages=messages,
        temperature=0.7,
        max_tokens=456,
        response_format={"type": "json_object"},
    )

    assert recorder.calls == [
        {
            "model": "third-party-chat-model",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 456,
            "response_format": {"type": "json_object"},
        }
    ]


def test_provider_error_is_propagated_without_guessing_or_retrying():
    provider_error = RuntimeError("unsupported max_tokens due to a server outage")
    recorder = CompletionRecorder(error=provider_error)

    with pytest.raises(RuntimeError) as captured:
        create_chat_completion(
            client_for(recorder),
            model="legacy-model",
            messages=[],
            max_tokens=10,
        )

    assert captured.value is provider_error
    assert len(recorder.calls) == 1


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5", True),
        (" GPT-5.1-mini ", True),
        ("gpt-4.1", False),
        ("my-gpt-5-proxy", False),
        (None, False),
    ],
)
def test_gpt5_family_detection(model, expected):
    assert is_gpt5_family(model) is expected


def test_extracts_text_from_supported_content_shapes():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[
                        {"text": {"value": "first"}},
                        {"content": " second"},
                        SimpleNamespace(text=" third"),
                    ]
                )
            )
        ]
    )

    assert extract_chat_completion_text(response) == "first second third"
    assert extract_chat_completion_text(SimpleNamespace(choices=[])) == ""


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def anthropic_response(text="{}", stop_reason="end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


def anthropic_client():
    return SimpleNamespace(
        base_url="https://api.anthropic.com/v1/",
        api_key="test-key",
        chat=SimpleNamespace(completions=CompletionRecorder()),
    )


def test_anthropic_endpoint_uses_native_messages_with_cached_system(monkeypatch):
    from app.utils import openai_chat_compat as compat

    fake = FakeMessages(anthropic_response('{"a": 1}'))
    monkeypatch.setattr(
        compat, "_get_native_anthropic_client", lambda _c: SimpleNamespace(messages=fake)
    )
    client = anthropic_client()

    result = create_chat_completion(
        client,
        model="claude-sonnet-5-5",
        messages=[
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ],
        temperature=0.7,
        max_tokens=None,
        response_format={"type": "json_object"},
    )

    assert fake.calls == [
        {
            "model": "claude-sonnet-5-5",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8192,
            "system": [
                {
                    "type": "text",
                    "text": "be terse",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]
    assert client.chat.completions.calls == []
    assert extract_chat_completion_text(result) == '{"a": 1}'
    assert result.choices[0].finish_reason == "stop"


def test_anthropic_max_tokens_stop_reason_maps_to_length(monkeypatch):
    from app.utils import openai_chat_compat as compat

    fake = FakeMessages(anthropic_response("{", stop_reason="max_tokens"))
    monkeypatch.setattr(
        compat, "_get_native_anthropic_client", lambda _c: SimpleNamespace(messages=fake)
    )

    result = create_chat_completion(
        anthropic_client(),
        model="claude-sonnet-5-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
    )

    assert result.choices[0].finish_reason == "length"
    assert "system" not in fake.calls[0]
    assert fake.calls[0]["max_tokens"] == 100


def test_native_client_strips_v1_suffix(monkeypatch):
    import sys
    from app.utils import openai_chat_compat as compat

    seen = {}

    class FakeAnthropic:
        def __init__(self, api_key, base_url):
            seen.update(api_key=api_key, base_url=base_url)

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))
    monkeypatch.setattr(compat, "_native_anthropic_clients", {})

    compat._get_native_anthropic_client(anthropic_client())

    assert seen == {"api_key": "test-key", "base_url": "https://api.anthropic.com"}
