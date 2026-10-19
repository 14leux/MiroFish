"""
OpenAI Chat Completions compatibility helpers.

This module keeps existing behavior for legacy models/providers while
gracefully adapting request parameters for GPT-5 family models.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ANTHROPIC_DEFAULT_MAX_TOKENS = 8192
_ANTHROPIC_STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
}
_native_anthropic_clients: Dict[Any, Any] = {}


def is_gpt5_family(model: Optional[str]) -> bool:
    """Return True when model belongs to GPT-5 family aliases/snapshots."""
    if not model:
        return False
    return model.strip().lower().startswith("gpt-5")


def is_anthropic_endpoint(client: Any) -> bool:
    """Return True when the client points at api.anthropic.com."""
    return "api.anthropic.com" in str(getattr(client, "base_url", "")).lower()


def _get_native_anthropic_client(client: Any) -> Any:
    """Return a cached native Anthropic SDK client for an OpenAI-style client.

    Anthropic's OpenAI-compatible endpoint ignores `cache_control`, so prompt
    caching only works through the native Messages API. The SDK appends
    `/v1/messages` itself, so the OpenAI-style `/v1` suffix is stripped.
    """
    base_url = str(getattr(client, "base_url", "")).rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[: -len("/v1")]
    api_key = getattr(client, "api_key", None)
    key = (base_url, api_key)
    native = _native_anthropic_clients.get(key)
    if native is None:
        import anthropic  # lazy: only needed when the provider is Anthropic

        native = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        _native_anthropic_clients[key] = native
    return native


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return str(content or "")


def _create_anthropic_completion(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: Optional[int],
) -> Any:
    """Call the native Messages API and return an OpenAI-shaped response.

    The system prompt carries a cache breakpoint so repeated calls that share
    it are billed as cache reads. Temperature and response_format are not sent,
    matching the earlier compatibility-endpoint behaviour; prompts already ask
    for JSON. The result exposes `choices[0].message.content` and
    `choices[0].finish_reason` so every existing caller keeps working.
    """
    system_text = "\n\n".join(
        _message_text(m.get("content")) for m in messages if m.get("role") == "system"
    )
    chat_messages = [m for m in messages if m.get("role") != "system"]

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": chat_messages,
        "max_tokens": max_tokens or _ANTHROPIC_DEFAULT_MAX_TOKENS,
    }
    if system_text:
        kwargs["system"] = [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    response = _get_native_anthropic_client(client).messages.create(**kwargs)

    usage = getattr(response, "usage", None)
    logger.info(
        "Anthropic usage: input=%s cache_write=%s cache_read=%s output=%s",
        getattr(usage, "input_tokens", 0),
        getattr(usage, "cache_creation_input_tokens", 0) or 0,
        getattr(usage, "cache_read_input_tokens", 0) or 0,
        getattr(usage, "output_tokens", 0),
    )

    text = "".join(
        block.text
        for block in getattr(response, "content", None) or []
        if getattr(block, "type", None) == "text"
    )
    stop_reason = getattr(response, "stop_reason", None)
    finish_reason = _ANTHROPIC_STOP_REASONS.get(stop_reason, stop_reason)
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


def create_chat_completion(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Create a chat completion with model-specific request parameters.

    Compatibility strategy:
    - For GPT-5 family, avoid sending temperature by default.
    - For token limit, use `max_completion_tokens` on GPT-5, `max_tokens` otherwise.
    - Preserve the legacy request shape for every non-GPT-5 model/provider.
    - Propagate provider errors unchanged instead of guessing from message text.
    - api.anthropic.com is routed to the native Messages API (prompt caching).
    """
    anthropic = is_anthropic_endpoint(client)
    if anthropic:
        # Native Messages API so prompt caching works; see helper docstring.
        return _create_anthropic_completion(
            client, model=model, messages=messages, max_tokens=max_tokens
        )

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    if response_format is not None:
        kwargs["response_format"] = response_format

    gpt5_family = is_gpt5_family(model)

    if temperature is not None and not gpt5_family:
        kwargs["temperature"] = temperature

    if max_tokens is not None:
        if gpt5_family:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

    return client.chat.completions.create(**kwargs)


def extract_chat_completion_text(response: Any) -> str:
    """Extract plain text from chat completion response across SDK content shapes."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text_obj = item.get("text")
                if isinstance(text_obj, dict):
                    text_obj = text_obj.get("value")
                if isinstance(text_obj, str):
                    chunks.append(text_obj)
                elif isinstance(item.get("content"), str):
                    chunks.append(item["content"])
                continue

            text_obj = getattr(item, "text", None)
            if isinstance(text_obj, dict):
                text_obj = text_obj.get("value")
            if isinstance(text_obj, str):
                chunks.append(text_obj)
                continue

            content_obj = getattr(item, "content", None)
            if isinstance(content_obj, str):
                chunks.append(content_obj)

        return "".join(chunks).strip()

    return str(content or "")
