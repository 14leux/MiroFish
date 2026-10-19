"""
Native Anthropic model backend for camel / OASIS simulations.

camel's own Anthropic backend still talks to Anthropic's OpenAI-compatible
endpoint, which ignores `cache_control`. Every OASIS agent resends its persona
plus its growing memory on each round, so that is the traffic prompt caching
helps most. This backend calls the native Messages API instead and places two
cache breakpoints per request:

* on the system block (tools render first, so the breakpoint also covers them);
* on the last content block of the last message (a rolling breakpoint, so each
  round reads the previous round's prefix and writes only the new tail).

Only api.anthropic.com base URLs use it; every other provider keeps the stock
OpenAI-compatible camel backend (see `create_simulation_model`).

The module avoids importing the Flask `app` package on purpose: simulation
scripts run as standalone subprocesses.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from camel.models import ModelFactory
from camel.models.openai_compatible_model import OpenAICompatibleModel
from camel.types import ModelPlatformType
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 4096
_STATS_LOG_EVERY = 100
_FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}
_CACHE = {"type": "ephemeral"}


def is_anthropic_url(url: Optional[str]) -> bool:
    return "api.anthropic.com" in (url or "").lower()


def _native_base_url(url: Optional[str]) -> str:
    """The SDK appends /v1/messages itself, so drop the OpenAI-style /v1."""
    base = (url or "https://api.anthropic.com").rstrip("/")
    return base[: -len("/v1")] if base.endswith("/v1") else base


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return str(content)


def convert_tools(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """OpenAI function tool schemas -> Anthropic tool definitions."""
    converted = []
    for tool in tools or []:
        function = tool.get("function", tool)
        converted.append(
            {
                "name": function["name"],
                "description": function.get("description", ""),
                "input_schema": function.get("parameters")
                or {"type": "object", "properties": {}},
            }
        )
    return converted


def convert_messages(
    messages: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """OpenAI chat messages -> (system blocks, Anthropic messages).

    Tool results are folded into a single user message per assistant turn, as
    the Messages API requires, and adjacent same-role messages are merged.
    The two cache breakpoints are attached here.
    """
    system_text = "\n\n".join(
        _text_of(m.get("content")) for m in messages if m.get("role") == "system"
    ).strip()

    out: List[Dict[str, Any]] = []

    def append(role: str, blocks: List[Dict[str, Any]]) -> None:
        if not blocks:
            return
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        if role == "tool":
            append(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": _text_of(message.get("content")) or "(no output)",
                    }
                ],
            )
        elif role == "assistant":
            blocks: List[Dict[str, Any]] = []
            text = _text_of(message.get("content")).rstrip()
            if text:
                blocks.append({"type": "text", "text": text})
            for call in message.get("tool_calls") or []:
                arguments = call["function"].get("arguments") or "{}"
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["function"]["name"],
                        "input": json.loads(arguments)
                        if isinstance(arguments, str)
                        else arguments,
                    }
                )
            append("assistant", blocks)
        else:
            text = _text_of(message.get("content"))
            if text.strip():
                append("user", [{"type": "text", "text": text}])

    system: List[Dict[str, Any]] = []
    if system_text:
        system.append({"type": "text", "text": system_text, "cache_control": _CACHE})
    if out:
        out[-1]["content"][-1] = {**out[-1]["content"][-1], "cache_control": _CACHE}
    return system, out


def convert_response(response: Any, model: str) -> ChatCompletion:
    """Anthropic Message -> OpenAI ChatCompletion, as camel's agent expects."""
    text = "".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    )
    tool_calls = [
        ChatCompletionMessageToolCall(
            id=block.id,
            type="function",
            function=Function(name=block.name, arguments=json.dumps(block.input)),
        )
        for block in response.content
        if getattr(block, "type", None) == "tool_use"
    ]
    usage = response.usage
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    prompt_tokens = (usage.input_tokens or 0) + cache_read + cache_write
    return ChatCompletion(
        id=response.id,
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                finish_reason=_FINISH_REASONS.get(response.stop_reason, "stop"),
                message=ChatCompletionMessage(
                    role="assistant",
                    content=text or None,
                    tool_calls=tool_calls or None,
                ),
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=usage.output_tokens or 0,
            total_tokens=prompt_tokens + (usage.output_tokens or 0),
        ),
    )


class NativeAnthropicModel(OpenAICompatibleModel):
    """camel model backend that calls Anthropic's native Messages API."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._native_client = None
        self._native_async_client = None
        self.cache_stats = {"calls": 0, "input": 0, "cache_write": 0, "cache_read": 0}

    def _clients(self):
        if self._native_client is None:
            import anthropic  # imported lazily: only needed for Anthropic URLs

            options = dict(
                api_key=self._api_key,
                base_url=_native_base_url(self._url),
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
            self._native_client = anthropic.Anthropic(**options)
            self._native_async_client = anthropic.AsyncAnthropic(**options)
        return self._native_client, self._native_async_client

    def _build_request(
        self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        # Sampling parameters are deliberately not forwarded: newer Claude
        # models reject non-default temperature/top_p.
        system, chat_messages = convert_messages(messages)
        request: Dict[str, Any] = {
            "model": str(self.model_type),
            "messages": chat_messages,
            "max_tokens": self.model_config_dict.get("max_tokens")
            or DEFAULT_MAX_TOKENS,
        }
        if system:
            request["system"] = system
        converted_tools = convert_tools(tools)
        if converted_tools:
            request["tools"] = converted_tools
        return request

    def _record(self, response: Any) -> None:
        usage = response.usage
        stats = self.cache_stats
        stats["calls"] += 1
        stats["input"] += usage.input_tokens or 0
        stats["cache_write"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
        stats["cache_read"] += getattr(usage, "cache_read_input_tokens", 0) or 0
        if stats["calls"] % _STATS_LOG_EVERY == 0:
            total = stats["input"] + stats["cache_write"] + stats["cache_read"]
            logger.info(
                "Anthropic cache after %d calls: read=%d write=%d uncached=%d (%.0f%% read)",
                stats["calls"],
                stats["cache_read"],
                stats["cache_write"],
                stats["input"],
                100 * stats["cache_read"] / total if total else 0,
            )

    def _request_chat_completion(self, messages, tools=None):
        sync_client, _ = self._clients()
        response = sync_client.messages.create(**self._build_request(messages, tools))
        self._record(response)
        return convert_response(response, str(self.model_type))

    async def _arequest_chat_completion(self, messages, tools=None):
        _, async_client = self._clients()
        response = await async_client.messages.create(
            **self._build_request(messages, tools)
        )
        self._record(response)
        return convert_response(response, str(self.model_type))


def create_simulation_model(
    *, model_name: str, api_key: str, base_url: Optional[str]
):
    """Return the camel model for a simulation run.

    Anthropic URLs get the native, cache-aware backend; everything else keeps
    the stock OpenAI-compatible backend configured through the environment by
    the calling script.
    """
    if is_anthropic_url(base_url):
        return NativeAnthropicModel(
            model_type=model_name, api_key=api_key, url=base_url
        )
    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=model_name,
    )
