"""Adapter protocol plus deterministic and OpenAI-wire helpers."""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from typing import Any, Protocol

import httpx

from .errors import AdapterProtocolError, AdapterTimeoutError, AdapterTransportError
from .models import AgentTurn, Message, ToolCall


class AgentAdapter(Protocol):
    async def respond(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> AgentTurn: ...


class ScriptedAdapter:
    """Deterministic adapter used to verify the conformance runtime itself."""

    def __init__(self, turns: Sequence[AgentTurn]) -> None:
        self._turns = list(turns)
        self.requests: list[tuple[tuple[Message, ...], tuple[dict[str, Any], ...]]] = []

    async def respond(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> AgentTurn:
        self.requests.append((tuple(messages), tuple(copy.deepcopy(tools))))
        if not self._turns:
            raise RuntimeError("scripted adapter has no turns left")
        return self._turns.pop(0)


class OpenAICompatibleAdapter:
    """Minimal live adapter for OpenAI-compatible chat-completions endpoints."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 60,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self._owns_client = client is None
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers=headers,
            timeout=timeout,
        )

    async def respond(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> AgentTurn:
        payload = {
            "model": self.model,
            "messages": [openai_message(message) for message in messages],
            "tools": [openai_tool_definition(tool) for tool in tools],
            "tool_choice": "auto",
            "temperature": 0,
        }
        try:
            response = await self._client.post("chat/completions", json=payload)
        except httpx.TimeoutException as error:
            raise AdapterTimeoutError("provider request timed out") from error
        except httpx.HTTPError as error:
            raise AdapterTransportError(
                f"provider transport failed with {type(error).__name__}"
            ) from error

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise AdapterTransportError(
                f"provider returned HTTP {response.status_code}"
            ) from error

        try:
            body = response.json()
        except ValueError as error:
            raise AdapterProtocolError("provider response is not valid JSON") from error
        message = _chat_message(body)
        return parse_openai_turn(message)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def openai_tool_definition(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert a neutral definition without normalizing or dropping schema fields."""

    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": copy.deepcopy(tool["input_schema"]),
        },
    }


def openai_message(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name is not None:
        payload["name"] = message.name
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(
                        call.arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                },
            }
            for call in message.tool_calls
        ]
    return payload


def parse_openai_turn(payload: dict[str, Any]) -> AgentTurn:
    """Parse one OpenAI-compatible assistant message into the neutral contract."""

    content = payload.get("content")
    if content is not None and not isinstance(content, str):
        raise AdapterProtocolError("assistant content must be a string or null")

    raw_calls = payload.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        raise AdapterProtocolError("assistant tool_calls must be an array")

    calls: list[ToolCall] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            raise AdapterProtocolError("each tool call must be an object")
        function = raw_call.get("function")
        if not isinstance(function, dict):
            raise AdapterProtocolError("tool call function must be an object")
        call_id = raw_call.get("id")
        name = function.get("name")
        if not isinstance(call_id, str) or not call_id:
            raise AdapterProtocolError("tool call id must be a non-empty string")
        if not isinstance(name, str) or not name:
            raise AdapterProtocolError("tool call name must be a non-empty string")
        raw_arguments = function.get("arguments", "{}")
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as error:
                raise AdapterProtocolError("tool call arguments are not valid JSON") from error
        else:
            arguments = raw_arguments
        if not isinstance(arguments, dict):
            raise AdapterProtocolError("tool call arguments must decode to an object")
        calls.append(
            ToolCall(
                call_id=call_id,
                name=name,
                arguments=arguments,
            )
        )
    return AgentTurn(content=content or "", tool_calls=tuple(calls))


def _chat_message(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise AdapterProtocolError("provider response must be an object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AdapterProtocolError("provider response must contain at least one choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise AdapterProtocolError("provider choice must be an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise AdapterProtocolError("provider choice must contain an assistant message")
    return message
