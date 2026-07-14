"""Adapter protocol plus deterministic and OpenAI-wire helpers."""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from typing import Any, Protocol

import httpx

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
        response = await self._client.post("chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()
        return parse_openai_turn(body["choices"][0]["message"])

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

    calls: list[ToolCall] = []
    for raw_call in payload.get("tool_calls", []):
        function = raw_call["function"]
        raw_arguments = function.get("arguments", "{}")
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        if not isinstance(arguments, dict):
            raise ValueError("tool call arguments must decode to an object")
        calls.append(
            ToolCall(
                call_id=str(raw_call["id"]),
                name=str(function["name"]),
                arguments=arguments,
            )
        )
    return AgentTurn(content=str(payload.get("content") or ""), tool_calls=tuple(calls))
