from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ruagent_compat.adapters import (
    OpenAICompatibleAdapter,
    ScriptedAdapter,
    openai_message,
    openai_tool_definition,
    parse_openai_turn,
)
from ruagent_compat.errors import (
    AdapterProtocolError,
    AdapterTimeoutError,
    AdapterTransportError,
)
from ruagent_compat.models import AgentTurn, Message, ToolCall


def test_openai_wire_preserves_all_schema_keywords() -> None:
    schema = {
        "$defs": {"value": {"type": ["string", "null"]}},
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"$ref": "#/$defs/value"}},
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    wire = openai_tool_definition(
        {"name": "collect", "description": "Собрать", "input_schema": schema}
    )
    assert wire["function"]["parameters"] == schema


def test_parse_openai_turn_accepts_json_string_arguments() -> None:
    turn = parse_openai_turn(
        {
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {"name": "search", "arguments": '{"query":"Москва"}'},
                }
            ],
        }
    )
    assert turn.tool_calls[0].arguments == {"query": "Москва"}


def test_parse_openai_turn_accepts_mapping_arguments() -> None:
    turn = parse_openai_turn(
        {
            "tool_calls": [
                {"id": "call-1", "function": {"name": "search", "arguments": {"q": 1}}}
            ]
        }
    )
    assert turn.tool_calls[0].arguments == {"q": 1}


def test_parse_openai_turn_rejects_non_object_arguments() -> None:
    with pytest.raises(AdapterProtocolError, match="must decode to an object"):
        parse_openai_turn(
            {
                "tool_calls": [
                    {"id": "call-1", "function": {"name": "search", "arguments": "[]"}}
                ]
            }
        )


@pytest.mark.asyncio
async def test_scripted_adapter_fails_when_script_is_exhausted() -> None:
    adapter = ScriptedAdapter([AgentTurn(content="one")])
    await adapter.respond([], [])
    with pytest.raises(RuntimeError, match="no turns left"):
        await adapter.respond([], [])


def test_openai_message_preserves_assistant_tool_calls() -> None:
    message = Message(
        role="assistant",
        content="",
        tool_calls=(ToolCall("call-1", "search", {"запрос": "Москва"}),),
    )
    wire = openai_message(message)
    assert wire["tool_calls"][0]["id"] == "call-1"
    assert wire["tool_calls"][0]["function"]["arguments"] == '{"запрос":"Москва"}'


@pytest.mark.asyncio
async def test_live_adapter_sends_complete_wire_contract() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-2",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"query":"Москва"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    client = httpx.AsyncClient(
        base_url="http://provider.test/v1",
        transport=httpx.MockTransport(handler),
    )
    adapter = OpenAICompatibleAdapter(
        "http://provider.test/v1",
        "model",
        client=client,
    )
    turn = await adapter.respond(
        [
            Message(
                role="assistant",
                content="",
                tool_calls=(ToolCall("call-1", "search", {"query": "Казань"}),),
            ),
            Message(role="tool", content='{"ok":true}', tool_call_id="call-1"),
        ],
        [
            {
                "name": "search",
                "description": "Search",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            }
        ],
    )
    await client.aclose()

    assert captured["messages"][0]["tool_calls"][0]["id"] == "call-1"
    assert captured["tools"][0]["function"]["parameters"]["additionalProperties"] is False
    assert turn.tool_calls == (ToolCall("call-2", "search", {"query": "Москва"}),)


@pytest.mark.asyncio
async def test_live_adapter_wraps_http_status_without_response_body() -> None:
    client = httpx.AsyncClient(
        base_url="http://provider.test/v1/",
        transport=httpx.MockTransport(lambda request: httpx.Response(500, text="secret")),
    )
    adapter = OpenAICompatibleAdapter("http://provider.test/v1", "model", client=client)

    with pytest.raises(AdapterTransportError, match="HTTP 500") as caught:
        await adapter.respond([], [])
    await client.aclose()

    assert "secret" not in str(caught.value)


@pytest.mark.asyncio
async def test_live_adapter_rejects_malformed_response_contract() -> None:
    client = httpx.AsyncClient(
        base_url="http://provider.test/v1/",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": []})),
    )
    adapter = OpenAICompatibleAdapter("http://provider.test/v1", "model", client=client)

    with pytest.raises(AdapterProtocolError, match="at least one choice"):
        await adapter.respond([], [])
    await client.aclose()


@pytest.mark.asyncio
async def test_live_adapter_wraps_timeout() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("secret upstream detail", request=request)

    client = httpx.AsyncClient(
        base_url="http://provider.test/v1/",
        transport=httpx.MockTransport(timeout),
    )
    adapter = OpenAICompatibleAdapter("http://provider.test/v1", "model", client=client)

    with pytest.raises(AdapterTimeoutError, match="provider request timed out") as caught:
        await adapter.respond([], [])
    await client.aclose()

    assert "secret" not in str(caught.value)
