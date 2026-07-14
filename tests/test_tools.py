from __future__ import annotations

from typing import Any

import pytest

from ruagent_compat.errors import (
    IdempotencyConflictError,
    ToolExecutionError,
    ToolPermissionError,
    ToolValidationError,
    UnknownToolError,
)
from ruagent_compat.models import ToolCall
from ruagent_compat.tools import ToolExecutor, ToolRegistry, ToolSpec


def integer_tool(handler: Any, **kwargs: Any) -> ToolSpec:
    return ToolSpec(
        "increment",
        "Increment",
        {
            "type": "object",
            "properties": {"amount": {"type": "integer"}},
            "required": ["amount"],
            "additionalProperties": False,
        },
        handler,
        **kwargs,
    )


def test_tool_spec_rejects_invalid_schema() -> None:
    with pytest.raises(ValueError, match="invalid schema"):
        ToolSpec("broken", "Broken", {"type": "not-a-type"}, lambda arguments, state: {})


def test_registry_rejects_duplicate_names() -> None:
    tool = integer_tool(lambda arguments, state: {})
    with pytest.raises(ValueError, match="duplicate tool"):
        ToolRegistry([tool, tool])


def test_registry_preserves_declared_tool_order() -> None:
    first = ToolSpec("z_first", "First", {"type": "object"}, lambda arguments, state: {})
    second = ToolSpec("a_second", "Second", {"type": "object"}, lambda arguments, state: {})
    registry = ToolRegistry([first, second])

    assert [tool["name"] for tool in registry.wire_definitions()] == ["z_first", "a_second"]


@pytest.mark.asyncio
async def test_executor_supports_nested_cyrillic_schema() -> None:
    schema = {
        "$defs": {
            "строка": {
                "type": "object",
                "properties": {"название": {"type": "string"}},
                "required": ["название"],
                "additionalProperties": False,
            }
        },
        "type": "object",
        "properties": {"позиции": {"type": "array", "items": {"$ref": "#/$defs/строка"}}},
        "required": ["позиции"],
    }

    def save(arguments: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        state.update(arguments)
        return arguments

    executor = ToolExecutor(ToolRegistry([ToolSpec("save", "Сохранить", schema, save)]), {})
    outcome = await executor.execute(
        ToolCall("cyrillic", "save", {"позиции": [{"название": "Тест"}]})
    )
    assert outcome.result == {"позиции": [{"название": "Тест"}]}
    assert executor.state == outcome.result


@pytest.mark.asyncio
async def test_validation_blocks_handler_and_state() -> None:
    calls = 0

    def handler(arguments: dict[str, Any], state: dict[str, Any]) -> None:
        nonlocal calls
        calls += 1
        state["called"] = True

    executor = ToolExecutor(ToolRegistry([integer_tool(handler)]), {})
    with pytest.raises(ToolValidationError, match="invalid arguments"):
        await executor.execute(ToolCall("bad", "increment", {"amount": "one"}))
    assert calls == 0
    assert executor.state == {}


@pytest.mark.asyncio
async def test_permission_gate_runs_before_side_effect() -> None:
    def handler(arguments: dict[str, Any], state: dict[str, Any]) -> None:
        state["paid"] = arguments["amount"]

    executor = ToolExecutor(
        ToolRegistry([integer_tool(handler, requires_approval=True)]),
        {},
    )
    with pytest.raises(ToolPermissionError, match="requires approval"):
        await executor.execute(ToolCall("pay", "increment", {"amount": 10}))
    assert executor.state == {}


@pytest.mark.asyncio
async def test_duplicate_call_is_cached_without_second_side_effect() -> None:
    def handler(arguments: dict[str, Any], state: dict[str, Any]) -> int:
        state["counter"] = state.get("counter", 0) + arguments["amount"]
        return state["counter"]

    executor = ToolExecutor(ToolRegistry([integer_tool(handler)]), {"counter": 0})
    call = ToolCall("same", "increment", {"amount": 1})
    first = await executor.execute(call)
    second = await executor.execute(call)
    assert first.cached is False
    assert second.cached is True
    assert executor.state == {"counter": 1}


@pytest.mark.asyncio
async def test_tool_result_is_detached_from_committed_state() -> None:
    def handler(arguments: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        state["items"] = [arguments["amount"]]
        return {"items": state["items"]}

    executor = ToolExecutor(ToolRegistry([integer_tool(handler)]), {})
    outcome = await executor.execute(ToolCall("detached", "increment", {"amount": 1}))
    outcome.result["items"].append(2)

    assert executor.state == {"items": [1]}


@pytest.mark.asyncio
async def test_handler_cannot_mutate_state_through_leaked_working_copy() -> None:
    leaked: dict[str, dict[str, Any]] = {}

    def handler(arguments: dict[str, Any], state: dict[str, Any]) -> dict[str, bool]:
        state["items"] = [arguments["amount"]]
        leaked["state"] = state
        return {"ok": True}

    executor = ToolExecutor(ToolRegistry([integer_tool(handler)]), {})
    await executor.execute(ToolCall("leaked", "increment", {"amount": 1}))
    leaked["state"]["items"].append(2)

    assert executor.state == {"items": [1]}


@pytest.mark.asyncio
async def test_non_json_result_does_not_commit_state() -> None:
    def handler(arguments: dict[str, Any], state: dict[str, Any]) -> set[int]:
        state["committed"] = True
        return {arguments["amount"]}

    executor = ToolExecutor(ToolRegistry([integer_tool(handler)]), {})

    with pytest.raises(ToolExecutionError, match="non-JSON-serializable"):
        await executor.execute(ToolCall("not-json", "increment", {"amount": 1}))

    assert executor.state == {}


@pytest.mark.asyncio
async def test_call_id_reuse_with_other_arguments_is_conflict() -> None:
    executor = ToolExecutor(
        ToolRegistry([integer_tool(lambda arguments, state: arguments)]),
        {},
    )
    await executor.execute(ToolCall("same", "increment", {"amount": 1}))
    with pytest.raises(IdempotencyConflictError, match="different arguments"):
        await executor.execute(ToolCall("same", "increment", {"amount": 2}))


@pytest.mark.asyncio
async def test_retry_rolls_back_partial_state() -> None:
    attempts = 0

    def handler(arguments: dict[str, Any], state: dict[str, Any]) -> int:
        nonlocal attempts
        attempts += 1
        state["counter"] = state.get("counter", 0) + 1
        if attempts == 1:
            raise ToolExecutionError("temporary", retryable=True)
        return state["counter"]

    executor = ToolExecutor(
        ToolRegistry([integer_tool(handler, max_retries=1)]),
        {"counter": 0},
    )
    outcome = await executor.execute(ToolCall("retry", "increment", {"amount": 1}))
    assert outcome.attempts == 2
    assert executor.state == {"counter": 1}


@pytest.mark.asyncio
async def test_unknown_tool_is_typed_error() -> None:
    executor = ToolExecutor(ToolRegistry(), {})
    with pytest.raises(UnknownToolError, match="unknown tool"):
        await executor.execute(ToolCall("missing", "missing", {}))
