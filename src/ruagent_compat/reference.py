"""Offline reference cases that prove runtime semantics, not model quality."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .adapters import ScriptedAdapter, openai_tool_definition
from .errors import ToolExecutionError
from .models import AgentTurn, RunStatus, ToolCall
from .runner import AgentRunner
from .tools import ToolRegistry, ToolSpec


@dataclass(frozen=True, slots=True)
class ReferenceResult:
    case: str
    status: str
    assertion: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def run_reference_suite() -> list[ReferenceResult]:
    checks = [
        _schema_preservation_case,
        _duplicate_idempotency_case,
        _permission_gate_case,
        _transactional_retry_case,
        _serialization_boundary_case,
        _invalid_arguments_case,
        _loop_budget_case,
    ]
    results: list[ReferenceResult] = []
    for check in checks:
        try:
            details, assertion = await check()
        except AssertionError as error:
            results.append(
                ReferenceResult(
                    case=check.__name__.removeprefix("_").removesuffix("_case"),
                    status="fail",
                    assertion=str(error) or "assertion failed",
                    details={},
                )
            )
        else:
            results.append(
                ReferenceResult(
                    case=check.__name__.removeprefix("_").removesuffix("_case"),
                    status="pass",
                    assertion=assertion,
                    details=details,
                )
            )
    return results


async def _schema_preservation_case() -> tuple[dict[str, Any], str]:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": {
            "item": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "название": {"type": "string"},
                    "количество": {"type": "integer", "minimum": 1},
                },
                "required": ["название", "количество"],
            }
        },
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {"type": "array", "items": {"$ref": "#/$defs/item"}},
            "priority": {"enum": ["низкий", "обычный", "высокий"]},
        },
        "required": ["items", "priority"],
    }
    tool = ToolSpec("create_order", "Создать заказ", schema, lambda arguments, state: {})
    wire = openai_tool_definition(tool.wire_definition())
    assert wire["function"]["parameters"] == schema, "wire adapter changed nested JSON Schema"
    return {"schema_keys": sorted(schema), "cyrillic": True}, "schema preserved byte-for-value"


async def _duplicate_idempotency_case() -> tuple[dict[str, Any], str]:
    def increment(arguments: dict[str, Any], state: dict[str, Any]) -> dict[str, int]:
        state["counter"] = state.get("counter", 0) + arguments["amount"]
        return {"counter": state["counter"]}

    registry = ToolRegistry(
        [
            ToolSpec(
                "increment",
                "Increment once",
                {
                    "type": "object",
                    "properties": {"amount": {"type": "integer"}},
                    "required": ["amount"],
                    "additionalProperties": False,
                },
                increment,
            )
        ]
    )
    duplicate = ToolCall("same-call", "increment", {"amount": 1})
    adapter = ScriptedAdapter(
        [AgentTurn(tool_calls=(duplicate, duplicate)), AgentTurn(content="done")]
    )
    result = await AgentRunner(registry).run(adapter, "increment", initial_state={"counter": 0})
    cached_events = [event for event in result.events if event.data.get("cached") is True]
    assert result.state == {"counter": 1}, "duplicate call caused a second side effect"
    assert len(cached_events) == 1, "duplicate result was not served from idempotency ledger"
    return {"counter": 1, "cached_results": 1}, "duplicate side effect prevented"


async def _permission_gate_case() -> tuple[dict[str, Any], str]:
    def pay(arguments: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        state["payments"] = [arguments]
        return arguments

    registry = ToolRegistry(
        [
            ToolSpec(
                "pay",
                "Payment mock",
                {
                    "type": "object",
                    "properties": {"amount": {"type": "integer", "minimum": 1}},
                    "required": ["amount"],
                },
                pay,
                requires_approval=True,
            )
        ]
    )
    adapter = ScriptedAdapter(
        [
            AgentTurn(tool_calls=(ToolCall("pay-1", "pay", {"amount": 100}),)),
            AgentTurn(content="approval required"),
        ]
    )
    result = await AgentRunner(registry).run(adapter, "pay")
    errors = [event for event in result.events if event.kind == "tool_error"]
    assert result.state == {}, "permission failure mutated state"
    assert errors[0].data["error"] == "permission_denied", "wrong permission error code"
    return (
        {"state": result.state, "error": errors[0].data["error"]},
        "approval precedes side effect",
    )


async def _transactional_retry_case() -> tuple[dict[str, Any], str]:
    attempts = 0

    def flaky(arguments: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        state["tickets"] = state.get("tickets", 0) + 1
        if attempts == 1:
            raise ToolExecutionError("temporary failure", retryable=True)
        return {"tickets": state["tickets"], "title": arguments["title"]}

    registry = ToolRegistry(
        [
            ToolSpec(
                "create_ticket",
                "Create ticket",
                {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                },
                flaky,
                max_retries=1,
            )
        ]
    )
    adapter = ScriptedAdapter(
        [
            AgentTurn(
                tool_calls=(ToolCall("ticket-1", "create_ticket", {"title": "Ошибка"}),)
            ),
            AgentTurn(content="created"),
        ]
    )
    result = await AgentRunner(registry).run(adapter, "create")
    tool_result = next(event for event in result.events if event.kind == "tool_result")
    assert result.state == {"tickets": 1}, "failed attempt leaked partial state"
    assert tool_result.data["attempts"] == 2, "retry count was not recorded"
    return {"attempts": attempts, "committed_tickets": 1}, "failed attempt rolled back"


async def _serialization_boundary_case() -> tuple[dict[str, Any], str]:
    def invalid_result(arguments: dict[str, Any], state: dict[str, Any]) -> set[str]:
        state["leaked"] = arguments["value"]
        return {arguments["value"]}

    registry = ToolRegistry(
        [
            ToolSpec(
                "invalid_result",
                "Return a deliberately non-JSON result",
                {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
                invalid_result,
            )
        ]
    )
    adapter = ScriptedAdapter(
        [
            AgentTurn(
                tool_calls=(ToolCall("invalid-json-1", "invalid_result", {"value": "x"}),)
            ),
            AgentTurn(content="rejected"),
        ]
    )
    result = await AgentRunner(registry).run(adapter, "run invalid result")
    error = next(event for event in result.events if event.kind == "tool_error")
    assert result.state == {}, "non-JSON result committed partial state"
    assert error.data["error"] == "tool_execution_error", "wrong serialization error code"
    return (
        {"state": result.state, "error": error.data["error"]},
        "non-JSON result rejected before commit",
    )


async def _invalid_arguments_case() -> tuple[dict[str, Any], str]:
    registry = ToolRegistry(
        [
            ToolSpec(
                "set_priority",
                "Set priority",
                {
                    "type": "object",
                    "properties": {"priority": {"enum": ["low", "high"]}},
                    "required": ["priority"],
                    "additionalProperties": False,
                },
                lambda arguments, state: state.update(arguments),
            )
        ]
    )
    adapter = ScriptedAdapter(
        [
            AgentTurn(tool_calls=(ToolCall("priority-1", "set_priority", {"other": 1}),)),
            AgentTurn(content="invalid"),
        ]
    )
    result = await AgentRunner(registry).run(adapter, "set")
    error = next(event for event in result.events if event.kind == "tool_error")
    assert result.state == {}, "invalid arguments reached the handler"
    assert error.data["error"] == "invalid_arguments", "wrong validation error code"
    return {"state": result.state, "error": error.data["error"]}, "invalid call blocked"


async def _loop_budget_case() -> tuple[dict[str, Any], str]:
    registry = ToolRegistry(
        [
            ToolSpec(
                "noop",
                "No operation",
                {"type": "object", "additionalProperties": False},
                lambda arguments, state: {},
            )
        ]
    )
    adapter = ScriptedAdapter(
        [
            AgentTurn(tool_calls=(ToolCall(f"noop-{index}", "noop", {}),))
            for index in range(3)
        ]
    )
    result = await AgentRunner(registry, max_steps=3).run(adapter, "loop")
    assert result.status is RunStatus.MAX_STEPS, "loop did not stop at max_steps"
    return {"steps": result.steps, "status": result.status.value}, "agent loop bounded"
