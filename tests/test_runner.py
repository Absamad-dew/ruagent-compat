from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ruagent_compat.adapters import ScriptedAdapter
from ruagent_compat.models import AgentTurn, RunStatus, ToolCall
from ruagent_compat.runner import AgentRunner
from ruagent_compat.tools import ToolRegistry, ToolSpec


def registry() -> ToolRegistry:
    def set_value(arguments: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        state["value"] = arguments["value"]
        return {"value": state["value"]}

    return ToolRegistry(
        [
            ToolSpec(
                "set_value",
                "Set value",
                {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
                set_value,
            )
        ]
    )


@pytest.mark.asyncio
async def test_runner_completes_without_tool_call() -> None:
    result = await AgentRunner(registry()).run(
        ScriptedAdapter([AgentTurn(content="готово")]),
        "задача",
    )
    assert result.status is RunStatus.COMPLETED
    assert result.final_text == "готово"
    assert [event.kind for event in result.events] == ["model_request", "model_response"]


@pytest.mark.asyncio
async def test_runner_records_deterministic_tool_trace() -> None:
    adapter = ScriptedAdapter(
        [
            AgentTurn(tool_calls=(ToolCall("set-1", "set_value", {"value": 7}),)),
            AgentTurn(content="done"),
        ]
    )
    result = await AgentRunner(registry()).run(adapter, "set")
    assert result.state == {"value": 7}
    assert [event.sequence for event in result.events] == list(range(len(result.events)))
    assert [event.kind for event in result.events] == [
        "model_request",
        "model_response",
        "tool_start",
        "tool_result",
        "model_request",
        "model_response",
    ]


@pytest.mark.asyncio
async def test_runner_returns_validation_error_to_next_model_turn() -> None:
    adapter = ScriptedAdapter(
        [
            AgentTurn(tool_calls=(ToolCall("set-1", "set_value", {"value": "bad"}),)),
            AgentTurn(content="corrected"),
        ]
    )
    result = await AgentRunner(registry()).run(adapter, "set")
    assert result.state == {}
    second_request_messages = adapter.requests[1][0]
    assert '"error": "invalid_arguments"' in second_request_messages[-1].content


@pytest.mark.asyncio
async def test_runner_stops_at_step_budget() -> None:
    turns = [
        AgentTurn(tool_calls=(ToolCall(f"call-{index}", "set_value", {"value": index}),))
        for index in range(2)
    ]
    result = await AgentRunner(registry(), max_steps=2).run(ScriptedAdapter(turns), "loop")
    assert result.status is RunStatus.MAX_STEPS
    assert result.steps == 2


@pytest.mark.asyncio
async def test_runner_propagates_cancellation() -> None:
    class SlowAdapter:
        async def respond(self, messages: Any, tools: Any) -> AgentTurn:
            await asyncio.sleep(10)
            return AgentTurn(content="late")

    task = asyncio.create_task(AgentRunner(registry()).run(SlowAdapter(), "cancel"))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_runner_returns_redacted_adapter_error_trace() -> None:
    class BrokenAdapter:
        async def respond(self, messages: Any, tools: Any) -> AgentTurn:
            raise RuntimeError("token=must-not-appear")

    result = await AgentRunner(registry()).run(BrokenAdapter(), "fail safely")

    assert result.status is RunStatus.ADAPTER_ERROR
    assert result.state == {}
    assert [event.kind for event in result.events] == ["model_request", "model_error"]
    assert result.events[-1].data == {
        "error": "adapter_error",
        "message": "unhandled adapter error: RuntimeError",
    }
    assert "must-not-appear" not in str(result.to_dict())
