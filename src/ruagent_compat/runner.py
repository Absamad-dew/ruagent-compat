"""Deterministic state-based runner used by provider compatibility cases."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from typing import Any

from .adapters import AgentAdapter
from .errors import AdapterError, CompatError
from .models import Event, Message, RunResult, RunStatus
from .tools import ToolExecutor, ToolRegistry


class AgentRunner:
    def __init__(self, registry: ToolRegistry, *, max_steps: int = 8) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")
        self.registry = registry
        self.max_steps = max_steps

    async def run(
        self,
        adapter: AgentAdapter,
        user_message: str,
        *,
        initial_state: dict[str, Any] | None = None,
        approved_tools: Iterable[str] = (),
    ) -> RunResult:
        state = copy.deepcopy(initial_state or {})
        executor = ToolExecutor(self.registry, state, approved_tools=approved_tools)
        messages = [Message(role="user", content=user_message)]
        events: list[Event] = []

        def emit(step: int, kind: str, data: dict[str, Any]) -> None:
            events.append(Event(sequence=len(events), step=step, kind=kind, data=data))

        for step in range(1, self.max_steps + 1):
            emit(step, "model_request", {"message_count": len(messages)})
            try:
                turn = await adapter.respond(messages, self.registry.wire_definitions())
            except AdapterError as error:
                emit(
                    step,
                    "model_error",
                    {"error": error.code, "message": str(error)},
                )
                return RunResult(
                    status=RunStatus.ADAPTER_ERROR,
                    final_text="",
                    state=copy.deepcopy(state),
                    events=tuple(events),
                    steps=step,
                )
            except Exception as error:
                emit(
                    step,
                    "model_error",
                    {
                        "error": "adapter_error",
                        "message": f"unhandled adapter error: {type(error).__name__}",
                    },
                )
                return RunResult(
                    status=RunStatus.ADAPTER_ERROR,
                    final_text="",
                    state=copy.deepcopy(state),
                    events=tuple(events),
                    steps=step,
                )
            emit(
                step,
                "model_response",
                {"content": turn.content, "tool_call_count": len(turn.tool_calls)},
            )
            messages.append(
                Message(
                    role="assistant",
                    content=turn.content,
                    tool_calls=turn.tool_calls,
                )
            )

            if not turn.tool_calls:
                return RunResult(
                    status=RunStatus.COMPLETED,
                    final_text=turn.content,
                    state=copy.deepcopy(state),
                    events=tuple(events),
                    steps=step,
                )

            for call in turn.tool_calls:
                emit(
                    step,
                    "tool_start",
                    {"call_id": call.call_id, "name": call.name, "arguments": call.arguments},
                )
                try:
                    outcome = await executor.execute(call)
                except CompatError as error:
                    payload = {"ok": False, "error": error.code, "message": str(error)}
                    emit(step, "tool_error", {"call_id": call.call_id, **payload})
                else:
                    payload = {
                        "ok": True,
                        "result": outcome.result,
                        "cached": outcome.cached,
                        "attempts": outcome.attempts,
                    }
                    emit(step, "tool_result", {"call_id": call.call_id, **payload})

                messages.append(
                    Message(
                        role="tool",
                        content=json.dumps(payload, ensure_ascii=False, sort_keys=True),
                        name=call.name,
                        tool_call_id=call.call_id,
                    )
                )

        return RunResult(
            status=RunStatus.MAX_STEPS,
            final_text="",
            state=copy.deepcopy(state),
            events=tuple(events),
            steps=self.max_steps,
        )
