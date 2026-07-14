"""Small serializable types shared by adapters, runners, and reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AgentTurn:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class Event:
    sequence: int
    step: int
    kind: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "step": self.step,
            "kind": self.kind,
            "data": self.data,
        }


class RunStatus(StrEnum):
    COMPLETED = "completed"
    ADAPTER_ERROR = "adapter_error"
    MAX_STEPS = "max_steps"


@dataclass(frozen=True, slots=True)
class RunResult:
    status: RunStatus
    final_text: str
    state: dict[str, Any]
    events: tuple[Event, ...]
    steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "final_text": self.final_text,
            "state": self.state,
            "steps": self.steps,
            "events": [event.to_dict() for event in self.events],
        }
