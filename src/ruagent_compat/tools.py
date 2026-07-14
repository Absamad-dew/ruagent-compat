"""Validated, permission-aware, transactional tool execution."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from .errors import (
    IdempotencyConflictError,
    ToolExecutionError,
    ToolPermissionError,
    ToolValidationError,
    UnknownToolError,
)
from .models import ToolCall

ToolHandler = Callable[[dict[str, Any], dict[str, Any]], Any | Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    requires_approval: bool = False
    max_retries: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name cannot be empty")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        try:
            Draft202012Validator.check_schema(self.input_schema)
        except SchemaError as error:
            raise ValueError(f"invalid schema for {self.name}: {error.message}") from error

    def wire_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": copy.deepcopy(self.input_schema),
        }


class ToolRegistry:
    def __init__(self, tools: Iterable[ToolSpec] = ()) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as error:
            raise UnknownToolError(f"unknown tool: {name}") from error

    def wire_definitions(self) -> list[dict[str, Any]]:
        return [tool.wire_definition() for tool in self._tools.values()]


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    result: Any
    cached: bool
    attempts: int


@dataclass(frozen=True, slots=True)
class _LedgerEntry:
    fingerprint: str
    result: Any
    attempts: int


class ToolExecutor:
    """Executes tools against copied state and commits only successful attempts."""

    def __init__(
        self,
        registry: ToolRegistry,
        state: dict[str, Any],
        *,
        approved_tools: Iterable[str] = (),
    ) -> None:
        self.registry = registry
        self.state = state
        self.approved_tools = frozenset(approved_tools)
        self._ledger: dict[str, _LedgerEntry] = {}

    async def execute(self, call: ToolCall) -> ToolOutcome:
        tool = self.registry.get(call.name)
        self._validate(tool, call.arguments)

        if tool.requires_approval and tool.name not in self.approved_tools:
            raise ToolPermissionError(f"tool requires approval: {tool.name}")

        fingerprint = _fingerprint(call.name, call.arguments)
        previous = self._ledger.get(call.call_id)
        if previous is not None:
            if previous.fingerprint != fingerprint:
                raise IdempotencyConflictError(
                    f"call_id {call.call_id!r} was reused with different arguments"
                )
            return ToolOutcome(
                result=copy.deepcopy(previous.result),
                cached=True,
                attempts=previous.attempts,
            )

        attempts = 0
        while True:
            attempts += 1
            working_state = copy.deepcopy(self.state)
            try:
                result = tool.handler(copy.deepcopy(call.arguments), working_state)
                if inspect.isawaitable(result):
                    result = await result
            except ToolExecutionError as error:
                if error.retryable and attempts <= tool.max_retries:
                    continue
                raise
            except Exception as error:
                raise ToolExecutionError(
                    f"tool {tool.name} handler failed with {type(error).__name__}"
                ) from error

            _ensure_json_serializable(
                {"result": result, "state": working_state},
                tool_name=tool.name,
            )
            committed_state = copy.deepcopy(working_state)
            self.state.clear()
            self.state.update(committed_state)
            self._ledger[call.call_id] = _LedgerEntry(
                fingerprint=fingerprint,
                result=copy.deepcopy(result),
                attempts=attempts,
            )
            return ToolOutcome(
                result=copy.deepcopy(result),
                cached=False,
                attempts=attempts,
            )

    @staticmethod
    def _validate(tool: ToolSpec, arguments: Mapping[str, Any]) -> None:
        try:
            Draft202012Validator(tool.input_schema).validate(dict(arguments))
        except ValidationError as error:
            path = ".".join(str(part) for part in error.absolute_path)
            location = f" at {path}" if path else ""
            raise ToolValidationError(
                f"invalid arguments for {tool.name}{location}: {error.message}"
            ) from error


def _fingerprint(name: str, arguments: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {"name": name, "arguments": arguments},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ensure_json_serializable(value: Any, *, tool_name: str) -> None:
    try:
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ToolExecutionError(
            f"tool {tool_name} produced a non-JSON-serializable result or state"
        ) from error
