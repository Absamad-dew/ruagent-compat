"""Concurrency regressions for one public in-memory executor."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ruagent_compat.models import ToolCall
from ruagent_compat.tools import ToolExecutor, ToolRegistry, ToolSpec


@pytest.mark.asyncio
async def test_concurrent_duplicate_enters_handler_once() -> None:
    entered = 0

    async def increment(arguments: dict[str, Any], state: dict[str, Any]) -> int:
        nonlocal entered
        entered += 1
        await asyncio.sleep(0.01)
        state["count"] = state.get("count", 0) + 1
        return int(state["count"])

    executor = ToolExecutor(
        ToolRegistry([ToolSpec("increment", "Increment", {"type": "object"}, increment)]), {}
    )
    outcomes = await asyncio.gather(
        *(executor.execute(ToolCall("same", "increment", {})) for _ in range(8))
    )
    assert entered == 1
    assert executor.state == {"count": 1}
    assert sum(outcome.cached for outcome in outcomes) == 7


@pytest.mark.asyncio
async def test_concurrent_distinct_calls_do_not_lose_committed_state() -> None:
    async def increment(arguments: dict[str, Any], state: dict[str, Any]) -> int:
        previous = int(state.get("count", 0))
        await asyncio.sleep(0.005)
        state["count"] = previous + 1
        return previous + 1

    executor = ToolExecutor(
        ToolRegistry([ToolSpec("increment", "Increment", {"type": "object"}, increment)]), {}
    )
    outcomes = await asyncio.gather(
        *(executor.execute(ToolCall(str(i), "increment", {})) for i in range(8))
    )
    assert executor.state == {"count": 8}
    assert sorted(outcome.result for outcome in outcomes) == list(range(1, 9))
