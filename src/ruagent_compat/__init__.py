"""Executable compatibility contracts for tool-using agents."""

from .adapters import (
    OpenAICompatibleAdapter,
    ScriptedAdapter,
    openai_message,
    openai_tool_definition,
    parse_openai_turn,
)
from .models import AgentTurn, Event, Message, RunResult, RunStatus, ToolCall
from .runner import AgentRunner
from .tools import ToolExecutor, ToolRegistry, ToolSpec

__all__ = [
    "AgentRunner",
    "AgentTurn",
    "Event",
    "Message",
    "OpenAICompatibleAdapter",
    "RunResult",
    "RunStatus",
    "ScriptedAdapter",
    "ToolCall",
    "ToolExecutor",
    "ToolRegistry",
    "ToolSpec",
    "openai_tool_definition",
    "openai_message",
    "parse_openai_turn",
]
