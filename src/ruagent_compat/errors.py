"""Typed failures exposed by the conformance runtime."""

from __future__ import annotations


class CompatError(Exception):
    """Base error with a stable machine-readable code."""

    code = "compat_error"


class UnknownToolError(CompatError):
    code = "unknown_tool"


class ToolValidationError(CompatError):
    code = "invalid_arguments"


class ToolPermissionError(CompatError):
    code = "permission_denied"


class IdempotencyConflictError(CompatError):
    code = "idempotency_conflict"


class ToolExecutionError(CompatError):
    code = "tool_execution_error"

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable

