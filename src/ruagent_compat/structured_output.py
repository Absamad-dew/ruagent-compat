"""Strict, provider-neutral auditing for structured JSON model output."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

_FULL_JSON_FENCE = re.compile(
    r"\A\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)
_TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens", "max_output_tokens"})


class StructuredOutputKind(StrEnum):
    """Mutually exclusive outcome of one structured-output audit."""

    DIRECT_JSON = "direct_json"
    FENCED_JSON = "fenced_json"
    REASONING_CONTENT_FALLBACK = "reasoning_content_fallback"
    EMPTY = "empty"
    TRUNCATED_JSON = "truncated_json"
    INVALID_JSON = "invalid_json"


class StructuredOutputSource(StrEnum):
    """Provider message field that was inspected for the classified output."""

    NONE = "none"
    CONTENT = "content"
    REASONING_CONTENT = "reasoning_content"


class StructuredOutputNormalization(StrEnum):
    """Explicit transformation used before accepting JSON."""

    NONE = "none"
    FULL_JSON_FENCE = "full_json_fence"
    REASONING_CONTENT_FALLBACK = "reasoning_content_fallback"


@dataclass(frozen=True, slots=True)
class StructuredOutputProvenance:
    """Caller-supplied provider facts retained with the audit decision."""

    provider: str | None = None
    model: str | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "request_id": self.request_id,
            "finish_reason": self.finish_reason,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class StructuredOutputAudit:
    """Serializable audit result without silent JSON repair."""

    kind: StructuredOutputKind
    source: StructuredOutputSource
    normalization: StructuredOutputNormalization
    value: Any = None
    json_text: str | None = None
    schema_valid: bool | None = None
    error: str | None = None
    provenance: StructuredOutputProvenance = field(default_factory=StructuredOutputProvenance)

    @property
    def json_valid(self) -> bool:
        return self.kind in {
            StructuredOutputKind.DIRECT_JSON,
            StructuredOutputKind.FENCED_JSON,
            StructuredOutputKind.REASONING_CONTENT_FALLBACK,
        }

    @property
    def accepted(self) -> bool:
        return self.json_valid and self.schema_valid is not False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "source": self.source.value,
            "normalization": self.normalization.value,
            "json_valid": self.json_valid,
            "accepted": self.accepted,
            "value": copy.deepcopy(self.value),
            "json_text": self.json_text,
            "schema_valid": self.schema_valid,
            "error": self.error,
            "provenance": self.provenance.to_dict(),
        }


def audit_structured_output(
    *,
    content: str | None,
    reasoning_content: str | None = None,
    allow_reasoning_fallback: bool = False,
    schema: Mapping[str, Any] | None = None,
    provenance: StructuredOutputProvenance | None = None,
) -> StructuredOutputAudit:
    """Classify and optionally schema-validate one provider-neutral JSON response.

    Only complete direct JSON and a complete outer ``json`` Markdown fence are
    accepted from ``content``. ``reasoning_content`` is considered only when
    ``content`` is empty and the caller explicitly opts in. Embedded JSON and
    malformed JSON are never extracted or repaired.
    """

    if content is not None and not isinstance(content, str):
        raise TypeError("content must be a string or None")
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        raise TypeError("reasoning_content must be a string or None")

    recorded_provenance = _copy_provenance(provenance)
    validator = _schema_validator(schema)
    direct_text = (content or "").strip()

    if direct_text:
        return _audit_content_candidate(direct_text, validator, recorded_provenance)

    if not allow_reasoning_fallback:
        return _failure(
            StructuredOutputKind.EMPTY,
            StructuredOutputSource.NONE,
            "assistant content is empty",
            recorded_provenance,
        )

    fallback_text = (reasoning_content or "").strip()
    if not fallback_text:
        return _failure(
            StructuredOutputKind.EMPTY,
            StructuredOutputSource.NONE,
            "assistant content and reasoning_content are empty",
            recorded_provenance,
        )

    try:
        value = json.loads(fallback_text)
    except json.JSONDecodeError as error:
        return _json_failure(
            error,
            StructuredOutputSource.REASONING_CONTENT,
            recorded_provenance,
        )

    return _success(
        kind=StructuredOutputKind.REASONING_CONTENT_FALLBACK,
        source=StructuredOutputSource.REASONING_CONTENT,
        normalization=StructuredOutputNormalization.REASONING_CONTENT_FALLBACK,
        value=value,
        json_text=fallback_text,
        validator=validator,
        provenance=recorded_provenance,
    )


def _audit_content_candidate(
    text: str,
    validator: Draft202012Validator | None,
    provenance: StructuredOutputProvenance,
) -> StructuredOutputAudit:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as direct_error:
        fence_match = _FULL_JSON_FENCE.fullmatch(text)
        if fence_match is None:
            return _json_failure(direct_error, StructuredOutputSource.CONTENT, provenance)

        fenced_text = fence_match.group("body").strip()
        try:
            value = json.loads(fenced_text)
        except json.JSONDecodeError as fenced_error:
            return _json_failure(fenced_error, StructuredOutputSource.CONTENT, provenance)

        return _success(
            kind=StructuredOutputKind.FENCED_JSON,
            source=StructuredOutputSource.CONTENT,
            normalization=StructuredOutputNormalization.FULL_JSON_FENCE,
            value=value,
            json_text=fenced_text,
            validator=validator,
            provenance=provenance,
        )

    return _success(
        kind=StructuredOutputKind.DIRECT_JSON,
        source=StructuredOutputSource.CONTENT,
        normalization=StructuredOutputNormalization.NONE,
        value=value,
        json_text=text,
        validator=validator,
        provenance=provenance,
    )


def _success(
    *,
    kind: StructuredOutputKind,
    source: StructuredOutputSource,
    normalization: StructuredOutputNormalization,
    value: Any,
    json_text: str,
    validator: Draft202012Validator | None,
    provenance: StructuredOutputProvenance,
) -> StructuredOutputAudit:
    schema_valid: bool | None = None
    error: str | None = None
    if validator is not None:
        validation_error = next(iter(validator.iter_errors(value)), None)
        schema_valid = validation_error is None
        if validation_error is not None:
            error = _schema_error(validation_error)

    return StructuredOutputAudit(
        kind=kind,
        source=source,
        normalization=normalization,
        value=value,
        json_text=json_text,
        schema_valid=schema_valid,
        error=error,
        provenance=provenance,
    )


def _json_failure(
    error: json.JSONDecodeError,
    source: StructuredOutputSource,
    provenance: StructuredOutputProvenance,
) -> StructuredOutputAudit:
    kind = (
        StructuredOutputKind.TRUNCATED_JSON
        if (
            provenance.finish_reason is not None
            and provenance.finish_reason.casefold() in _TRUNCATED_FINISH_REASONS
        )
        else StructuredOutputKind.INVALID_JSON
    )
    return _failure(kind, source, f"invalid JSON: {error.msg}", provenance)


def _failure(
    kind: StructuredOutputKind,
    source: StructuredOutputSource,
    error: str,
    provenance: StructuredOutputProvenance,
) -> StructuredOutputAudit:
    return StructuredOutputAudit(
        kind=kind,
        source=source,
        normalization=StructuredOutputNormalization.NONE,
        error=error,
        provenance=provenance,
    )


def _schema_validator(
    schema: Mapping[str, Any] | None,
) -> Draft202012Validator | None:
    if schema is None:
        return None
    schema_copy = copy.deepcopy(dict(schema))
    try:
        Draft202012Validator.check_schema(schema_copy)
    except SchemaError as error:
        raise ValueError("schema must be a valid Draft 2020-12 JSON Schema") from error
    return Draft202012Validator(schema_copy)


def _schema_error(error: ValidationError) -> str:
    path = "$"
    for component in error.absolute_path:
        if isinstance(component, int):
            path += f"[{component}]"
        else:
            path += f".{component}"
    return f"schema validation failed at {path} ({error.validator})"


def _copy_provenance(
    provenance: StructuredOutputProvenance | None,
) -> StructuredOutputProvenance:
    if provenance is None:
        return StructuredOutputProvenance()
    return StructuredOutputProvenance(
        provider=provenance.provider,
        model=provenance.model,
        request_id=provenance.request_id,
        finish_reason=provenance.finish_reason,
        metadata=provenance.metadata,
    )
