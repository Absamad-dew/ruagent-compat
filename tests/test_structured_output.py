from __future__ import annotations

import pytest

from ruagent_compat import (
    StructuredOutputKind,
    StructuredOutputNormalization,
    StructuredOutputProvenance,
    StructuredOutputSource,
    __version__,
    audit_structured_output,
)

OBJECT_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def test_package_version_is_0_2_0() -> None:
    assert __version__ == "0.2.0"


def test_direct_json_is_accepted_without_normalization() -> None:
    result = audit_structured_output(
        content='  {"answer":"Москва"}\n',
        schema=OBJECT_SCHEMA,
        provenance=StructuredOutputProvenance(
            provider="local",
            model="model-a",
            request_id="req-1",
            finish_reason="stop",
        ),
    )

    assert result.kind is StructuredOutputKind.DIRECT_JSON
    assert result.source is StructuredOutputSource.CONTENT
    assert result.normalization is StructuredOutputNormalization.NONE
    assert result.value == {"answer": "Москва"}
    assert result.schema_valid is True
    assert result.accepted is True
    assert result.provenance.request_id == "req-1"


def test_complete_json_fence_is_the_only_content_normalization() -> None:
    result = audit_structured_output(content='```json\n{"answer":"да"}\n```')

    assert result.kind is StructuredOutputKind.FENCED_JSON
    assert result.source is StructuredOutputSource.CONTENT
    assert result.normalization is StructuredOutputNormalization.FULL_JSON_FENCE
    assert result.json_text == '{"answer":"да"}'
    assert result.accepted is True


@pytest.mark.parametrize(
    "content",
    [
        'Префикс {"answer":"да"}',
        '{"answer":"да"} хвост',
        '```javascript\n{"answer":"да"}\n```',
        'prefix ```json\n{"answer":"да"}\n``` suffix',
    ],
)
def test_embedded_or_non_json_fenced_content_is_not_repaired(content: str) -> None:
    result = audit_structured_output(content=content)

    assert result.kind is StructuredOutputKind.INVALID_JSON
    assert result.json_valid is False
    assert result.normalization is StructuredOutputNormalization.NONE
    assert result.value is None


def test_malformed_json_fence_is_not_repaired() -> None:
    result = audit_structured_output(content='```json\n{"answer":}\n```')

    assert result.kind is StructuredOutputKind.INVALID_JSON
    assert result.source is StructuredOutputSource.CONTENT
    assert result.error == "invalid JSON: Expecting value"


def test_reasoning_content_is_ignored_without_explicit_opt_in() -> None:
    result = audit_structured_output(
        content=None,
        reasoning_content='{"answer":"reasoning"}',
    )

    assert result.kind is StructuredOutputKind.EMPTY
    assert result.source is StructuredOutputSource.NONE
    assert result.accepted is False


def test_explicit_reasoning_fallback_accepts_only_complete_json() -> None:
    result = audit_structured_output(
        content=" ",
        reasoning_content='{"answer":"reasoning"}',
        allow_reasoning_fallback=True,
        provenance=StructuredOutputProvenance(finish_reason="stop"),
    )

    assert result.kind is StructuredOutputKind.REASONING_CONTENT_FALLBACK
    assert result.source is StructuredOutputSource.REASONING_CONTENT
    assert result.normalization is StructuredOutputNormalization.REASONING_CONTENT_FALLBACK
    assert result.value == {"answer": "reasoning"}


def test_invalid_content_never_falls_back_to_reasoning() -> None:
    result = audit_structured_output(
        content='{"answer":}',
        reasoning_content='{"answer":"would-hide-the-final-error"}',
        allow_reasoning_fallback=True,
    )

    assert result.kind is StructuredOutputKind.INVALID_JSON
    assert result.source is StructuredOutputSource.CONTENT
    assert result.value is None


def test_fenced_reasoning_content_is_not_normalized() -> None:
    result = audit_structured_output(
        content=None,
        reasoning_content='```json\n{"answer":"hidden"}\n```',
        allow_reasoning_fallback=True,
    )

    assert result.kind is StructuredOutputKind.INVALID_JSON
    assert result.source is StructuredOutputSource.REASONING_CONTENT


def test_invalid_json_is_truncated_only_with_explicit_provider_signal() -> None:
    truncated = audit_structured_output(
        content='{"answer":"cut',
        provenance=StructuredOutputProvenance(finish_reason="length"),
    )
    malformed = audit_structured_output(
        content='{"answer":"cut',
        provenance=StructuredOutputProvenance(finish_reason="stop"),
    )

    assert truncated.kind is StructuredOutputKind.TRUNCATED_JSON
    assert malformed.kind is StructuredOutputKind.INVALID_JSON


def test_schema_failure_preserves_json_source_and_rejects_result() -> None:
    result = audit_structured_output(
        content='{"answer":7}',
        schema=OBJECT_SCHEMA,
    )

    assert result.kind is StructuredOutputKind.DIRECT_JSON
    assert result.json_valid is True
    assert result.schema_valid is False
    assert result.accepted is False
    assert result.error == "schema validation failed at $.answer (type)"
    assert "7" not in result.error


def test_invalid_schema_fails_before_audit() -> None:
    with pytest.raises(ValueError, match="valid Draft 2020-12"):
        audit_structured_output(
            content='{"answer":"ok"}',
            schema={"type": "not-a-real-type"},
        )


def test_provenance_metadata_is_copied_and_serialized() -> None:
    metadata = {"runtime": {"name": "llama.cpp"}}
    provenance = StructuredOutputProvenance(metadata=metadata)
    metadata["runtime"]["name"] = "changed"

    result = audit_structured_output(content="null", provenance=provenance)
    provenance.metadata["runtime"]["name"] = "changed-after-audit"
    serialized = result.to_dict()

    assert result.kind is StructuredOutputKind.DIRECT_JSON
    assert result.value is None
    assert serialized["json_valid"] is True
    assert serialized["provenance"]["metadata"] == {
        "runtime": {"name": "llama.cpp"}
    }


def test_non_string_provider_fields_raise_protocol_type_error() -> None:
    with pytest.raises(TypeError, match="content must"):
        audit_structured_output(content={"answer": "not-wire-content"})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="reasoning_content must"):
        audit_structured_output(content=None, reasoning_content=[])  # type: ignore[arg-type]
