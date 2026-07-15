"""Runnable structured-output protocol proof without a provider call."""

from __future__ import annotations

import json

from ruagent_compat import StructuredOutputProvenance, audit_structured_output

schema = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

audit = audit_structured_output(
    content="",
    reasoning_content='{"answer":"Москва"}',
    allow_reasoning_fallback=True,
    schema=schema,
    provenance=StructuredOutputProvenance(
        provider="local-openai-compatible",
        model="example-model",
        request_id="example-1",
        finish_reason="stop",
        metadata={"runtime": "example"},
    ),
)

print(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2))
