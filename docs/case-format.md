# Case format

Every provider case should preserve four separate layers:

- **neutral contract** — user message, tools, initial state, approvals, and step budget;
- **wire capture** — exact provider request/response after secret redaction;
- **evaluation** — final state, call validity, duplicate side effects, recovery, and trace;
- **provenance** — model, API/SDK/runtime versions, date, sampling, and environment.

Aggregate scores are secondary. A case is useful when another developer can
reproduce it and decide whether the behavior belongs to the model, adapter,
runtime, or evaluation layer.

Provider adapters must not silently remove JSON Schema keywords. If a provider
does not support a keyword, the adapter should return an explicit compatibility
result rather than rewriting the contract without evidence.

