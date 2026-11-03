# Changelog

## 0.2.1

Fix concurrent execution in one `ToolExecutor`: identical calls now enter the handler once, and different calls no longer overwrite each other’s committed state. Per-executor serialization covers validation, permission checks, cache lookup, execution and commit. Added regression tests that reproduced both failures in 0.2.0.

The executor is scoped to one process and event loop. Its rollback covers copied in-memory state, not external API effects. Durable cross-process coordination belongs at the integration boundary.
