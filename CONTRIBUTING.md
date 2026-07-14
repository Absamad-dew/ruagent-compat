# Contributing

The smallest useful contribution is one failure that becomes one deterministic
test. Please include:

1. provider, model, SDK/runtime, and exact versions;
2. the smallest tool schema and conversation that reproduce the behavior;
3. raw assistant/tool output with secrets removed;
4. expected final state and actual final state;
5. at least one schema or prompt mutation that checks the failure is not a
   single-string artifact.

Run before opening a pull request:

```bash
python -m pytest -q
python -m ruff check .
python -m mypy src
```

Do not submit private conversations, credentials, production endpoints, or
security findings that have not completed responsible disclosure.

