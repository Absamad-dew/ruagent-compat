# ruagent-compat

[![CI](https://github.com/Absamad-dew/ruagent-compat/actions/workflows/ci.yml/badge.svg)](https://github.com/Absamad-dew/ruagent-compat/actions/workflows/ci.yml)

`ruagent-compat` is a small, provider-neutral executable contract for tool-using
agents. It focuses on failures that language-only leaderboards miss:

- nested JSON Schema and Cyrillic tool contracts;
- permission checks before side effects;
- idempotent duplicate tool calls;
- transactional retries after partial failure;
- deterministic event traces and state diffs;
- bounded agent loops.

The project deliberately starts smaller than a full agent framework. The first
goal is to turn one failure into one reproducible test that can be attached to an
SDK issue or pull request.

## What is proven today

The repository includes a deterministic scripted reference adapter. It verifies
the conformance runner itself without network access or private data. It does
**not** claim to measure Yandex, GigaChat, or T-Tech model quality yet.

Run the reference suite:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest
.venv/Scripts/ruagent-compat verify --json reports/reference.json --html reports/reference.html
```

## Contract

An adapter receives the same conversation and an unmodified list of Draft 2020-12
JSON Schemas. It returns an assistant turn with zero or more tool calls. The runner
then records deterministic events, validates each call, enforces approval and
idempotency rules, executes against a copy of state, and commits state only after
success.

The core success measures are final state, invalid calls, duplicate side effects,
recovery, number of attempts, and terminal status. Text quality is intentionally
outside the reference layer.

## First external adapters

1. T-lite 2.1 local tool calling, with exact inference version pins.
2. GigaChat API structured output/function calling.
3. Yandex AI Studio Responses/OpenAI-compatible API.

The provider matrix will only be published after a runnable baseline exists.

## Related upstream contribution

[Yandex AI Studio SDK PR #235](https://github.com/yandex-cloud/yandex-ai-studio-sdk/pull/235)
implements an optional per-attempt timeout for
[issue #216](https://github.com/yandex-cloud/yandex-ai-studio-sdk/issues/216),
while keeping every attempt bounded by the overall retry deadline. This is the
intended workflow: upstream reliability test first, broader cross-provider
comparison second.

## Non-goals

- replacing provider SDKs;
- claiming model improvements from scripted cases;
- fuzzing production endpoints without permission;
- hiding negative results.

## Кратко по-русски

Это не «ещё один фреймворк агентов», а небольшой исполняемый набор контрактов.
Он проверяет схемы инструментов, retries, permissions, idempotency и конечное
состояние одинаковым способом. Сначала публикуются воспроизводимые тесты и
negative results; сравнительные заявления о моделях появятся только после
реальных прогонов.
