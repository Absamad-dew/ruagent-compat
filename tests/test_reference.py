from __future__ import annotations

import pytest

from ruagent_compat.reference import run_reference_suite


@pytest.mark.asyncio
async def test_reference_suite_passes_all_contracts() -> None:
    results = await run_reference_suite()
    assert len(results) == 6
    assert {result.status for result in results} == {"pass"}
