"""Portable JSON and HTML output for compatibility runs."""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from pathlib import Path

from .reference import ReferenceResult


def write_json(path: Path, results: Sequence[ReferenceResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_html(path: Path, results: Sequence[ReferenceResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        "<tr>"
        f"<td><code>{html.escape(result.case)}</code></td>"
        f"<td class='{html.escape(result.status)}'>{html.escape(result.status.upper())}</td>"
        f"<td>{html.escape(result.assertion)}</td>"
        "<td><code>"
        f"{html.escape(json.dumps(result.details, ensure_ascii=False, sort_keys=True))}"
        "</code></td>"
        "</tr>"
        for result in results
    )
    passed = sum(result.status == "pass" for result in results)
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ruagent-compat reference report</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:40px auto;
padding:0 20px;color:#172033}}
h1{{margin-bottom:4px}}p{{color:#536078}}
table{{border-collapse:collapse;width:100%;margin-top:24px}}
th,td{{border:1px solid #d8deea;padding:10px;text-align:left;vertical-align:top}}
th{{background:#f3f6fb}}.pass{{color:#08783e;font-weight:700}}
.fail{{color:#b42318;font-weight:700}}code{{overflow-wrap:anywhere}}
</style>
</head>
<body>
<h1>ruagent-compat reference report</h1>
<p>{passed}/{len(results)} deterministic runtime contracts passed.
Scripted reference only; no provider quality claim.</p>
<table><thead><tr><th>Case</th><th>Status</th><th>Assertion</th><th>Details</th></tr></thead>
<tbody>{rows}</tbody></table>
</body></html>
"""
    path.write_text(document, encoding="utf-8")
