"""Command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .reference import run_reference_suite
from .reporting import write_html, write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ruagent-compat")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="run deterministic reference contracts")
    verify.add_argument("--json", type=Path, dest="json_path")
    verify.add_argument("--html", type=Path, dest="html_path")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command != "verify":
        raise RuntimeError(f"unsupported command: {args.command}")

    results = asyncio.run(run_reference_suite())
    if args.json_path:
        write_json(args.json_path, results)
    if args.html_path:
        write_html(args.html_path, results)

    for result in results:
        print(f"{result.status.upper():4} {result.case}: {result.assertion}")
    passed = sum(result.status == "pass" for result in results)
    print(f"\n{passed}/{len(results)} reference contracts passed")
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
