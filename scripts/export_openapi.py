#!/usr/bin/env python3
"""Export the FastAPI OpenAPI schema to docs/api/openapi.json.

Usage:
    python scripts/export_openapi.py
    python scripts/export_openapi.py --check
    make openapi
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "api" / "openapi.json"


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise SystemExit("Could not find version in pyproject.toml")
    return match.group(1)


def _normalize_descriptions(obj):
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if k == "422" and isinstance(v, dict) and v.get("description") == "Unprocessable Content":
                v = {**v, "description": "Unprocessable Entity"}
            elif k == "413" and isinstance(v, dict) and v.get("description") == "Content Too Large":
                v = {**v, "description": "Request Entity Too Large"}
            new_obj[k] = _normalize_descriptions(v)
        return new_obj
    elif isinstance(obj, list):
        return [_normalize_descriptions(x) for x in obj]
    return obj


def build_spec() -> dict:
    from nightmarenet import __version__
    from nightmarenet.api.app import app

    spec = app.openapi()
    pkg_version = _pyproject_version()
    if __version__ != pkg_version:
        raise SystemExit(
            f"nightmarenet.__version__ ({__version__}) != pyproject.toml ({pkg_version})"
        )
    if spec.get("info", {}).get("version") != pkg_version:
        raise SystemExit(
            f"OpenAPI info.version ({spec.get('info', {}).get('version')}) "
            f"!= pyproject.toml ({pkg_version})"
        )
    return _normalize_descriptions(spec)


def dump_spec(spec: dict) -> str:
    return json.dumps(spec, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if committed spec differs from the live schema",
    )
    args = parser.parse_args()

    try:
        text = dump_spec(build_spec())
    except ImportError as exc:
        print(
            f"Failed to import API app (install with: pip install -e '.[api]'): {exc}",
            file=sys.stderr,
        )
        return 1

    if args.check:
        if not args.output.exists():
            print(f"Missing committed spec: {args.output}", file=sys.stderr)
            print("Run: python scripts/export_openapi.py", file=sys.stderr)
            return 1
        committed = args.output.read_text(encoding="utf-8")
        if committed != text:
            import difflib

            diff = difflib.unified_diff(
                committed.splitlines(keepends=True),
                text.splitlines(keepends=True),
                fromfile=str(args.output),
                tofile="generated",
            )
            print(
                f"OpenAPI drift detected: {args.output} is out of date.",
                file=sys.stderr,
            )
            print("".join(diff), file=sys.stderr)
            print("Run: make openapi  (or python scripts/export_openapi.py)", file=sys.stderr)
            return 1
        print(f"OpenAPI spec is up to date: {args.output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
