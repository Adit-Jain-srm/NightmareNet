#!/usr/bin/env python3
"""NightmareNet robustness-check GitHub Action entrypoint.

Runs ``nightmarenet evaluate --json``, writes a report, sets Action outputs,
emits a markdown summary table, and exits non-zero when the score is below
the configured threshold.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

MARKER = "<!-- nightmarenet-robustness-check -->"


def _parse_csv(raw: str) -> List[str]:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def extract_score(report: Dict[str, Any]) -> float:
    """Pull a [0, 1] robustness score from evaluate --json payloads."""
    if "robustness_score" in report:
        return float(report["robustness_score"])
    # Config/evaluator path may nest metrics differently.
    for key in ("score", "avg_robustness", "robustness"):
        if key in report:
            return float(report[key])
    raise KeyError("Report JSON is missing robustness_score")


def per_distortion_rows(
    report: Dict[str, Any],
    distortion_types: Sequence[str],
) -> List[Tuple[str, float]]:
    """Build (label, score) rows for the PR comment table."""
    rows: List[Tuple[str, float]] = []
    strengths = report.get("strengths") or []
    types = [t.lower() for t in distortion_types] if distortion_types else ["dream", "nightmare"]

    if strengths and isinstance(strengths, list):
        for entry in strengths:
            if not isinstance(entry, dict):
                continue
            strength = entry.get("strength", "?")
            if "dream" in types and "dream_similarity" in entry:
                rows.append((f"dream@{strength}", float(entry["dream_similarity"])))
            if "nightmare" in types and "nightmare_similarity" in entry:
                rows.append((f"nightmare@{strength}", float(entry["nightmare_similarity"])))
        if rows:
            return rows

    # Aggregates when per-strength detail is absent.
    if "dream" in types and "avg_dream_similarity" in report:
        rows.append(("dream (avg)", float(report["avg_dream_similarity"])))
    if "nightmare" in types and "avg_nightmare_similarity" in report:
        rows.append(("nightmare (avg)", float(report["avg_nightmare_similarity"])))
    return rows


def filter_aggregate(
    report: Dict[str, Any],
    distortion_types: Sequence[str],
) -> float:
    """Recompute aggregate score when the caller restricts distortion types."""
    types = [t.lower() for t in distortion_types] if distortion_types else []
    if not types or set(types) >= {"dream", "nightmare"}:
        return extract_score(report)

    parts: List[float] = []
    if "dream" in types and "avg_dream_similarity" in report:
        parts.append(float(report["avg_dream_similarity"]))
    if "nightmare" in types and "avg_nightmare_similarity" in report:
        parts.append(float(report["avg_nightmare_similarity"]))
    if parts:
        return round(sum(parts) / len(parts), 4)
    return extract_score(report)


def format_markdown(
    *,
    model: str,
    score: float,
    threshold: float,
    passed: bool,
    rows: Sequence[Tuple[str, float]],
    report_path: str,
) -> str:
    status = "PASSED" if passed else "FAILED"
    lines = [
        MARKER,
        "## NightmareNet Robustness Check",
        "",
        f"**Status:** {status}  ",
        f"**Model:** `{model}`  ",
        f"**Overall score:** `{score:.4f}` (threshold `{threshold:.4f}`)  ",
        f"**Report:** `{report_path}`",
        "",
        "| Distortion | Score |",
        "| --- | ---: |",
    ]
    if rows:
        for label, value in rows:
            lines.append(f"| {label} | {value:.4f} |")
    else:
        lines.append("| (no per-distortion breakdown) | — |")
    lines.append("")
    return "\n".join(lines)


def _append_output(name: str, value: str) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if not out:
        print(f"::notice::{name}={value}")
        return
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(f"{name}={value}\n")


def _append_summary(markdown: str) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return
    with open(summary, "a", encoding="utf-8") as fh:
        fh.write(markdown)
        fh.write("\n")


def build_evaluate_cmd(
    *,
    model_path: str,
    config_path: str,
    strengths: str,
    text: str,
    dataset: str,
) -> List[str]:
    # Prefer the console script; fall back to module invocation.
    cmd = ["nightmarenet", "evaluate", "--json"]
    try:
        subprocess.run(
            ["nightmarenet", "--help"],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        cmd = [sys.executable, "-m", "nightmarenet.cli", "evaluate", "--json"]

    if config_path:
        cmd.extend(["--config", config_path])
    if model_path:
        cmd.extend(["--model", model_path])
    if dataset:
        cmd.extend(["--dataset", dataset])
    if strengths:
        cmd.extend(["--strengths", strengths])
    if text:
        cmd.extend(["--text", text])
    return cmd


def run_evaluate(cmd: Sequence[str], report_path: Path) -> Dict[str, Any]:
    result = subprocess.run(
        list(cmd),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr or result.stdout or "evaluate failed\n")
        raise SystemExit(result.returncode or 1)

    stdout = (result.stdout or "").strip()
    if not stdout:
        sys.stderr.write("evaluate produced empty stdout\n")
        raise SystemExit(1)

    # Last non-empty line should be the JSON object (logging may be disabled).
    payload_line = stdout.splitlines()[-1]
    report = json.loads(payload_line)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    del argv  # env-driven; argv unused
    model_path = os.environ.get("INPUT_MODEL_PATH", "").strip()
    config_path = os.environ.get("INPUT_CONFIG_PATH", "").strip()
    threshold = float(os.environ.get("INPUT_THRESHOLD", "0.7"))
    distortion_types = _parse_csv(os.environ.get("INPUT_DISTORTION_TYPES", "dream,nightmare"))
    strengths = os.environ.get("INPUT_STRENGTHS", "0.1,0.3,0.5,0.7,0.9").strip()
    text = os.environ.get("INPUT_TEXT", "").strip()
    dataset = os.environ.get("INPUT_DATASET", "sst2").strip()

    if not model_path and not config_path:
        sys.stderr.write("::error::model_path or config_path is required\n")
        return 1

    temp_root = Path(os.environ.get("RUNNER_TEMP") or os.environ.get("TMPDIR") or "/tmp")
    report_path = temp_root / "nightmarenet-robustness-report.json"
    comment_path = Path(
        os.environ.get("INPUT_COMMENT_PATH") or (temp_root / "robustness-comment.md")
    )

    cmd = build_evaluate_cmd(
        model_path=model_path,
        config_path=config_path,
        strengths=strengths,
        text=text,
        dataset=dataset,
    )
    print(f"Running: {' '.join(cmd)}", flush=True)
    report = run_evaluate(cmd, report_path)

    score = filter_aggregate(report, distortion_types)
    passed = score >= threshold
    rows = per_distortion_rows(report, distortion_types)
    markdown = format_markdown(
        model=model_path or config_path or report.get("model", ""),
        score=score,
        threshold=threshold,
        passed=passed,
        rows=rows,
        report_path=str(report_path),
    )
    comment_path.write_text(markdown, encoding="utf-8")
    _append_summary(markdown)

    _append_output("robustness_score", f"{score:.4f}")
    _append_output("passed", "true" if passed else "false")
    _append_output("report_path", str(report_path))

    print(markdown)
    if not passed:
        sys.stderr.write(
            f"::error::Robustness score {score:.4f} is below threshold {threshold:.4f}\n"
        )
        return 1
    print(f"Robustness score {score:.4f} >= threshold {threshold:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
