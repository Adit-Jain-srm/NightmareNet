"""Unit tests for the robustness-check Action helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / ".github" / "actions" / "robustness-check" / "entrypoint.py"


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location("robustness_check_entrypoint", ENTRY)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ep():
    return _load_entrypoint()


def test_extract_score(ep):
    assert ep.extract_score({"robustness_score": 0.81}) == 0.81


def test_filter_aggregate_respects_distortion_types(ep):
    report = {
        "robustness_score": 0.5,
        "avg_dream_similarity": 0.9,
        "avg_nightmare_similarity": 0.1,
    }
    assert ep.filter_aggregate(report, ["dream"]) == 0.9
    assert ep.filter_aggregate(report, ["nightmare"]) == 0.1
    assert ep.filter_aggregate(report, ["dream", "nightmare"]) == 0.5


def test_per_distortion_rows(ep):
    report = {
        "strengths": [
            {
                "strength": 0.5,
                "dream_similarity": 0.8,
                "nightmare_similarity": 0.4,
            }
        ]
    }
    rows = ep.per_distortion_rows(report, ["dream", "nightmare"])
    assert ("dream@0.5", 0.8) in rows
    assert ("nightmare@0.5", 0.4) in rows


def test_format_markdown_contains_table_and_marker(ep):
    md = ep.format_markdown(
        model="distilbert-base-uncased",
        score=0.75,
        threshold=0.7,
        passed=True,
        rows=[("dream@0.5", 0.8)],
        report_path="/tmp/report.json",
    )
    assert ep.MARKER in md
    assert "PASSED" in md
    assert "| dream@0.5 | 0.8000 |" in md
    assert "0.7500" in md


def test_entrypoint_fails_below_threshold(ep, tmp_path, monkeypatch):
    report = {
        "model": "m",
        "robustness_score": 0.4,
        "avg_dream_similarity": 0.4,
        "avg_nightmare_similarity": 0.4,
        "strengths": [],
    }

    def fake_run(cmd, report_path):
        report_path.write_text("{}", encoding="utf-8")
        return report

    monkeypatch.setenv("INPUT_MODEL_PATH", "distilbert-base-uncased")
    monkeypatch.setenv("INPUT_THRESHOLD", "0.7")
    monkeypatch.setenv("INPUT_DISTORTION_TYPES", "dream,nightmare")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("INPUT_COMMENT_PATH", str(tmp_path / "comment.md"))
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setattr(ep, "run_evaluate", fake_run)
    monkeypatch.setattr(ep, "build_evaluate_cmd", lambda **kwargs: ["nightmarenet", "evaluate"])

    assert ep.main() == 1
    assert (tmp_path / "comment.md").is_file()
