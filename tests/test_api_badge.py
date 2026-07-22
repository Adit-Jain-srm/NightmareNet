"""Tests for the live robustness badge endpoint (/api/v1/badge/latest.svg).

Covers:
- Dynamic SVG badge generation for the most recent completed run
- "no data" fallback badge when no completed runs exist
- Score threshold colors: green (>80%), yellow (50-80%), red (<50%)
- Cache-Control header presence (max-age=300)
- Public access without API key
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402, I001

from nightmarenet.api.app import app  # noqa: E402, I001
import nightmarenet.pipeline_runner as pr  # noqa: E402, I001

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clean_runs(monkeypatch, tmp_path):
    """Ensure clean runner registry and temporary runs directory for each test."""
    monkeypatch.setenv("NIGHTMARENET_RUNS_DIR", str(tmp_path))
    pr._runners.clear()


class TestLatestBadgeSvg:
    """Tests for ``GET /api/v1/badge/latest.svg``."""

    def test_latest_svg_no_runs(self):
        """Returns a 'no data' badge when no completed runs exist."""
        r = client.get("/api/v1/badge/latest.svg")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/svg+xml")
        assert "max-age=300" in r.headers.get("cache-control", "")
        body = r.text
        assert "robustness" in body
        assert "no data" in body
        assert "#9f9f9f" in body
        ET.fromstring(body)

    def test_latest_svg_green_score(self):
        """Score > 0.80 returns a green badge (#22c55e)."""
        now = time.time()
        pr._persist_run_state("run-green", {}, "running", now)
        pr._update_run_state(
            "run-green",
            "complete",
            now,
            metrics={"robustness_score": 0.85},
        )

        r = client.get("/api/v1/badge/latest.svg")
        assert r.status_code == 200
        body = r.text
        assert "0.85" in body
        assert "#22c55e" in body
        ET.fromstring(body)

    def test_latest_svg_yellow_score(self):
        """Score between 0.50 and 0.80 returns a yellow badge (#eab308)."""
        now = time.time()
        pr._persist_run_state("run-yellow", {}, "running", now)
        pr._update_run_state(
            "run-yellow",
            "complete",
            now,
            metrics={"auc_robustness": 0.65},
        )

        r = client.get("/api/v1/badge/latest.svg")
        assert r.status_code == 200
        body = r.text
        assert "0.65" in body
        assert "#eab308" in body
        ET.fromstring(body)

    def test_latest_svg_red_score(self):
        """Score < 0.50 returns a red badge (#ef4444)."""
        now = time.time()
        pr._persist_run_state("run-red", {}, "running", now)
        pr._update_run_state(
            "run-red",
            "complete",
            now,
            metrics={
                "trained_results": {"robustness": {"auc_robustness": 0.35}},
            },
        )

        r = client.get("/api/v1/badge/latest.svg")
        assert r.status_code == 200
        body = r.text
        assert "0.35" in body
        assert "#ef4444" in body
        ET.fromstring(body)

    def test_latest_svg_selects_most_recent_completed_run(self):
        """Picks the score from the most recent completed run."""
        old_time = time.time() - 300
        new_time = time.time()

        # Older run: score 0.30
        pr._persist_run_state("run-old", {}, "running", old_time)
        pr._update_run_state("run-old", "complete", old_time, metrics={"robustness_score": 0.30})

        # Newer run: score 0.92
        pr._persist_run_state("run-new", {}, "running", new_time)
        pr._update_run_state("run-new", "complete", new_time, metrics={"robustness_score": 0.92})

        r = client.get("/api/v1/badge/latest.svg")
        assert r.status_code == 200
        body = r.text
        assert "0.92" in body
        assert "#22c55e" in body

    def test_latest_svg_ignores_non_completed_runs(self):
        """Ignores running/failed/cancelled runs and uses latest completed run."""
        old_time = time.time() - 300
        new_time = time.time()

        # Older completed run
        pr._persist_run_state("run-complete", {}, "running", old_time)
        pr._update_run_state(
            "run-complete", "complete", old_time, metrics={"robustness_score": 0.75}
        )

        # Newer failed run
        pr._persist_run_state("run-failed", {}, "running", new_time)
        pr._update_run_state("run-failed", "failed", new_time, metrics={"robustness_score": 0.10})

        r = client.get("/api/v1/badge/latest.svg")
        assert r.status_code == 200
        body = r.text
        assert "0.75" in body
        assert "#eab308" in body

    def test_latest_svg_public_access(self, monkeypatch):
        """Endpoint remains publicly accessible even when API key is set."""
        monkeypatch.setenv("NIGHTMARENET_API_KEY", "rk_test_key_123")
        r = client.get("/api/v1/badge/latest.svg")
        assert r.status_code == 200
