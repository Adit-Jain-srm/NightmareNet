"""Tests for distortion quality vs model size correlation study script."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.distortion_quality_vs_size import (
    calculate_metrics,
    main,
    run_study,
    write_svg_plot,
)


def test_calculate_metrics() -> None:
    orig = "The quick brown fox jumps over the lazy dog"
    distortions = [
        "The quick red fox jumps over the lazy dog",
        "The fast brown fox jumps over a lazy dog",
    ]
    metrics = calculate_metrics(orig, distortions)

    assert "semantic_preservation" in metrics
    assert "grammaticality" in metrics
    assert "diversity" in metrics

    for _key, val in metrics.items():
        assert 0.0 <= val <= 1.0


def test_calculate_metrics_empty() -> None:
    metrics = calculate_metrics("test", [])
    assert metrics["semantic_preservation"] == 0.0
    assert metrics["grammaticality"] == 0.0
    assert metrics["diversity"] == 0.0


def test_write_svg_plot(tmp_path: Path) -> None:
    data = {
        "tiny": {"semantic_preservation": 0.6, "grammaticality": 0.5, "diversity": 0.4},
        "base": {"semantic_preservation": 0.8, "grammaticality": 0.8, "diversity": 0.8},
    }
    plot_file = tmp_path / "plot.svg"
    write_svg_plot(data, plot_file)

    assert plot_file.exists()
    content = plot_file.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "Distortion Quality vs Model Size" in content


def test_run_study_calibrate() -> None:
    res = run_study(calibrate=True)
    assert "models" in res
    assert "minimum_production_size" in res
    assert res["minimum_production_size"] == "base"
    assert "tiny" in res["models"]
    assert "large" in res["models"]


def test_main_cli(tmp_path: Path, monkeypatch) -> None:
    out_json = tmp_path / "results.json"
    out_svg = tmp_path / "plot.svg"

    monkeypatch.setattr(
        "sys.argv",
        [
            "distortion_quality_vs_size.py",
            "--calibrate",
            "--output",
            str(out_json),
            "--plot",
            str(out_svg),
        ],
    )

    exit_code = main()
    assert exit_code == 0
    assert out_json.exists()
    assert out_svg.exists()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["minimum_production_size"] == "base"
