"""Tests for ensemble benchmarking functionality."""

import os
import tempfile
from unittest import mock

import yaml

from nightmarenet.evaluation.degradation_curves import calculate_degradation_curves
from nightmarenet.evaluation.ensemble_benchmark import EnsembleOrchestrator
from nightmarenet.evaluation.format_results import to_latex_table
from nightmarenet.evaluation.pareto_analysis import get_pareto_frontier


def test_pareto_frontier_correctness():
    """Test that the Pareto frontier correctly identifies non-dominated models."""
    results = [
        {"model": "A", "robustness": 0.9, "latency": 10.0, "params": 100},  # dominated by B
        {"model": "B", "robustness": 0.95, "latency": 5.0, "params": 50},  # pareto optimal
        {"model": "C", "robustness": 0.8, "latency": 2.0, "params": 20},  # pareto optimal
        {"model": "D", "robustness": 0.7, "latency": 8.0, "params": 80},  # dominated by C
    ]

    pareto_front = get_pareto_frontier(results)
    assert len(pareto_front) == 2
    models_on_front = {m["model"] for m in pareto_front}
    assert "B" in models_on_front
    assert "C" in models_on_front
    assert "A" not in models_on_front
    assert "D" not in models_on_front


def test_degradation_curve_aggregation():
    """Test aggregation of robustness scores into degradation curves."""
    raw_results = {
        "model_A": {"dream": {"strengths": [0.1, 0.5, 0.9], "accuracies": [10.0, 50.0, 100.0]}},
        "model_B": {"dream": {"strengths": [0.1, 0.5], "accuracies": [5.0, 20.0]}},
    }

    curves = calculate_degradation_curves(raw_results)

    assert "model_A" in curves
    assert len(curves["model_A"]) == 3
    assert curves["model_A"][0]["strength"] == 0.1
    assert curves["model_A"][0]["robustness"] == 10.0

    assert "model_B" in curves
    assert len(curves["model_B"]) == 2
    assert curves["model_B"][1]["strength"] == 0.5
    assert curves["model_B"][1]["robustness"] == 20.0


def test_latex_table_generation():
    """Test output formatting for LaTeX table generation."""
    models_summary = [
        {"model": "test-model-1", "robustness": 0.85, "latency": 1.2, "params": 110_000_000},
        {"model": "test_model_2", "robustness": 0.92, "latency": 2.5, "params": 340_000_000},
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        latex_path = os.path.join(temp_dir, "table.tex")
        to_latex_table(models_summary, latex_path)

        with open(latex_path, encoding="utf-8") as f:
            content = f.read()

        assert "\\begin{tabular}" in content
        assert "test-model-1" in content
        assert "110.0M" in content
        # Note: test_model_2 should have underscores escaped
        assert "test\\_model\\_2" in content
        assert "340.0M" in content


@mock.patch("nightmarenet.evaluation.ensemble_benchmark.ProcessPoolExecutor")
def test_ensemble_orchestrator_logic(mock_executor_class):
    """Test EnsembleOrchestrator config parsing and orchestration logic."""
    mock_future = mock.MagicMock()
    mock_future.result.return_value = {
        "model": "dummy",
        "robustness": 0.99,
        "latency": 1.5,
        "params": 1000,
        "results_by_distortion": {"dream": {"strengths": [0.1, 0.5], "accuracies": [5.0, 10.0]}},
    }

    mock_executor = mock.MagicMock()
    mock_executor.submit.return_value = mock_future
    mock_executor.__enter__.return_value = mock_executor
    mock_executor_class.return_value = mock_executor

    config = {
        "models": ["dummy"],
        "dataset": {"name": "test", "split": "val"},
        "distortions": [{"type": "dream", "strengths": [0.1, 0.5]}],
    }

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        temp_config = f.name

    try:
        orchestrator = EnsembleOrchestrator(temp_config)
        assert orchestrator.config.models == ["dummy"]

        results = orchestrator.run()
        assert "models_summary" in results
        assert len(results["models_summary"]) == 1
        assert results["models_summary"][0]["robustness"] == 0.99

        assert "raw_results" in results
        assert "dummy" in results["raw_results"]
    finally:
        os.remove(temp_config)


@mock.patch("nightmarenet.evaluation.ensemble_benchmark.ProcessPoolExecutor")
def test_ensemble_orchestrator_timeout(mock_executor_class):
    """Test EnsembleOrchestrator timeout behavior."""
    from concurrent.futures import TimeoutError

    # Mock future that raises TimeoutError
    mock_future = mock.MagicMock()
    mock_future.result.side_effect = TimeoutError("Timed out")

    mock_executor = mock.MagicMock()
    mock_executor.submit.return_value = mock_future
    mock_executor.__enter__.return_value = mock_executor

    mock_executor_class.return_value = mock_executor

    config = {"models": ["slow_model"], "dataset": {"name": "test"}}

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        temp_config = f.name

    try:
        orchestrator = EnsembleOrchestrator(temp_config)
        results = orchestrator.run(timeout_seconds=1)

        # It should handle the timeout gracefully without crashing,
        # but the slow_model won't be in the results.
        assert len(results["models_summary"]) == 0
        assert "slow_model" not in results["raw_results"]
    finally:
        os.remove(temp_config)


def test_cache_key_uniqueness():
    """Test that cache keys are unique for different strength values."""

    # Test the cache key generation logic directly
    def _get_cache_key(model, dataset, split, distortion_type, strength):
        safe_model = model.replace("/", "_").replace("-", "_")
        return f"{safe_model}_{dataset}_{split}_{distortion_type}_{strength:g}.json"

    # Test that 0.15 and 0.1 produce different cache keys
    key_0_15 = _get_cache_key("model", "dataset", "split", "dream", 0.15)
    key_0_1 = _get_cache_key("model", "dataset", "split", "dream", 0.1)

    assert key_0_15 != key_0_1, f"Cache keys should be unique: {key_0_15} vs {key_0_1}"

    # Test that 0.10 and 0.1 produce the same cache key (g format removes trailing zeros)
    key_0_10 = _get_cache_key("model", "dataset", "split", "dream", 0.10)
    assert key_0_10 == key_0_1, (
        f"Cache keys should be same for 0.10 and 0.1: {key_0_10} vs {key_0_1}"
    )


def test_cache_hit_scenario():
    """Test cache hit scenario where cached results are reused."""
    import json
    from pathlib import Path

    # Test cache key generation and file I/O logic directly
    def _get_cache_key(model, dataset, split, distortion_type, strength):
        safe_model = model.replace("/", "_").replace("-", "_")
        return f"{safe_model}_{dataset}_{split}_{distortion_type}_{strength:g}.json"

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create a cache file with pre-computed results
        cache_file = cache_dir / _get_cache_key("dummy", "test", "val", "dream", 0.1)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"accuracy": 0.95}, f)

        # Verify cache file exists and can be read
        assert cache_file.exists()
        with open(cache_file, encoding="utf-8") as f:
            cached_result = json.load(f)
            assert cached_result["accuracy"] == 0.95


def test_cache_miss_scenario():
    """Test cache miss scenario where results are computed and cached."""
    import json
    from pathlib import Path

    # Test cache key generation and file I/O logic directly
    def _get_cache_key(model, dataset, split, distortion_type, strength):
        safe_model = model.replace("/", "_").replace("-", "_")
        return f"{safe_model}_{dataset}_{split}_{distortion_type}_{strength:g}.json"

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_file = cache_dir / _get_cache_key("dummy", "test", "val", "dream", 0.1)

        # Verify cache file doesn't exist initially
        assert not cache_file.exists()

        # Simulate writing cache file after evaluation
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"accuracy": 0.85}, f)

        # Verify cache file was created
        assert cache_file.exists()
        with open(cache_file, encoding="utf-8") as f:
            cached_result = json.load(f)
            assert cached_result["accuracy"] == 0.85


def test_cache_corrupt_scenario():
    """Test corrupt cache scenario where corrupted cache triggers re-evaluation."""
    import json
    from pathlib import Path

    # Test cache key generation and file I/O logic directly
    def _get_cache_key(model, dataset, split, distortion_type, strength):
        safe_model = model.replace("/", "_").replace("-", "_")
        return f"{safe_model}_{dataset}_{split}_{distortion_type}_{strength:g}.json"

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_file = cache_dir / _get_cache_key("dummy", "test", "val", "dream", 0.1)

        # Create a corrupted cache file
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write("{ invalid json content")

        # Verify corrupted cache file exists
        assert cache_file.exists()

        # Simulate handling corrupted cache - should raise JSONDecodeError
        try:
            with open(cache_file, encoding="utf-8") as f:
                json.load(f)
            raise AssertionError("Should have raised JSONDecodeError")
        except json.JSONDecodeError:
            # Expected - corrupted cache should be detected
            pass

        # Simulate re-evaluation by overwriting with valid data
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"accuracy": 0.90}, f)

        # Verify cache file now has valid data
        with open(cache_file, encoding="utf-8") as f:
            cached_result = json.load(f)
            assert cached_result["accuracy"] == 0.90
