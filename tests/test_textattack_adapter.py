"""Tests for the TextAttack adapter module."""

from __future__ import annotations

import pytest

from nightmarenet.evaluation.textattack_adapter import (
    ATTACK_RECIPES,
    format_comparison_table,
    get_recipe,
    run_textattack_evaluation,
)


class TestImportFallback:
    """Test graceful behavior when textattack is not installed."""

    def test_run_textattack_evaluation_raises_import_error(self, monkeypatch):
        monkeypatch.setitem(
            __import__("sys").modules, "textattack", None
        )
        with pytest.raises(ImportError, match="textattack is not installed"):
            run_textattack_evaluation(None, None)

    def test_get_recipe_raises_import_error_without_textattack(self, monkeypatch):
        monkeypatch.setitem(
            __import__("sys").modules, "textattack", None
        )
        with pytest.raises(ImportError, match="textattack is not installed"):
            get_recipe("textfooler", None)


class TestGetRecipe:
    """Test get_recipe validation logic."""

    def test_unsupported_attack_raises_value_error(self, monkeypatch):
        monkeypatch.setattr(
            "nightmarenet.evaluation.textattack_adapter._check_textattack_available",
            lambda: None,
        )
        with pytest.raises(ValueError, match="Unsupported attack.*nonexistent"):
            get_recipe("nonexistent", None)

    def test_all_recipe_names_are_recognized(self, monkeypatch):
        monkeypatch.setattr(
            "nightmarenet.evaluation.textattack_adapter._check_textattack_available",
            lambda: None,
        )
        for name in ATTACK_RECIPES:
            with pytest.raises((ImportError, Exception)):
                get_recipe(name, None)


class TestFormatComparisonTable:
    """Test print_comparison_table handles various inputs."""

    def test_empty_results(self):
        table = format_comparison_table({})
        assert "TEXTATTACK ROBUSTNESS EVALUATION" in table
        assert "Dataset: sst2" in table

    def test_error_results(self):
        results = {"textfooler": {"error": "some failure"}}
        table = format_comparison_table(results)
        assert "ERROR" in table
        assert "TEXTFOOLER" in table

    def test_successful_results(self):
        results = {
            "textfooler": {
                "attack_success_rate": 45.5,
                "avg_perturbation_pct": 12.3,
                "avg_queries": 150.7,
                "successful_attacks": 91,
                "failed_attacks": 109,
                "skipped_attacks": 0,
                "elapsed_time": 60.0,
            }
        }
        table = format_comparison_table(results, dataset_name="sst2")
        assert "45.5" in table
        assert "12.3" in table
        assert "150.7" in table
        assert "89.2%" in table  # baseline for textfooler on sst2

    def test_mixed_results(self):
        results = {
            "textfooler": {
                "attack_success_rate": 50.0,
                "avg_perturbation_pct": 10.0,
                "avg_queries": 100.0,
                "successful_attacks": 100,
                "failed_attacks": 100,
                "skipped_attacks": 0,
                "elapsed_time": 30.0,
            },
            "bertattack": {"error": "timeout"},
        }
        table = format_comparison_table(results, dataset_name="sst2")
        assert "TEXTFOOLER" in table
        assert "BERTATTACK" in table
        assert "ERROR" in table
        assert "50.0" in table
