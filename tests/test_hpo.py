"""Tests for the HPO (Hyperparameter Optimization) module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _optuna_available() -> bool:
    try:
        import optuna  # noqa: F401
        return True
    except ImportError:
        return False


class TestHyperparameterOptimizer:
    """Tests for HyperparameterOptimizer initialization and optimization."""

    def test_import_error_without_optuna(self, monkeypatch):
        """Raises ImportError with install instructions when optuna is missing."""
        monkeypatch.setattr(
            "nightmarenet.optimization.hpo.OPTUNA_AVAILABLE", False
        )
        from nightmarenet.optimization.hpo import HyperparameterOptimizer

        with pytest.raises(ImportError, match="Optuna is required"):
            HyperparameterOptimizer("configs/default.yaml")

    @pytest.mark.skipif(
        not _optuna_available(), reason="optuna not installed"
    )
    def test_optimizer_loads_config(self, tmp_path):
        """Optimizer reads hpo section from config."""
        import yaml

        config = {
            "model": {"name": "gpt2", "type": "causal_lm", "max_length": 64, "device": "cpu"},
            "dataset": {"name": "wikitext", "config": "wikitext-2-raw-v1", "text_column": "text"},
            "training": {
                "wake_epochs": 1, "dream_epochs": 0, "nightmare_epochs": 0,
                "compression_rounds": 0, "num_cycles": 1, "batch_size": 4,
                "learning_rate": 5e-5,
            },
            "distortion": {"dream_strength": 0.2, "nightmare_strength": 0.7},
            "compression": {"pruning_ratio": 0.1, "bottleneck_rank_ratio": 0.5},
            "seed": 42,
            "hpo": {
                "n_trials": 2,
                "study_name": "test-study",
                "storage": f"sqlite:///{tmp_path / 'test.db'}",
                "search_space": {
                    "training.learning_rate": {
                        "type": "float", "low": 1e-5, "high": 1e-3, "log": True,
                    },
                },
            },
        }
        cfg_path = tmp_path / "test_config.yaml"
        cfg_path.write_text(yaml.dump(config))

        from nightmarenet.optimization.hpo import HyperparameterOptimizer

        opt = HyperparameterOptimizer(str(cfg_path))
        assert opt.n_trials == 2
        assert opt.study_name == "test-study"
        assert "training.learning_rate" in opt.search_space

    @pytest.mark.skipif(
        not _optuna_available(), reason="optuna not installed"
    )
    def test_optimize_runs_trials(self, tmp_path):
        """optimize() executes trials and returns a study."""
        import yaml

        config = {
            "model": {"name": "gpt2", "type": "causal_lm", "max_length": 64, "device": "cpu"},
            "dataset": {"name": "wikitext", "config": "wikitext-2-raw-v1", "text_column": "text"},
            "training": {
                "wake_epochs": 1, "dream_epochs": 0, "nightmare_epochs": 0,
                "compression_rounds": 0, "num_cycles": 1, "batch_size": 4,
                "learning_rate": 5e-5,
            },
            "distortion": {"dream_strength": 0.2, "nightmare_strength": 0.7},
            "compression": {"pruning_ratio": 0.1, "bottleneck_rank_ratio": 0.5},
            "seed": 42,
            "hpo": {
                "n_trials": 2,
                "study_name": "trial-test",
                "storage": f"sqlite:///{tmp_path / 'trial.db'}",
                "search_space": {
                    "distortion.nightmare_strength": {"type": "float", "low": 0.5, "high": 0.9}
                },
            },
        }
        cfg_path = tmp_path / "trial_config.yaml"
        cfg_path.write_text(yaml.dump(config))

        mock_comparison = {"robustness_delta": 0.05}

        with patch("nightmarenet.optimization.hpo.Pipeline") as mock_pipeline:
            mock_instance = MagicMock()
            mock_instance.run.return_value = mock_comparison
            mock_instance._context = MagicMock(cancelled=False)
            mock_pipeline.return_value = mock_instance

            from nightmarenet.optimization.hpo import HyperparameterOptimizer

            opt = HyperparameterOptimizer(str(cfg_path))
            study = opt.optimize()

            assert study.best_value == 0.05
            assert mock_pipeline.call_count == 2
