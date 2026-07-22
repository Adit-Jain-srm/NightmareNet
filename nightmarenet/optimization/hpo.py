"""Optuna Hyperparameter Optimization integration for NightmareNet."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import optuna

from nightmarenet.pipeline import Pipeline
from nightmarenet.utils.config import load_config

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None  # type: ignore[assignment]
    OPTUNA_AVAILABLE = False


logger = logging.getLogger(__name__)


def _set_nested(config: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dictionary using dotted key notation."""
    keys = dotted_key.split(".")
    current = config
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


class HyperparameterOptimizer:
    """Manages Optuna hyperparameter optimization for the NightmareNet pipeline."""

    def __init__(self, config_path: str):
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for HPO. Install it with: pip install 'nightmarenet[hpo]'"
            )
        self.config_path = config_path
        self.base_config = load_config(config_path)
        self.hpo_config = self.base_config.get("hpo", {})

        self.study_name = self.hpo_config.get("study_name", "nightmarenet-optimization")
        # Default to a local SQLite database if storage is not configured.
        self.storage = self.hpo_config.get("storage", "sqlite:///nightmarenet_hpo.db")
        self.direction = self.hpo_config.get("direction", "maximize")
        self.n_trials = self.hpo_config.get("n_trials", 20)
        self.pruning_enabled = self.hpo_config.get("pruning", True)
        self.search_space = self.hpo_config.get("search_space", {})

        pruner = (
            optuna.pruners.MedianPruner() if self.pruning_enabled else optuna.pruners.NopPruner()
        )

        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=self.direction,
            pruner=pruner,
            load_if_exists=True,
        )

    def _suggest_parameters(self, trial: optuna.Trial) -> dict:
        """Parse search space from config and suggest parameters."""
        trial_params = {}
        for param_key, param_def in self.search_space.items():
            param_type = param_def.get("type")
            if param_type == "float":
                trial_params[param_key] = trial.suggest_float(
                    param_key,
                    param_def["low"],
                    param_def["high"],
                    log=param_def.get("log", False),
                )
            elif param_type == "int":
                trial_params[param_key] = trial.suggest_int(
                    param_key,
                    param_def["low"],
                    param_def["high"],
                    log=param_def.get("log", False),
                )
            elif param_type == "categorical":
                trial_params[param_key] = trial.suggest_categorical(param_key, param_def["choices"])
            else:
                logger.warning("Unknown parameter type '%s' for %s", param_type, param_key)
        return trial_params

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        suggested_params = self._suggest_parameters(trial)

        trial_config = copy.deepcopy(self.base_config)
        for k, v in suggested_params.items():
            _set_nested(trial_config, k, v)

        # Pruning coordination
        pruned_flag = [False]
        pipeline = None

        def on_event(metrics: dict) -> None:
            if not self.pruning_enabled:
                return

            # The pipeline emits status via metrics
            status = metrics.get("status")
            if status == "training":
                phase_loss = metrics.get("phase_loss", 0.0)
                current_cycle = metrics.get("current_cycle", 0)

                if current_cycle > 0:
                    trial.report(phase_loss, step=current_cycle)
                    if trial.should_prune():
                        pruned_flag[0] = True
                        if pipeline is not None:
                            pipeline.cancel()

        pipeline = Pipeline(config=trial_config, on_event=on_event)

        try:
            comparison = pipeline.run()
        except Exception as e:
            if pruned_flag[0] or pipeline._context.cancelled:
                raise optuna.exceptions.TrialPruned() from e
            logger.exception("Trial failed")
            raise

        if pruned_flag[0] or pipeline._context.cancelled:
            raise optuna.exceptions.TrialPruned()

        metric_val = comparison.get("robustness_delta")
        if metric_val is None:
            for fallback in ("robustness", "avg_robustness", "mean_robustness"):
                if fallback in comparison:
                    metric_val = comparison[fallback]
                    break

        if metric_val is None:
            logger.warning("robustness_delta not found in comparison. Returning 0.0")
            return 0.0

        return float(metric_val)

    def optimize(self) -> optuna.Study:
        """Run the optimization study."""
        logger.info("Starting optimization for %d trials.", self.n_trials)
        self.study.optimize(self._objective, n_trials=self.n_trials)

        try:
            best_trial = self.study.best_trial
            logger.info("Optimization finished. Best trial: %d", best_trial.number)
            logger.info("  Value: %s", best_trial.value)
            logger.info("  Params:")
            for k, v in best_trial.params.items():
                logger.info("    %s: %s", k, v)
        except ValueError:
            logger.info("Optimization finished but no completed trials were found.")

        return self.study
