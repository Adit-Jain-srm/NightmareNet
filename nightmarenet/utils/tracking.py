"""Experiment tracking abstraction for NightmareNet.

Supports wandb, tensorboard, and none (no-op) backends via a unified interface.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Unified experiment tracker wrapping wandb/tensorboard/none/mlflow backends.

    Args:
        backend: Tracking backend name: "none", "wandb", "tensorboard", or "mlflow".
        project: Project name for the tracking platform.
        run_name: Optional name for this experiment run.
        config: Optional hyperparameter config to log.
        log_dir: Directory for tensorboard logs. Ignored for wandb.
    """

    def __init__(
        self,
        backend: str = "none",
        project: str = "nightmarenet",
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
        log_dir: str = "logs/runs",
    ):
        self.backend = backend.lower()
        self.project = project
        self.run_name = run_name
        self._step = 0
        self._writer: Any = None
        self._run: Any = None

        self.run_id = str(uuid.uuid4())

        self.lineage: dict[str, Any] = {
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": {},
            "phases": [],
        }

        if self.backend == "wandb":
            try:
                import wandb  # type: ignore[import-untyped]

                self._run = wandb.init(
                    project=project,
                    name=run_name,
                    config=config or {},
                    reinit=True,
                )
                logger.info("Wandb tracking initialized (project=%s).", project)
            except ImportError:
                logger.warning("wandb not installed; falling back to no-op tracker.")
                self.backend = "none"
            except Exception as exc:
                logger.warning("Failed to init wandb: %s; falling back to no-op.", exc)
                self.backend = "none"

        elif self.backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(log_dir=log_dir)
                logger.info("TensorBoard tracking initialized (log_dir=%s).", log_dir)
            except ImportError:
                logger.warning("tensorboard not installed; falling back to no-op tracker.")
                self.backend = "none"

        elif self.backend == "mlflow":
            try:
                import mlflow

                tracking_cfg = config.get("tracking", {}) if config else {}
                tracking_uri = tracking_cfg.get("tracking_uri")
                experiment_name = tracking_cfg.get("experiment", self.project)
                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(experiment_name)
                self._run = mlflow.start_run(run_name=self.run_name)
                logger.info("MLflow tracking initialized (experiment=%s).", experiment_name)
            except ImportError:
                logger.warning("mlflow not installed; falling back to no-op tracker.")
                self.backend = "none"
            except Exception as exc:
                logger.warning("Failed to init mlflow: %s; falling back to no-op.", exc)
                self.backend = "none"

        elif self.backend != "none":
            logger.warning("Unknown tracking backend '%s'; using no-op.", self.backend)
            self.backend = "none"

        if self.backend == "none":
            logger.info("Experiment tracking disabled (backend=none).")

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """Log a dict of metrics at the given step.

        Args:
            metrics: Dictionary of metric name → value.
            step: Global step number. Auto-increments if None.
        """
        if step is None:
            step = self._step
            self._step += 1
        else:
            self._step = step + 1

        if self.backend == "wandb":
            import wandb  # type: ignore[import-untyped]

            wandb.log(metrics, step=step)

        elif self.backend == "tensorboard" and self._writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._writer.add_scalar(key, value, global_step=step)
        elif self.backend == "mlflow":
            try:
                import mlflow

                num_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                if num_metrics:
                    mlflow.log_metrics(num_metrics, step=step)
            except Exception as exc:
                logger.warning("Failed to log metrics to MLflow: %s", exc)

    def log_phase(self, cycle: int, phase: str, metrics: dict[str, Any]) -> None:
        """Log metrics for a specific training phase.

        Args:
            cycle: Current training cycle number.
            phase: Phase name (wake, dream, nightmare, compress).
            metrics: Phase result metrics dict.
        """
        prefixed = {f"{phase}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
        prefixed["cycle"] = cycle

        self.lineage["phases"].append(
            {
                "cycle": cycle,
                "phase": phase,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics,
            }
        )

        self.log_metrics(prefixed)

    def log_config(self, config: dict) -> None:
        """Log hyperparameters / full config.

        Args:
            config: Configuration dictionary.
        """
        self.lineage["config"] = config
        if self.backend == "wandb":
            import wandb  # type: ignore[import-untyped]

            wandb.config.update(config, allow_val_change=True)

        elif self.backend == "tensorboard" and self._writer is not None:
            # Flatten config for tensorboard hparams
            flat = {}
            for section, values in config.items():
                if isinstance(values, dict):
                    for k, v in values.items():
                        if isinstance(v, (str, int, float, bool)):
                            flat[f"{section}/{k}"] = v
                elif isinstance(values, (str, int, float, bool)):
                    flat[section] = values
            self._writer.add_hparams(flat, {})

        elif self.backend == "mlflow":
            try:
                import mlflow

                flat = {}
                for section, values in config.items():
                    if isinstance(values, dict):
                        for k, v in values.items():
                            if isinstance(v, (str, int, float, bool)):
                                flat[f"{section}/{k}"] = v
                    elif isinstance(values, (str, int, float, bool)):
                        flat[section] = values
                mlflow.log_params(flat)
            except Exception as exc:
                logger.warning("Failed to log config to MLflow: %s", exc)

    def log_artifact(self, path: str) -> None:
        """Log an artifact file or directory.

        Args:
            path: Local path to the file or directory.
        """
        if self.backend == "mlflow":
            try:
                import os

                import mlflow

                if os.path.isdir(path):
                    # Preserve the directory name to prevent root-level overwrites
                    artifact_path = os.path.basename(os.path.normpath(path)) or None
                    mlflow.log_artifacts(path, artifact_path=artifact_path)
                elif os.path.exists(path):
                    mlflow.log_artifact(path)
                else:
                    logger.warning("Artifact path does not exist: %s", path)
            except Exception as exc:
                logger.warning("Failed to log artifact to MLflow: %s", exc)

    def get_lineage(self) -> dict:
        """Return stored training lineage."""
        return self.lineage

    def finish(self) -> None:
        """Finalize and close the tracking session."""
        if self.backend == "wandb" and self._run is not None:
            import wandb  # type: ignore[import-untyped]

            wandb.finish()
            logger.info("Wandb run finished.")

        elif self.backend == "tensorboard" and self._writer is not None:
            self._writer.close()
            logger.info("TensorBoard writer closed.")

        elif self.backend == "mlflow":
            try:
                import mlflow

                mlflow.end_run()
                logger.info("MLflow run finished.")
            except Exception as exc:
                logger.warning("Failed to finish MLflow run: %s", exc)


def create_tracker_from_config(config: dict) -> ExperimentTracker:
    """Create an ExperimentTracker from a NightmareNet config dict.

    Args:
        config: Full configuration dictionary with optional 'tracking' section.

    Returns:
        Configured ExperimentTracker instance.
    """
    tracking_config = config.get("tracking", {})
    return ExperimentTracker(
        backend=tracking_config.get("backend", "none"),
        project=tracking_config.get("project", "nightmarenet"),
        run_name=tracking_config.get("run_name"),
        config=config,
        log_dir=tracking_config.get("log_dir", "logs/runs"),
    )
