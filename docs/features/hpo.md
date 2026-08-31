# Hyperparameter Optimization (HPO) Guide

NightmareNet integrates [Optuna](https://optuna.org/) to search for the configuration that maximizes robustness. You describe a search space in YAML, and the optimizer runs repeated pipeline trials, pruning unpromising ones early and persisting results to a study database.

## Overview

HPO lives in `nightmarenet/optimization/hpo.py`. The `HyperparameterOptimizer` reads an `hpo` section from a standard training config, samples parameters for each trial, runs the full pipeline, and maximizes the `robustness_delta` returned by the run.

| Concept | Description |
|---|---|
| Study | An Optuna study persisted to a storage backend (SQLite by default). |
| Trial | One pipeline run with a sampled set of hyperparameters. |
| Objective | Maximizes `robustness_delta` (falls back to `robustness`, `avg_robustness`, or `mean_robustness`). |
| Pruning | Optuna's `MedianPruner` stops trials that under-perform mid-run. |

> [!NOTE]
> HPO requires the optional Optuna dependency. Install it with `pip install 'nightmarenet[hpo]'`.

---

## Quick Start

```bash
pip install 'nightmarenet[hpo]'
nightmarenet optimize --config configs/examples/hpo-config.yaml
```

Override the number of trials from the command line:

```bash
nightmarenet optimize --config configs/examples/hpo-config.yaml --n-trials 50
```

---

## Configuration

HPO reuses your training config and adds an `hpo` section. A complete example ships at [`configs/examples/hpo-config.yaml`](../../configs/examples/hpo-config.yaml):

```yaml
hpo:
  study_name: nightmarenet-robustness-search
  storage: sqlite:///nightmarenet_hpo.db
  direction: maximize
  n_trials: 20
  pruning: true
  search_space:
    training.learning_rate:
      type: float
      low: 1.0e-5
      high: 1.0e-3
      log: true
    distortion.nightmare_strength:
      type: float
      low: 0.5
      high: 0.95
    training.batch_size:
      type: categorical
      choices: [4, 8, 16]
```

### `hpo` keys

| Key | Default | Description |
|---|---|---|
| `study_name` | `nightmarenet-optimization` | Optuna study name. |
| `storage` | `sqlite:///nightmarenet_hpo.db` | Optuna storage URL (studies resume via `load_if_exists`). |
| `direction` | `maximize` | Optimization direction. |
| `n_trials` | `20` | Number of trials (overridable with `--n-trials`). |
| `pruning` | `true` | Enable `MedianPruner`; when `false`, uses `NopPruner`. |
| `search_space` | `{}` | Map of dotted config keys to parameter definitions. |

### Search-space parameter types

Each entry in `search_space` is keyed by a **dotted path** into the base config (e.g. `training.learning_rate`) and supports three types:

| `type` | Required fields | Optional fields |
|---|---|---|
| `float` | `low`, `high` | `log` (log-uniform sampling) |
| `int` | `low`, `high` | `log` |
| `categorical` | `choices` | — |

Sampled values are written back into a deep copy of the base config using the dotted path, so any config key the pipeline reads can be tuned.

---

## API Reference

### `HyperparameterOptimizer`

```python
from nightmarenet.optimization import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(config_path="configs/examples/hpo-config.yaml")
study = optimizer.optimize()   # returns an optuna.Study
```

| Member | Description |
|---|---|
| `HyperparameterOptimizer(config_path)` | Loads the config, builds the Optuna study, and raises `ImportError` if Optuna is not installed. |
| `optimize()` | Runs `n_trials` trials and returns the `optuna.Study`. |
| `study` | The underlying Optuna study (created with `load_if_exists=True`). |

Each trial builds a fresh pipeline via `nightmarenet.pipeline.Pipeline`, and — when pruning is enabled — reports intermediate `phase_loss` per cycle so Optuna can prune weak trials early.

---

## Examples

### Inspect the best trial

```python
from nightmarenet.optimization import HyperparameterOptimizer

optimizer = HyperparameterOptimizer("configs/examples/hpo-config.yaml")
study = optimizer.optimize()

print("Best value:", study.best_value)
print("Best params:", study.best_params)
```

### Resume an existing study

Because the study is created with `load_if_exists=True`, re-running the same command against the same `storage` URL continues the existing study rather than starting over:

```bash
nightmarenet optimize --config configs/examples/hpo-config.yaml --n-trials 20
# ... later, add more trials to the same study ...
nightmarenet optimize --config configs/examples/hpo-config.yaml --n-trials 30
```

---

## Related Documentation

- [Model Compression](model-compression.md) — tune `compression.pruning_ratio` as part of a search.
- [Getting Started](../tutorials/getting-started.md) — install and run your first cycle.
- Config reference: [`configs/examples/hpo-config.yaml`](../../configs/examples/hpo-config.yaml)
