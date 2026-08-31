# Configuration Guide

NightmareNet is driven by YAML configuration files. This directory holds the
base config, ready-to-run benchmark configs, and a set of annotated
[`examples/`](examples/). This guide explains the schema, the
inheritance/override model, and the validation rules.

## How configs are loaded

Configs are loaded by `load_config()` in
[`../nightmarenet/utils/config.py`](../nightmarenet/utils/config.py):

```python
from nightmarenet.utils.config import load_config

config = load_config("configs/examples/custom-distortion-chain.yaml")
```

`load_config()`:

1. Reads your YAML file into a dict.
2. **Deep-merges it over the built-in `DEFAULT_CONFIG`** (see below).
3. Validates the merged result and raises `ValueError` on any schema violation.

## Inheritance / override model

There is no `extends:` key. Instead, every config **inherits from the built-in
defaults** (`DEFAULT_CONFIG` in `utils/config.py`, mirrored by
[`default.yaml`](default.yaml)). Your file only needs to specify the keys you
want to change.

- The merge is **recursive (deep)**: setting `distortion.dream_strength` leaves
  the rest of the `distortion` block at its default values.
- Nested dictionaries are merged key-by-key; scalars and lists are replaced
  wholesale.
- Because of the merge, a minimal config is valid — any key you omit is filled
  from the defaults.

```yaml
# A complete, valid config — everything else comes from the defaults.
model:
  name: distilbert-base-uncased
  type: seq_classification
dataset:
  name: glue
  text_column: sentence
```

> **Unknown top-level keys** (anything not listed in [Top-level keys](#top-level-keys))
> produce a **warning** with a "did you mean …?" suggestion, but do **not** fail
> loading. Unknown *nested* keys (e.g. `dataset.config`, `compression.distillation`)
> are merged through and left for the relevant subsystem to consume.

## Top-level keys

These are the top-level sections defined in [`default.yaml`](default.yaml):

| Key | Purpose |
|-----|---------|
| `model` | Backbone model, task type, sequence length, device. |
| `dataset` | Dataset name/subset, text and label columns, sampling. |
| `training` | Cycle counts, epochs, batch size, optimizer, checkpointing, distributed. |
| `distortion` | Dream/Nightmare strengths, schedule, and per-type engine weights. |
| `compression` | Pruning and (optional) distillation settings. |
| `evaluation` | Which metrics to run and the robustness strength sweep. |
| `tracking` | Experiment tracking backend and compliance reporting. |
| `notifications` | Webhook notifications (see [webhooks guide](../docs/features/webhooks.md)). |
| `hpo` | Optuna hyperparameter search (consumed by `nightmarenet optimize`; see [`examples/hpo-config.yaml`](examples/hpo-config.yaml)). |
| `seed` | Global random seed. |
| `compliance` | Compliance signing-key path. |
| `observability` | Structured logging and OpenTelemetry export. |
| `adaption` | Optional Adaption Labs dataset optimization. |

### `model`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `name` | str (required) | `gpt2` | HF model id or local path. |
| `type` | str (required) | `causal_lm` | `causal_lm` \| `masked_lm` \| `seq_classification` \| `image_classification`. |
| `num_labels` | int | `2` | Used for `seq_classification`. |
| `max_length` | int | `128` | Required unless `type: image_classification`. |
| `device` | str (required) | `auto` | `auto` \| `cuda` \| `cpu`. |

### `dataset`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `name` | str (required) | `wikitext` | HF dataset name. |
| `subset` | str | `wikitext-2-raw-v1` | Dataset subset/config. |
| `text_column` | str | `text` | Required unless `type: image_classification`. |
| `max_samples` | int \| null | `null` | Cap samples for fast iteration; `null` = full. |
| `streaming` | bool | `false` | Stream instead of loading into memory. |

### `training`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `wake_epochs` | int (required) | `3` | Wake-phase epochs. |
| `dream_epochs` | int (required) | `2` | Dream-phase epochs. |
| `nightmare_epochs` | int (required) | `1` | Nightmare-phase epochs. |
| `num_cycles` | int (required, ≥ 1) | `3` | Number of full cycles. |
| `compression_rounds` | int (required) | `1` | Compress-phase rounds. |
| `batch_size` | int (required, ≥ 1) | `8` | Batch size. |
| `learning_rate` | float (required) | `5e-5` | Range `1e-10`–`1.0`. |
| `nightmare_lr_multiplier` | float (required) | `2.0` | Range `0.1`–`100`. |
| `weight_decay` | float | `0.01` | Optimizer weight decay. |
| `warmup_steps` | int | `100` | LR warmup steps. |
| `gradient_accumulation_steps` | int (required, ≥ 1) | `4` | Gradient accumulation. |
| `max_grad_norm` | float (required) | `1.0` | Gradient clipping norm. |
| `use_amp` | bool | `false` | Mixed-precision training (CUDA). |
| `gradient_checkpointing` | bool | `false` | Trade compute for memory. |
| `early_stopping` | bool | `false` | Stop when loss plateaus. |
| `distributed` | bool | `false` | Enable native DDP (see [`distributed-training.yaml`](examples/distributed-training.yaml)). |
| `save_every_phase` | bool | `true` | Checkpoint after each phase. |
| `checkpoint_dir` | str | `checkpoints` | Checkpoint directory. |
| `log_dir` | str | `logs` | Log directory. |

### `distortion`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `dream_strength` | float (required) | `0.25` | Range `0.0`–`1.0`. |
| `nightmare_strength` | float (required) | `0.8` | Range `0.0`–`1.0`. |
| `dream_dp_epsilon` | float \| null | `null` | `null` = off; range `1e-6`–`1e6` when set. |
| `dream_dp_delta` | float | `1e-5` | DP delta. |
| `dream_dp_sensitivity` | float | `1.0` | DP sensitivity. |
| `language` | str | `english` | `keyboard_typo` language. |
| `keyboard_layout` | str \| null | `null` | Override layout; `null` = language default. |
| `strength_schedule` | str | `uniform` | `uniform` \| `linear` \| `cosine` \| `step`. |
| `schedule_across_cycles` | bool | `false` | Per-cycle strength scheduling. |
| `strength_min` | float | `0.2` | Range `0.0`–`1.0`. |
| `strength_max` | float | `0.9` | Range `0.0`–`1.0`. |
| `text` | map | see `default.yaml` | Per-engine weights: `char_swap`, `char_insert`, `char_delete`, `keyboard_typo`, `word_shuffle`, `token_mask`. |
| `semantic` | map | see `default.yaml` | `synonym_replace`, `negation_inject`, `topic_splice`. |
| `adversarial` | map | see `default.yaml` | `contradiction`, `ambiguity`, `cross_domain`, `misleading_context`, `learned`, `learned_model`. |

### `compression`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `pruning_ratio` | float (required) | `0.2` | Range `0.0`–`0.9999`. |
| `pruning_method` | str | `magnitude` | `magnitude` \| `bottleneck`. |
| `bottleneck_rank_ratio` | float (required) | `0.5` | Range `0.01`–`1.0`. |
| `finetune_after_prune` | bool | `true` | Fine-tune after pruning. |
| `finetune_epochs` | int | `1` | Fine-tuning epochs. |

The optional `distillation` keys (`distillation`, `distillation_temperature`,
`distillation_alpha`, `distillation_epochs`) are documented inline in
[`default.yaml`](default.yaml).

### `evaluation`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `metrics` | list[str] | `[recall, generalization, robustness, hallucination]` | Enabled metrics. |
| `robustness_strengths` | list[float] | `[0.1 … 0.9]` | Strength sweep for the robustness AUC. |
| `output_dir` | str | `results` | Where results are written. |
| `output_format` | str | `json` | `json` \| `markdown`. |

### `tracking`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `backend` | str | `none` | `none` \| `wandb` \| `tensorboard`. |
| `project` | str | `nightmarenet` | Project name. |
| `log_dir` | str | `logs/runs` | Run log directory. |

### `notifications`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `webhooks` | list[map] | `[]` | Each entry needs `url` (str) and optional `events`. |

`events` values must be one of `run_complete`, `regression_detected`, `alert`,
`deploy`. See the [webhooks guide](../docs/features/webhooks.md).

### `hpo`

An optional map consumed by `nightmarenet optimize`. See
[`examples/hpo-config.yaml`](examples/hpo-config.yaml).

### `seed`

Global integer random seed (default `42`).

### `compliance`, `observability`, `adaption`

Optional sections documented inline in [`default.yaml`](default.yaml). They are
not part of the validated schema (unknown-key warnings are expected if used at
the top level), but are merged through for the relevant subsystems.

## Validation rules

Validation happens in `validate_config()`
([`../nightmarenet/utils/config.py`](../nightmarenet/utils/config.py)) after the
merge:

- **Required keys** must be present after merging with defaults. The `required`
  fields above (`model.name`, `model.type`, `model.device`, the core `training.*`
  fields, `distortion.dream_strength`/`nightmare_strength`,
  `compression.pruning_ratio`/`bottleneck_rank_ratio`, `seed`) are supplied by
  the defaults, so you only hit these errors if you override them with `null`.
- **`model.max_length` and `dataset.text_column`** are required unless
  `model.type: image_classification`.
- **Types** are checked per field (an `int` is accepted where a `float` is
  expected).
- **Numeric ranges** are enforced for the fields marked with a range above.
- **Webhooks** must be a list of mappings with a string `url` and, if present,
  an `events` list of the allowed event strings.

Any violation raises `ValueError` with a bulleted list of every problem.

## Validate your config

```bash
python -c "from nightmarenet.utils.config import load_config; load_config('configs/examples/custom-distortion-chain.yaml')"
```

No output (beyond an informational log line) means the config is valid.

## Examples

Annotated, ready-to-run configs live in [`examples/`](examples/):

| File | Demonstrates |
|------|--------------|
| [`custom-distortion-chain.yaml`](examples/custom-distortion-chain.yaml) | Chaining/weighting multiple distortion engines and scheduling nightmare strength. |
| [`multi-dataset-training.yaml`](examples/multi-dataset-training.yaml) | Hardening a model across several datasets (one run per dataset). |
| [`distributed-training.yaml`](examples/distributed-training.yaml) | Native DDP / multi-GPU setup. |
| [`transfer-learning.yaml`](examples/transfer-learning.yaml) | Hardening a foundation backbone for robustness transfer. |
| [`convergence-study.yaml`](examples/convergence-study.yaml) | Adaptive cycle termination / convergence. |
| [`cross-arch-eval.yaml`](examples/cross-arch-eval.yaml) | Cross-architecture evaluation. |
| [`dp-training.yaml`](examples/dp-training.yaml) | Differential-privacy Dream noise. |
| [`gpt2-robustness.yaml`](examples/gpt2-robustness.yaml) | Causal-LM (GPT-2) robustness cycle. |
| [`hpo-config.yaml`](examples/hpo-config.yaml) | Optuna hyperparameter search. |
| [`mlflow-tracking.yaml`](examples/mlflow-tracking.yaml) | Experiment tracking. |

## Related Documentation

- [Getting Started](../docs/tutorials/getting-started.md) — install and run your first cycle.
- [`default.yaml`](default.yaml) — the fully-commented base config.
- [Webhook notifications guide](../docs/features/webhooks.md) — configuring the `notifications` section.
