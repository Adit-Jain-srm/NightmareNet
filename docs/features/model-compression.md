# Model Compression Guide

The Compress phase of the NightmareNet cycle shrinks a hardened model while preserving its robustness. It combines magnitude-based pruning (or low-rank bottlenecks) with optional RSLAD-style adversarial distillation, so the compressed student inherits the robust features of its teacher rather than trading them away.

## Overview

Compression utilities live in `nightmarenet/compression/`:

| Component | Source | Responsibility |
|---|---|---|
| `MagnitudePruner` | `pruning.py` | Zeros out the smallest-magnitude weights. |
| `BottleneckWrapper` / `apply_bottleneck_to_model` | `pruning.py` | Inserts low-rank bottleneck projections. |
| `run_distillation` / `fgsm_perturb` | `distillation.py` | RSLAD-style adversarial robust distillation. |

The Compress phase (`nightmarenet/training/phases.py`) reads a `compression` config section, prunes (or bottlenecks) the model, snapshots the pre-pruning model as a teacher, and optionally distills the pruned student on adversarial inputs.

---

## Quick Start

Compression runs automatically as the final phase of `nightmarenet train`. Control it through the `compression` section of your YAML config:

```yaml
compression:
  pruning_ratio: 0.2          # prune the smallest 20% of weights
  pruning_method: magnitude   # "magnitude" or "bottleneck"
  distillation: true          # RSLAD-style robust distillation after pruning
  finetune_after_prune: true
  finetune_epochs: 1
```

```bash
nightmarenet train --config configs/benchmark_sst2.yaml
```

You can also apply compression directly to any PyTorch model:

```python
from nightmarenet.compression import MagnitudePruner

pruner = MagnitudePruner(pruning_ratio=0.2)
stats = pruner.apply(model)
print(stats)   # {'pruned_params': ..., 'total_params': ..., 'sparsity': ...}
```

---

## Configuration

The compress phase honours the following keys under `compression`:

| Key | Default | Description |
|---|---|---|
| `pruning_ratio` | `0.2` | Fraction of weights to zero out, in `[0, 1)`. |
| `pruning_method` | `magnitude` | `magnitude` (weight zeroing) or `bottleneck` (low-rank projection). |
| `bottleneck_rank_ratio` | `0.5` | Bottleneck dim as a ratio of the hidden dim (used when `pruning_method: bottleneck`). |
| `distillation` | `false` | Enable RSLAD-style distillation after pruning. |
| `finetune_after_prune` | `true` | Fine-tune the pruned model to recover accuracy. |
| `finetune_epochs` | `1` | Number of fine-tuning epochs. |

---

## API Reference

### `MagnitudePruner`

```python
MagnitudePruner(pruning_ratio: float = 0.2)
```

`pruning_ratio` must be in `[0, 1)`; a value outside that range raises `ValueError`.

| Method | Returns | Description |
|---|---|---|
| `apply(model)` | `dict` | Prunes weights in-place; returns `{pruned_params, total_params, sparsity}`. Parameters with fewer than 2 dimensions are skipped. |

### `BottleneckWrapper`

```python
BottleneckWrapper(original_layer, bottleneck_dim: Optional[int] = None, rank_ratio: float = 0.5)
```

Wraps a layer with a down-project → up-project pair. When `bottleneck_dim` is omitted it is derived from `rank_ratio`. `rank_ratio` must be in `(0, 1]`.

### `apply_bottleneck_to_model`

```python
apply_bottleneck_to_model(model, rank_ratio: float = 0.5, target_modules: Optional[list] = None) -> dict
```

Recursively wraps modules whose names match `target_modules` (default `["mlp", "attn"]`). Returns `{wrapped_count, rank_ratio}`.

### `run_distillation`

```python
run_distillation(
    teacher, student, dataloader, optimizer, device,
    epochs: int = 1, temperature: float = 4.0, alpha: float = 0.7,
    epsilon: float = 0.01, scaler=None,
) -> dict
```

Runs RSLAD-style distillation: the frozen `teacher` supervises the pruned `student` on FGSM-perturbed inputs. The loss blends KL divergence (temperature-scaled) and the task loss via `alpha`. Returns `{distillation_loss}`.

### `fgsm_perturb`

```python
fgsm_perturb(model, batch: dict, epsilon: float = 0.01) -> dict
```

Generates FGSM adversarial examples. For text batches it perturbs input embeddings and returns `inputs_embeds`; for vision batches (`pixel_values` present) it perturbs and clamps pixels to `[0, 1]`.

---

## Examples

### Prune, then distill

```python
import copy
import torch
from nightmarenet.compression import MagnitudePruner
from nightmarenet.compression.distillation import run_distillation

# Snapshot the teacher before pruning
teacher = copy.deepcopy(model).eval()
for p in teacher.parameters():
    p.requires_grad = False

# Prune the student
MagnitudePruner(pruning_ratio=0.2).apply(model)

# Distill robustness back into the pruned student
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
result = run_distillation(
    teacher=teacher, student=model, dataloader=dataloader,
    optimizer=optimizer, device=torch.device("cuda"),
    epochs=1, temperature=4.0, alpha=0.7, epsilon=0.01,
)
print(result["distillation_loss"])
```

### Low-rank bottleneck instead of pruning

```yaml
compression:
  pruning_method: bottleneck
  bottleneck_rank_ratio: 0.5
```

```python
from nightmarenet.compression import apply_bottleneck_to_model

stats = apply_bottleneck_to_model(model, rank_ratio=0.5, target_modules=["mlp", "attn"])
print(stats)   # {'wrapped_count': ..., 'rank_ratio': 0.5}
```

---

## Related Documentation

- [Distributed Training](distributed-training.md) — the compress phase always runs on a single GPU.
- [Getting Started](../tutorials/getting-started.md) — install and run your first cycle.
- Config reference: [`configs/benchmark_sst2.yaml`](../../configs/benchmark_sst2.yaml)
