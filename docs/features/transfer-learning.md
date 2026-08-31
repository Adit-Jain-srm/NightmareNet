# Transfer Learning Guide

Robustness transfer learning lets you reuse the hardened representations from a completed NightmareNet cycle instead of re-running the full Wake → Dream → Nightmare → Compress loop for every new task. You register a hardened model as a *foundation backbone*, attach a fresh task head, and fine-tune — optionally freezing the bottom layers to preserve the robustness the foundation already learned.

## Overview

The transfer module lives in `nightmarenet/transfer/`:

| Component | Source | Responsibility |
|---|---|---|
| `TransferConfig` / `load_config` | `config.py` | Dataclass config and YAML loader. |
| `create_transfer_model` | `head_factory.py` | Attaches a task head to a foundation backbone. |
| `FoundationRegistry` / `get_registry` | `registry.py` | Stores and loads foundation backbones. |
| `TransferFineTuner` | `fine_tune.py` | Fine-tuning loop with layer freezing. |
| `calculate_transfer_ratio` / `evaluate_transfer_efficiency` | `measurement.py` | Transfer efficiency metrics. |
| `generate_transfer_report` | `report.py` | Markdown efficiency report. |

---

## Quick Start

```bash
# 1. Register a hardened model as a foundation backbone
nightmarenet foundation register --model ./output/sst2-hardened --name sst2-robust

# 2. List registered foundations
nightmarenet foundation list

# 3. Fine-tune the foundation on a downstream task
nightmarenet transfer --foundation sst2-robust --config configs/transfer_ag_news.yaml
```

Registering a model extracts **only** the base backbone (no task head) via `AutoModel`, so the same robust backbone can seed many downstream tasks.

---

## Configuration

Transfer runs are configured with a YAML file loaded into `TransferConfig`. See [`configs/transfer_ag_news.yaml`](../../configs/transfer_ag_news.yaml) for a complete example:

```yaml
task_type: "seq_classification"   # or "token_classification"
dataset: "ag_news"
num_labels: 4
batch_size: 16
num_epochs: 3
freeze_bottom_n: 4                # freeze the bottom 4 backbone layers
unfreeze_after_epoch: 2           # then unfreeze everything after epoch 2
learning_rate: 0.00003
output_dir: "./output/transfer_ag_news"
device: "cuda"
strict_layer_freezing: true       # error if freezable layers cannot be found
```

| Field | Default | Description |
|---|---|---|
| `task_type` | `seq_classification` | `seq_classification` or `token_classification`. |
| `dataset` | `sst2` | Downstream dataset name. |
| `num_labels` | `2` | Number of output labels. |
| `batch_size` | `8` | Training batch size. |
| `num_epochs` | `3` | Fine-tuning epochs. |
| `freeze_bottom_n` | `0` | Number of bottom backbone layers to freeze. |
| `unfreeze_after_epoch` | `1` | Epoch (1-indexed) after which all layers unfreeze. |
| `learning_rate` | `3e-5` | Optimizer learning rate. |
| `output_dir` | `./output/transfer` | Where fine-tuned artifacts are written. |
| `device` | `cuda` | Training device. |
| `strict_layer_freezing` | `false` | Raise `RuntimeError` if freezable layers are not found. |

Foundation models are stored under `~/.nightmarenet/foundation` by default; override the location with the `NIGHTMARENET_FOUNDATION_DIR` environment variable.

Layer freezing supports BERT-like backbones (`encoder.layer`) and GPT-like backbones (`transformer.h`).

---

## API Reference

### Foundation registry (`registry.py`)

```python
from nightmarenet.transfer import FoundationRegistry, get_registry

registry = get_registry()                      # singleton
registry.register(model_path="./output/hardened", name="sst2-robust",
                  metadata={"robustness_score": 0.68})
backbone, tokenizer, metadata = registry.load("sst2-robust")
names = registry.list_models()
```

### Model creation (`head_factory.py`)

```python
create_transfer_model(foundation_path, task_type="seq_classification", **kwargs) -> PreTrainedModel
```

Instantiates `AutoModelForSequenceClassification` or `AutoModelForTokenClassification` on top of the foundation backbone. Extra keyword arguments (e.g. `num_labels`) are forwarded to `from_pretrained`.

### Fine-tuning (`fine_tune.py`)

```python
TransferFineTuner(model, optimizer, device, scaler=None)

fine_tuner.run(
    dataloader,
    num_epochs,
    freeze_bottom_n=0,
    unfreeze_after_epoch=1,
    strict_layer_freezing=False,
) -> dict   # {avg_loss, final_epoch_loss, per_epoch_losses, layers_frozen}
```

### Measurement (`measurement.py`)

```python
calculate_transfer_ratio(transferred_robustness, baseline_robustness) -> float
evaluate_transfer_efficiency(transfer_ratio) -> str
```

`evaluate_transfer_efficiency` classifies the ratio as *Highly Efficient* (> 0.7), *Moderately Efficient* (> 0.3), or *Weak*.

### Reporting (`report.py`)

```python
generate_transfer_report(
    transferred_robustness, baseline_robustness,
    clean_accuracy_transferred, clean_accuracy_baseline,
    transferred_time_s, baseline_time_s,
) -> str   # markdown report
```

---

## Examples

### End-to-end fine-tuning in Python

```python
import torch
from torch.utils.data import DataLoader
from nightmarenet.transfer import get_registry, create_transfer_model, TransferFineTuner
from nightmarenet.transfer.config import load_config

config = load_config("configs/transfer_ag_news.yaml")
registry = get_registry()
foundation_path = registry.cache_dir / "sst2-robust"

model = create_transfer_model(
    str(foundation_path),
    task_type=config.task_type,
    num_labels=config.num_labels,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

fine_tuner = TransferFineTuner(model, optimizer, torch.device(config.device))
result = fine_tuner.run(
    dataloader,
    num_epochs=config.num_epochs,
    freeze_bottom_n=config.freeze_bottom_n,
    unfreeze_after_epoch=config.unfreeze_after_epoch,
    strict_layer_freezing=config.strict_layer_freezing,
)
print(result["per_epoch_losses"])
```

### Measure transfer efficiency

```bash
nightmarenet transfer --measure \
    --transferred ./output/transfer_ag_news/eval.json \
    --baseline ./output/full_cycle/eval.json
```

Both JSON files must contain `robustness_score` and `clean_accuracy` keys. The command prints a markdown report summarizing the transfer ratio, efficiency classification, and compute savings.

---

## Related Documentation

- [Distributed Training](distributed-training.md) — scale the original cycle across GPUs.
- [Getting Started](../tutorials/getting-started.md) — install and run your first cycle.
- Config reference: [`configs/transfer_ag_news.yaml`](../../configs/transfer_ag_news.yaml)
