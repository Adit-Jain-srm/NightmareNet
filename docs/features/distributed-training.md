# Distributed Training Guide

NightmareNet can run the Wake → Dream → Nightmare → Compress cycle across multiple GPUs. Each phase picks the parallelism strategy best suited to its workload, and every phase boundary is checkpointed atomically so an interrupted run can resume without corruption.

## Overview

The distributed layer lives in `nightmarenet/distributed/` and is composed of four cooperating pieces:

| Component | Source | Responsibility |
|---|---|---|
| `DevicePool` | `device_pool.py` | Discovers available GPUs and estimates memory requirements. |
| `DDPWrapper` | `ddp_wrapper.py` | Manages the PyTorch `DistributedDataParallel` process group. |
| `apply_phase_strategy` | `strategies.py` | Selects the right parallelism strategy per phase. |
| `AtomicCheckpointer` / `ResumeManager` | `checkpoint.py`, `resume.py` | Atomic checkpoint saves and validated resume. |

### Per-phase strategy

`apply_phase_strategy(phase, model, device_pool, ddp_wrapper, distributed_enabled)` maps each phase to a strategy:

| Phase | Strategy | Why |
|---|---|---|
| `wake` | `DistributedDataParallel` (DDP) | Data-parallel supervised training. |
| `nightmare` | `DistributedDataParallel` (DDP) | Data-parallel adversarial training. |
| `dream` | `torch.nn.DataParallel` | Embarrassingly parallel augmentation/inference. |
| `compress` | Single GPU | Pruning and distillation run on one device. |

If `distributed_enabled` is `False` or only one device is available, the model is returned unwrapped.

---

## Quick Start

Enable multi-GPU execution by passing `--distributed` to `nightmarenet train`. Pass `auto` to use every visible GPU, or a comma-separated device list to pin specific GPUs:

```bash
# Use all available GPUs
nightmarenet train --config configs/benchmark_sst2.yaml --distributed auto

# Pin specific GPU IDs
nightmarenet train --config configs/benchmark_sst2.yaml --distributed 0,1,2
```

DDP requires launching under `torchrun`, which sets the `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` environment variables. `DDPWrapper.setup()` detects these variables and initializes the process group; if they are absent, DDP is skipped and training falls back to a single device.

```bash
torchrun --nproc_per_node=2 -m nightmarenet.cli train \
    --config configs/benchmark_sst2.yaml --distributed auto
```

---

## Configuration

### Device pool

`DevicePool` discovers devices automatically or accepts an explicit override list:

```python
from nightmarenet.distributed import DevicePool

# Auto-discover all CUDA devices
pool = DevicePool()
print(pool.get_num_devices())        # e.g. 2
print(pool.should_use_ddp())         # True when more than one device is present

# Explicit device IDs
pool = DevicePool(override_devices=[0, 1])

# Estimate VRAM (GB) for a model of N parameters
# (4 bytes/param x 3 for model+grads+optimizer x 1.2 buffer)
print(pool.estimate_memory_requirements(num_params=66_000_000))
```

### DDP backend

`DDPWrapper` defaults to the NCCL backend, which is recommended for NVIDIA GPUs:

```python
from nightmarenet.distributed import DDPWrapper

ddp = DDPWrapper(backend="nccl")
ddp.setup()                # no-op unless launched via torchrun
if ddp.is_initialized:
    model = ddp.wrap_model(model)
# ... train ...
ddp.teardown()             # destroys the process group
```

---

## API Reference

### `DevicePool`

```python
DevicePool(override_devices: Optional[list[int]] = None)
```

| Method | Returns | Description |
|---|---|---|
| `get_num_devices()` | `int` | Number of usable devices. |
| `estimate_memory_requirements(num_params)` | `float` | Estimated VRAM in GB. |
| `should_use_ddp()` | `bool` | `True` when more than one device is available. |

### `DDPWrapper`

```python
DDPWrapper(backend: str = "nccl")
```

| Method | Description |
|---|---|
| `setup()` | Initializes the process group when launched via `torchrun`. |
| `wrap_model(model)` | Wraps a model in `DistributedDataParallel`. |
| `teardown()` | Destroys the process group. |
| `is_initialized` | Attribute — `True` once `setup()` succeeds. |

### Strategy helpers (`strategies.py`)

```python
apply_phase_strategy(phase, model, device_pool, ddp_wrapper, distributed_enabled) -> nn.Module
unwrap_model(model) -> nn.Module   # recursively unwraps DDP / DataParallel
```

### Checkpointing (`checkpoint.py`, `resume.py`)

```python
from nightmarenet.distributed import AtomicCheckpointer, ResumeManager

checkpointer = AtomicCheckpointer(base_dir="./checkpoints")
target = checkpointer.save(
    run_id="sst2-v1", cycle=1, phase="dream",
    model=model, optimizer=optimizer, config=config,
    metrics={"loss": 0.42}, devices_used=[0, 1],
)

resume = ResumeManager(resume_dir=target)
metadata = resume.verify_and_load(model, optimizer, current_config=config)
```

`AtomicCheckpointer.save()` writes to a temporary directory, records SHA-256 hashes of every file, drops a `.complete` sentinel, and then atomically swaps the directory into place. `ResumeManager.verify_and_load()` calls `validate_checkpoint_integrity()` to confirm the sentinel, metadata, version compatibility, and checksums before loading model, optimizer, and RNG state.

Module-level helpers available in `checkpoint.py`:

| Function | Description |
|---|---|
| `load_model_weights(model, checkpoint_dir, device)` | Loads `model.pt`, `model.safetensors`, or `pytorch_model.bin`. |
| `validate_checkpoint_integrity(checkpoint_dir, config=None)` | Structural, version, and checksum validation. |
| `compute_config_hash(config)` | Deterministic SHA-256 of the config. |
| `compute_file_sha256(filepath)` | SHA-256 of a single file. |
| `compute_dir_hashes(directory)` | SHA-256 of every file in a directory. |
| `check_version_compatibility(checkpoint_version, current_version)` | Raises on major (or 0.x minor) mismatch. |

---

## Examples

### Resume an interrupted run

Checkpoints are saved at each phase boundary. To resume, point `--resume` at the checkpoint directory:

```bash
nightmarenet train \
    --config configs/benchmark_sst2.yaml \
    --distributed auto \
    --resume ./checkpoints/sst2-v1/cycle-1-dream
```

### Choosing a device count from memory

```python
from nightmarenet.distributed import DevicePool

pool = DevicePool()
required_gb = pool.estimate_memory_requirements(num_params=124_000_000)
print(f"Estimated {required_gb:.1f} GB VRAM required per replica")
if not pool.should_use_ddp():
    print("Single device detected — DDP will be skipped.")
```

---

## Related Documentation

- [Model Compression](model-compression.md) — the single-GPU compress phase.
- [Getting Started](../tutorials/getting-started.md) — install and run your first cycle.
- Config reference: [`configs/benchmark_sst2.yaml`](../../configs/benchmark_sst2.yaml)
