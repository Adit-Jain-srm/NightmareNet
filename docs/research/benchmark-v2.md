# NightmareNet Benchmark v2 — Full Wake–Dream–Nightmare–Compress Pipeline

**Date:** 2026-07-17  
**Hardware:** NVIDIA Tesla T4 (16 GB VRAM, Kaggle)  
**Software:** Python 3.12, PyTorch, Transformers  
**Reproducibility:** `python scripts/train.py --config configs/benchmark_sst2_full_cycle.yaml`

---

# TL;DR

Benchmark v2 validates the **complete NightmareNet training pipeline** described in the paper by executing the canonical **Wake → Dream → Nightmare → Compression** workflow across **3 full training cycles**.

Unlike Benchmark v1, which used a lightweight benchmark script implementing only **Wake → Nightmare**, this benchmark exercises the project's production training pipeline (`scripts/train.py`) using a dedicated benchmark configuration.

The benchmark completed successfully on a Tesla T4 GPU without runtime failures, validating that the complete multi-cycle pipeline is reproducible on commodity GPU hardware.

---

# Motivation

Benchmark v1 demonstrated the effectiveness of the Wake → Nightmare adversarial training path but did not benchmark the complete sleep-inspired learning cycle proposed in the project architecture.

This benchmark addresses that gap by validating the full four-phase pipeline:

```mermaid
flowchart LR
    Wake --> Dream --> Nightmare --> Compression
```

executed for **3 consecutive cycles**.

---

# Experimental Setup

| Parameter | Value |
|-----------|-------|
| Model | `distilbert-base-uncased` |
| Dataset | GLUE SST-2 |
| Training samples | 500 |
| Batch size | 8 |
| Wake epochs | 2 |
| Dream epochs | 1 |
| Nightmare epochs | 1 |
| Compression rounds | 1 |
| Number of cycles | 3 |
| Seed | 42 |
| Device | CUDA (Tesla T4, 16 GB VRAM) |

---

# Methodology

The benchmark uses the canonical NightmareNet training pipeline:

```bash
python scripts/train.py --config configs/benchmark_sst2_full_cycle.yaml
```

Each training cycle executes:

1. Wake
2. Dream
3. Nightmare
4. Compression

The configured benchmark performs **3 complete cycles**, exercising every stage of the production training workflow.

Unlike Benchmark v1, this benchmark does **not** use the lightweight `scripts/run_gpu_benchmark.py` helper. Instead, it validates the same Trainer-based pipeline intended for normal NightmareNet training.

---

# Engineering Notes

Benchmark v2 required additional engineering work beyond creating a benchmark configuration.

During implementation, several inconsistencies between the benchmark requirements and the existing training pipeline were identified and resolved:

- Added a dedicated benchmark configuration (`configs/benchmark_sst2_full_cycle.yaml`) matching the benchmark requirements (DistilBERT, SST-2, 500 training samples, seed 42, CUDA).
- Extended the training pipeline to support sequence-classification models, enabling end-to-end benchmarking with DistilBERT on SST-2.
- Resolved configuration inconsistencies between benchmark settings and the Trainer pipeline to ensure the benchmark configuration was correctly interpreted.
- Updated dataset loading and tokenization to support sequence-classification datasets within the canonical training workflow.
- Verified CUDA execution and successful completion of all configured training phases across three complete cycles on Tesla T4 hardware.

These changes ensure that Benchmark v2 validates the production NightmareNet training pipeline rather than a standalone benchmark implementation, improving reproducibility and alignment with the project's architecture.

---

# Results

The benchmark successfully completed the full configured training workflow.

## Pipeline Validation

| Phase | Status |
|--------|--------|
| Wake | ✅ |
| Dream | ✅ |
| Nightmare | ✅ |
| Compression | ✅ |

Configured cycles completed: **3 / 3**

Execution completed successfully on NVIDIA Tesla T4 without runtime failures.

## Benchmark Outputs

Successful execution produced:

- Training history
- Intermediate checkpoints
- Final trained model
- GPU benchmark artifacts
- Training logs

These outputs confirm successful execution of the complete four-phase training pipeline using the project's canonical Trainer implementation.

## Quantitative Results

The benchmark was validated on an NVIDIA Tesla T4 GPU using 500 training samples and 200 evaluation samples.

| Metric | Baseline | NightmareNet |
|--------|---------:|-------------:|
| Clean Accuracy | 73.00% | 80.50% |
| Average Distorted Accuracy | 57.25% | 64.25% |
| Robustness Improvement (Δ) | — | **+7.00 percentage points (+12.23%)** |

Total benchmark wall time: **57.3 seconds**.

> **Per-phase timing is not reported.** The canonical Trainer (`scripts/train.py`)
> emits only aggregate wall time, not a per-phase (Wake/Dream/Nightmare/Compression)
> breakdown. Adding per-phase instrumentation is tracked as future work rather than
> fabricated here, so only the measured total is stated.



# Comparison with Benchmark v1

| Aspect | Benchmark v1 | Benchmark v2 |
|--------|--------------|--------------|
| Objective | Validate adversarial robustness of the Wake → Nightmare pipeline | Validate execution of the complete NightmareNet training pipeline |
| Training workflow | Wake → Nightmare | Wake → Dream → Nightmare → Compression |
| Training cycles | 1 | 3 |
| Execution path | Lightweight benchmark script (`scripts/run_gpu_benchmark.py`) | Canonical Trainer pipeline (`scripts/train.py`) |
| Configuration | CLI parameters | Dedicated YAML configuration (`benchmark_sst2_full_cycle.yaml`) |
| Dream phase | ✗ Not included | ✓ Included |
| Compression phase | ✗ Not included | ✓ Included |
| Multi-cycle training | ✗ | ✓ |
| Production pipeline validation | Partial | Complete |
| Primary outcome | Adversarial robustness benchmark | End-to-end validation of the full sleep-cycle training workflow |
| Robustness Improvement | Baseline established in v1 | +7.00 pp (+12.23%) under benchmark configuration |


Benchmark v1 established the effectiveness of the lightweight Wake → Nightmare adversarial training strategy. Benchmark v2 extends this work by validating the complete production training workflow described in the NightmareNet architecture, demonstrating that the full Wake → Dream → Nightmare → Compression pipeline executes successfully across three consecutive training cycles using the canonical Trainer implementation.

---

# Reproducibility

Run:

```bash
python scripts/train.py \
    --config configs/benchmark_sst2_full_cycle.yaml
```

Hardware used for this benchmark:

- GPU: NVIDIA Tesla T4 (16 GB)
- Batch size: 8
- Seed: 42

The benchmark completed successfully without modification on Kaggle GPU infrastructure.

---

# Limitations

This benchmark validates successful execution of the complete multi-cycle training pipeline and reports aggregate (end-to-end) robustness improvement.

**Scope note — robustness accumulation across cycles is out of scope for this benchmark.** The original objective included validating the "robustness accumulates across cycles" claim, but per-cycle robustness measurements are not currently recorded by the training pipeline. Rather than infer or fabricate a per-cycle progression, this benchmark reports only the measured end-to-end delta (+7.00 pp). Quantifying accumulation requires evaluating intermediate checkpoints after each cycle.

Future work: instrument the Trainer to evaluate robustness on the held-out set after each cycle (and per phase), producing both a per-cycle accumulation curve and a per-phase timing breakdown. This is deferred to a follow-up PR to keep this change focused on validating full-pipeline execution.

---

# Conclusion

Benchmark v2 successfully validates NightmareNet's complete Wake–Dream–Nightmare–Compress training pipeline using the project's canonical Trainer implementation.

Beyond introducing a dedicated full-cycle benchmark configuration, this work resolved several configuration and pipeline inconsistencies required to execute sequence-classification benchmarking through the production training workflow.

Compared with Benchmark v1, which focused on a lightweight Wake → Nightmare benchmark, Benchmark v2 demonstrates that the complete multi-cycle training pipeline executes successfully and reproducibly on commodity GPU hardware.