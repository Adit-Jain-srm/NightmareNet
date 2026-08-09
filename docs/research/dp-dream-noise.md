# Differential privacy noise in the Dream phase (ε-calibrated)

**Date:** 2026-08-05  
**Issue:** [#546](https://github.com/Adit-Jain-srm/NightmareNet/issues/546)  
**Config:** [`configs/examples/dp-training.yaml`](../../configs/examples/dp-training.yaml)  
**Script:** [`scripts/run_dp_dream_noise.py`](../../scripts/run_dp_dream_noise.py)  
**Results:** [`results/dp_dream_noise.json`](../../results/dp_dream_noise.json)

---

## TL;DR

NightmareNet’s Dream phase normally applies mild text/semantic corruption with **no privacy guarantee**. This work adds an optional **Gaussian mechanism** whose noise scale σ is calibrated to a target privacy budget ε (`distortion.dream_dp_epsilon`). A `PrivacyAccountant` records cumulative ε across Dream cycles so multi-cycle runs stay auditable.

Smaller ε → larger σ → stronger privacy → typically lower robustness. Larger ε → milder noise → weaker privacy → robustness closer to the no-DP baseline.

> Sweep rows below are **calibrated** from `results/gpu_benchmark.json` (no GPU here). Replace with measured trainings:
>
> ```bash
> python scripts/run_dp_dream_noise.py --run --device cuda
> python scripts/train.py --config configs/examples/dp-training.yaml
> ```

---

## Mechanism

| Symbol | Meaning |
|--------|---------|
| ε | Privacy budget per Dream-phase spend (`dream_dp_epsilon`) |
| δ | Failure probability (`dream_dp_delta`, default `1e-5`) |
| Δ | Assumed L2 sensitivity (`dream_dp_sensitivity`, default `1.0`) |
| σ | Gaussian noise std: `Δ · √(2 ln(1.25/δ)) / ε` |

Text path: after standard Dream text/semantic distortions, each non-space codepoint is treated as a scalar query and perturbed with N(0, σ²), then rounded back. This is a **research approximation** for Dream-phase DP noise (not a claim of Opacus-style DP-SGD end-to-end).

Config:

```yaml
distortion:
  dream_dp_epsilon: 3.0   # null disables DP
  dream_dp_delta: 1.0e-5
  dream_dp_sensitivity: 1.0
```

The trainer charges the accountant once per Dream phase and writes `privacy_epsilon_spent` / `privacy_epsilon_cumulative` into the phase history.

---

## ε ↔ σ ↔ robustness (calibrated)

Anchor: no-DP NightmareNet DistilBERT SST-2 robustness improvement **+14.49%** (`gpu_benchmark.json`).

| ε | σ (Δ=1, δ=1e-5) | Privacy | Robustness improvement (calibrated) |
|--:|----------------:|---------|------------------------------------:|
| 1.0 | ≈ 4.845 | strong | 5.48% |
| 3.0 | ≈ 1.615 | moderate | 10.16% |
| 8.0 | ≈ 0.606 | weak | 12.61% |
| ∞ / null | 0 | none | 14.49% (anchor) |

**Takeaway:** Privacy and Dream-time robustness trade off smoothly under this calibration. ε=8 stays close to the no-DP band; ε=1 buys strong privacy at a clear robustness cost.

---

## Reproduce

```bash
# Schema + example config
python scripts/run_dp_dream_noise.py --validate

# Offline privacy–robustness table
python scripts/run_dp_dream_noise.py --calibrate

# DP noise smoke on sample text
python scripts/run_dp_dream_noise.py --run --device cpu

# Full cyclic training with DP Dream noise (set dream_dp_epsilon per sweep)
python scripts/train.py --config configs/examples/dp-training.yaml
```

### Seeds and hardware

| Item | Value |
|------|-------|
| Seed | **42** |
| Anchor | `results/gpu_benchmark.json` (CUDA DistilBERT SST-2) |
| This PR | Calibrated sweep; measure after `--run` / full `train.py` |

---

## Files

| Path | Role |
|------|------|
| `nightmarenet/distortions/dream.py` | σ calibration, Gaussian text noise, `PrivacyAccountant` |
| `configs/examples/dp-training.yaml` | Example SST-2 config with ε=3.0 |
| `scripts/run_dp_dream_noise.py` | validate / calibrate / run |
| `results/dp_dream_noise.json` | Machine-readable sweep |
| `docs/research/dp-dream-noise.md` | This report |
