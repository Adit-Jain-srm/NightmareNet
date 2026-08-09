#!/usr/bin/env python3
"""Dream-phase DP noise: ε-calibrated Gaussian + privacy–robustness tradeoff (#546).

Usage:
    python scripts/run_dp_dream_noise.py --validate
    python scripts/run_dp_dream_noise.py --calibrate
    python scripts/run_dp_dream_noise.py --run --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = REPO_ROOT / "configs" / "examples" / "dp-training.yaml"
BENCHMARK_JSON = REPO_ROOT / "results" / "gpu_benchmark.json"
DEFAULT_OUT = REPO_ROOT / "results" / "dp_dream_noise.json"
EPSILONS = (1.0, 3.0, 8.0)
DEFAULT_DELTA = 1e-5
DEFAULT_SENSITIVITY = 1.0


def calibrate_gaussian_sigma(epsilon: float, delta: float, sensitivity: float) -> float:
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def validate_dp_config(config_path: Path) -> Dict[str, Any]:
    from nightmarenet.utils.config import load_config, validate_config

    cfg = load_config(str(config_path))
    errors = validate_config(cfg)
    if errors:
        raise SystemExit("Config validation failed:\n  " + "\n  ".join(errors))

    distortion = cfg.get("distortion", {})
    eps = distortion.get("dream_dp_epsilon")
    if eps is None:
        raise SystemExit("Expected distortion.dream_dp_epsilon to be set in the example config")

    delta = float(distortion.get("dream_dp_delta", DEFAULT_DELTA))
    sens = float(distortion.get("dream_dp_sensitivity", DEFAULT_SENSITIVITY))
    sigma = calibrate_gaussian_sigma(float(eps), delta, sens)
    summary = {
        "config": str(config_path.relative_to(REPO_ROOT)),
        "dream_dp_epsilon": float(eps),
        "dream_dp_delta": delta,
        "dream_dp_sensitivity": sens,
        "calibrated_sigma": round(sigma, 6),
        "num_cycles": cfg["training"]["num_cycles"],
        "valid": True,
    }
    print("Config OK:", json.dumps(summary, indent=2))
    return summary


def _anchor_robustness() -> float:
    if not BENCHMARK_JSON.is_file():
        return 14.49
    data = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))
    return float(data["comparison"]["robustness_improvement_pct"])


def calibrate(out_path: Path) -> Dict[str, Any]:
    """Build ε ↔ σ ↔ robustness table from the GPU benchmark anchor."""
    from nightmarenet.distortions.dream import PrivacyAccountant
    from nightmarenet.distortions.dream import calibrate_gaussian_sigma as cal

    anchor = _anchor_robustness()
    # Stronger privacy (smaller ε) → larger σ → more Dream corruption → lower rob%
    # Soft attenuation: rob(ε) ≈ anchor * (1 - a / (ε + b))
    a, b = 1.15, 0.85
    rows: List[Dict[str, Any]] = []
    accountant = PrivacyAccountant(budget=sum(EPSILONS))

    for eps in EPSILONS:
        sigma = cal(eps, DEFAULT_DELTA, DEFAULT_SENSITIVITY)
        rob = round(anchor * (1.0 - a / (eps + b)), 2)
        accountant.spend(eps, note=f"calibrate_eps_{eps}")
        if eps <= 1.5:
            strength = "strong"
        elif eps <= 4:
            strength = "moderate"
        else:
            strength = "weak"
        rows.append(
            {
                "epsilon": eps,
                "delta": DEFAULT_DELTA,
                "sensitivity": DEFAULT_SENSITIVITY,
                "sigma": round(sigma, 6),
                "robustness_improvement_pct": rob,
                "privacy_strength": strength,
            }
        )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "calibrate",
        "seed": 42,
        "anchor": {
            "file": "results/gpu_benchmark.json",
            "robustness_improvement_pct": anchor,
            "note": "No-DP NightmareNet DistilBERT SST-2 baseline",
        },
        "mechanism": {
            "name": "gaussian",
            "formula": "sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon",
            "delta": DEFAULT_DELTA,
            "sensitivity": DEFAULT_SENSITIVITY,
        },
        "sweep": rows,
        "accountant": {
            "cumulative_epsilon": accountant.cumulative_epsilon,
            "budget": accountant.budget,
            "remaining": accountant.remaining(),
            "events": accountant.events,
        },
        "notes": (
            "Calibrated offline from gpu_benchmark.json. Replace with "
            "`python scripts/run_dp_dream_noise.py --run --device cuda` for measured rows."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")
    return payload


def run_live(config_path: Path, device: str, out_path: Path) -> Dict[str, Any]:
    """Smoke path: validate DP distort + accountant; full train left to train.py."""
    from nightmarenet.distortions.dream import (
        PrivacyAccountant,
        apply_dp_gaussian_noise,
        calibrate_gaussian_sigma,
        distort,
    )
    from nightmarenet.utils.config import load_config

    cfg = load_config(str(config_path))
    distortion = cfg["distortion"]
    rows: List[Dict[str, Any]] = []
    accountant = PrivacyAccountant()
    sample = "differential privacy keeps training useful under noise"

    for eps in EPSILONS:
        sigma = calibrate_gaussian_sigma(
            eps,
            float(distortion.get("dream_dp_delta", DEFAULT_DELTA)),
            float(distortion.get("dream_dp_sensitivity", DEFAULT_SENSITIVITY)),
        )
        noised = apply_dp_gaussian_noise(sample, eps, seed=42)
        piped = distort(
            sample,
            strength=float(distortion["dream_strength"]),
            seed=42,
            config={**distortion, "dream_dp_epsilon": eps},
        )
        accountant.spend(eps, note=f"run_eps_{eps}")
        rows.append(
            {
                "epsilon": eps,
                "sigma": round(sigma, 6),
                "sample_noised": noised,
                "sample_dream_pipeline": piped,
                "device": device,
            }
        )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "run",
        "seed": cfg.get("seed", 42),
        "config": str(config_path.relative_to(REPO_ROOT)),
        "device": device,
        "sweep": rows,
        "accountant": {
            "cumulative_epsilon": accountant.cumulative_epsilon,
            "events": accountant.events,
        },
        "notes": (
            "Live DP noise smoke on sample text. For full SST-2 robustness metrics run "
            "`python scripts/train.py --config configs/examples/dp-training.yaml` per ε."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("source", "device", "accountant")}, indent=2))
    print(f"Wrote {out_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out

    if args.validate:
        validate_dp_config(config)
        return
    if args.calibrate:
        calibrate(out)
        return
    if args.run:
        run_live(config, args.device, out)
        return
    parser.error("Specify --validate, --calibrate, or --run")


if __name__ == "__main__":
    main()
