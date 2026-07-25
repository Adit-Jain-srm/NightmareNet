#!/usr/bin/env python3
"""Run SST-2 benchmark: NightmareNet 4-phase vs wake-only baseline.

Writes results to results/benchmark-v1.json.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --quick   # distortion-only smoke (no training)
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _robustness_smoke() -> dict:
    """Fast robustness proxy without full training."""
    from nightmarenet.distortions.registry import get_registry

    registry = get_registry()
    text = "This movie was absolutely wonderful and inspiring."
    strengths = [0.1, 0.3, 0.5, 0.7, 0.9]
    scores = []
    for s in strengths:
        d = registry.apply("dream", text, strength=s, seed=42)
        n = registry.apply("nightmare", text, strength=s, seed=42)
        # Character retention as simple robustness proxy
        dream_ret = len(set(text) & set(d)) / len(set(text) | set(d) or {" "})
        nightmare_ret = len(set(text) & set(n)) / len(set(text) | set(n) or {" "})
        scores.append(
            {
                "strength": s,
                "dream_retention": round(dream_ret, 4),
                "nightmare_retention": round(nightmare_ret, 4),
            }
        )
    return {"mode": "distortion_smoke", "scores": scores}


def _run_training(config_path: Path, label: str) -> dict:
    """Run full training pipeline from YAML config."""
    from nightmarenet.pipeline import Pipeline
    from nightmarenet.utils.config import load_config

    config = load_config(str(config_path))
    pipeline = Pipeline(config=config)
    pipeline.run()
    metrics = pipeline.metrics.to_dict()
    return {
        "label": label,
        "config": str(config_path),
        "status": metrics.get("status"),
        "history_len": len(metrics.get("history", [])),
        "comparison": pipeline.metrics.comparison,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="NightmareNet SST-2 benchmark runner")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Distortion smoke test only (seconds, no GPU training)",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "results" / "benchmark-v1.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "mode": "quick" if args.quick else "full",
    }

    if args.quick:
        result["smoke"] = _robustness_smoke()
    else:
        baseline_cfg = REPO_ROOT / "configs" / "benchmark_sst2_baseline.yaml"
        full_cfg = REPO_ROOT / "configs" / "benchmark_sst2.yaml"
        result["baseline"] = _run_training(baseline_cfg, "wake_only")
        result["nightmarenet"] = _run_training(full_cfg, "four_phase")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Benchmark results written to {output_path}")

    # Trigger deploy webhook if notifications.webhooks are configured
    try:
        from nightmarenet.utils.config import load_config
        from nightmarenet.utils.webhooks import trigger_webhook

        config = load_config(str(REPO_ROOT / "configs" / "default.yaml"))
        trigger_webhook(
            config,
            "deploy",
            "SST-2 benchmark run finished.",
            {
                "mode": result.get("mode", "unknown"),
                "timestamp": result.get("timestamp"),
                "output_path": str(output_path),
            },
        )
    except Exception as e:
        print(f"Warning: Failed to trigger webhook notification: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
