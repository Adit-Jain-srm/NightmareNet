#!/usr/bin/env python3
"""Compute FLOP analysis for NightmareNet training cycles.

This module provides pure functions for calculating FLOP estimates
based on model parameters, sample count, epochs, and cycle count.

Usage:
    python scripts/compute_cost_analysis.py --config configs/benchmark_sst2_full_cycle.yaml
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ruff: noqa: E402 - Allow module-level import after sys.path manipulation
from nightmarenet.utils.config import load_config

# FLOP estimates per sample per epoch (approximate, model-dependent)
# These are reasonable estimates for common transformer models
BASELINE_WAKE_EPOCHS = 3

FLOPS_PER_SAMPLE_PER_EPOCH = {
    "distilbert-base-uncased": 2.5e12,  # ~2.5 trillion FLOPs
    "distilgpt2": 1.2e12,  # ~1.2 trillion FLOPs
    "bert-base-uncased": 4.5e12,  # ~4.5 trillion FLOPs
    "gpt2": 2.0e12,  # ~2.0 trillion FLOPs
    "gpt2-medium": 5.0e12,  # ~5.0 trillion FLOPs
    "gpt2-large": 10.0e12,  # ~10.0 trillion FLOPs
    "gpt2-xl": 20.0e12,  # ~20.0 trillion FLOPs
}


def get_flops_per_sample(model_name: str) -> float:
    """Get FLOPs per sample per epoch for a given model.

    Args:
        model_name: HuggingFace model identifier (e.g., 'distilbert-base-uncased')

    Returns:
        Estimated FLOPs per sample per epoch
    """
    # Try to find model in lookup table (case-insensitive match)
    model_lower = model_name.lower()

    # Check for exact model names first
    if model_lower in FLOPS_PER_SAMPLE_PER_EPOCH:
        return FLOPS_PER_SAMPLE_PER_EPOCH[model_lower]

    # Check if any lookup key is contained in the model name (prefer longest key)
    best_match = None
    best_key_length = 0
    for key, value in FLOPS_PER_SAMPLE_PER_EPOCH.items():
        if key in model_lower:
            if len(key) > best_key_length:
                best_key_length = len(key)
                best_match = value

    if best_match is not None:
        return best_match

    # Check if model name is contained in any lookup key (for "bert" matching "bert-base-uncased")
    for key, value in FLOPS_PER_SAMPLE_PER_EPOCH.items():
        if model_lower in key:
            return value

    # Default estimate: 2.5e12 FLOPs (similar to DistilBERT)
    return 2.5e12


def calculate_phase_flops(
    model_name: str,
    num_samples: int,
    num_epochs: int,
    flops_per_sample: float = None,
) -> float:
    """Calculate FLOPs for a single training phase.

    Args:
        model_name: HuggingFace model identifier
        num_samples: Number of training samples
        num_epochs: Number of epochs
        flops_per_sample: Optional override for FLOPs per sample

    Returns:
        Total FLOPs for the phase
    """
    if flops_per_sample is None:
        flops_per_sample = get_flops_per_sample(model_name)

    return flops_per_sample * num_samples * num_epochs


def calculate_cycle_flops(
    model_name: str,
    num_samples: int,
    wake_epochs: int,
    dream_epochs: int,
    nightmare_epochs: int,
    compression_rounds: int,
    flops_per_sample: float = None,
) -> dict:
    """Calculate FLOPs for one complete training cycle.

    Args:
        model_name: HuggingFace model identifier
        num_samples: Number of training samples
        wake_epochs: Wake phase epochs
        dream_epochs: Dream phase epochs
        nightmare_epochs: Nightmare phase epochs
        compression_rounds: Compression phase rounds
        flops_per_sample: Optional override for FLOPs per sample

    Returns:
        Dict with FLOPs for each phase and total
    """
    if flops_per_sample is None:
        flops_per_sample = get_flops_per_sample(model_name)

    wake_flops = calculate_phase_flops(model_name, num_samples, wake_epochs, flops_per_sample)
    dream_flops = calculate_phase_flops(model_name, num_samples, dream_epochs, flops_per_sample)
    nightmare_flops = calculate_phase_flops(
        model_name, num_samples, nightmare_epochs, flops_per_sample
    )
    compress_flops = calculate_phase_flops(
        model_name, num_samples, compression_rounds, flops_per_sample
    )

    total = wake_flops + dream_flops + nightmare_flops + compress_flops

    return {
        "phases": {
            "wake": wake_flops,
            "dream": dream_flops,
            "nightmare": nightmare_flops,
            "compress": compress_flops,
        },
        "total": total,
    }


def calculate_total_flops(
    model_name: str,
    num_samples: int,
    num_cycles: int,
    wake_epochs: int,
    dream_epochs: int,
    nightmare_epochs: int,
    compression_rounds: int,
    flops_per_sample: float = None,
) -> dict:
    """Calculate total FLOPs for the entire training schedule.

    Args:
        model_name: HuggingFace model identifier
        num_samples: Number of training samples
        num_cycles: Total number of training cycles
        wake_epochs: Wake phase epochs per cycle
        dream_epochs: Dream phase epochs per cycle
        nightmare_epochs: Nightmare phase epochs per cycle
        compression_rounds: Compression rounds per cycle
        flops_per_sample: Optional override for FLOPs per sample

    Returns:
        Dict with per-cycle FLOPs, total FLOPs, and metadata
    """
    if flops_per_sample is None:
        flops_per_sample = get_flops_per_sample(model_name)

    # Calculate per-cycle FLOPs
    cycle_flops = calculate_cycle_flops(
        model_name,
        num_samples,
        wake_epochs,
        dream_epochs,
        nightmare_epochs,
        compression_rounds,
        flops_per_sample,
    )

    # Calculate total across all cycles
    total_flops = cycle_flops["total"] * num_cycles

    # Calculate baseline (standard fine-tuning equivalent)
    # Standard FT: 1 cycle, BASELINE_WAKE_EPOCHS wake epochs, 0 dream, 0 nightmare, 0 compress
    baseline_cycle_flops = calculate_cycle_flops(
        model_name,
        num_samples,
        wake_epochs=BASELINE_WAKE_EPOCHS,  # Typical fine-tuning epochs
        dream_epochs=0,
        nightmare_epochs=0,
        compression_rounds=0,
        flops_per_sample=flops_per_sample,
    )
    baseline_total = baseline_cycle_flops["total"] * num_cycles

    return {
        "metadata": {
            "model": model_name,
            "num_samples": num_samples,
            "num_cycles": num_cycles,
            "flops_per_sample": flops_per_sample,
        },
        "schedule": {
            "wake_epochs": wake_epochs,
            "dream_epochs": dream_epochs,
            "nightmare_epochs": nightmare_epochs,
            "compression_rounds": compression_rounds,
        },
        "per_cycle": cycle_flops,
        "total": {
            "nightmarenet": total_flops,
            "baseline": baseline_total,
        },
        "comparison": {
            "nightmarenet_vs_baseline": total_flops / max(baseline_total, 1e-9),
            "cycle_total": cycle_flops["total"],
            "phase_sum": sum(cycle_flops["phases"].values()),
            "phases_match_cycle_total": (
                abs(cycle_flops["total"] - sum(cycle_flops["phases"].values())) < 1e-6
            ),
        },
    }


def format_flops(flops: float) -> str:
    """Format FLOPs value for human readability.

    Args:
        flops: Number of FLOPs

    Returns:
        Formatted string (e.g., "2.5 TFLOPs")
    """
    if flops >= 1e15:
        return f"{flops / 1e15:.2f} PFLOPs"
    elif flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    else:
        return f"{flops:.2f} FLOPs"


def print_analysis(analysis: dict) -> None:
    """Print FLOP analysis results in a readable format.

    Args:
        analysis: Result dict from calculate_total_flops
    """
    meta = analysis["metadata"]
    schedule = analysis["schedule"]
    per_cycle = analysis["per_cycle"]
    total = analysis["total"]
    comparison = analysis["comparison"]

    print("=" * 70)
    print("NIGHTMARENET FLOP ANALYSIS")
    print("=" * 70)
    print()
    print("MODEL CONFIGURATION")
    print(f"  Model: {meta['model']}")
    print(f"  Training Samples: {meta['num_samples']:,}")
    print(f"  FLOPs per Sample/Epoch: {format_flops(meta['flops_per_sample'])}")
    print()
    print("TRAINING SCHEDULE")
    print(f"  Cycles: {meta['num_cycles']}")
    print(f"  Wake epochs/cycle: {schedule['wake_epochs']}")
    print(f"  Dream epochs/cycle: {schedule['dream_epochs']}")
    print(f"  Nightmare epochs/cycle: {schedule['nightmare_epochs']}")
    print(f"  Compression rounds/cycle: {schedule['compression_rounds']}")
    print()
    print("PER-CYCLE FLOPS BREAKDOWN")
    print(f"  Wake:      {format_flops(per_cycle['phases']['wake'])}")
    print(f"  Dream:     {format_flops(per_cycle['phases']['dream'])}")
    print(f"  Nightmare: {format_flops(per_cycle['phases']['nightmare'])}")
    print(f"  Compress:  {format_flops(per_cycle['phases']['compress'])}")
    print(f"  Cycle Total: {format_flops(per_cycle['total'])}")
    print()
    print("TOTAL FLOPS")
    print(f"  NightmareNet (total): {format_flops(total['nightmarenet'])}")
    print(f"  Baseline ({BASELINE_WAKE_EPOCHS} epoch FT): {format_flops(total['baseline'])}")
    print()
    print("COMPARISON")
    print(f"  NightmareNet vs Baseline: {comparison['nightmarenet_vs_baseline']:.2f}x")
    print(f"  Cycle Total == Sum(Phases): {comparison['phases_match_cycle_total']}")
    print(f"  Phase Sum: {format_flops(comparison['phase_sum'])}")
    print()
    print("NOTE: FLOP comparisons are valid for equal epoch counts.")
    print("      The sample count used is: " + str(meta["num_samples"]))
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute FLOP analysis for NightmareNet training cycles"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "default.yaml"),
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name from config",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override sample count from config",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress printed output (useful for programmatic use)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Get values from config or override
    model_name = args.model or config.get("model", {}).get("name", "distilbert-base-uncased")
    num_samples = (
        config.get("dataset", {}).get("max_samples", 500) if args.samples is None else args.samples
    )
    num_cycles = config.get("training", {}).get("num_cycles", 3)
    wake_epochs = config.get("training", {}).get("wake_epochs", 3)
    dream_epochs = config.get("training", {}).get("dream_epochs", 2)
    nightmare_epochs = config.get("training", {}).get("nightmare_epochs", 1)
    compression_rounds = config.get("training", {}).get("compression_rounds", 1)

    # Calculate FLOPs
    analysis = calculate_total_flops(
        model_name=model_name,
        num_samples=num_samples,
        num_cycles=num_cycles,
        wake_epochs=wake_epochs,
        dream_epochs=dream_epochs,
        nightmare_epochs=nightmare_epochs,
        compression_rounds=compression_rounds,
    )

    # Print results
    if not args.quiet:
        print_analysis(analysis)

    # Write to file if specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nAnalysis written to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
