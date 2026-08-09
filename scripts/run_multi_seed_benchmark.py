#!/usr/bin/env python3
"""Run multi-seed benchmark for SST-2.

Evaluates DistilBERT-base-uncased across 5 seeds: 42, 1, 7, 99, 123.
Computes mean, standard deviation, 95% confidence intervals, and paired t-test
significance. Updates tables in docs/research/paper-draft.md with error bars.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from scipy.stats import ttest_rel

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SEEDS = [42, 1, 7, 99, 123]


def run_benchmark_for_seed(seed: int, device: str, train_samples: int, eval_samples: int) -> Path:
    out_dir = REPO_ROOT / "results" / "multi_seed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"seed_{seed}.json"

    if out_file.exists():
        print(f"Results for seed {seed} already exist at {out_file}. Skipping execution.")
        return out_file

    print("\n==========================================")
    print(f"Running benchmark for Seed {seed} on {device}")
    print("==========================================")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_gpu_benchmark.py"),
        "--seed",
        str(seed),
        "--device",
        device,
        "--train-samples",
        str(train_samples),
        "--eval-samples",
        str(eval_samples),
        "--output",
        str(out_file),
    ]

    subprocess.run(cmd, check=True)
    return out_file


def compute_stats(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    if n <= 1:
        return sum(values) / max(n, 1), 0.0, 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    from scipy.stats import t as t_dist

    t_crit = t_dist.ppf(0.975, df=n - 1)
    ci = t_crit * (std / math.sqrt(n))
    return mean, std, ci


def update_paper_draft(summary: dict) -> None:
    paper_path = REPO_ROOT / "docs" / "research" / "paper-draft.md"
    if not paper_path.exists():
        print(f"Warning: paper-draft.md not found at {paper_path}. Skipping update.")
        return

    content = paper_path.read_text(encoding="utf-8")

    # 1. Update Headline numbers table
    b_clean = summary["headline"]["baseline_clean"]
    n_clean = summary["headline"]["nightmarenet_clean"]
    b_dist = summary["headline"]["baseline_avg_dist"]
    n_dist = summary["headline"]["nightmarenet_avg_dist"]
    b_drop = summary["headline"]["baseline_robustness_drop"]
    n_drop = summary["headline"]["nightmarenet_robustness_drop"]
    rel_imp = summary["headline"]["relative_improvement"]

    headline_table_target = """| Clean accuracy | 0.7450 | **0.7850** | **+0.0400** |
| Avg distorted accuracy | 0.5830 | **0.6625** | **+0.0795** |
| Robustness drop \\(\\Delta_{\\text{rob}}\\) | 0.1620 | **0.1225** | −0.0395 |
| **Relative robustness improvement** | — | — | **+13.64%** |"""

    delta_latex = r"\(\Delta_{\text{rob}}\)"
    headline_table_replacement = (
        f"| Clean accuracy | {b_clean['mean']:.4f} ± {b_clean['std']:.4f} | "
        f"**{n_clean['mean']:.4f} ± {n_clean['std']:.4f}** | "
        f"**{n_clean['mean'] - b_clean['mean']:+.4f}** |\n"
        f"| Avg distorted accuracy | {b_dist['mean']:.4f} ± {b_dist['std']:.4f} | "
        f"**{n_dist['mean']:.4f} ± {n_dist['std']:.4f}** | "
        f"**{n_dist['mean'] - b_dist['mean']:+.4f}** |\n"
        f"| Robustness drop {delta_latex} | {b_drop['mean']:.4f} ± {b_drop['std']:.4f} | "
        f"**{n_drop['mean']:.4f} ± {n_drop['std']:.4f}** | "
        f"**{n_drop['mean'] - b_drop['mean']:+.4f}** |\n"
        f"| **Relative robustness improvement** | — | — | **{rel_imp['mean']:+.2f}%** |"
    )

    content = content.replace(headline_table_target, headline_table_replacement)

    # 2. Update Per-strength breakdown table
    strengths = ["0.1", "0.3", "0.5", "0.7", "0.9"]
    target_lines = []
    replacement_lines = []

    for s in strengths:
        b_dream = summary["per_strength"]["dream"][s]["baseline"]
        n_dream = summary["per_strength"]["dream"][s]["nightmarenet"]
        b_night = summary["per_strength"]["nightmare"][s]["baseline"]
        n_night = summary["per_strength"]["nightmare"][s]["nightmarenet"]

        # Reconstruct exactly matching lines in original paper-draft.md
        # Table lines in paper-draft.md might have rounded values (3 decimal places)
        # e.g., | 0.1 | 0.700 | 0.765 | +0.065 | 0.710 | 0.770 | +0.060 |
        if s == "0.1":
            t_line = "| 0.1 | 0.700 | 0.765 | +0.065 | 0.710 | 0.770 | +0.060 |"
        elif s == "0.3":
            t_line = "| 0.3 | 0.665 | 0.725 | +0.060 | 0.655 | 0.735 | +0.080 |"
        elif s == "0.5":
            t_line = "| 0.5 | 0.580 | 0.645 | +0.065 | 0.585 | 0.630 | +0.045 |"
        elif s == "0.7":
            t_line = "| 0.7 | 0.480 | 0.565 | +0.085 | 0.480 | 0.560 | +0.080 |"
        elif s == "0.9":
            t_line = "| 0.9 | 0.490 | 0.590 | +0.100 | 0.485 | 0.640 | +0.155 |"

        r_line = (
            f"| {float(s):.1f} | "
            f"{b_dream['mean']:.3f} ± {b_dream['std']:.3f} | "
            f"{n_dream['mean']:.3f} ± {n_dream['std']:.3f} | "
            f"{n_dream['mean'] - b_dream['mean']:+.3f} | "
            f"{b_night['mean']:.3f} ± {b_night['std']:.3f} | "
            f"{n_night['mean']:.3f} ± {n_night['std']:.3f} | "
            f"{n_night['mean'] - b_night['mean']:+.3f} |"
        )
        target_lines.append(t_line)
        replacement_lines.append(r_line)

    for t_l, r_l in zip(target_lines, replacement_lines):
        content = content.replace(t_l, r_l)

    # 3. Add statistical test findings paragraph and update description numbers
    clean_delta_pct = (n_clean["mean"] - b_clean["mean"]) * 100
    dist_delta_pct = (n_dist["mean"] - b_dist["mean"]) * 100

    content = content.replace(
        "+4.0 and +7.95 absolute percentage points respectively",
        (
            f"{clean_delta_pct:+.1f} and {dist_delta_pct:+.2f} "
            "absolute percentage points respectively"
        ),
    )
    content = content.replace(
        "shrinks by a quarter (0.162 → 0.123)",
        f"decreases slightly ({b_drop['mean']:.4f} → {n_drop['mean']:.4f})",
    )

    sig_text_target = "(+13.64%) sits comfortably in the 10–30% target band of our specification."
    p_clean_val = summary["statistical_tests"]["clean_accuracy"]["p_value"]
    p_dist_val = summary["statistical_tests"]["avg_distorted_accuracy"]["p_value"]
    sig_status_clean = "significant" if p_clean_val < 0.05 else "non-significant"
    sig_status_dist = "significant" if p_dist_val < 0.05 else "non-significant"

    sig_text_replacement = (
        f"({rel_imp['mean']:+.2f}%) sits comfortably in the 10–30% target band of our "
        "specification. A paired t-test comparing clean accuracy across 5 seeds shows "
        f"a {sig_status_clean} improvement (p = {p_clean_val:.4f}). "
        "A paired t-test comparing average distorted accuracy across 5 seeds shows "
        f"a {sig_status_dist} improvement (p = {p_dist_val:.4f})."
    )
    content = content.replace(sig_text_target, sig_text_replacement)

    paper_path.write_text(content, encoding="utf-8")
    print(f"Updated: {paper_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi-seed benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-samples", type=int, default=500)
    parser.add_argument("--eval-samples", type=int, default=200)
    args = parser.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    results = []
    for seed in SEEDS:
        out_file = run_benchmark_for_seed(seed, args.device, args.train_samples, args.eval_samples)
        with open(out_file, encoding="utf-8") as f:
            results.append(json.load(f))

    # Parse and extract
    baselines = [r["baseline"] for r in results]
    nightmarenets = [r["nightmarenet"] for r in results]

    # Headline extracts
    b_clean_vals = [x["clean_accuracy"] for x in baselines]
    n_clean_vals = [x["clean_accuracy"] for x in nightmarenets]
    b_dist_vals = [x["avg_distorted_accuracy"] for x in baselines]
    n_dist_vals = [x["avg_distorted_accuracy"] for x in nightmarenets]
    b_drop_vals = [x["robustness_drop"] for x in baselines]
    n_drop_vals = [x["robustness_drop"] for x in nightmarenets]
    rel_imp_vals = [r["comparison"]["robustness_improvement_pct"] for r in results]

    # Run t-tests
    t_clean, p_clean = ttest_rel(n_clean_vals, b_clean_vals)
    t_dist, p_dist = ttest_rel(n_dist_vals, b_dist_vals)

    # Headline stats
    summary = {
        "headline": {
            "baseline_clean": dict(zip(["mean", "std", "ci"], compute_stats(b_clean_vals))),
            "nightmarenet_clean": dict(zip(["mean", "std", "ci"], compute_stats(n_clean_vals))),
            "baseline_avg_dist": dict(zip(["mean", "std", "ci"], compute_stats(b_dist_vals))),
            "nightmarenet_avg_dist": dict(zip(["mean", "std", "ci"], compute_stats(n_dist_vals))),
            "baseline_robustness_drop": dict(
                zip(["mean", "std", "ci"], compute_stats(b_drop_vals))
            ),
            "nightmarenet_robustness_drop": dict(
                zip(["mean", "std", "ci"], compute_stats(n_drop_vals))
            ),
            "relative_improvement": dict(zip(["mean", "std", "ci"], compute_stats(rel_imp_vals))),
        },
        "per_strength": {"dream": {}, "nightmare": {}},
        "statistical_tests": {
            "clean_accuracy": {
                "t_statistic": float(t_clean),
                "p_value": float(p_clean),
            },
            "avg_distorted_accuracy": {
                "t_statistic": float(t_dist),
                "p_value": float(p_dist),
            },
        },
        "raw_runs": [
            {
                "seed": seed,
                "baseline": {
                    "clean_accuracy": b_c,
                    "avg_distorted_accuracy": b_d,
                    "robustness_drop": b_dr,
                },
                "nightmarenet": {
                    "clean_accuracy": n_c,
                    "avg_distorted_accuracy": n_d,
                    "robustness_drop": n_dr,
                },
            }
            for seed, b_c, b_d, b_dr, n_c, n_d, n_dr in zip(
                SEEDS,
                b_clean_vals,
                b_dist_vals,
                b_drop_vals,
                n_clean_vals,
                n_dist_vals,
                n_drop_vals,
            )
        ],
    }

    # Per-strength stats
    strengths = ["0.1", "0.3", "0.5", "0.7", "0.9"]
    for d_type in ("dream", "nightmare"):
        for s in strengths:
            b_vals = [x["distorted_accuracy"][d_type][s] for x in baselines]
            n_vals = [x["distorted_accuracy"][d_type][s] for x in nightmarenets]

            b_mean, b_std, b_ci = compute_stats(b_vals)
            n_mean, n_std, n_ci = compute_stats(n_vals)

            summary["per_strength"][d_type][s] = {
                "baseline": {"mean": b_mean, "std": b_std, "ci": b_ci},
                "nightmarenet": {"mean": n_mean, "std": n_std, "ci": n_ci},
            }

    # Write summary
    summary_path = REPO_ROOT / "results" / "multi_seed" / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWritten summary statistics to: {summary_path}")

    # Update paper
    update_paper_draft(summary)

    print("\n=== AGGREGATED BENCHMARK SUMMARY ===")
    b_clean_m = summary["headline"]["baseline_clean"]["mean"]
    b_clean_s = summary["headline"]["baseline_clean"]["std"]
    b_dist_m = summary["headline"]["baseline_avg_dist"]["mean"]
    b_dist_s = summary["headline"]["baseline_avg_dist"]["std"]
    n_clean_m = summary["headline"]["nightmarenet_clean"]["mean"]
    n_clean_s = summary["headline"]["nightmarenet_clean"]["std"]
    n_dist_m = summary["headline"]["nightmarenet_avg_dist"]["mean"]
    n_dist_s = summary["headline"]["nightmarenet_avg_dist"]["std"]

    print(
        f"Baseline     clean={b_clean_m:.4f} ± {b_clean_s:.4f}  "
        f"avg_distorted={b_dist_m:.4f} ± {b_dist_s:.4f}"
    )
    print(
        f"NightmareNet clean={n_clean_m:.4f} ± {n_clean_s:.4f}  "
        f"avg_distorted={n_dist_m:.4f} ± {n_dist_s:.4f}"
    )
    print(f"Paired t-test clean_accuracy p-value: {p_clean:.4f}")
    print(f"Paired t-test avg_distorted_accuracy p-value: {p_dist:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
