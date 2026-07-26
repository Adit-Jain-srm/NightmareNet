"""Certified-robustness experiment: baseline vs. trained model (issue #162).

Certifies the same fixed sample subset under both a baseline (wake-only) model
and a NightmareNet-trained (wake + nightmare) model via randomized smoothing,
and reports whether the certified radius increases after adversarial training
-- the core hypothesis of issue #153's certified-robustness sub-issue 3.

Also runs a correlation analysis between each model's own per-sample *empirical*
robustness (1 / perplexity under a fixed `nightmare` distortion) and its
per-sample *certified* radius, to check whether the two robustness notions --
one observational, one a formal guarantee -- agree on which samples are hard.

Usage (real models -- requires network access to the HF Hub):

    python scripts/run_certification_experiment.py \\
        --baseline-model distilbert-base-uncased \\
        --trained-model ./checkpoints/nightmarenet-sst2 \\
        --dataset glue --dataset-config sst2 --split validation \\
        --text-column sentence --subset-size 50 \\
        --sigma 0.1 --n 1000 --n0 100 \\
        --output-dir results/certification

Usage (offline smoke test -- no network, no GPU, deterministic synthetic
model + dataset; validates the certification -> reporting -> export pipeline
end-to-end without claiming any real robustness result):

    python scripts/run_certification_experiment.py --smoke

Outputs (written to --output-dir, default results/certification/):
    certification_experiment.json   Raw results: config, aggregates, per-sample
                                     radii for both models, correlation stats.
    certification_experiment_report.md
                                     Markdown report via Evaluator.generate_report()
                                     -- the exact section that ships in real runs.
    baseline_certification.csv / trained_certification.csv
                                     Per-sample CSV export via format_results.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nightmarenet.evaluation.certification import certify_dataset  # noqa: E402
from nightmarenet.evaluation.evaluator import Evaluator  # noqa: E402
from nightmarenet.evaluation.format_results import (  # noqa: E402
    certification_to_json_dict,
    to_csv_certification,
)
from nightmarenet.evaluation.metrics import compute_perplexity  # noqa: E402

logger = logging.getLogger("run_certification_experiment")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-mode assets: deterministic, offline, no network/GPU required.
#
# NOT a substitute for the real SST-2 / DistilBERT experiment -- this exists so
# the certification -> reporting -> export pipeline can be validated end-to-end
# in environments without HF Hub access (see docs/research/paper-draft.md §5.5
# for why the full-scale numbers are marked "pending compute").
# ─────────────────────────────────────────────────────────────────────────────


class _SmokeConfig:
    num_labels = 2


class _SmokeClassifier(nn.Module):
    """Tiny deterministic (given a seed) classifier standing in for DistilBERT.

    `robustness` in [0, 1] controls how much the logits' margin shrinks under
    the noise the certifier and the perplexity proxy both inject -- a stand-in
    for "how adversarially hardened the model is", used only to give the
    smoke run a directionally sensible (baseline < trained) certified radius,
    not to claim a real measurement.
    """

    def __init__(self, vocab_size: int = 64, hidden_dim: int = 16, robustness: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.config = _SmokeConfig()
        self._robustness = robustness
        generator = torch.Generator().manual_seed(1234)
        with torch.no_grad():
            # Deterministic init: alternating +/- weights scaled by hidden_dim,
            # so the mean-pooled embedding sum is a stable, informative logit
            # margin rather than noise from random init.
            weights = (
                torch.randn(vocab_size, hidden_dim, generator=generator) * (1.0 + robustness) * 0.5
            )
            self.embedding.weight.copy_(weights)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        embeds = self.embedding(input_ids)
        margin = embeds.mean(dim=(1, 2)) * (1.0 + self._robustness)
        logits = torch.stack([-margin, margin], dim=1)
        # compute_perplexity (metrics.py) calls model(**batch, labels=input_ids)
        # and reads outputs.loss -- language-model-style next-token loss doesn't
        # apply to this 2-class classifier, so a cheap proxy loss (variance of
        # the per-token embedding norm) stands in: still deterministic given
        # (input_ids, robustness), still varies per distorted sample, which is
        # all compute_perplexity's exp(loss) needs to produce a non-degenerate
        # per-sample robustness signal for the correlation analysis.
        loss = embeds.pow(2).mean() / (1.0 + self._robustness)
        output_cls = type("Output", (), {"logits": logits, "loss": loss})
        return output_cls()


class _SmokeTokenizer:
    """Minimal HF-tokenizer-shaped callable: hashes text into a fixed-length
    deterministic token sequence so identical text always yields identical ids."""

    vocab_size = 64
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, truncation=True, max_length=32, padding=None, return_tensors=None):
        ids = [
            (abs(hash((text, i))) % (self.vocab_size - 1)) + 1  # avoid the pad id
            for i in range(min(max_length, 16))
        ]
        input_ids = torch.tensor([ids])
        attention_mask = torch.ones_like(input_ids)
        if return_tensors == "pt":
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class _SmokeDataset:
    """Minimal HF-Dataset-shaped wrapper (mirrors the fakes in
    tests/test_certification.py / tests/test_evaluation.py)."""

    _POSITIVE = [
        "a genuinely charming and well-acted film",
        "surprisingly delightful from start to finish",
        "a warm, funny, and deeply human story",
        "an accomplished and moving piece of cinema",
        "wonderfully paced with a strong emotional core",
    ]
    _NEGATIVE = [
        "a tedious and poorly acted mess",
        "dull, overlong, and never finds its footing",
        "an unpleasant and forgettable viewing experience",
        "clumsily plotted with wooden performances",
        "a disappointing and joyless slog",
    ]

    def __init__(self, n: int = 50, seed: int = 42):
        rng = np.random.default_rng(seed)
        examples = []
        for i in range(n):
            label = int(rng.integers(0, 2))
            pool = self._POSITIVE if label == 1 else self._NEGATIVE
            template = pool[i % len(pool)]
            examples.append({"text": f"{template} (sample {i})", "label": label})
        self._examples = examples

    def __len__(self):
        return len(self._examples)

    def __iter__(self):
        return iter(self._examples)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [ex[key] for ex in self._examples]
        return self._examples[key]

    @property
    def column_names(self):
        return list(self._examples[0].keys()) if self._examples else []

    def shuffle(self, seed=42):
        return self

    def select(self, indices):
        return _select(self, indices)

    def map(self, fn, desc=None, **kwargs):
        return _SmokeDataset.from_examples([fn(ex) for ex in self._examples])

    @classmethod
    def from_examples(cls, examples):
        obj = cls.__new__(cls)
        obj._examples = examples
        return obj


def _select(dataset: _SmokeDataset, indices) -> _SmokeDataset:
    return _SmokeDataset.from_examples([dataset._examples[i] for i in indices])


def _build_smoke_assets(seed: int = 42):
    tokenizer = _SmokeTokenizer()
    baseline_model = _SmokeClassifier(robustness=0.0)
    trained_model = _SmokeClassifier(robustness=0.6)
    dataset = _SmokeDataset(n=50, seed=seed)
    return baseline_model, trained_model, tokenizer, dataset


# ─────────────────────────────────────────────────────────────────────────────
# Real-model loading (network/GPU required -- not exercised in offline runs).
# ─────────────────────────────────────────────────────────────────────────────


def _load_real_assets(args: argparse.Namespace):
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.baseline_model)
    baseline_model = AutoModelForSequenceClassification.from_pretrained(args.baseline_model)
    trained_model = AutoModelForSequenceClassification.from_pretrained(args.trained_model)
    baseline_model.to(args.device).eval()
    trained_model.to(args.device).eval()

    if args.dataset_config:
        dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    return baseline_model, trained_model, tokenizer, dataset


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample empirical robustness (for the correlation analysis)
# ─────────────────────────────────────────────────────────────────────────────


def per_sample_empirical_robustness(
    model,
    tokenizer,
    subset,
    *,
    distortion_fn,
    strength: float,
    text_column: str,
    max_length: int,
    device: str,
) -> list:
    """Computes a per-sample empirical robustness proxy: 1 / perplexity of the
    model on a single distorted example, for each example in `subset` (in order).

    This mirrors metrics.robustness_score's inv-perplexity convention (higher =
    more robust) but per-sample rather than dataset-averaged, so it can be
    correlated 1:1 against certify_dataset's per-sample certified_radius --
    robustness_score/quick_robustness_score don't expose a per-sample breakdown,
    since AUC-over-strengths is inherently a dataset-level aggregate.
    """
    from torch.utils.data import DataLoader

    scores = []
    for example in subset:
        text = example[text_column]
        distorted_text = distortion_fn(text, strength=strength)
        encoded = tokenizer(
            distorted_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        loader = DataLoader([{k: v.squeeze(0) for k, v in encoded.items()}], batch_size=1)
        ppl = compute_perplexity(model, loader, device=device)
        scores.append(1.0 / max(ppl, 1e-8))
    return scores


def _correlation_analysis(empirical: list, certified: list) -> dict:
    """Pearson + Spearman correlation between per-sample empirical robustness
    and per-sample certified radius. Spearman is reported alongside Pearson
    since certified_radius is a monotonic-but-nonlinear (Phi^-1) function of
    the underlying probability estimate -- rank correlation is the more
    appropriate primary statistic here, Pearson is included for reference."""
    if len(empirical) < 2 or len(certified) < 2:
        return {
            "n": len(empirical),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_r": None,
            "spearman_p": None,
            "note": "Insufficient samples for correlation (need >= 2).",
        }

    pearson_r, pearson_p = stats.pearsonr(empirical, certified)
    spearman_r, spearman_p = stats.spearmanr(empirical, certified)

    return {
        "n": len(empirical),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────


def run_experiment(args: argparse.Namespace) -> dict:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    if args.smoke:
        logger.info("Running in --smoke mode: synthetic offline model + dataset.")
        baseline_model, trained_model, tokenizer, dataset = _build_smoke_assets(seed=args.seed)
        text_column, label_column = "text", "label"
    else:
        baseline_model, trained_model, tokenizer, dataset = _load_real_assets(args)
        text_column, label_column = args.text_column, args.label_column

    # Pre-select the fixed subset ONCE so both certification and the per-sample
    # empirical-robustness pass operate on identical, index-aligned samples --
    # required for the correlation analysis to be meaningful. certify_dataset
    # is then called with subset_size=None so it doesn't re-shuffle/re-select.
    subset = dataset.shuffle(seed=args.seed).select(range(min(args.subset_size, len(dataset))))

    cert_kwargs = dict(
        text_column=text_column,
        label_column=label_column,
        sigma=args.sigma,
        n=args.n,
        n0=args.n0,
        alpha=args.alpha,
        subset_size=None,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )

    logger.info("Certifying baseline model on %d samples...", len(subset))
    baseline_cert = certify_dataset(baseline_model, tokenizer, subset, **cert_kwargs)

    logger.info("Certifying trained model on %d samples...", len(subset))
    trained_cert = certify_dataset(trained_model, tokenizer, subset, **cert_kwargs)

    def _registry_distortion():
        from nightmarenet.distortions.registry import get_registry

        registry = get_registry()

        def _apply(text, strength):
            return registry.apply("nightmare", text, strength=strength, seed=args.seed)

        return _apply

    distortion_fn = _registry_distortion()

    logger.info("Computing per-sample empirical robustness (baseline)...")
    baseline_empirical = per_sample_empirical_robustness(
        baseline_model,
        tokenizer,
        subset,
        distortion_fn=distortion_fn,
        strength=args.distortion_strength,
        text_column=text_column,
        max_length=args.max_length,
        device=args.device,
    )
    logger.info("Computing per-sample empirical robustness (trained)...")
    trained_empirical = per_sample_empirical_robustness(
        trained_model,
        tokenizer,
        subset,
        distortion_fn=distortion_fn,
        strength=args.distortion_strength,
        text_column=text_column,
        max_length=args.max_length,
        device=args.device,
    )

    baseline_radii = [r.certified_radius for r in baseline_cert["results"]]
    trained_radii = [r.certified_radius for r in trained_cert["results"]]

    correlation = {
        "baseline": _correlation_analysis(baseline_empirical, baseline_radii),
        "trained": _correlation_analysis(trained_empirical, trained_radii),
    }

    # ── Aggregate summary in the same shape _run_certification returns, so the
    # real Evaluator.generate_report() code path can render it unmodified. ────
    def _summary(cert: dict) -> dict:
        return {
            "certified_radius_mean": cert["certified_radius_mean"],
            "certified_radius_median": cert["certified_radius_median"],
            "certification_abstain_rate": cert["certification_abstain_rate"],
            "certified_accuracy": cert["certified_accuracy"],
            "samples_certified": cert["n_samples"],
            "budget_exceeded": False,
        }

    baseline_summary = _summary(baseline_cert)
    trained_summary = _summary(trained_cert)
    deltas = {
        k: trained_summary[k] - baseline_summary[k]
        for k in baseline_summary
        if isinstance(baseline_summary[k], (int, float))
        and not isinstance(baseline_summary[k], bool)
    }

    comparison = {
        "baseline_label": "baseline (wake-only)",
        "trained_label": "nightmarenet (wake + nightmare)",
        "metrics": {
            "certification": {
                "baseline": baseline_summary,
                "trained": trained_summary,
                "deltas": deltas,
            }
        },
    }

    hypothesis_confirmed = (
        trained_summary["certified_radius_mean"] > baseline_summary["certified_radius_mean"]
    )

    experiment_result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke" if args.smoke else "full",
        "seed": args.seed,
        "config": {
            "sigma": args.sigma,
            "n": args.n,
            "n0": args.n0,
            "alpha": args.alpha,
            "subset_size": args.subset_size,
            "distortion_strength": args.distortion_strength,
        },
        "baseline": certification_to_json_dict(baseline_cert),
        "trained": certification_to_json_dict(trained_cert),
        "correlation_analysis": correlation,
        "hypothesis": {
            "claim": "Certified radius increases after wake+nightmare adversarial training.",
            "confirmed": hypothesis_confirmed,
            "baseline_mean_radius": baseline_summary["certified_radius_mean"],
            "trained_mean_radius": trained_summary["certified_radius_mean"],
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)

    json_path = os.path.join(args.output_dir, "certification_experiment.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(experiment_result, f, indent=2, default=str)
    logger.info("Wrote %s", json_path)

    to_csv_certification(baseline_cert, os.path.join(args.output_dir, "baseline_certification.csv"))
    to_csv_certification(trained_cert, os.path.join(args.output_dir, "trained_certification.csv"))
    logger.info("Wrote per-sample CSVs to %s", args.output_dir)

    report_evaluator = Evaluator(
        model=baseline_model,
        tokenizer=tokenizer,
        config={
            "evaluation": {
                "metrics": ["certification"],
                "output_dir": args.output_dir,
                "certification": {
                    "sigma": args.sigma,
                    "n": args.n,
                    "n0": args.n0,
                    "alpha": args.alpha,
                    "subset_size": args.subset_size,
                },
            }
        },
    )
    report_md = report_evaluator.generate_report(comparison)
    report_path = os.path.join(args.output_dir, "certification_experiment_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info("Wrote %s", report_path)

    logger.info(
        "Hypothesis (%s > %s? radius increases after training): %s",
        f"{trained_summary['certified_radius_mean']:.4f}",
        f"{baseline_summary['certified_radius_mean']:.4f}",
        "CONFIRMED" if hypothesis_confirmed else "NOT confirmed",
    )

    return experiment_result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Offline synthetic smoke run (no network/GPU)."
    )
    parser.add_argument(
        "--baseline-model", default=None, help="HF model name/path (baseline, wake-only)."
    )
    parser.add_argument(
        "--trained-model", default=None, help="HF model name/path (trained, wake+nightmare)."
    )
    parser.add_argument("--dataset", default="glue")
    parser.add_argument("--dataset-config", default="sst2")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--text-column", default="sentence")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--subset-size", type=int, default=50)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--n0", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--distortion-strength", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="results/certification")
    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.smoke and (not args.baseline_model or not args.trained_model):
        parser.error("--baseline-model and --trained-model are required unless --smoke is set.")

    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
