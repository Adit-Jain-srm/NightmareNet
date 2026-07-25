"""Benchmark gradient- and attention-based learned adversarial generators.

This script compares target-model loss increase and generation latency under the
same text samples, strength, device, and model. It is intentionally separate from
the full training benchmark so contributors can validate the attack generator
before spending a full cycle's compute budget.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from nightmarenet.distortions.learned import LearnedAdversarialGenerator

DEFAULT_SAMPLES = [
    "The film was a triumph of restraint and vision.",
    "The customer service team solved the problem quickly.",
    "The new policy improved outcomes for most participants.",
    "The product is reliable, affordable, and easy to use.",
    "The research findings strongly support the original hypothesis.",
    "The restaurant delivered an excellent dining experience.",
    "The software update made the application faster and safer.",
    "The treatment produced encouraging results in the trial.",
]


@dataclass
class StrategyResult:
    """Aggregate metrics for one learned-adversarial strategy."""

    strategy: str
    samples: int
    mean_seconds_per_sample: float
    median_seconds_per_sample: float
    mean_target_loss_before: float
    mean_target_loss_after: float
    mean_target_loss_increase: float
    changed_outputs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="distilbert-base-uncased",
        help="Masked language model used as both target and attention fallback.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--max-slowdown",
        type=float,
        default=3.0,
        help="Maximum allowed gradient/attention latency ratio.",
    )
    parser.add_argument(
        "--enforce-slowdown",
        action="store_true",
        help="Exit non-zero when gradient generation exceeds --max-slowdown.",
    )
    return parser.parse_args()


def load_samples(path: Optional[Path]) -> list[str]:
    if path is None:
        return DEFAULT_SAMPLES
    samples = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [sample for sample in samples if sample]


def target_loss(model, tokenizer, text: str, device: str) -> float:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded, labels=encoded["input_ids"])
    return float(outputs.loss.item())


def run_strategy(
    strategy: str,
    samples: list[str],
    model,
    tokenizer,
    args: argparse.Namespace,
) -> StrategyResult:
    generator = LearnedAdversarialGenerator(
        model_name=args.model,
        device=args.device,
        strength=args.strength,
        target_model=model if strategy == "gradient" else None,
        target_tokenizer=tokenizer if strategy == "gradient" else None,
        strategy=strategy,
        cache_enabled=False,
        seed=args.seed,
    )

    timings: list[float] = []
    losses_before: list[float] = []
    losses_after: list[float] = []
    changed_outputs = 0

    for _ in range(args.repeats):
        for text in samples:
            before = target_loss(model, tokenizer, text, args.device)
            started = time.perf_counter()
            adversarial = generator.generate(text, strength=args.strength)
            elapsed = time.perf_counter() - started
            after = target_loss(model, tokenizer, adversarial, args.device)

            timings.append(elapsed)
            losses_before.append(before)
            losses_after.append(after)
            changed_outputs += int(adversarial != text)

    mean_before = statistics.fmean(losses_before)
    mean_after = statistics.fmean(losses_after)
    return StrategyResult(
        strategy=strategy,
        samples=len(timings),
        mean_seconds_per_sample=statistics.fmean(timings),
        median_seconds_per_sample=statistics.median(timings),
        mean_target_loss_before=mean_before,
        mean_target_loss_after=mean_after,
        mean_target_loss_increase=mean_after - mean_before,
        changed_outputs=changed_outputs,
    )


def main() -> int:
    args = parse_args()
    samples = load_samples(args.input_file)
    if not samples:
        raise ValueError("At least one non-empty benchmark sample is required")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(args.device)
    model.eval()

    attention = run_strategy("attention", samples, model, tokenizer, args)
    gradient = run_strategy("gradient", samples, model, tokenizer, args)
    slowdown = gradient.mean_seconds_per_sample / max(
        attention.mean_seconds_per_sample,
        1e-12,
    )

    report = {
        "model": args.model,
        "device": args.device,
        "strength": args.strength,
        "seed": args.seed,
        "repeats": args.repeats,
        "attention": asdict(attention),
        "gradient": asdict(gradient),
        "gradient_to_attention_slowdown": slowdown,
        "within_requested_3x_budget": slowdown <= args.max_slowdown,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    if args.enforce_slowdown and slowdown > args.max_slowdown:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
