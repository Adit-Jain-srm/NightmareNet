#!/usr/bin/env python3
"""Real GPU benchmark: SST-2 baseline vs NightmareNet adversarial training.

Trains DistilBERT-base-uncased in two regimes, evaluates clean accuracy and
robustness under dream/nightmare distortions, and writes a comparison JSON
to ``results/gpu_benchmark.json``.

Optimized for RTX 3050 Ti Laptop (4 GB VRAM):
    - Batch size 8
    - max_length 128
    - mixed precision FP16 via autocast
    - 1 epoch over ``--train-samples`` rows (default 1000)

Usage:
    python scripts/run_gpu_benchmark.py
    python scripts/run_gpu_benchmark.py --train-samples 500 --eval-samples 200
    python scripts/run_gpu_benchmark.py --device cpu  # for CI smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_data(train_samples: int, eval_samples: int) -> tuple[list[dict], list[dict]]:
    from datasets import load_dataset

    # HF datasets 4.x requires a namespaced repo id; the bare "glue" alias was removed.
    try:
        raw = load_dataset("nyu-mll/glue", "sst2")
    except Exception:
        raw = load_dataset("glue", "sst2")
    train = raw["train"].shuffle(seed=42).select(range(min(train_samples, len(raw["train"]))))
    val = raw["validation"].select(range(min(eval_samples, len(raw["validation"]))))
    return list(train), list(val)


def _tokenize_batch(tokenizer: Any, texts: list[str], device: str) -> dict:
    enc = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def _train_epoch(
    model: Any,
    tokenizer: Any,
    train: list[dict],
    device: str,
    batch_size: int,
    lr: float,
    use_amp: bool,
    distort_fn: Any = None,
) -> float:
    import torch

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device == "cuda") else None
    total_loss = 0.0
    steps = 0

    for i in range(0, len(train), batch_size):
        batch = train[i : i + batch_size]
        texts = [row["sentence"] for row in batch]
        if distort_fn is not None:
            texts = [distort_fn(t) for t in texts]
        labels = torch.tensor([row["label"] for row in batch], device=device)
        enc = _tokenize_batch(tokenizer, texts, device)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(**enc, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(**enc, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)


def _evaluate(
    model: Any,
    tokenizer: Any,
    examples: list[dict],
    device: str,
    batch_size: int,
    distort_fn: Any = None,
) -> float:
    import torch

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]
            texts = [row["sentence"] for row in batch]
            if distort_fn is not None:
                texts = [distort_fn(t) for t in texts]
            labels = torch.tensor([row["label"] for row in batch], device=device)
            enc = _tokenize_batch(tokenizer, texts, device)
            logits = model(**enc).logits
            preds = logits.argmax(dim=-1)
            correct += int((preds == labels).sum().item())
            total += len(labels)
    return correct / max(total, 1)


def _build_distorter(distortion: str, strength: float):
    """Build a fast distortion callable that skips the slow learned-adversarial path.

    The learned-adversarial DistilBERT loader is correct for production runs but
    too slow for a benchmark sweep (per-call attention extraction). We disable it
    via explicit config and rely on rule-based contradictions/ambiguity instead.
    """
    from nightmarenet.distortions import dream as dream_mod
    from nightmarenet.distortions import nightmare as nightmare_mod

    if distortion == "dream":

        def fn_dream(text: str) -> str:
            return dream_mod.distort(text, strength=strength, seed=42)

        return fn_dream

    # Nightmare with learned: 0 to avoid per-call model loading
    cfg = {
        "adversarial": {
            "contradiction": 0.3,
            "ambiguity": 0.3,
            "cross_domain": 0.2,
            "misleading_context": 0.2,
            "learned": 0.0,
        }
    }

    def fn_nightmare(text: str) -> str:
        return nightmare_mod.distort(text, strength=strength, seed=42, config=cfg)

    return fn_nightmare


def _train_model(label: str, train: list[dict], val: list[dict], args: argparse.Namespace) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"\n[{label}] loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(
        args.device
    )

    use_amp = args.device == "cuda" and not args.no_amp
    t0 = time.time()

    if label == "baseline":
        print(
            f"[{label}] wake epoch over {len(train)} examples (bs={args.batch_size}, amp={use_amp})"
        )
        loss = _train_epoch(
            model,
            tokenizer,
            train,
            args.device,
            args.batch_size,
            args.lr,
            use_amp,
        )
        history = [{"phase": "wake", "loss": loss}]
    else:
        print(
            f"[{label}] wake epoch over {len(train)} examples (bs={args.batch_size}, amp={use_amp})"
        )
        wake_loss = _train_epoch(
            model,
            tokenizer,
            train,
            args.device,
            args.batch_size,
            args.lr,
            use_amp,
        )
        nightmare_distorter = _build_distorter("nightmare", strength=0.5)
        print(f"[{label}] nightmare epoch (adversarial hardening, strength=0.5)")
        nightmare_loss = _train_epoch(
            model,
            tokenizer,
            train,
            args.device,
            args.batch_size,
            args.lr * 0.5,
            use_amp,
            distort_fn=nightmare_distorter,
        )
        history = [
            {"phase": "wake", "loss": wake_loss},
            {"phase": "nightmare", "loss": nightmare_loss},
        ]

    train_seconds = time.time() - t0
    print(f"[{label}] training done in {train_seconds:.1f}s")

    print(f"[{label}] evaluating on {len(val)} val examples...")
    clean_acc = _evaluate(model, tokenizer, val, args.device, args.batch_size)

    distorted: dict[str, dict[str, float]] = {}
    for d_type in ("dream", "nightmare"):
        per_strength: dict[str, float] = {}
        for s in (0.1, 0.3, 0.5, 0.7, 0.9):
            fn = _build_distorter(d_type, strength=s)
            acc = _evaluate(model, tokenizer, val, args.device, args.batch_size, distort_fn=fn)
            per_strength[f"{s:.1f}"] = round(acc, 4)
            print(f"  {d_type}@{s:.1f}: acc={acc:.4f}")
        distorted[d_type] = per_strength

    avg_distorted = (
        sum(v for d in distorted.values() for v in d.values()) / 10  # 2 * 5
    )
    robustness_drop = clean_acc - avg_distorted

    # Free VRAM before next run
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "label": label,
        "model": args.model,
        "train_samples": len(train),
        "eval_samples": len(val),
        "train_seconds": round(train_seconds, 2),
        "history": history,
        "clean_accuracy": round(clean_acc, 4),
        "distorted_accuracy": distorted,
        "avg_distorted_accuracy": round(avg_distorted, 4),
        "robustness_drop": round(robustness_drop, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="NightmareNet GPU benchmark on SST-2")
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "results" / "gpu_benchmark.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _set_seed(args.seed)

    # Verify device
    try:
        import torch

        if args.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available; falling back to CPU")
            args.device = "cpu"
        if args.device == "cuda":
            free_mb = torch.cuda.mem_get_info()[0] // (1024 * 1024)
            print(f"GPU: {torch.cuda.get_device_name(0)} | {free_mb} MB free")
    except ImportError:
        print("PyTorch not installed", file=sys.stderr)
        return 1

    print(f"Loading SST-2: train={args.train_samples} val={args.eval_samples}")
    train, val = _load_data(args.train_samples, args.eval_samples)

    baseline = _train_model("baseline", train, val, args)
    nightmarenet = _train_model("nightmarenet", train, val, args)

    improvement = nightmarenet["avg_distorted_accuracy"] - baseline["avg_distorted_accuracy"]
    relative_pct = (improvement / max(baseline["avg_distorted_accuracy"], 1e-9)) * 100

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "seed": args.seed,
        "baseline": baseline,
        "nightmarenet": nightmarenet,
        "comparison": {
            "clean_delta": round(nightmarenet["clean_accuracy"] - baseline["clean_accuracy"], 4),
            "avg_distorted_delta": round(improvement, 4),
            "robustness_improvement_pct": round(relative_pct, 2),
            "robustness_drop_reduction": round(
                baseline["robustness_drop"] - nightmarenet["robustness_drop"], 4
            ),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(
        f"Baseline       clean={baseline['clean_accuracy']:.4f}  "
        f"avg_distorted={baseline['avg_distorted_accuracy']:.4f}"
    )
    print(
        f"NightmareNet   clean={nightmarenet['clean_accuracy']:.4f}  "
        f"avg_distorted={nightmarenet['avg_distorted_accuracy']:.4f}"
    )
    print(f"Robustness improvement: {improvement:+.4f}  ({relative_pct:+.2f}%)")
    print(f"Written: {out_path}")

    # Trigger deploy webhook if notifications.webhooks are configured
    try:
        from nightmarenet.utils.config import load_config
        from nightmarenet.utils.webhooks import trigger_webhook

        config = load_config(str(REPO_ROOT / "configs" / "default.yaml"))
        trigger_webhook(
            config,
            "deploy",
            "SST-2 GPU benchmark run finished.",
            {
                "model": args.model,
                "device": args.device,
                "timestamp": result.get("timestamp"),
                "avg_distorted_delta": f"{improvement:+.4f}",
                "improvement_pct": f"{relative_pct:.2f}%",
                "output_path": str(out_path),
            },
        )
    except Exception as e:
        print(f"Warning: Failed to trigger webhook notification: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
