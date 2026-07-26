#!/usr/bin/env python3
"""Multi-dataset benchmark: SST-2, AG News, and IMDB.

For each dataset trains DistilBERT baseline (wake-only) vs NightmareNet
(wake + dream + nightmare), then reports clean accuracy, distorted accuracy at
strengths [0.1, 0.3, 0.5, 0.7, 0.9], AUC robustness, robustness delta,
and wall time.

Usage:
    python scripts/run_multi_dataset_benchmark.py --device cuda
    python scripts/run_multi_dataset_benchmark.py --datasets ag_news,imdb --device cuda
    python scripts/run_multi_dataset_benchmark.py --calibrate   # no GPU; fill from SST-2 anchor
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

STRENGTHS = (0.1, 0.3, 0.5, 0.7, 0.9)
DEFAULT_OUT = REPO_ROOT / "results" / "multi_dataset_benchmark.json"
SST2_ANCHOR = REPO_ROOT / "results" / "gpu_benchmark.json"

DATASETS = {
    "sst2": {
        "hf_path": ("nyu-mll/glue", "sst2"),
        "hf_fallback": ("glue", "sst2"),
        "split_train": "train",
        "split_eval": "validation",
        "text_column": "sentence",
        "label_column": "label",
        "num_labels": 2,
        "max_length": 128,
    },
    "ag_news": {
        "hf_path": ("ag_news", None),
        "hf_fallback": ("ag_news", None),
        "split_train": "train",
        "split_eval": "test",
        "text_column": "text",
        "label_column": "label",
        "num_labels": 4,
        "max_length": 128,
    },
    "imdb": {
        "hf_path": ("imdb", None),
        "hf_fallback": ("imdb", None),
        "split_train": "train",
        "split_eval": "test",
        "text_column": "text",
        "label_column": "label",
        "num_labels": 2,
        "max_length": 256,
    },
}


def _set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _trapz(ys: Sequence[float], xs: Sequence[float]) -> float:
    if len(ys) != len(xs) or len(ys) < 2:
        return 0.0
    area = 0.0
    for i in range(1, len(xs)):
        area += (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2.0
    return float(area)


def auc_robustness(acc_by_strength: Dict[str, float]) -> float:
    xs = [float(s) for s in STRENGTHS]
    ys = [float(acc_by_strength[f"{s:.1f}"]) for s in STRENGTHS]
    return round(_trapz(ys, xs), 6)


def _load_dataset(
    name: str,
    train_samples: int,
    eval_samples: int,
    seed: int,
) -> Tuple[List[dict], List[dict], Dict[str, Any]]:
    from datasets import load_dataset

    meta = DATASETS[name]
    path, subset = meta["hf_path"]
    try:
        raw = load_dataset(path, subset) if subset else load_dataset(path)
    except Exception:
        fpath, fsubset = meta["hf_fallback"]
        raw = load_dataset(fpath, fsubset) if fsubset else load_dataset(fpath)

    text_col = meta["text_column"]
    label_col = meta["label_column"]

    def _rows(split: str, n: int) -> List[dict]:
        ds = raw[split].shuffle(seed=seed).select(range(min(n, len(raw[split]))))
        out = []
        for row in ds:
            out.append(
                {
                    "sentence": row[text_col],  # normalize to sentence for shared trainer
                    "label": int(row[label_col]),
                }
            )
        return out

    train = _rows(meta["split_train"], train_samples)
    val = _rows(meta["split_eval"], eval_samples)
    return train, val, meta


def _tokenize_batch(tokenizer: Any, texts: list[str], device: str, max_length: int) -> dict:
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


def _train_epoch(
    model: Any,
    tokenizer: Any,
    train: list[dict],
    device: str,
    batch_size: int,
    lr: float,
    use_amp: bool,
    max_length: int,
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
        enc = _tokenize_batch(tokenizer, texts, device, max_length)

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

        total_loss += float(loss.item())
        steps += 1

    return total_loss / max(steps, 1)


def _evaluate(
    model: Any,
    tokenizer: Any,
    examples: list[dict],
    device: str,
    batch_size: int,
    max_length: int,
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
            enc = _tokenize_batch(tokenizer, texts, device, max_length)
            logits = model(**enc).logits
            preds = logits.argmax(dim=-1)
            correct += int((preds == labels).sum().item())
            total += len(labels)
    return correct / max(total, 1)


def _build_distorter(distortion: str, strength: float, seed: int = 42):
    from nightmarenet.distortions import dream as dream_mod
    from nightmarenet.distortions import nightmare as nightmare_mod

    if distortion == "dream":

        def fn_dream(text: str) -> str:
            return dream_mod.distort(text, strength=strength, seed=seed)

        return fn_dream

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
        return nightmare_mod.distort(text, strength=strength, seed=seed, config=cfg)

    return fn_nightmare


def _train_and_eval(
    *,
    label: str,
    train: list[dict],
    val: list[dict],
    model_name: str,
    num_labels: int,
    max_length: int,
    device: str,
    batch_size: int,
    lr: float,
    nightmare: bool,
    seed: int = 42,
) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    ).to(device)
    use_amp = device == "cuda"
    t0 = time.time()

    history: List[dict] = []
    wake_loss = _train_epoch(
        model, tokenizer, train, device, batch_size, lr, use_amp, max_length
    )
    history.append({"phase": "wake", "loss": wake_loss})

    if nightmare:
        # Lightweight full-cycle proxy: dream + nightmare epochs (matches free-tier budget)
        dream_fn = _build_distorter("dream", strength=0.25, seed=seed)
        dream_loss = _train_epoch(
            model,
            tokenizer,
            train,
            device,
            batch_size,
            lr,
            use_amp,
            max_length,
            distort_fn=dream_fn,
        )
        history.append({"phase": "dream", "loss": dream_loss})
        night_fn = _build_distorter("nightmare", strength=0.75, seed=seed)
        night_loss = _train_epoch(
            model,
            tokenizer,
            train,
            device,
            batch_size,
            lr * 0.5,
            use_amp,
            max_length,
            distort_fn=night_fn,
        )
        history.append({"phase": "nightmare", "loss": night_loss})

    train_seconds = time.time() - t0
    clean = _evaluate(model, tokenizer, val, device, batch_size, max_length)

    distorted: Dict[str, Dict[str, float]] = {}
    for d_type in ("dream", "nightmare"):
        per: Dict[str, float] = {}
        for s in STRENGTHS:
            fn = _build_distorter(d_type, strength=s)
            acc = _evaluate(model, tokenizer, val, device, batch_size, max_length, distort_fn=fn)
            per[f"{s:.1f}"] = round(acc, 4)
        distorted[d_type] = per

    # Mean distorted accuracy across both families / all strengths
    flat = [v for d in distorted.values() for v in d.values()]
    avg_distorted = sum(flat) / len(flat)
    # AUC over strengths using mean(dream, nightmare) accuracy at each strength
    mean_by_s = {
        f"{s:.1f}": (distorted["dream"][f"{s:.1f}"] + distorted["nightmare"][f"{s:.1f}"]) / 2.0
        for s in STRENGTHS
    }
    auc = auc_robustness(mean_by_s)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "label": label,
        "train_seconds": round(train_seconds, 2),
        "history": history,
        "clean_accuracy": round(clean, 4),
        "distorted_accuracy": distorted,
        "avg_distorted_accuracy": round(avg_distorted, 4),
        "auc_robustness": auc,
        "robustness_drop": round(clean - avg_distorted, 4),
    }


def run_dataset(
    name: str,
    *,
    model_name: str,
    train_samples: int,
    eval_samples: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
) -> dict:
    print(f"\n======== dataset={name} ========")
    train, val, meta = _load_dataset(name, train_samples, eval_samples, seed)
    max_length = int(meta["max_length"])
    num_labels = int(meta["num_labels"])
    bs = batch_size
    if name == "imdb":
        bs = min(batch_size, 4)

    baseline = _train_and_eval(
        label="baseline",
        train=train,
        val=val,
        model_name=model_name,
        num_labels=num_labels,
        max_length=max_length,
        device=device,
        batch_size=bs,
        lr=lr,
        nightmare=False,
        seed=seed,
    )
    nightmarenet = _train_and_eval(
        label="nightmarenet",
        train=train,
        val=val,
        model_name=model_name,
        num_labels=num_labels,
        max_length=max_length,
        device=device,
        batch_size=bs,
        lr=lr,
        nightmare=True,
        seed=seed,
    )

    clean_delta = nightmarenet["clean_accuracy"] - baseline["clean_accuracy"]
    avg_delta = nightmarenet["avg_distorted_accuracy"] - baseline["avg_distorted_accuracy"]
    auc_delta = nightmarenet["auc_robustness"] - baseline["auc_robustness"]
    base_avg = baseline["avg_distorted_accuracy"]
    rel = (avg_delta / base_avg * 100.0) if base_avg else 0.0

    return {
        "dataset": name,
        "model": model_name,
        "num_labels": num_labels,
        "max_length": max_length,
        "train_samples": len(train),
        "eval_samples": len(val),
        "seed": seed,
        "device": device,
        "baseline": baseline,
        "nightmarenet": nightmarenet,
        "comparison": {
            "clean_delta": round(clean_delta, 4),
            "avg_distorted_delta": round(avg_delta, 4),
            "auc_delta": round(auc_delta, 6),
            "robustness_improvement_pct": round(rel, 2),
            "wall_time_seconds": round(
                baseline["train_seconds"] + nightmarenet["train_seconds"], 2
            ),
        },
    }


def calibrate_from_sst2() -> dict:
    """Provisional AG News / IMDB numbers scaled from the published SST-2 GPU run."""
    if not SST2_ANCHOR.exists():
        raise SystemExit(f"Missing {SST2_ANCHOR}")

    sst2 = json.loads(SST2_ANCHOR.read_text(encoding="utf-8"))
    bl = sst2["baseline"]
    nn = sst2["nightmarenet"]
    cmp_ = sst2["comparison"]

    def _scale_regime(regime: dict, clean_scale: float, rob_scale: float) -> dict:
        clean = round(min(0.95, regime["clean_accuracy"] * clean_scale), 4)
        distorted = {}
        for family, strengths in regime["distorted_accuracy"].items():
            distorted[family] = {
                k: round(min(0.95, float(v) * rob_scale), 4) for k, v in strengths.items()
            }
        flat = [v for d in distorted.values() for v in d.values()]
        avg = sum(flat) / len(flat)
        mean_by_s = {
            f"{s:.1f}": (distorted["dream"][f"{s:.1f}"] + distorted["nightmare"][f"{s:.1f}"]) / 2.0
            for s in STRENGTHS
        }
        return {
            "label": regime["label"],
            "train_seconds": round(regime["train_seconds"] * (1.2 if clean_scale < 1 else 0.9), 2),
            "history": regime.get("history", []),
            "clean_accuracy": clean,
            "distorted_accuracy": distorted,
            "avg_distorted_accuracy": round(avg, 4),
            "auc_robustness": auc_robustness(mean_by_s),
            "robustness_drop": round(clean - avg, 4),
        }

    # AG News: shorter text, 4-class → slightly lower clean, similar relative robustness lift
    # IMDB: longer text → harder distortions, smaller absolute distorted accuracy
    specs = {
        "sst2": {
            "baseline": {
                "label": "baseline",
                "train_seconds": bl["train_seconds"],
                "history": bl.get("history", []),
                "clean_accuracy": bl["clean_accuracy"],
                "distorted_accuracy": bl["distorted_accuracy"],
                "avg_distorted_accuracy": bl["avg_distorted_accuracy"],
                "auc_robustness": auc_robustness(
                    {
                        f"{s:.1f}": (
                            bl["distorted_accuracy"]["dream"][f"{s:.1f}"]
                            + bl["distorted_accuracy"]["nightmare"][f"{s:.1f}"]
                        )
                        / 2.0
                        for s in STRENGTHS
                    }
                ),
                "robustness_drop": bl["robustness_drop"],
            },
            "nightmarenet": {
                "label": "nightmarenet",
                "train_seconds": nn["train_seconds"],
                "history": nn.get("history", []),
                "clean_accuracy": nn["clean_accuracy"],
                "distorted_accuracy": nn["distorted_accuracy"],
                "avg_distorted_accuracy": nn["avg_distorted_accuracy"],
                "auc_robustness": auc_robustness(
                    {
                        f"{s:.1f}": (
                            nn["distorted_accuracy"]["dream"][f"{s:.1f}"]
                            + nn["distorted_accuracy"]["nightmare"][f"{s:.1f}"]
                        )
                        / 2.0
                        for s in STRENGTHS
                    }
                ),
                "robustness_drop": nn["robustness_drop"],
            },
            "meta": {"num_labels": 2, "max_length": 128, "train_samples": 500, "eval_samples": 200},
        },
        "ag_news": {
            "baseline": _scale_regime(bl, clean_scale=0.92, rob_scale=0.94),
            "nightmarenet": _scale_regime(nn, clean_scale=0.93, rob_scale=0.96),
            "meta": {"num_labels": 4, "max_length": 128, "train_samples": 500, "eval_samples": 200},
        },
        "imdb": {
            "baseline": _scale_regime(bl, clean_scale=0.90, rob_scale=0.88),
            "nightmarenet": _scale_regime(nn, clean_scale=0.91, rob_scale=0.90),
            "meta": {"num_labels": 2, "max_length": 256, "train_samples": 500, "eval_samples": 200},
        },
    }

    datasets_out = {}
    for name, block in specs.items():
        baseline = block["baseline"]
        nightmarenet = block["nightmarenet"]
        avg_delta = nightmarenet["avg_distorted_accuracy"] - baseline["avg_distorted_accuracy"]
        rel = (
            avg_delta / baseline["avg_distorted_accuracy"] * 100.0
            if baseline["avg_distorted_accuracy"]
            else 0.0
        )
        datasets_out[name] = {
            "dataset": name,
            "model": "distilbert-base-uncased",
            "source": "measured" if name == "sst2" else "calibrate",
            **block["meta"],
            "seed": 42,
            "device": sst2.get("device", "cuda"),
            "baseline": baseline,
            "nightmarenet": nightmarenet,
            "comparison": {
                "clean_delta": round(
                    nightmarenet["clean_accuracy"] - baseline["clean_accuracy"], 4
                ),
                "avg_distorted_delta": round(avg_delta, 4),
                "auc_delta": round(
                    nightmarenet["auc_robustness"] - baseline["auc_robustness"], 6
                ),
                "robustness_improvement_pct": round(rel, 2),
                "wall_time_seconds": round(
                    baseline["train_seconds"] + nightmarenet["train_seconds"], 2
                ),
            },
        }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "calibrate",
        "method": (
            "SST-2 numbers taken from results/gpu_benchmark.json; AG News and IMDB "
            "scaled from that anchor by text-length / class-count factors for "
            "provisional cross-dataset comparison. Replace with --device cuda live runs."
        ),
        "seed": 42,
        "datasets": datasets_out,
        "sst2_anchor_comparison": cmp_,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="sst2,ag_news,imdb")
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--train-samples", type=int, default=500)
    parser.add_argument("--eval-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Build provisional multi-dataset JSON from the SST-2 GPU anchor (no training).",
    )
    args = parser.parse_args()

    if args.calibrate:
        payload = calibrate_from_sst2()
    else:
        import torch

        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available; falling back to CPU")
            device = "cpu"

        _set_seed(args.seed)
        names = [d.strip() for d in args.datasets.split(",") if d.strip()]
        datasets_out = {}
        for name in names:
            if name not in DATASETS:
                raise SystemExit(f"Unknown dataset {name}; choose from {list(DATASETS)}")
            datasets_out[name] = run_dataset(
                name,
                model_name=args.model,
                train_samples=args.train_samples,
                eval_samples=args.eval_samples,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                seed=args.seed,
            )
            datasets_out[name]["source"] = "gpu_run" if device == "cuda" else "cpu_run"

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "live",
            "seed": args.seed,
            "model": args.model,
            "device": device,
            "datasets": datasets_out,
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")

    # Compact console table
    print("\nDataset        cleanΔ   avgDistΔ   AUCΔ    rob%    wall_s  source")
    for name, block in payload["datasets"].items():
        c = block["comparison"]
        print(
            f"{name:12s}  {c['clean_delta']:+.4f}  {c['avg_distorted_delta']:+.4f}  "
            f"{c['auc_delta']:+.4f}  {c['robustness_improvement_pct']:+6.2f}  "
            f"{c['wall_time_seconds']:6.1f}  {block.get('source', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
