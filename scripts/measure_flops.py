#!/usr/bin/env python3
"""Script to measure and analyze FLOPs per phase and cycle in NightmareNet.

Uses fvcore to profile the forward pass FLOPs of the model, and scales
this count according to training dynamics (backward pass = 2x forward FLOPs,
reference model evaluation, epochs, batch sizes, etc.).
"""

import argparse
import sys

import numpy as np
import torch

# Try to import fvcore or torchprofile
try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False

try:
    from torchprofile import profile_macs
    HAS_TORCHPROFILE = True
except ImportError:
    HAS_TORCHPROFILE = False

from transformers import AutoConfig, AutoModelForSequenceClassification


class HFModelWrapper(torch.nn.Module):
    """Wrapper to expose keyword inputs as positional inputs for JIT tracing."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


def parse_args():
    parser = argparse.ArgumentParser(description="Profile NightmareNet Compute Cost (FLOPs)")
    parser.add_argument(
        "--model",
        type=str,
        default="distilbert-base-uncased",
        help="Hugging Face model to profile (default: distilbert-base-uncased)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length (default: 128)",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=2000,
        help="Number of training samples per epoch (default: 2000)",
    )
    parser.add_argument(
        "--wake-epochs",
        type=int,
        default=2,
        help="Wake phase epochs per cycle (default: 2)",
    )
    parser.add_argument(
        "--dream-epochs",
        type=int,
        default=1,
        help="Dream phase epochs per cycle (default: 1)",
    )
    parser.add_argument(
        "--nightmare-epochs",
        type=int,
        default=1,
        help="Nightmare phase epochs per cycle (default: 1)",
    )
    parser.add_argument(
        "--compress-epochs",
        type=int,
        default=1,
        help="Compression fine-tuning epochs per cycle (default: 1)",
    )
    parser.add_argument(
        "--num-cycles",
        type=int,
        default=3,
        help="Number of training cycles (default: 3)",
    )
    parser.add_argument(
        "--pruning-ratio",
        type=float,
        default=0.15,
        help="Compression pruning ratio (default: 0.15)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Profiling device (cpu or cuda)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.batch_size <= 0:
        raise SystemExit("error: --batch-size must be positive")
    if args.wake_epochs < 0 or args.dream_epochs < 0 or args.nightmare_epochs < 0:
        raise SystemExit("error: --*-epochs must be non-negative")
    if not (0.0 <= args.pruning_ratio <= 1.0):
        raise SystemExit("error: --pruning-ratio must be between 0.0 and 1.0")

    has_cuda = torch.cuda.is_available()
    device = torch.device(
        args.device if has_cuda or args.device == "cpu" else "cpu"
    )
    print(f"Profiling on device: {device}")

    if not HAS_FVCORE and not HAS_TORCHPROFILE:
        print("Error: Either 'fvcore' or 'torchprofile' must be installed to count FLOPs.")
        sys.exit(1)

    print(f"Loading model '{args.model}'...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=2
        )
    except Exception as e:
        print(
            f"Warning: Failed to load pretrained weights for '{args.model}' ({e}). "
            "Creating model from config instead..."
        )
        config = AutoConfig.from_pretrained(args.model, num_labels=2)
        model = AutoModelForSequenceClassification.from_config(config)

    model.eval()
    model.to(device)
    wrapper = HFModelWrapper(model)

    # Generate dummy input
    vocab_size = getattr(model.config, "vocab_size", 30522)
    dummy_input_ids = torch.randint(
        0, vocab_size, (args.batch_size, args.seq_len), dtype=torch.long, device=device
    )
    dummy_attention_mask = torch.ones(
        (args.batch_size, args.seq_len), dtype=torch.long, device=device
    )
    inputs = (dummy_input_ids, dummy_attention_mask)

    print("Measuring forward pass FLOPs...")
    forward_flops = 0
    if HAS_FVCORE:
        try:
            # Silence internal warnings
            import logging
            logging.getLogger("fvcore").setLevel(logging.ERROR)
            analysis = FlopCountAnalysis(wrapper, inputs)
            forward_flops = analysis.total()
            print(f"fvcore forward FLOPs: {forward_flops:,}")
        except Exception as e:
            print(f"fvcore profiling failed: {e}")

    if forward_flops == 0 and HAS_TORCHPROFILE:
        try:
            macs = profile_macs(wrapper, inputs)
            # 1 MAC is approximately 2 FLOPs (1 multiply + 1 add)
            forward_flops = macs * 2
            print(f"torchprofile MACs: {macs:,} (~{forward_flops:,} FLOPs)")
        except Exception as e:
            print(f"torchprofile profiling failed: {e}")

    if forward_flops == 0:
        print("Error: Failed to measure FLOPs.")
        sys.exit(1)

    # FLOPs counting logic:
    # 1. Forward pass = forward_flops
    # 2. Backward pass = 2 * forward_flops
    # 3. Training step (forward + backward) = 3 * forward_flops
    # 4. Reference model evaluation (forward only, no grad) = 1 * forward_flops

    # Steps per epoch
    steps_per_epoch = int(np.ceil(args.train_samples / args.batch_size))
    print("\nConfiguration:")
    print(f"  Train Samples/Epoch: {args.train_samples}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Steps/Epoch: {steps_per_epoch}")
    print(f"  Wake Epochs: {args.wake_epochs}")
    print(f"  Dream Epochs: {args.dream_epochs}")
    print(f"  Nightmare Epochs: {args.nightmare_epochs}")
    print(f"  Compress Epochs: {args.compress_epochs}")
    print(f"  Num Cycles: {args.num_cycles}")
    print(f"  Pruning Ratio: {args.pruning_ratio}")

    # FLOPs per step for each phase
    step_flops_wake = 3 * forward_flops
    # model (forward+backward) + reference (forward only)
    step_flops_dream = 3 * forward_flops + 1 * forward_flops
    step_flops_nightmare = 3 * forward_flops

    # Compress phase: unstructured pruning fine-tuning (dense calculation)
    # Under unstructured pruning, the tensor sizes do not change, so raw FLOPs are identical.
    # We also report effective sparse FLOPs: scaled by (1 - pruning_ratio)
    step_flops_compress_dense = 3 * forward_flops
    step_flops_compress_sparse = 3 * forward_flops * (1.0 - args.pruning_ratio)

    # Compute per-phase totals for one cycle (in FLOPs)
    cycle_flops_wake = args.wake_epochs * steps_per_epoch * step_flops_wake
    cycle_flops_dream = args.dream_epochs * steps_per_epoch * step_flops_dream
    cycle_flops_nightmare = args.nightmare_epochs * steps_per_epoch * step_flops_nightmare
    cycle_flops_compress_dense = (
        args.compress_epochs * steps_per_epoch * step_flops_compress_dense
    )
    cycle_flops_compress_sparse = (
        args.compress_epochs * steps_per_epoch * step_flops_compress_sparse
    )

    cycle_total_dense = (
        cycle_flops_wake + cycle_flops_dream + cycle_flops_nightmare
        + cycle_flops_compress_dense
    )
    cycle_total_sparse = (
        cycle_flops_wake + cycle_flops_dream + cycle_flops_nightmare
        + cycle_flops_compress_sparse
    )

    total_dense = args.num_cycles * cycle_total_dense
    total_sparse = args.num_cycles * cycle_total_sparse

    # Print results in TeraFLOPs (1 TFLOP = 1e12 FLOPs)
    to_tflops = 1e12

    print("\n--- Compute Cost per Phase & Cycle ---")
    print(f"Single Forward Pass: {forward_flops / 1e9:.3f} GFLOPs")
    wake_tflops = steps_per_epoch * step_flops_wake / to_tflops
    print(f"Wake Epoch FLOPs: {wake_tflops:.4f} TFLOPs")
    dream_tflops = steps_per_epoch * step_flops_dream / to_tflops
    print(f"Dream Epoch FLOPs: {dream_tflops:.4f} TFLOPs (incl. Reference Model)")
    nightmare_tflops = steps_per_epoch * step_flops_nightmare / to_tflops
    print(f"Nightmare Epoch FLOPs: {nightmare_tflops:.4f} TFLOPs")
    compress_dense_tflops = steps_per_epoch * step_flops_compress_dense / to_tflops
    print(f"Compress Epoch FLOPs (dense): {compress_dense_tflops:.4f} TFLOPs")
    compress_sparse_tflops = (
        steps_per_epoch * step_flops_compress_sparse / to_tflops
    )
    print(f"Compress Epoch FLOPs (sparse effective): {compress_sparse_tflops:.4f} TFLOPs")

    print(f"\nTotal per Cycle (Dense): {cycle_total_dense / to_tflops:.4f} TFLOPs")
    print(f"Total per Cycle (Sparse): {cycle_total_sparse / to_tflops:.4f} TFLOPs")
    print(f"Total across {args.num_cycles} Cycles (Dense): {total_dense / to_tflops:.4f} TFLOPs")
    print(f"Total across {args.num_cycles} Cycles (Sparse): {total_sparse / to_tflops:.4f} TFLOPs")

    # Comparisons with baseline methods
    # Standard Fine-Tuning: 2 epochs of Wake only (as in configs/benchmark_sst2_baseline.yaml)
    baseline_ft_epochs = 2
    baseline_ft_flops = baseline_ft_epochs * steps_per_epoch * step_flops_wake

    # Standard Adversarial Training (Wake + PGD-10)
    # A PGD-10 step requires:
    # - 10 steps of forward+backward (gradient w.r.t input) for generation
    # - 1 step of forward+backward (gradient w.r.t weights) for training
    # Total = 11 * forward_flops + 11 * backward_flops = 33 * forward_flops per step
    step_flops_at = 11 * 3 * forward_flops  # 33 * forward_flops
    baseline_at_epochs = 2
    baseline_at_flops = baseline_at_epochs * steps_per_epoch * step_flops_at

    # TRADES (Standard)
    # Similar to PGD-10, requires 10 steps of generation + 1 training step w/ KL loss
    # TRADES training is mathematically equivalent in step complexity to standard AT
    baseline_trades_flops = baseline_at_flops

    print("\n--- Compute Cost Comparison ---")
    print(f"Standard FT (2 epochs): {baseline_ft_flops / to_tflops:.4f} TFLOPs")
    print(f"Standard AT / PGD-10 (2 epochs): {baseline_at_flops / to_tflops:.4f} TFLOPs")
    print(f"TRADES (2 epochs): {baseline_trades_flops / to_tflops:.4f} TFLOPs")
    cycle_dense_tflops = cycle_total_dense / to_tflops
    print(f"NightmareNet Full Cycle (1 cycle, 5 total epochs): {cycle_dense_tflops:.4f} TFLOPs")
    print(
        f"NightmareNet Full Cycle ({args.num_cycles} cycles, "
        f"{args.num_cycles * 5} total epochs): {total_dense / to_tflops:.4f} TFLOPs"
    )

    # Compute Overhead Ratios
    overhead_nn_vs_ft = cycle_total_dense / baseline_ft_flops
    overhead_at_vs_ft = baseline_at_flops / baseline_ft_flops
    savings_nn_vs_at = baseline_at_flops / cycle_total_dense

    print("\n--- Overhead & Efficiency Analysis ---")
    print(f"NightmareNet Cycle vs. Standard FT Overhead: {overhead_nn_vs_ft:.2f}x")
    print(
        f"Standard AT / TRADES vs. Standard FT Overhead: {overhead_at_vs_ft:.2f}x "
        "(Gradient-based PGD)"
    )
    print(
        f"NightmareNet Cycle vs. Standard AT Compute Ratio: {1/savings_nn_vs_at:.2f}x "
        f"(saves {savings_nn_vs_at:.2f}x compute)"
    )

    # Print markdown format
    print("\nMarkdown Table for Appendix F:")
    print("| Method | Epochs | Steps / Epoch | TFLOPs / Epoch | Total TFLOPs | Overhead vs. FT |")
    print("|---|---|---|---|---|---|")

    # Format table values
    wake_ep_tflops = steps_per_epoch * step_flops_wake / to_tflops
    at_ep_tflops = steps_per_epoch * step_flops_at / to_tflops

    row_ft = (
        f"| Standard Fine-Tuning | {baseline_ft_epochs} | {steps_per_epoch} | "
        f"{wake_ep_tflops:.4f} | {baseline_ft_flops / to_tflops:.4f} | 1.0x (Baseline) |"
    )
    row_at = (
        f"| Standard AT (PGD-10) | {baseline_at_epochs} | {steps_per_epoch} | "
        f"{at_ep_tflops:.4f} | {baseline_at_flops / to_tflops:.4f} | {overhead_at_vs_ft:.2f}x |"
    )
    row_trades = (
        f"| TRADES | {baseline_at_epochs} | {steps_per_epoch} | "
        f"{at_ep_tflops:.4f} | {baseline_trades_flops / to_tflops:.4f} | {overhead_at_vs_ft:.2f}x |"
    )
    row_nn1 = (
        f"| **NightmareNet (1 cycle)** | 5 | {steps_per_epoch} | Multi-phase | "
        f"{cycle_total_dense / to_tflops:.4f} | {overhead_nn_vs_ft:.2f}x |"
    )
    row_nn3 = (
        f"| **NightmareNet ({args.num_cycles} cycles)** | {args.num_cycles * 5} | "
        f"{steps_per_epoch} | Multi-phase | {total_dense / to_tflops:.4f} | "
        f"{total_dense / baseline_ft_flops:.2f}x |"
    )

    print(row_ft)
    print(row_at)
    print(row_trades)
    print(row_nn1)
    print(row_nn3)



if __name__ == "__main__":
    main()
