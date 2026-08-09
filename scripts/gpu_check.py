#!/usr/bin/env python3
"""GPU readiness probe for NightmareNet.

Prints a structured report on CUDA availability, device specs, mixed precision
support, and recommended config knobs for the detected device.

Usage:
    python scripts/gpu_check.py
    python scripts/gpu_check.py --json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys


def _device_report() -> dict:
    report: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    try:
        import torch
    except ImportError as e:
        report["torch"] = {"installed": False, "error": str(e)}
        return report

    report["torch"] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_compiled": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }

    if not torch.cuda.is_available():
        report["recommendation"] = (
            "CPU-only PyTorch detected. For GPU training, install CUDA wheels: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
        return report

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)

    report["device"] = {
        "name": torch.cuda.get_device_name(device),
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_mb": total // (1024 * 1024),
        "free_memory_mb": free // (1024 * 1024),
        "multiprocessor_count": props.multi_processor_count,
    }

    report["mixed_precision"] = {
        "fp16_supported": True,
        "bf16_supported": props.major >= 8,
    }

    vram_gb = total / (1024**3)
    if vram_gb < 6:
        report["tier"] = "low-vram"
        report["recommendation"] = (
            "Detected <6GB VRAM. Use mixed_precision: fp16, gradient_checkpointing: true, "
            "batch_size 4-16 max for DistilBERT-class models."
        )
    elif vram_gb < 16:
        report["tier"] = "mid-vram"
        report["recommendation"] = (
            "Detected 6-16GB VRAM. Comfortable for DistilBERT/GPT-2 (124M). "
            "Use mixed_precision: fp16, batch_size 16-32."
        )
    else:
        report["tier"] = "high-vram"
        report["recommendation"] = "Detected >=16GB VRAM. Suitable for GPT-2 medium and larger."

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU readiness probe")
    parser.add_argument("--json", action="store_true", help="JSON output for tooling")
    args = parser.parse_args()

    report = _device_report()

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("NightmareNet GPU Readiness Report")
    print("=" * 50)
    print(f"Python:   {report['python']}")
    print(f"Platform: {report['platform']}")
    torch_info = report.get("torch", {})
    if not torch_info.get("installed"):
        print(f"PyTorch:  NOT INSTALLED ({torch_info.get('error')})")
        return 1
    print(f"PyTorch:  {torch_info['version']}  (compiled with CUDA {torch_info['cuda_compiled']})")
    print(f"CUDA available: {torch_info['cuda_available']}")
    if torch_info["cuda_available"]:
        dev = report["device"]
        print(f"Device:   {dev['name']}  (CC {dev['compute_capability']})")
        print(f"Memory:   {dev['free_memory_mb']}/{dev['total_memory_mb']} MB free")
        print(f"Tier:     {report['tier']}")
        print(
            f"Mixed precision: fp16={report['mixed_precision']['fp16_supported']}, "
            f"bf16={report['mixed_precision']['bf16_supported']}"
        )
    print()
    print("Recommendation:")
    print(f"  {report.get('recommendation', 'OK')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
