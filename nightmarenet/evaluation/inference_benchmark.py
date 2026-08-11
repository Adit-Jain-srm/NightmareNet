"""Inference performance benchmarking utilities."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any, Optional, Union

import torch


def _synchronize(device: torch.device) -> None:
    """Synchronize CUDA before/after timed inference when using a GPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _safe_model_name(model_name: str) -> str:
    """Convert a model path/name into a filesystem-safe identifier."""
    name = Path(model_name).name or model_name
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "model"


def _move_inputs_to_device(inputs: Any, device: torch.device) -> Any:
    """Move tensor inputs recursively to the benchmark device."""
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {key: _move_inputs_to_device(value, device) for key, value in inputs.items()}
    if isinstance(inputs, (list, tuple)):
        converted = [_move_inputs_to_device(value, device) for value in inputs]
        return type(inputs)(converted)
    return inputs

def _get_cpu_memory_mb() -> Optional[float]:
    """Return peak process memory in MB when available."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        # Linux reports KB; macOS reports bytes.
        if usage.ru_maxrss < 1024 * 1024:
            return usage.ru_maxrss / 1024.0
        return usage.ru_maxrss / (1024 * 1024)
    except (ImportError, AttributeError, OSError):
        return None
        
def _run_model(model: torch.nn.Module, inputs: Any) -> Any:
    """Run a model with either tensor or mapping inputs."""
    if isinstance(inputs, torch.Tensor):
        return model(inputs)
    return model(**inputs)


def _build_inputs(
    model: torch.nn.Module,
    tokenizer: Optional[Any],
    batch_size: int,
    max_length: int,
) -> Any:
    """Build deterministic inputs suitable for text or generic PyTorch models."""
    if tokenizer is not None:
        text = "The quick brown fox jumps over the lazy dog."
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError(
                    "Tokenizer has no pad_token or eos_token; cannot create padded benchmark inputs."
                )
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer(
            [text] * batch_size,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    return torch.randn(batch_size, 10)


@torch.inference_mode()
def benchmark_batch(
    model: torch.nn.Module,
    tokenizer: Optional[Any],
    device: torch.device,
    batch_size: int,
    warmup_iterations: int = 5,
    measurement_iterations: int = 20,
    max_length: int = 128,
) -> dict[str, Any]:
    """Measure latency, throughput, and peak memory for one batch size."""
    model.eval()

    inputs = _build_inputs(model, tokenizer, batch_size, max_length)
    inputs = _move_inputs_to_device(inputs, device)

    for _ in range(max(5, warmup_iterations)):
        _run_model(model, inputs)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    timings = []

    for _ in range(measurement_iterations):
        _synchronize(device)
        start = time.perf_counter()
        _run_model(model, inputs)
        _synchronize(device)
        timings.append(time.perf_counter() - start)

    average_latency_ms = mean(timings) * 1000.0
    throughput = batch_size / mean(timings)

    if device.type == "cuda":
        peak_memory_mb: Optional[float] = (
            torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        )
    else:
        peak_memory_mb = _get_cpu_memory_mb()

    return {
        "batch_size": batch_size,
        "average_latency_ms": average_latency_ms,
        "throughput_samples_per_sec": throughput,
        "peak_memory_mb": peak_memory_mb,
        "device": str(device),
        "warmup_iterations": max(5, warmup_iterations),
        "measurement_iterations": measurement_iterations,
    }


def run_benchmark(
    model: torch.nn.Module,
    tokenizer: Optional[Any],
    device: torch.device,
    batch_sizes: list[int],
    *,
    model_name: str,
    max_length: int = 128,
    warmup_iterations: int = 5,
    measurement_iterations: int = 20,
) -> dict[str, Any]:
    """Run inference benchmarks for all requested batch sizes."""
    if not batch_sizes:
        raise ValueError("At least one batch size is required.")

    if any(size <= 0 for size in batch_sizes):
        raise ValueError("Batch sizes must be positive integers.")

    results = [
        benchmark_batch(
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
            max_length=max_length,
        )
        for batch_size in batch_sizes
    ]

    return {
        "model": model_name,
        "device": str(device),
        "results": results,
    }


def save_results(
    results: dict[str, Any],
    model_name: str,
    output_dir: Union[str, Path] = "results",
) -> Path:
    """Save benchmark results to a timestamped JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"benchmark_{_safe_model_name(model_name)}_{timestamp}"

    for attempt in range(100):
        suffix = "" if attempt == 0 else f"_{attempt}"
        result_path = output_path / f"{base_name}{suffix}.json"

        try:
            with result_path.open("x", encoding="utf-8") as handle:
                json.dump(results, handle, indent=2)
            return result_path
        except FileExistsError:
            continue

    raise OSError("Could not create a unique benchmark result filename.")
