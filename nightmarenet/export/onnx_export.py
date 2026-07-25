"""ONNX export functionality."""

import logging
import os
from importlib.util import find_spec

import torch

from nightmarenet.export.utils import unwrap_output

logger = logging.getLogger(__name__)


def check_dependencies() -> None:
    missing = []

    if find_spec("onnx") is None:
        missing.append("onnx")

    if find_spec("onnxruntime") is None:
        missing.append("onnxruntime")

    if missing:
        raise ImportError(
            "Optional export dependencies are missing. "
            "Install them with: pip install 'nightmarenet[export]' "
            f"(missing: {', '.join(missing)})"
        )


def export_to_onnx(
    model: torch.nn.Module, output_path: str, dummy_input: dict, opset: int = 14
) -> None:
    """Export a PyTorch model to ONNX format."""
    check_dependencies()

    import numpy as np
    import onnx
    import onnxruntime

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    model.eval()

    dynamic_axes = {}
    for key in dummy_input.keys():
        dynamic_axes[key] = {0: "batch_size", 1: "sequence_length"}

    input_names = list(dummy_input.keys())
    output_names = ["output"]
    dynamic_axes["output"] = {0: "batch_size", 1: "sequence_length"}

    # Optional wrapper to extract tensor from dict/dataclass output
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, *args, **kwargs):
            out = self.m(*args, **kwargs)
            return unwrap_output(out)

    wrapped_model = ONNXWrapper(model)
    wrapped_model.eval()

    try:
        logger.info("Exporting model to %s (opset=%s)", output_path, opset)
        torch.onnx.export(
            wrapped_model,
            tuple(dummy_input.values()),
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    except Exception as e:
        logger.error("Failed to export ONNX model: %s", e)
        raise RuntimeError(f"ONNX export failed: {e}") from e

    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX checker validation passed.")
    except Exception as e:
        logger.error("ONNX validation failed: %s", e)
        raise RuntimeError(f"ONNX validation failed: {e}") from e

    logger.info("Running output validation...")
    ort_session = onnxruntime.InferenceSession(output_path)
    ort_inputs = {k: v.cpu().numpy() for k, v in dummy_input.items()}

    ort_outs = ort_session.run(None, ort_inputs)
    ort_out = ort_outs[0]

    with torch.no_grad():
        pt_outs = wrapped_model(*tuple(dummy_input.values()))
        pt_out = pt_outs.cpu().numpy()

    max_diff = np.max(np.abs(pt_out - ort_out))
    logger.info("Maximum absolute difference: %s", max_diff)

    if max_diff >= 1e-5:
        raise RuntimeError(f"Output tolerance exceeded. Max diff: {max_diff}")

    logger.info("ONNX export and validation completed successfully.")
