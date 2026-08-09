"""TorchScript export functionality."""

import logging
import os

import numpy as np
import torch

from nightmarenet.export.utils import unwrap_output

logger = logging.getLogger(__name__)


def export_to_torchscript(model: torch.nn.Module, output_path: str, dummy_input: dict) -> None:
    """Export a PyTorch model to TorchScript format using tracing."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    model.eval()

    class TraceWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, *args):
            # Try to pass as kwargs if we can infer keys from dummy_input
            # otherwise pass as positional args
            if len(args) == len(dummy_input):
                kwargs = dict(zip(dummy_input.keys(), args))
                try:
                    out = self.m(**kwargs)
                except Exception:
                    out = self.m(*args)
            else:
                out = self.m(*args)

            return unwrap_output(out)

    wrapped_model = TraceWrapper(model)
    wrapped_model.eval()
    inputs_tuple = tuple(dummy_input.values())

    try:
        logger.info("Tracing model for TorchScript export to %s", output_path)
        traced_model = torch.jit.trace(wrapped_model, inputs_tuple, strict=False)
        traced_model.save(output_path)
    except Exception as e:
        logger.error("Failed to export TorchScript model: %s", e)
        raise RuntimeError(f"TorchScript export failed: {e}") from e

    logger.info("Running output validation...")
    loaded_model = torch.jit.load(output_path)
    loaded_model.eval()

    with torch.no_grad():
        ts_out = loaded_model(*inputs_tuple)
        if isinstance(ts_out, tuple):
            ts_out = ts_out[0].cpu().numpy()
        else:
            ts_out = ts_out.cpu().numpy()

        pt_outs = wrapped_model(*inputs_tuple)
        pt_out = pt_outs.cpu().numpy()

    max_diff = np.max(np.abs(pt_out - ts_out))
    logger.info("Maximum absolute difference: %s", max_diff)

    if max_diff >= 1e-5:
        raise RuntimeError(f"Output tolerance exceeded. Max diff: {max_diff}")

    logger.info("TorchScript export and validation completed successfully.")
