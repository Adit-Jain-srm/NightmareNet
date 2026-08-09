"""Shared utilities for model export."""

from typing import Any


def unwrap_output(out: Any) -> Any:
    """Unwrap model output to extract the main tensor."""
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, dict):
        return out[list(out.keys())[0]]
    if isinstance(out, tuple):
        return out[0]
    return out
