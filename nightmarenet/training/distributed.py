"""Distributed training utilities via HuggingFace Accelerate.

Provides a thin wrapper that enables multi-GPU / DDP training when
``accelerate`` is installed and ``training.distributed`` is ``True``
in the configuration.  Falls back transparently to single-device
training otherwise.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

_ACCELERATE_AVAILABLE: bool = False
Accelerator: Any = None
try:
    from accelerate import Accelerator  # type: ignore[no-redef]

    _ACCELERATE_AVAILABLE = True
except ImportError:
    pass


def is_accelerate_available() -> bool:
    """Return ``True`` if the ``accelerate`` library is importable."""
    return _ACCELERATE_AVAILABLE


class DistributedContext:
    """Lightweight wrapper around :class:`accelerate.Accelerator`.

    When ``enabled=True`` and accelerate is installed, calling
    :meth:`prepare` wraps the model, optimizer, and dataloaders for
    distributed training.  Otherwise, all methods are no-ops and the
    originals are returned unmodified.

    Args:
        enabled: Whether distributed training is requested.
        mixed_precision: AMP mode forwarded to Accelerator (``"no"``,
            ``"fp16"``, ``"bf16"``).
        gradient_accumulation_steps: Accumulation steps forwarded to
            Accelerator so it can normalise losses correctly.
    """

    def __init__(
        self,
        enabled: bool = False,
        mixed_precision: str = "no",
        gradient_accumulation_steps: int = 1,
    ) -> None:
        self.enabled = enabled and _ACCELERATE_AVAILABLE
        self._accelerator: Optional[Any] = None

        if enabled and not _ACCELERATE_AVAILABLE:
            logger.warning(
                "distributed=True but `accelerate` is not installed; "
                "falling back to single-device training."
            )
            self.enabled = False

        if self.enabled:
            accel = Accelerator(
                mixed_precision=mixed_precision,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            self._accelerator = accel
            logger.info(
                "Accelerate distributed context created "
                "(device=%s, num_processes=%d, mixed_precision=%s).",
                accel.device,
                accel.num_processes,
                accel.mixed_precision,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """The device managed by Accelerator, or a sensible default."""
        if self._accelerator is not None:
            return self._accelerator.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def is_main_process(self) -> bool:
        """``True`` on rank-0 (or always when distributed is disabled)."""
        if self._accelerator is not None:
            return self._accelerator.is_main_process
        return True

    @property
    def num_processes(self) -> int:
        """Total number of processes in the group."""
        if self._accelerator is not None:
            return self._accelerator.num_processes
        return 1

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def prepare(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *dataloaders: DataLoader,
    ) -> tuple:
        """Wrap model, optimizer, and dataloaders for DDP.

        Returns:
            Tuple of ``(model, optimizer, *dataloaders)`` — prepared for
            distributed training when enabled, or the originals otherwise.
        """
        if self._accelerator is not None:
            prepared = self._accelerator.prepare(model, optimizer, *dataloaders)
            logger.info("Model, optimizer, and %d dataloaders prepared for DDP.", len(dataloaders))
            return prepared
        return (model, optimizer, *dataloaders)

    def backward(self, loss: torch.Tensor) -> None:
        """Call ``accelerator.backward`` or plain ``loss.backward``."""
        if self._accelerator is not None:
            self._accelerator.backward(loss)
        else:
            loss.backward()

    def clip_grad_norm(
        self,
        parameters: Any,
        max_norm: float,
    ) -> None:
        """Clip gradients, using Accelerator when available."""
        if self._accelerator is not None:
            self._accelerator.clip_grad_norm_(parameters, max_norm)
        else:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def save_model(self, model: torch.nn.Module, path: str) -> None:
        """Save on rank-0 only."""
        if self.is_main_process:
            unwrapped: Any = self.unwrap_model(model)
            if hasattr(unwrapped, "save_pretrained"):
                unwrapped.save_pretrained(path)
            else:
                torch.save(unwrapped.state_dict(), path)

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Return the original model from a DDP-wrapped one."""
        if self._accelerator is not None:
            return self._accelerator.unwrap_model(model)
        return model

    def wait_for_everyone(self) -> None:
        """Barrier — blocks until all processes reach this point."""
        if self._accelerator is not None:
            self._accelerator.wait_for_everyone()

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print only on rank-0."""
        if self._accelerator is not None:
            self._accelerator.print(*args, **kwargs)
        else:
            print(*args, **kwargs)
