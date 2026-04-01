"""Phase alternation scheduling for the sleep-inspired training pipeline.

Controls the sequence and timing of Wake → Dream → Nightmare → Compress cycles.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

from nightmarenet.utils.validation import validate_positive_int

logger = logging.getLogger(__name__)


class CyclicScheduler:
    """Runs wake → dream → nightmare → compress cycles.

    Manages the phase ordering and provides iteration over the
    full training schedule.

    Args:
        num_cycles: Number of full sleep cycles to run.
        wake_epochs: Epochs per wake phase.
        dream_epochs: Epochs per dream phase.
        nightmare_epochs: Epochs per nightmare phase.
        compression_rounds: Number of compression rounds per cycle.
    """

    PHASE_ORDER = ["wake", "dream", "nightmare", "compress"]

    def __init__(
        self,
        num_cycles: int = 3,
        wake_epochs: int = 3,
        dream_epochs: int = 2,
        nightmare_epochs: int = 1,
        compression_rounds: int = 1,
    ):
        validate_positive_int(num_cycles, "num_cycles")
        validate_positive_int(wake_epochs, "wake_epochs", allow_zero=True)
        validate_positive_int(dream_epochs, "dream_epochs", allow_zero=True)
        validate_positive_int(nightmare_epochs, "nightmare_epochs", allow_zero=True)
        validate_positive_int(compression_rounds, "compression_rounds", allow_zero=True)
        self.num_cycles = num_cycles
        self.wake_epochs = wake_epochs
        self.dream_epochs = dream_epochs
        self.nightmare_epochs = nightmare_epochs
        self.compression_rounds = compression_rounds
        self._current_cycle = 0
        self._current_phase_idx = 0

    @property
    def current_cycle(self) -> int:
        """Return the current cycle number (0-indexed)."""
        return self._current_cycle

    @property
    def current_phase(self) -> str:
        """Return the name of the current phase."""
        return self.PHASE_ORDER[self._current_phase_idx]

    def get_epochs_for_phase(self, phase: str) -> int:
        """Return the number of epochs for a given phase.

        Args:
            phase: Phase name ("wake", "dream", "nightmare", "compress").

        Returns:
            Number of epochs/rounds for the phase.
        """
        return {
            "wake": self.wake_epochs,
            "dream": self.dream_epochs,
            "nightmare": self.nightmare_epochs,
            "compress": self.compression_rounds,
        }.get(phase, 1)

    def __iter__(self) -> Iterator[tuple[int, str, int]]:
        """Iterate over all (cycle, phase, epochs) tuples in the schedule.

        Yields:
            Tuple of (cycle_number, phase_name, num_epochs).
        """
        for cycle in range(self.num_cycles):
            self._current_cycle = cycle
            for phase_idx, phase in enumerate(self.PHASE_ORDER):
                self._current_phase_idx = phase_idx
                epochs = self.get_epochs_for_phase(phase)
                logger.info(
                    "Schedule: Cycle %d/%d - Phase: %s (%d epochs/rounds)",
                    cycle + 1,
                    self.num_cycles,
                    phase,
                    epochs,
                )
                yield cycle, phase, epochs

    def __len__(self) -> int:
        """Return total number of phase executions."""
        return self.num_cycles * len(self.PHASE_ORDER)

    def summary(self) -> str:
        """Return a human-readable summary of the training schedule."""
        lines = [f"Training Schedule: {self.num_cycles} cycles"]
        for cycle in range(self.num_cycles):
            lines.append(f"  Cycle {cycle + 1}:")
            for phase in self.PHASE_ORDER:
                epochs = self.get_epochs_for_phase(phase)
                lines.append(f"    {phase}: {epochs} epochs/rounds")
        return "\n".join(lines)


class AdaptiveScheduler:
    """Adaptive scheduler that adjusts phase durations based on validation loss.

    Extends the base cyclic schedule by monitoring validation metrics
    and dynamically adjusting epoch counts.

    Args:
        base_scheduler: A CyclicScheduler instance to adapt.
        patience: Number of phases without improvement before adjusting.
        adjustment_factor: Factor to increase/decrease epochs by.
    """

    def __init__(
        self,
        base_scheduler: Optional[CyclicScheduler] = None,
        patience: int = 2,
        adjustment_factor: float = 0.5,
        max_epochs: int = 50,
    ):
        self.base_scheduler = base_scheduler or CyclicScheduler()
        self.patience = patience
        self.adjustment_factor = adjustment_factor
        self.max_epochs = max_epochs
        self._loss_history: list[dict] = []
        self._no_improvement_count = 0
        self._best_loss = float("inf")

    def update(self, phase: str, loss: float) -> None:
        """Update the scheduler with the latest validation loss.

        Args:
            phase: The phase that just completed.
            loss: The validation loss after this phase.
        """
        self._loss_history.append({"phase": phase, "loss": loss})

        if loss < self._best_loss:
            self._best_loss = loss
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1

        if self._no_improvement_count >= self.patience:
            # Increase dream and nightmare epochs to provide more training signal
            self.base_scheduler.dream_epochs = min(
                self.max_epochs,
                max(
                    1,
                    int(
                        self.base_scheduler.dream_epochs
                        * (1 + self.adjustment_factor)
                    ),
                ),
            )
            self.base_scheduler.nightmare_epochs = min(
                self.max_epochs,
                max(
                    1,
                    int(
                        self.base_scheduler.nightmare_epochs
                        * (1 + self.adjustment_factor)
                    ),
                ),
            )
            logger.info(
                "Adaptive scheduler: increased dream_epochs=%d, nightmare_epochs=%d",
                self.base_scheduler.dream_epochs,
                self.base_scheduler.nightmare_epochs,
            )
            self._no_improvement_count = 0

    def __iter__(self) -> Iterator[tuple[int, str, int]]:
        """Delegate iteration to the base scheduler."""
        return iter(self.base_scheduler)

    def __len__(self) -> int:
        """Delegate length to the base scheduler."""
        return len(self.base_scheduler)


def create_scheduler_from_config(config: dict) -> CyclicScheduler:
    """Create a CyclicScheduler from a configuration dictionary.

    Args:
        config: Full configuration dictionary.

    Returns:
        Configured CyclicScheduler instance.
    """
    training_config = config.get("training", {})
    return CyclicScheduler(
        num_cycles=training_config.get("num_cycles", 3),
        wake_epochs=training_config.get("wake_epochs", 3),
        dream_epochs=training_config.get("dream_epochs", 2),
        nightmare_epochs=training_config.get("nightmare_epochs", 1),
        compression_rounds=training_config.get("compression_rounds", 1),
    )
