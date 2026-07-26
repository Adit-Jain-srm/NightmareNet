"""Training callbacks for event-driven pipeline communication.

Enables real-time streaming of training progress to UI, WebSocket,
and logging systems without coupling the training loop to any
specific transport.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class EventType(Enum):
    PHASE_START = "phase_start"
    PHASE_END = "phase_end"
    EPOCH_START = "epoch_start"
    EPOCH_END = "epoch_end"
    STEP = "step"
    METRIC = "metric"
    CHECKPOINT = "checkpoint"
    ERROR = "error"
    CANCEL = "cancel"


@dataclass
class TrainingEvent:
    """Immutable event emitted by the training loop."""

    event_type: EventType
    phase: str
    timestamp: float = field(default_factory=time.time)
    epoch: Optional[int] = None
    step: Optional[int] = None
    total_steps: Optional[int] = None
    metrics: Optional[Dict[str, float]] = None
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def progress_pct(self) -> float:
        if self.step is not None and self.total_steps and self.total_steps > 0:
            return (self.step / self.total_steps) * 100.0
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "phase": self.phase,
            "timestamp": self.timestamp,
            "epoch": self.epoch,
            "step": self.step,
            "total_steps": self.total_steps,
            "progress_pct": round(self.progress_pct, 1),
            "metrics": self.metrics,
            "message": self.message,
        }


EventHandler = Callable[[TrainingEvent], None]


class CallbackManager:
    """Manages event handlers for training loop communication.

    Usage:
        mgr = CallbackManager()
        mgr.on(EventType.STEP, lambda e: None)
        mgr.emit(TrainingEvent(event_type=EventType.STEP, phase="wake", step=10, total_steps=100))
    """

    def __init__(self) -> None:
        self._handlers: Dict[EventType, List[EventHandler]] = {}
        self._global_handlers: List[EventHandler] = []

    def on(self, event_type: EventType, handler: EventHandler) -> None:
        """Register a handler for a specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def on_all(self, handler: EventHandler) -> None:
        """Register a handler for all events."""
        self._global_handlers.append(handler)

    def emit(self, event: TrainingEvent) -> None:
        """Emit an event to all registered handlers."""
        for handler in self._global_handlers:
            try:
                handler(event)
            except Exception:
                pass

        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                pass

    def clear(self) -> None:
        """Remove all registered handlers."""
        self._handlers.clear()
        self._global_handlers.clear()
