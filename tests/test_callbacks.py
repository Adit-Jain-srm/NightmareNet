"""Tests for training callback event system."""

import pytest
import torch

from nightmarenet.training.callbacks import CallbackManager, EventType, TrainingEvent


def test_event_progress_pct() -> None:
    event = TrainingEvent(
        event_type=EventType.STEP,
        phase="wake",
        step=25,
        total_steps=100,
    )
    assert event.progress_pct == 25.0


def test_callback_manager_emits_to_handlers() -> None:
    mgr = CallbackManager()
    received = []

    def handler(event: TrainingEvent) -> None:
        received.append(event.phase)

    mgr.on(EventType.STEP, handler)
    mgr.emit(TrainingEvent(event_type=EventType.STEP, phase="dream", step=1, total_steps=10))
    assert received == ["dream"]


def test_callback_manager_global_handler() -> None:
    mgr = CallbackManager()
    phases = []
    mgr.on_all(lambda e: phases.append(e.phase))
    mgr.emit(TrainingEvent(event_type=EventType.PHASE_START, phase="nightmare"))
    assert phases == ["nightmare"]


def test_event_to_dict() -> None:
    event = TrainingEvent(
        event_type=EventType.METRIC,
        phase="compress",
        metrics={"loss": 0.5},
    )
    d = event.to_dict()
    assert d["event_type"] == "metric"
    assert d["phase"] == "compress"
    assert d["metrics"]["loss"] == 0.5


def test_callback_manager_phase_start_event() -> None:
    mgr = CallbackManager()
    received = []

    def handler(event: TrainingEvent) -> None:
        received.append(event.event_type)

    mgr.on(EventType.PHASE_START, handler)

    mgr.emit(
        TrainingEvent(
            event_type=EventType.PHASE_START,
            phase="wake",
        )
    )

    assert received == [EventType.PHASE_START]


def test_callback_manager_phase_end_event() -> None:
    mgr = CallbackManager()
    received = []

    def handler(event: TrainingEvent) -> None:
        received.append(event.event_type)

    mgr.on(EventType.PHASE_END, handler)

    mgr.emit(
        TrainingEvent(
            event_type=EventType.PHASE_END,
            phase="wake",
        )
    )

    assert received == [EventType.PHASE_END]


def test_callback_manager_epoch_start_event() -> None:
    mgr = CallbackManager()
    received = []

    def handler(event: TrainingEvent) -> None:
        received.append(event.event_type)

    mgr.on(EventType.EPOCH_START, handler)

    mgr.emit(
        TrainingEvent(
            event_type=EventType.EPOCH_START,
            phase="wake",
            epoch=1,
        )
    )

    assert received == [EventType.EPOCH_START]


def test_callback_manager_epoch_end_event() -> None:
    mgr = CallbackManager()
    received = []

    def handler(event: TrainingEvent) -> None:
        received.append(event.event_type)

    mgr.on(EventType.EPOCH_END, handler)

    mgr.emit(
        TrainingEvent(
            event_type=EventType.EPOCH_END,
            phase="wake",
            epoch=1,
        )
    )

    assert received == [EventType.EPOCH_END]


def test_warmup_lr_caps_at_base_value():
    """Verify LR at step 0 < LR at step warmup_steps, and LR never exceeds base."""
    from torch.optim.lr_scheduler import LambdaLR

    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    warmup_steps = 10

    def warmup_lambda(current_step):
        return min(1.0, current_step / warmup_steps)

    scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # Step 0: LR should be 0 (0/10 = 0)
    lr_at_0 = scheduler.get_last_lr()[0]

    # Step through warmup
    for _ in range(warmup_steps):
        optimizer.step()
        scheduler.step()

    lr_at_warmup = scheduler.get_last_lr()[0]

    # Step well past warmup
    for _ in range(50):
        optimizer.step()
        scheduler.step()

    lr_past_warmup = scheduler.get_last_lr()[0]

    assert lr_at_0 < lr_at_warmup, "LR should increase during warmup"
    assert lr_at_warmup == pytest.approx(0.01), "LR should reach base_lr at warmup end"
    assert lr_past_warmup == pytest.approx(0.01), "LR must not exceed base_lr after warmup"
