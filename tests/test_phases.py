"""Tests for training phases and scheduler."""

from nightmarenet.training.scheduler import (
    AdaptiveScheduler,
    CyclicScheduler,
    create_scheduler_from_config,
)


class TestCyclicScheduler:
    """Test the CyclicScheduler."""

    def test_default_schedule(self):
        scheduler = CyclicScheduler()
        phases = list(scheduler)
        # 3 cycles × 4 phases = 12 phase executions
        assert len(phases) == 12

    def test_custom_schedule(self):
        scheduler = CyclicScheduler(
            num_cycles=2,
            wake_epochs=1,
            dream_epochs=1,
            nightmare_epochs=1,
            compression_rounds=1,
        )
        phases = list(scheduler)
        assert len(phases) == 8  # 2 cycles × 4 phases

    def test_phase_order(self):
        scheduler = CyclicScheduler(num_cycles=1)
        phases = list(scheduler)
        assert phases[0][1] == "wake"
        assert phases[1][1] == "dream"
        assert phases[2][1] == "nightmare"
        assert phases[3][1] == "compress"

    def test_epochs_per_phase(self):
        scheduler = CyclicScheduler(wake_epochs=3, dream_epochs=2, nightmare_epochs=1)
        assert scheduler.get_epochs_for_phase("wake") == 3
        assert scheduler.get_epochs_for_phase("dream") == 2
        assert scheduler.get_epochs_for_phase("nightmare") == 1
        assert scheduler.get_epochs_for_phase("compress") == 1

    def test_len(self):
        scheduler = CyclicScheduler(num_cycles=5)
        assert len(scheduler) == 20  # 5 × 4

    def test_summary(self):
        scheduler = CyclicScheduler(num_cycles=2)
        summary = scheduler.summary()
        assert "Cycle 1:" in summary
        assert "Cycle 2:" in summary
        assert "wake:" in summary
        assert "dream:" in summary
        assert "nightmare:" in summary
        assert "compress:" in summary

    def test_current_properties(self):
        scheduler = CyclicScheduler(num_cycles=1)
        for cycle, phase, _epochs in scheduler:
            assert scheduler.current_cycle == cycle
            assert scheduler.current_phase == phase


class TestAdaptiveScheduler:
    """Test the AdaptiveScheduler."""

    def test_wraps_cyclic_scheduler(self):
        base = CyclicScheduler(num_cycles=1)
        adaptive = AdaptiveScheduler(base_scheduler=base)
        phases = list(adaptive)
        assert len(phases) == 4

    def test_update_improves(self):
        base = CyclicScheduler(num_cycles=1, dream_epochs=2)
        adaptive = AdaptiveScheduler(base_scheduler=base, patience=2)

        original_dream = base.dream_epochs
        # Report improving losses
        adaptive.update("wake", 2.0)
        adaptive.update("dream", 1.5)

        # Epochs should not change with improvement
        assert base.dream_epochs == original_dream

    def test_update_stagnation_increases_epochs(self):
        base = CyclicScheduler(num_cycles=1, dream_epochs=2, nightmare_epochs=1)
        adaptive = AdaptiveScheduler(base_scheduler=base, patience=2, adjustment_factor=0.5)

        # Report stagnating losses
        adaptive.update("wake", 2.0)
        adaptive.update("dream", 2.5)  # Worse
        adaptive.update("nightmare", 3.0)  # Worse again → triggers adjustment

        # Epochs should increase
        assert base.dream_epochs >= 2
        assert base.nightmare_epochs >= 1

    def test_len(self):
        base = CyclicScheduler(num_cycles=3)
        adaptive = AdaptiveScheduler(base_scheduler=base)
        assert len(adaptive) == 12


class TestCreateSchedulerFromConfig:
    """Test the config-based scheduler factory."""

    def test_creates_from_config(self):
        config = {
            "training": {
                "num_cycles": 2,
                "wake_epochs": 5,
                "dream_epochs": 3,
                "nightmare_epochs": 2,
            }
        }
        scheduler = create_scheduler_from_config(config)
        assert isinstance(scheduler, CyclicScheduler)
        assert scheduler.num_cycles == 2
        assert scheduler.wake_epochs == 5
        assert scheduler.dream_epochs == 3
        assert scheduler.nightmare_epochs == 2

    def test_default_config(self):
        scheduler = create_scheduler_from_config({})
        assert isinstance(scheduler, CyclicScheduler)
        assert scheduler.num_cycles == 3

    def test_early_stopping_returns_adaptive(self):
        config = {
            "training": {
                "num_cycles": 2,
                "wake_epochs": 1,
                "dream_epochs": 1,
                "nightmare_epochs": 1,
                "early_stopping": True,
                "early_stopping_patience": 4,
                "early_stopping_min_delta": 0.001,
            }
        }
        scheduler = create_scheduler_from_config(config)
        assert isinstance(scheduler, AdaptiveScheduler)
        assert scheduler.early_stopping is True
        assert scheduler.early_stopping_patience == 4


class TestEarlyStopping:
    """Test early stopping in AdaptiveScheduler."""

    def test_no_stop_with_improvement(self):
        base = CyclicScheduler(num_cycles=1)
        adaptive = AdaptiveScheduler(
            base_scheduler=base,
            early_stopping=True,
            early_stopping_patience=2,
            early_stopping_min_delta=0.01,
        )
        adaptive.update("wake", 2.0)
        adaptive.update("dream", 1.5)
        adaptive.update("nightmare", 1.0)
        assert adaptive.should_stop is False

    def test_stop_on_plateau(self):
        base = CyclicScheduler(num_cycles=1)
        adaptive = AdaptiveScheduler(
            base_scheduler=base,
            early_stopping=True,
            early_stopping_patience=2,
            early_stopping_min_delta=0.1,
        )
        adaptive.update("wake", 2.0)
        adaptive.update("dream", 2.0)  # No improvement
        adaptive.update("nightmare", 2.0)  # Triggers stop (patience=2)
        assert adaptive.should_stop is True

    def test_stop_on_degradation(self):
        base = CyclicScheduler(num_cycles=1)
        adaptive = AdaptiveScheduler(
            base_scheduler=base,
            early_stopping=True,
            early_stopping_patience=3,
            early_stopping_min_delta=0.0,
        )
        adaptive.update("wake", 1.0)
        adaptive.update("dream", 1.5)  # Worse
        adaptive.update("nightmare", 2.0)  # Worse
        assert adaptive.should_stop is False  # patience=3, only 2 bad updates
        adaptive.update("compress", 2.5)  # Worse → triggers stop
        assert adaptive.should_stop is True

    def test_min_delta_sensitivity(self):
        base = CyclicScheduler(num_cycles=1)
        adaptive = AdaptiveScheduler(
            base_scheduler=base,
            early_stopping=True,
            early_stopping_patience=2,
            early_stopping_min_delta=0.5,
        )
        # Improve, but by less than min_delta
        adaptive.update("wake", 2.0)
        adaptive.update("dream", 1.8)  # Improved by 0.2 < delta 0.5
        adaptive.update("nightmare", 1.6)  # Still < delta from best
        assert adaptive.should_stop is True

    def test_disabled_early_stopping(self):
        base = CyclicScheduler(num_cycles=1)
        adaptive = AdaptiveScheduler(
            base_scheduler=base,
            early_stopping=False,
            early_stopping_patience=1,
        )
        adaptive.update("wake", 2.0)
        adaptive.update("dream", 5.0)
        assert adaptive.should_stop is False
