"""Edge-case and error-path tests for NightmareNet.

Covers validation boundaries, config errors, distortion corner cases,
generator edge conditions, scheduler limits, and integration flows.
"""

from __future__ import annotations

import random

import pytest
from datasets import Dataset

from nightmarenet.data.generator import (
    DreamDatasetGenerator,
    NightmareDatasetGenerator,
    create_generators_from_config,
)
from nightmarenet.distortions.adversarial import apply_adversarial_distortions
from nightmarenet.distortions.semantic import apply_semantic_distortions
from nightmarenet.distortions.text import apply_text_distortions
from nightmarenet.training.scheduler import (
    AdaptiveScheduler,
    CyclicScheduler,
)
from nightmarenet.utils.config import (
    DEFAULT_CONFIG,
    _deep_merge,
    load_config,
    validate_config,
)
from nightmarenet.utils.logging_config import reset_logging, setup_logging
from nightmarenet.utils.validation import (
    validate_config_keys,
    validate_dataloader,
    validate_dataset_columns,
    validate_non_empty_dataset,
    validate_positive_float,
    validate_positive_int,
    validate_ratio,
    validate_strength,
    validate_text,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = 42

SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog. Paris is the capital of France."

UNICODE_TEXT = "日本語テスト 🎉🚀 Ñoño café résumé naïve"

LONG_TEXT = " ".join(["word"] * 1200)


def _make_dataset(texts: list[str], col: str = "text") -> Dataset:
    return Dataset.from_dict({col: texts})


# ---------------------------------------------------------------------------
# 1. Validation Edge Cases
# ---------------------------------------------------------------------------


class TestValidation:
    """Edge cases for every validator in nightmarenet.utils.validation."""

    # -- validate_strength ---------------------------------------------------

    def test_strength_none(self):
        with pytest.raises(TypeError):
            validate_strength(None)

    def test_strength_string(self):
        with pytest.raises(TypeError):
            validate_strength("0.5")

    def test_strength_bool(self):
        """bool is a subclass of int, but the validator should still work."""
        # True == 1, which is a valid int/float; validate_strength accepts int/float
        # bool IS instance of (int, float), so True (1) is valid, False (0) is valid
        assert validate_strength(True) == 1.0
        assert validate_strength(False) == 0.0

    def test_strength_below_range(self):
        with pytest.raises(ValueError):
            validate_strength(-0.1)

    def test_strength_above_range(self):
        with pytest.raises(ValueError):
            validate_strength(1.1)

    def test_strength_lower_boundary(self):
        assert validate_strength(0.0) == 0.0

    def test_strength_upper_boundary(self):
        assert validate_strength(1.0) == 1.0

    # -- validate_positive_int -----------------------------------------------

    def test_positive_int_float_rejected(self):
        with pytest.raises(TypeError):
            validate_positive_int(1.5)

    def test_positive_int_negative(self):
        with pytest.raises(ValueError):
            validate_positive_int(-1)

    def test_positive_int_zero_disallowed(self):
        with pytest.raises(ValueError):
            validate_positive_int(0)

    def test_positive_int_zero_allowed(self):
        assert validate_positive_int(0, allow_zero=True) == 0

    def test_positive_int_bool_rejected(self):
        with pytest.raises(TypeError):
            validate_positive_int(True)

    # -- validate_positive_float ---------------------------------------------

    def test_positive_float_string_rejected(self):
        with pytest.raises(TypeError):
            validate_positive_float("1.0")

    def test_positive_float_negative(self):
        with pytest.raises(ValueError):
            validate_positive_float(-0.5)

    def test_positive_float_zero_disallowed(self):
        with pytest.raises(ValueError):
            validate_positive_float(0.0)

    def test_positive_float_zero_allowed(self):
        assert validate_positive_float(0.0, allow_zero=True) == 0.0

    def test_positive_float_bool_rejected(self):
        with pytest.raises(TypeError):
            validate_positive_float(True)

    # -- validate_ratio ------------------------------------------------------

    def test_ratio_below_range(self):
        with pytest.raises(ValueError):
            validate_ratio(-0.1)

    def test_ratio_upper_exclusive(self):
        with pytest.raises(ValueError):
            validate_ratio(1.0)

    def test_ratio_lower_boundary(self):
        assert validate_ratio(0.0) == 0.0

    def test_ratio_just_below_one(self):
        assert validate_ratio(0.999) == 0.999

    # -- validate_text -------------------------------------------------------

    def test_text_none(self):
        with pytest.raises(TypeError):
            validate_text(None)

    def test_text_int(self):
        with pytest.raises(TypeError):
            validate_text(42)

    def test_text_empty_allowed(self):
        assert validate_text("") == ""

    def test_text_empty_disallowed(self):
        with pytest.raises(ValueError):
            validate_text("", allow_empty=False)

    def test_text_whitespace_only_disallowed(self):
        with pytest.raises(ValueError):
            validate_text("   ", allow_empty=False)

    # -- validate_dataset_columns -------------------------------------------

    def test_dataset_columns_missing(self):
        ds = _make_dataset(["a", "b"])
        with pytest.raises(ValueError, match="missing required columns"):
            validate_dataset_columns(ds, ["text", "nonexistent"])

    def test_dataset_columns_no_attr(self):
        with pytest.raises(AttributeError):
            validate_dataset_columns("not a dataset", ["text"])

    # -- validate_non_empty_dataset -----------------------------------------

    def test_non_empty_dataset_empty(self):
        ds = Dataset.from_dict({"text": []})
        with pytest.raises(ValueError, match="empty"):
            validate_non_empty_dataset(ds)

    # -- validate_config_keys -----------------------------------------------

    def test_config_keys_missing(self):
        with pytest.raises(ValueError, match="missing required keys"):
            validate_config_keys({"a": 1}, ["a", "b"])

    def test_config_keys_non_dict(self):
        with pytest.raises(TypeError):
            validate_config_keys("not a dict", ["a"])

    # -- validate_dataloader ------------------------------------------------

    def test_dataloader_none(self):
        with pytest.raises(ValueError):
            validate_dataloader(None)


# ---------------------------------------------------------------------------
# 2. Config Edge Cases
# ---------------------------------------------------------------------------


class TestConfig:
    """Edge cases for config loading, validation, and merging."""

    def test_load_config_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_validate_config_invalid_type(self):
        cfg = _deep_merge(DEFAULT_CONFIG, {"model": {"max_length": "not_int"}})
        errors = validate_config(cfg)
        assert any("max_length" in e for e in errors)

    def test_validate_config_out_of_range(self):
        cfg = _deep_merge(DEFAULT_CONFIG, {"training": {"num_cycles": 0}})
        errors = validate_config(cfg)
        assert any("num_cycles" in e for e in errors)

    def test_deep_merge_nested_overrides(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99, "z": 100}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 99, "z": 100}, "b": 3}

    def test_deep_merge_does_not_mutate_base(self):
        base = {"a": {"x": 1}}
        _deep_merge(base, {"a": {"x": 2}})
        assert base["a"]["x"] == 1

    def test_validate_config_default_is_valid(self):
        errors = validate_config(DEFAULT_CONFIG)
        assert errors == []


# ---------------------------------------------------------------------------
# 3. Distortion Boundary Cases
# ---------------------------------------------------------------------------


class TestDistortionEdgeCases:
    """Edge cases for text, semantic, and adversarial distortions."""

    # -- strength=0.0 should preserve text (or very close) ------------------

    def test_text_distortion_zero_strength(self):
        random.seed(SEED)
        result = apply_text_distortions(SAMPLE_TEXT, strength=0.0)
        assert result == SAMPLE_TEXT

    def test_semantic_distortion_zero_strength(self):
        random.seed(SEED)
        result = apply_semantic_distortions(SAMPLE_TEXT, strength=0.0)
        assert result == SAMPLE_TEXT

    def test_adversarial_distortion_zero_strength(self):
        random.seed(SEED)
        result = apply_adversarial_distortions(SAMPLE_TEXT, strength=0.0)
        assert result == SAMPLE_TEXT

    # -- whitespace and minimal inputs --------------------------------------

    def test_text_distortion_whitespace_only(self):
        result = apply_text_distortions("   ", strength=0.5)
        assert result == "   "

    def test_semantic_distortion_single_word(self):
        result = apply_semantic_distortions("hello", strength=0.5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_adversarial_distortion_single_sentence(self):
        result = apply_adversarial_distortions("One sentence only", strength=0.5)
        assert isinstance(result, str)
        assert len(result) > 0

    # -- Unicode inputs -----------------------------------------------------

    def test_text_distortion_unicode(self):
        result = apply_text_distortions(UNICODE_TEXT, strength=0.3)
        assert isinstance(result, str)

    def test_semantic_distortion_unicode(self):
        result = apply_semantic_distortions(UNICODE_TEXT, strength=0.3)
        assert isinstance(result, str)

    def test_adversarial_distortion_unicode(self):
        result = apply_adversarial_distortions(UNICODE_TEXT, strength=0.3)
        assert isinstance(result, str)

    # -- Very long text -----------------------------------------------------

    def test_text_distortion_long_text(self):
        result = apply_text_distortions(LONG_TEXT, strength=0.3)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_semantic_distortion_long_text(self):
        result = apply_semantic_distortions(LONG_TEXT, strength=0.3)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_adversarial_distortion_long_text(self):
        result = apply_adversarial_distortions(LONG_TEXT, strength=0.3)
        assert isinstance(result, str)
        assert len(result) > 0

    # -- Invalid strength raises via validate_strength ----------------------

    def test_text_distortion_invalid_strength(self):
        with pytest.raises((TypeError, ValueError)):
            apply_text_distortions(SAMPLE_TEXT, strength="bad")

    def test_semantic_distortion_invalid_strength(self):
        with pytest.raises((TypeError, ValueError)):
            apply_semantic_distortions(SAMPLE_TEXT, strength=-1.0)

    def test_adversarial_distortion_invalid_strength(self):
        with pytest.raises((TypeError, ValueError)):
            apply_adversarial_distortions(SAMPLE_TEXT, strength=2.0)

    # -- Empty string -------------------------------------------------------

    def test_text_distortion_empty_string(self):
        assert apply_text_distortions("", strength=0.5) == ""

    def test_semantic_distortion_empty_string(self):
        assert apply_semantic_distortions("", strength=0.5) == ""

    def test_adversarial_distortion_empty_string(self):
        assert apply_adversarial_distortions("", strength=0.5) == ""


# ---------------------------------------------------------------------------
# 4. Generator Edge Cases
# ---------------------------------------------------------------------------


class TestGeneratorEdgeCases:
    """Edge cases for DreamDatasetGenerator and NightmareDatasetGenerator."""

    def test_dream_generator_strength_zero(self):
        gen = DreamDatasetGenerator(strength=0.0, seed=SEED)
        ds = _make_dataset(["Hello world. This is a test."])
        result = gen.generate(ds)
        assert len(result) == 1
        assert isinstance(result[0]["text"], str)

    def test_nightmare_generator_strength_one(self):
        gen = NightmareDatasetGenerator(strength=1.0, seed=SEED)
        ds = _make_dataset(["The quick brown fox jumps over the lazy dog."])
        result = gen.generate(ds)
        assert len(result) == 1
        assert isinstance(result[0]["text"], str)

    def test_generator_missing_text_column(self):
        gen = DreamDatasetGenerator(text_column="missing_col", seed=SEED)
        ds = _make_dataset(["hello"])
        with pytest.raises((ValueError, KeyError)):
            gen.generate(ds)

    def test_generator_single_sample_dataset(self):
        gen = DreamDatasetGenerator(strength=0.3, seed=SEED)
        ds = _make_dataset(["Single sample text for testing."])
        result = gen.generate(ds)
        assert len(result) == 1

    def test_nightmare_generator_single_sample(self):
        gen = NightmareDatasetGenerator(strength=0.8, seed=SEED)
        ds = _make_dataset(["Only one example here."])
        result = gen.generate(ds)
        assert len(result) == 1

    def test_create_generators_from_config_returns_both(self):
        dream_gen, nightmare_gen = create_generators_from_config(DEFAULT_CONFIG)
        assert isinstance(dream_gen, DreamDatasetGenerator)
        assert isinstance(nightmare_gen, NightmareDatasetGenerator)


# ---------------------------------------------------------------------------
# 5. Scheduler Edge Cases
# ---------------------------------------------------------------------------


class TestSchedulerEdgeCases:
    """Edge cases for CyclicScheduler and AdaptiveScheduler."""

    def test_cyclic_scheduler_min_cycles(self):
        sched = CyclicScheduler(num_cycles=1)
        phases = list(sched)
        assert len(phases) == 4
        assert all(p[0] == 0 for p in phases)

    def test_cyclic_scheduler_all_zero_except_wake(self):
        sched = CyclicScheduler(
            num_cycles=1,
            wake_epochs=2,
            dream_epochs=0,
            nightmare_epochs=0,
            compression_rounds=0,
        )
        phases = list(sched)
        assert len(phases) == 4
        epoch_counts = {name: epochs for _, name, epochs in phases}
        assert epoch_counts["wake"] == 2
        assert epoch_counts["dream"] == 0
        assert epoch_counts["nightmare"] == 0
        assert epoch_counts["compress"] == 0

    def test_adaptive_scheduler_no_updates(self):
        adaptive = AdaptiveScheduler()
        # Iterating without any update calls should work
        phases = list(adaptive)
        assert len(phases) == len(adaptive)

    def test_adaptive_scheduler_single_update(self):
        adaptive = AdaptiveScheduler(patience=2)
        adaptive.update("wake", 1.0)
        # No adaptation should happen after single update
        assert adaptive.base_scheduler.dream_epochs == 2

    def test_cyclic_scheduler_zero_num_cycles_rejected(self):
        with pytest.raises(ValueError):
            CyclicScheduler(num_cycles=0)


# ---------------------------------------------------------------------------
# 6. Logging Edge Cases
# ---------------------------------------------------------------------------


class TestLogging:
    """Edge cases for setup_logging and reset_logging."""

    def test_setup_logging_idempotent(self):
        reset_logging()
        setup_logging(console=True, file_logging=False)
        setup_logging(console=True, file_logging=False)  # second call is no-op
        reset_logging()

    def test_reset_logging(self):
        reset_logging()
        setup_logging(console=True, file_logging=False)
        reset_logging()
        # After reset, setup_logging should work again
        setup_logging(console=True, file_logging=False)
        reset_logging()


# ---------------------------------------------------------------------------
# 7. Integration Test
# ---------------------------------------------------------------------------


class TestIntegration:
    """Full pipeline: config → generators → dream data → nightmare data."""

    def test_full_pipeline(self):
        dream_gen, nightmare_gen = create_generators_from_config(DEFAULT_CONFIG)

        base_ds = _make_dataset(
            [
                "Machine learning transforms data into insight.",
                "Neural networks approximate complex functions.",
                "Deep learning is a subset of machine learning.",
                "Natural language processing handles human language.",
                "Transformers revolutionized the NLP landscape.",
            ]
        )

        dream_ds = dream_gen.generate(base_ds)
        assert len(dream_ds) == len(base_ds)
        assert "text" in dream_ds.column_names

        nightmare_ds = nightmare_gen.generate(base_ds)
        assert len(nightmare_ds) == len(base_ds)
        assert "text" in nightmare_ds.column_names

        # Dream and nightmare outputs should differ from each other
        dream_texts = dream_ds["text"]
        nightmare_texts = nightmare_ds["text"]
        assert any(d != n for d, n in zip(dream_texts, nightmare_texts))
