"""Tests for dream and nightmare dataset generators."""

import pytest
from datasets import Dataset

from nightmarenet.data.generator import (
    DreamDatasetGenerator,
    NightmareDatasetGenerator,
    create_generators_from_config,
)


def _make_sample_dataset(n=50):
    """Create a small sample dataset for testing."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Paris is the capital of France and a major European city.",
        "Deep learning uses neural networks with many layers.",
        "Natural language processing enables machines to understand text.",
    ]
    # Repeat to get n samples
    data = [texts[i % len(texts)] for i in range(n)]
    return Dataset.from_dict({"text": data})


class TestDreamDatasetGenerator:
    """Test DreamDatasetGenerator."""

    def test_generate_produces_dataset(self):
        dataset = _make_sample_dataset(20)
        gen = DreamDatasetGenerator(strength=0.25, seed=42)
        result = gen.generate(dataset)
        assert isinstance(result, Dataset)
        assert len(result) == len(dataset)

    def test_generate_modifies_text(self):
        dataset = _make_sample_dataset(20)
        gen = DreamDatasetGenerator(strength=0.5, seed=42)
        result = gen.generate(dataset)
        # At least some texts should be modified
        original_texts = dataset["text"]
        generated_texts = result["text"]
        differences = sum(1 for o, g in zip(original_texts, generated_texts) if o != g)
        # With strength=0.5 and seed=42, expect at least some modifications
        assert differences > 0, "Expected some texts to be modified at strength=0.5"

    def test_generate_preserves_schema(self):
        dataset = _make_sample_dataset(10)
        gen = DreamDatasetGenerator(strength=0.25, seed=42)
        result = gen.generate(dataset)
        assert list(result.column_names) == list(dataset.column_names)

    def test_generate_with_config(self):
        dataset = _make_sample_dataset(10)
        config = {"text": {"char_swap": 1.0, "word_shuffle": 0.0}}
        gen = DreamDatasetGenerator(strength=0.3, config=config, seed=42)
        result = gen.generate(dataset)
        assert len(result) == 10

    def test_generate_with_custom_text_column(self):
        data = Dataset.from_dict({"content": ["Hello world.", "Test data."]})
        gen = DreamDatasetGenerator(strength=0.3, text_column="content", seed=42)
        result = gen.generate(data)
        assert "content" in result.column_names

    def test_generate_handles_empty_text(self):
        data = Dataset.from_dict({"text": ["", "Hello world.", ""]})
        gen = DreamDatasetGenerator(strength=0.3, seed=42)
        result = gen.generate(data)
        assert len(result) == 3

    def test_generate_and_save(self, tmp_path):
        dataset = _make_sample_dataset(10)
        gen = DreamDatasetGenerator(strength=0.25, seed=42)
        result = gen.generate_and_save(dataset, str(tmp_path))
        assert (tmp_path / "dream").exists()
        assert isinstance(result, Dataset)


class TestNightmareDatasetGenerator:
    """Test NightmareDatasetGenerator."""

    def test_generate_produces_dataset(self):
        dataset = _make_sample_dataset(20)
        gen = NightmareDatasetGenerator(strength=0.8, seed=42)
        result = gen.generate(dataset)
        assert isinstance(result, Dataset)
        assert len(result) == len(dataset)

    def test_generate_applies_stronger_distortions(self):
        dataset = _make_sample_dataset(20)
        dream_gen = DreamDatasetGenerator(strength=0.2, seed=42)
        nightmare_gen = NightmareDatasetGenerator(strength=0.8, seed=42)

        dream_result = dream_gen.generate(dataset)
        nightmare_result = nightmare_gen.generate(dataset)

        # Both should produce datasets
        assert isinstance(dream_result, Dataset)
        assert isinstance(nightmare_result, Dataset)
        assert len(dream_result) == len(nightmare_result)

    def test_generate_preserves_schema(self):
        dataset = _make_sample_dataset(10)
        gen = NightmareDatasetGenerator(strength=0.8, seed=42)
        result = gen.generate(dataset)
        assert list(result.column_names) == list(dataset.column_names)

    def test_generate_handles_empty_text(self):
        data = Dataset.from_dict({"text": ["", "Hello world.", ""]})
        gen = NightmareDatasetGenerator(strength=0.8, seed=42)
        result = gen.generate(data)
        assert len(result) == 3

    def test_generate_and_save(self, tmp_path):
        dataset = _make_sample_dataset(10)
        gen = NightmareDatasetGenerator(strength=0.8, seed=42)
        result = gen.generate_and_save(dataset, str(tmp_path))
        assert (tmp_path / "nightmare").exists()
        assert isinstance(result, Dataset)

    def test_uniform_schedule_backward_compatible(self):
        """Test that uniform schedule produces identical behavior to default."""
        dataset = _make_sample_dataset(10)
        gen_default = NightmareDatasetGenerator(strength=0.8, seed=42)
        gen_uniform = NightmareDatasetGenerator(strength=0.8, seed=42, strength_schedule="uniform")

        result_default = gen_default.generate(dataset)
        result_uniform = gen_uniform.generate(dataset)

        assert list(result_default["text"]) == list(result_uniform["text"])

    def test_linear_schedule_produces_varying_strengths(self):
        """Test that linear schedule produces different per-sample distortions."""
        dataset = _make_sample_dataset(20)
        gen = NightmareDatasetGenerator(
            strength=0.8,
            seed=42,
            strength_schedule="linear",
            strength_min=0.3,
            strength_max=0.9,
        )
        result = gen.generate(dataset)

        # Verify strengths are computed correctly
        strengths = gen._compute_strengths(20)
        assert len(strengths) == 20
        assert strengths[0] == pytest.approx(0.3)  # min at start
        assert strengths[-1] == pytest.approx(0.9)  # max at end
        # For 20 samples, index 10 is at position 10/19 ≈ 0.526
        expected_mid = 0.3 + (10 / 19) * (0.9 - 0.3)
        assert strengths[10] == pytest.approx(expected_mid)

        # Verify dataset was generated
        assert len(result) == 20

    def test_cosine_schedule_produces_varying_strengths(self):
        """Test that cosine schedule produces different per-sample distortions."""
        dataset = _make_sample_dataset(20)
        gen = NightmareDatasetGenerator(
            strength=0.8,
            seed=42,
            strength_schedule="cosine",
            strength_min=0.3,
            strength_max=0.9,
        )
        result = gen.generate(dataset)

        # Verify strengths are computed correctly
        strengths = gen._compute_strengths(20)
        assert len(strengths) == 20
        assert strengths[0] == pytest.approx(0.3)  # min at start
        assert strengths[-1] == pytest.approx(0.9)  # max at end
        # Cosine should be different from linear at midpoint
        linear_mid = 0.3 + 0.5 * (0.9 - 0.3)
        assert strengths[10] != pytest.approx(linear_mid)

        # Verify dataset was generated
        assert len(result) == 20

    def test_step_schedule_produces_varying_strengths(self):
        """Test that step schedule produces discrete jumps."""
        dataset = _make_sample_dataset(20)
        gen = NightmareDatasetGenerator(
            strength=0.8,
            seed=42,
            strength_schedule="step",
            strength_min=0.3,
            strength_max=0.9,
        )
        result = gen.generate(dataset)

        # Verify strengths are computed correctly
        strengths = gen._compute_strengths(20)
        assert len(strengths) == 20
        # First half should be min
        assert all(s == 0.3 for s in strengths[:10])
        # Second half should be max
        assert all(s == 0.9 for s in strengths[10:])

        # Verify dataset was generated
        assert len(result) == 20

    def test_invalid_schedule_raises_error(self):
        """Test that invalid schedule raises ValueError."""
        with pytest.raises(ValueError, match="Invalid strength_schedule"):
            NightmareDatasetGenerator(strength=0.8, seed=42, strength_schedule="invalid")

    def test_schedule_determinism(self):
        """Test that each schedule variant is deterministic with same seed."""
        dataset = _make_sample_dataset(10)

        for schedule in ["uniform", "linear", "cosine", "step"]:
            gen1 = NightmareDatasetGenerator(
                strength=0.8,
                seed=42,
                strength_schedule=schedule,
                strength_min=0.3,
                strength_max=0.9,
            )
            gen2 = NightmareDatasetGenerator(
                strength=0.8,
                seed=42,
                strength_schedule=schedule,
                strength_min=0.3,
                strength_max=0.9,
            )

            result1 = gen1.generate(dataset)
            result2 = gen2.generate(dataset)

            assert list(result1["text"]) == list(result2["text"])

    def test_linear_schedule_lower_start_higher_end(self):
        """Test that linear schedule produces lower strength at start, higher at end."""
        gen = NightmareDatasetGenerator(
            strength=0.8,
            seed=42,
            strength_schedule="linear",
            strength_min=0.3,
            strength_max=0.9,
        )

        strengths = gen._compute_strengths(10)
        assert strengths[0] < strengths[-1]
        assert strengths[0] == pytest.approx(0.3)
        assert strengths[-1] == pytest.approx(0.9)


class TestCreateGeneratorsFromConfig:
    """Test the config-based generator factory."""

    def test_creates_both_generators(self):
        config = {
            "distortion": {
                "dream_strength": 0.25,
                "nightmare_strength": 0.8,
            },
            "dataset": {"text_column": "text"},
            "seed": 42,
        }
        dream_gen, nightmare_gen = create_generators_from_config(config)
        assert isinstance(dream_gen, DreamDatasetGenerator)
        assert isinstance(nightmare_gen, NightmareDatasetGenerator)
        assert dream_gen.strength == 0.25
        assert nightmare_gen.strength == 0.8

    def test_default_config(self):
        config = {}
        dream_gen, nightmare_gen = create_generators_from_config(config)
        assert isinstance(dream_gen, DreamDatasetGenerator)
        assert isinstance(nightmare_gen, NightmareDatasetGenerator)

    def test_config_with_strength_schedule(self):
        """Test that config reads strength_schedule parameters."""
        config = {
            "distortion": {
                "dream_strength": 0.25,
                "nightmare_strength": 0.8,
                "strength_schedule": "linear",
                "strength_min": 0.3,
                "strength_max": 0.9,
            },
            "dataset": {"text_column": "text"},
            "seed": 42,
        }
        dream_gen, nightmare_gen = create_generators_from_config(config)
        assert isinstance(dream_gen, DreamDatasetGenerator)
        assert isinstance(nightmare_gen, NightmareDatasetGenerator)
        assert nightmare_gen.strength_schedule == "linear"
        assert nightmare_gen.strength_min == 0.3
        assert nightmare_gen.strength_max == 0.9
