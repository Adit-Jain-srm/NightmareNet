"""Tests for compute_cost_analysis module."""

from scripts.compute_cost_analysis import (
    BASELINE_WAKE_EPOCHS,
    calculate_cycle_flops,
    calculate_phase_flops,
    calculate_total_flops,
    format_flops,
    get_flops_per_sample,
)


class TestGetFlopsPerSample:
    """Tests for get_flops_per_sample function."""

    def test_known_model_distilbert(self):
        """Test FLOPs lookup for DistilBERT."""
        flops = get_flops_per_sample("distilbert-base-uncased")
        assert flops == 2.5e12

    def test_known_model_distilgpt2(self):
        """Test FLOPs lookup for DistilGPT-2."""
        flops = get_flops_per_sample("distilgpt2")
        assert flops == 1.2e12

    def test_known_model_bert(self):
        """Test FLOPs lookup for BERT."""
        flops = get_flops_per_sample("bert-base-uncased")
        assert flops == 4.5e12

    def test_case_insensitive_match(self):
        """Test that model name matching is case-insensitive."""
        flops = get_flops_per_sample("DISTILBERT-BASE-UNCASED")
        assert flops == 2.5e12

    def test_fallback_model(self):
        """Test fallback for unknown model."""
        flops = get_flops_per_sample("unknown-model-name")
        assert flops == 2.5e12  # Default fallback

    def test_partial_name_match(self):
        """Test partial name matching."""
        flops = get_flops_per_sample("distilbert")
        assert flops == 2.5e12


class TestCalculatePhaseFlops:
    """Tests for calculate_phase_flops function."""

    def test_exact_calculation(self):
        """Test exact FLOP calculation for known inputs."""
        flops = calculate_phase_flops(
            model_name="distilbert-base-uncased",
            num_samples=1000,
            num_epochs=1,
            flops_per_sample=2.5e12,
        )
        assert flops == 2.5e12 * 1000 * 1

    def test_multiple_epochs(self):
        """Test FLOPs for multiple epochs."""
        flops = calculate_phase_flops(
            model_name="distilbert-base-uncased",
            num_samples=1000,
            num_epochs=3,
            flops_per_sample=2.5e12,
        )
        assert flops == 2.5e12 * 1000 * 3

    def test_zero_epochs(self):
        """Test FLOPs for zero epochs (should be zero)."""
        flops = calculate_phase_flops(
            model_name="distilbert-base-uncased",
            num_samples=1000,
            num_epochs=0,
            flops_per_sample=2.5e12,
        )
        assert flops == 0

    def test_zero_samples(self):
        """Test FLOPs for zero samples (should be zero)."""
        flops = calculate_phase_flops(
            model_name="distilbert-base-uncased",
            num_samples=0,
            num_epochs=1,
            flops_per_sample=2.5e12,
        )
        assert flops == 0

    def test_default_flops_per_sample(self):
        """Test that default FLOPs per sample is used when not provided."""
        flops = calculate_phase_flops(
            model_name="distilbert-base-uncased",
            num_samples=1000,
            num_epochs=1,
        )
        assert flops == 2.5e12 * 1000 * 1


class TestCalculateCycleFlops:
    """Tests for calculate_cycle_flops function."""

    def test_cycle_total_equals_phase_sum(self):
        """Test that cycle total equals sum of phases."""
        result = calculate_cycle_flops(
            model_name="distilbert-base-uncased",
            num_samples=500,
            wake_epochs=3,
            dream_epochs=2,
            nightmare_epochs=1,
            compression_rounds=1,
            flops_per_sample=2.5e12,
        )

        phase_sum = sum(result["phases"].values())
        assert result["total"] == phase_sum

    def test_phase_breakdown(self):
        """Test individual phase FLOPs breakdown."""
        result = calculate_cycle_flops(
            model_name="distilbert-base-uncased",
            num_samples=500,
            wake_epochs=3,
            dream_epochs=2,
            nightmare_epochs=1,
            compression_rounds=1,
            flops_per_sample=2.5e12,
        )

        # Wake: 2.5e12 * 500 * 3 = 3.75e15
        assert result["phases"]["wake"] == 3.75e15
        # Dream: 2.5e12 * 500 * 2 = 2.5e15
        assert result["phases"]["dream"] == 2.5e15
        # Nightmare: 2.5e12 * 500 * 1 = 1.25e15
        assert result["phases"]["nightmare"] == 1.25e15
        # Compress: 2.5e12 * 500 * 1 = 1.25e15
        assert result["phases"]["compress"] == 1.25e15

    def test_zero_epochs_in_phase(self):
        """Test that zero epochs in a phase gives zero FLOPs for that phase."""
        result = calculate_cycle_flops(
            model_name="distilbert-base-uncased",
            num_samples=500,
            wake_epochs=0,
            dream_epochs=2,
            nightmare_epochs=1,
            compression_rounds=1,
            flops_per_sample=2.5e12,
        )

        assert result["phases"]["wake"] == 0
        assert result["total"] == (
            result["phases"]["dream"] + result["phases"]["nightmare"] + result["phases"]["compress"]
        )


class TestCalculateTotalFlops:
    """Tests for calculate_total_flops function."""

    def test_total_flops_calculation(self):
        """Test total FLOPs across multiple cycles."""
        result = calculate_total_flops(
            model_name="distilbert-base-uncased",
            num_samples=500,
            num_cycles=3,
            wake_epochs=3,
            dream_epochs=2,
            nightmare_epochs=1,
            compression_rounds=1,
            flops_per_sample=2.5e12,
        )

        # Per cycle: 3.75e15 + 2.5e15 + 1.25e15 + 1.25e15 = 8.75e15
        # Total for 3 cycles: 8.75e15 * 3 = 2.625e16
        expected_cycle = 8.75e15
        expected_total = 2.625e16

        assert result["per_cycle"]["total"] == expected_cycle
        assert result["total"]["nightmarenet"] == expected_total

    def test_baseline_comparison(self):
        """Test baseline calculation (3 epoch FT per cycle)."""
        result = calculate_total_flops(
            model_name="distilbert-base-uncased",
            num_samples=500,
            num_cycles=1,
            wake_epochs=3,
            dream_epochs=2,
            nightmare_epochs=1,
            compression_rounds=1,
            flops_per_sample=2.5e12,
        )

        # Baseline: BASELINE_WAKE_EPOCHS wake epochs only = 2.5e12 * 500 * 3 = 3.75e15
        expected_baseline = 3.75e15
        assert result["total"]["baseline"] == expected_baseline
        assert result["comparison"]["nightmarenet_vs_baseline"] > 1

    def test_baseline_multi_cycle_scaling(self):
        """Test baseline scaling semantics when num_cycles > 1."""
        num_cycles = 4
        num_samples = 500
        result = calculate_total_flops(
            model_name="distilbert-base-uncased",
            num_samples=num_samples,
            num_cycles=num_cycles,
            wake_epochs=3,
            dream_epochs=2,
            nightmare_epochs=1,
            compression_rounds=1,
            flops_per_sample=2.5e12,
        )

        single_cycle_baseline = 2.5e12 * num_samples * BASELINE_WAKE_EPOCHS
        assert result["total"]["baseline"] == single_cycle_baseline * num_cycles

    def test_cycle_total_matches_sum(self):
        """Test that cycle_total == sum(phases)."""
        result = calculate_total_flops(
            model_name="distilbert-base-uncased",
            num_samples=500,
            num_cycles=3,
            wake_epochs=3,
            dream_epochs=2,
            nightmare_epochs=1,
            compression_rounds=1,
        )

        assert result["comparison"]["phases_match_cycle_total"] is True
        assert result["comparison"]["cycle_total"] == result["comparison"]["phase_sum"]

    def test_metadata_fields(self):
        """Test that metadata contains expected fields."""
        result = calculate_total_flops(
            model_name="distilbert-base-uncased",
            num_samples=500,
            num_cycles=3,
            wake_epochs=3,
            dream_epochs=2,
            nightmare_epochs=1,
            compression_rounds=1,
        )

        assert "model" in result["metadata"]
        assert "num_samples" in result["metadata"]
        assert "num_cycles" in result["metadata"]
        assert "flops_per_sample" in result["metadata"]

    def test_schedule_fields(self):
        """Test that schedule contains expected fields."""
        result = calculate_total_flops(
            model_name="distilbert-base-uncased",
            num_samples=500,
            num_cycles=3,
            wake_epochs=3,
            dream_epochs=2,
            nightmare_epochs=1,
            compression_rounds=1,
        )

        assert "wake_epochs" in result["schedule"]
        assert "dream_epochs" in result["schedule"]
        assert "nightmare_epochs" in result["schedule"]
        assert "compression_rounds" in result["schedule"]


class TestFormatFlops:
    """Tests for format_flops function."""

    def test_femto_flops(self):
        """Test formatting of very small FLOPs."""
        result = format_flops(1e6)
        assert "MFLOPs" in result

    def test_giga_flops(self):
        """Test formatting of giga FLOPs."""
        result = format_flops(1e12)
        assert "TFLOPs" in result

    def test_tera_flops(self):
        """Test formatting of tera FLOPs."""
        result = format_flops(1e15)
        assert "PFLOPs" in result

    def test_peta_flops(self):
        """Test formatting of peta FLOPs."""
        result = format_flops(1e18)
        assert "PFLOPs" in result

    def test_zero_flops(self):
        """Test formatting of zero FLOPs."""
        result = format_flops(0)
        assert "FLOPs" in result

    def test_default_formatting(self):
        """Test default formatting falls back to FLOPs."""
        result = format_flops(1e6)
        assert "1.00" in result
        assert "MFLOPs" in result
