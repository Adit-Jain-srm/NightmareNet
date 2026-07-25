"""Tests for the IBM ART adapter module.

Tests use mock objects so ART and PyTorch are NOT required to run the
import-guard and validation tests. Tests that exercise actual ART
functionality are skipped when ART is not installed.
"""

from __future__ import annotations

from dataclasses import fields
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Detect whether ART is available for integration tests
# ---------------------------------------------------------------------------
try:
    import art  # noqa: F401

    HAS_ART = True
except ImportError:
    HAS_ART = False

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Import-guard tests (always run, even without ART)
# ---------------------------------------------------------------------------


class TestARTImportGuard:
    """Verify graceful errors when ART is not installed."""

    def test_check_art_available_raises_with_install_hint(self):
        """Importing the guard with ART missing should raise ImportError with install hint."""
        with patch.dict("sys.modules", {"art": None}):
            # Re-import to get fresh state
            from nightmarenet.evaluation.art_adapter import _check_art_available

            with pytest.raises(ImportError, match="pip install 'nightmarenet\\[art\\]'"):
                _check_art_available()

    def test_supported_attacks_dictionary_contains_expected_keys(self):
        from nightmarenet.evaluation.art_adapter import SUPPORTED_ATTACKS

        assert "pgd" in SUPPORTED_ATTACKS
        assert "fgsm" in SUPPORTED_ATTACKS
        assert "cw" in SUPPORTED_ATTACKS

    def test_art_attack_result_dataclass_fields(self):
        from nightmarenet.evaluation.art_adapter import ARTAttackResult

        field_names = {f.name for f in fields(ARTAttackResult)}
        assert "attack_name" in field_names
        assert "success_rate" in field_names
        assert "mean_perturbation" in field_names
        assert "median_perturbation" in field_names
        assert "query_count" in field_names
        assert "elapsed_seconds" in field_names
        assert "adversarial_examples" in field_names

    def test_invalid_attack_name_raises_value_error(self):
        from nightmarenet.evaluation.art_adapter import _build_attack

        with pytest.raises(ValueError, match="Unsupported attack"):
            _build_attack(MagicMock(), "nonexistent_attack")


# ---------------------------------------------------------------------------
# Mock-based functional tests (no real ART needed)
# ---------------------------------------------------------------------------


class TestARTAdapterMocked:
    """Tests using mocked ART modules to verify adapter logic."""

    def test_run_art_attack_returns_valid_result(self):
        """Verify run_art_attack returns correct ARTAttackResult with mocked attack."""
        from nightmarenet.evaluation.art_adapter import ARTAttackResult

        # Mock the entire ART import chain
        mock_art = MagicMock()
        mock_pgd_class = MagicMock()

        n_samples = 10
        n_classes = 3
        input_shape = (3, 4, 4)
        x = np.random.rand(n_samples, *input_shape).astype(np.float32)
        y = np.eye(n_classes)[np.random.randint(0, n_classes, n_samples)]

        # Mock attack.generate returns slightly perturbed inputs
        x_adv = x + np.random.normal(0, 0.01, x.shape).astype(np.float32)
        mock_pgd_instance = MagicMock()
        mock_pgd_instance.generate.return_value = x_adv
        mock_pgd_class.return_value = mock_pgd_instance

        # Mock classifier.predict returns valid logits
        mock_classifier = MagicMock()
        clean_logits = np.random.rand(n_samples, n_classes).astype(np.float32)
        adv_logits = np.random.rand(n_samples, n_classes).astype(np.float32)
        mock_classifier.predict.side_effect = [clean_logits, adv_logits]

        with patch.dict("sys.modules", {"art": mock_art}):
            with patch(
                "nightmarenet.evaluation.art_adapter._build_attack",
                return_value=mock_pgd_instance,
            ):
                from nightmarenet.evaluation.art_adapter import run_art_attack

                result = run_art_attack(mock_classifier, "pgd", x, y)

        assert isinstance(result, ARTAttackResult)
        assert result.attack_name == "pgd"
        assert 0.0 <= result.success_rate <= 1.0
        assert result.mean_perturbation >= 0.0
        assert result.median_perturbation >= 0.0
        assert result.query_count > 0
        assert result.elapsed_seconds >= 0.0
        assert result.adversarial_examples is not None

    def test_run_art_benchmark_returns_list_of_results(self):
        """Verify run_art_benchmark returns one result per attack."""
        from nightmarenet.evaluation.art_adapter import ARTAttackResult

        mock_art = MagicMock()

        n_samples = 5
        n_classes = 2
        input_shape = (1, 8, 8)
        x = np.random.rand(n_samples, *input_shape).astype(np.float32)
        y = np.eye(n_classes)[np.random.randint(0, n_classes, n_samples)]

        mock_attack_instance = MagicMock()
        mock_attack_instance.generate.return_value = x + 0.01

        mock_classifier = MagicMock()
        # predict is called twice per attack (clean + adversarial) * 3 attacks = 6 calls
        logits = np.random.rand(n_samples, n_classes).astype(np.float32)
        mock_classifier.predict.return_value = logits

        with patch.dict("sys.modules", {"art": mock_art}):
            with patch(
                "nightmarenet.evaluation.art_adapter._build_attack",
                return_value=mock_attack_instance,
            ):
                from nightmarenet.evaluation.art_adapter import run_art_benchmark

                results = run_art_benchmark(
                    mock_classifier, x, y, attacks=["pgd", "fgsm", "cw"]
                )

        assert len(results) == 3
        assert all(isinstance(r, ARTAttackResult) for r in results)
        attack_names = [r.attack_name for r in results]
        assert "pgd" in attack_names
        assert "fgsm" in attack_names
        assert "cw" in attack_names

    def test_run_art_attack_with_integer_labels(self):
        """Verify integer (non-one-hot) labels are handled correctly."""
        mock_art = MagicMock()

        n_samples = 8
        n_classes = 4
        input_shape = (3, 4, 4)
        x = np.random.rand(n_samples, *input_shape).astype(np.float32)
        y = np.random.randint(0, n_classes, n_samples)

        mock_attack_instance = MagicMock()
        mock_attack_instance.generate.return_value = x + 0.01

        mock_classifier = MagicMock()
        logits = np.random.rand(n_samples, n_classes).astype(np.float32)
        mock_classifier.predict.return_value = logits

        with patch.dict("sys.modules", {"art": mock_art}):
            with patch(
                "nightmarenet.evaluation.art_adapter._build_attack",
                return_value=mock_attack_instance,
            ):
                from nightmarenet.evaluation.art_adapter import run_art_attack

                result = run_art_attack(mock_classifier, "fgsm", x, y)

        assert result.attack_name == "fgsm"
        assert 0.0 <= result.success_rate <= 1.0

    def test_run_art_attack_zero_correct_predictions(self):
        """When the model classifies nothing correctly, success_rate should be 0."""
        mock_art = MagicMock()

        n_samples = 5
        n_classes = 3
        input_shape = (1, 4, 4)
        x = np.random.rand(n_samples, *input_shape).astype(np.float32)
        y = np.zeros(n_samples, dtype=int)  # all label 0

        mock_attack_instance = MagicMock()
        mock_attack_instance.generate.return_value = x

        mock_classifier = MagicMock()
        # Predict class 1 for everything (never matches label 0)
        wrong_logits = np.zeros((n_samples, n_classes), dtype=np.float32)
        wrong_logits[:, 1] = 1.0
        mock_classifier.predict.return_value = wrong_logits

        with patch.dict("sys.modules", {"art": mock_art}):
            with patch(
                "nightmarenet.evaluation.art_adapter._build_attack",
                return_value=mock_attack_instance,
            ):
                from nightmarenet.evaluation.art_adapter import run_art_attack

                result = run_art_attack(mock_classifier, "pgd", x, y)

        assert result.success_rate == 0.0


# ---------------------------------------------------------------------------
# Integration tests (skipped when ART / torch not installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_ART or not HAS_TORCH, reason="ART and/or PyTorch not installed")
class TestARTAdapterIntegration:
    """Integration tests that use real ART + PyTorch."""

    def _make_simple_model(self):
        """Create a tiny linear classifier for testing."""
        import torch

        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 4 * 4, 3),
        )
        model.eval()
        return model

    def test_nightmarenet_art_classifier_wraps_model(self):
        from nightmarenet.evaluation.art_adapter import NightmareNetARTClassifier

        model = self._make_simple_model()
        wrapper = NightmareNetARTClassifier(
            model=model,
            nb_classes=3,
            input_shape=(3, 4, 4),
        )
        assert wrapper.classifier is not None

    def test_from_nightmarenet_model_factory(self):
        from nightmarenet.evaluation.art_adapter import NightmareNetARTClassifier

        model = self._make_simple_model()
        wrapper = NightmareNetARTClassifier.from_nightmarenet_model(
            model=model,
            nb_classes=3,
            input_shape=(3, 4, 4),
        )
        assert wrapper.classifier is not None

    def test_run_fgsm_attack(self):
        from nightmarenet.evaluation.art_adapter import (
            NightmareNetARTClassifier,
            run_art_attack,
        )

        model = self._make_simple_model()
        wrapper = NightmareNetARTClassifier(
            model=model, nb_classes=3, input_shape=(3, 4, 4)
        )

        x = np.random.rand(8, 3, 4, 4).astype(np.float32)
        y = np.random.randint(0, 3, 8)

        result = run_art_attack(wrapper.classifier, "fgsm", x, y, eps=0.1)

        assert result.attack_name == "fgsm"
        assert 0.0 <= result.success_rate <= 1.0
        assert result.mean_perturbation >= 0.0
        assert result.adversarial_examples.shape == x.shape
