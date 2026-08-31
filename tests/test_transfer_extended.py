"""Extended unit tests for nightmarenet.transfer modules.

Covers head_factory.py, measurement.py, report.py, registry.py, and config.py.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch.nn as nn

from nightmarenet.transfer.config import TransferConfig, load_config
from nightmarenet.transfer.head_factory import create_transfer_model
from nightmarenet.transfer.measurement import (
    calculate_transfer_ratio,
    evaluate_transfer_efficiency,
)
from nightmarenet.transfer.registry import FoundationRegistry, get_registry
from nightmarenet.transfer.report import generate_transfer_report


class DummyBackboneModule(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)

    def save_pretrained(self, save_directory: str | Path) -> None:
        p = Path(save_directory)
        p.mkdir(parents=True, exist_ok=True)
        (p / "config.json").write_text("{}", encoding="utf-8")


class DummyTokenizer:
    def save_pretrained(self, save_directory: str | Path) -> None:
        p = Path(save_directory)
        p.mkdir(parents=True, exist_ok=True)
        (p / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class TestTransferHeadFactory(unittest.TestCase):
    @mock.patch(
        "nightmarenet.transfer.head_factory.AutoModelForSequenceClassification.from_pretrained"
    )
    def test_create_seq_classification_head(self, mock_from_pretrained: mock.MagicMock) -> None:
        """Test sequence classification model creation with num_labels parameter."""
        mock_model = mock.MagicMock()
        mock_from_pretrained.return_value = mock_model

        model = create_transfer_model(
            "dummy/foundation/path", task_type="seq_classification", num_labels=5
        )
        self.assertIs(model, mock_model)
        mock_from_pretrained.assert_called_once_with("dummy/foundation/path", num_labels=5)

    @mock.patch(
        "nightmarenet.transfer.head_factory.AutoModelForSequenceClassification.from_pretrained"
    )
    def test_create_regression_head(self, mock_from_pretrained: mock.MagicMock) -> None:
        """Test single-output regression head creation with 1 label."""
        mock_model = mock.MagicMock()
        mock_from_pretrained.return_value = mock_model

        model = create_transfer_model(
            "dummy/foundation/path", task_type="seq_classification", num_labels=1
        )
        self.assertIs(model, mock_model)
        mock_from_pretrained.assert_called_once_with("dummy/foundation/path", num_labels=1)

    @mock.patch(
        "nightmarenet.transfer.head_factory.AutoModelForTokenClassification.from_pretrained"
    )
    def test_create_token_classification_head(self, mock_from_pretrained: mock.MagicMock) -> None:
        """Test token classification model creation with custom label mappings."""
        mock_model = mock.MagicMock()
        mock_from_pretrained.return_value = mock_model

        model = create_transfer_model(
            "dummy/foundation/path",
            task_type="token_classification",
            num_labels=9,
            id2label={0: "O", 1: "B-PER"},
        )
        self.assertIs(model, mock_model)
        mock_from_pretrained.assert_called_once_with(
            "dummy/foundation/path",
            num_labels=9,
            id2label={0: "O", 1: "B-PER"},
        )

    def test_create_transfer_model_unsupported_task_type(self) -> None:
        """Test exception raised for unsupported task types."""
        with self.assertRaises(ValueError) as ctx:
            create_transfer_model("dummy/path", task_type="qa_extraction")
        self.assertIn("Unsupported task_type 'qa_extraction'", str(ctx.exception))
        self.assertIn("Supported types: seq_classification", str(ctx.exception))


class TestTransferMeasurement(unittest.TestCase):
    def test_calculate_transfer_ratio_normal(self) -> None:
        """Test standard transfer ratio calculations."""
        self.assertAlmostEqual(calculate_transfer_ratio(0.8, 1.0), 0.8)
        self.assertAlmostEqual(calculate_transfer_ratio(0.45, 0.90), 0.5)
        self.assertAlmostEqual(calculate_transfer_ratio(1.2, 1.0), 1.2)

    def test_calculate_transfer_ratio_zero_and_negative_baseline(self) -> None:
        """Test division by zero and negative baseline protection."""
        self.assertEqual(calculate_transfer_ratio(0.8, 0.0), 0.0)
        self.assertEqual(calculate_transfer_ratio(0.8, -0.5), 0.0)
        self.assertEqual(calculate_transfer_ratio(0.0, 1.0), 0.0)

    def test_evaluate_transfer_efficiency_boundaries(self) -> None:
        """Test boundary conditions for transfer efficiency evaluations."""
        msg_high = "Highly Efficient (Saves 70%+ of compute)"
        msg_mod = "Moderately Efficient (Partial transfer)"
        msg_weak = "Weak (Full cycle still needed)"

        self.assertEqual(evaluate_transfer_efficiency(0.71), msg_high)
        self.assertEqual(evaluate_transfer_efficiency(0.70), msg_mod)
        self.assertEqual(evaluate_transfer_efficiency(0.50), msg_mod)
        self.assertEqual(evaluate_transfer_efficiency(0.31), msg_mod)
        self.assertEqual(evaluate_transfer_efficiency(0.30), msg_weak)
        self.assertEqual(evaluate_transfer_efficiency(0.0), msg_weak)
        self.assertEqual(evaluate_transfer_efficiency(-0.1), msg_weak)


class TestTransferReport(unittest.TestCase):
    def test_generate_transfer_report_high_ratio(self) -> None:
        """Test markdown report generation when transfer ratio exceeds 0.6."""
        report = generate_transfer_report(
            transferred_robustness=0.85,
            baseline_robustness=1.0,
            clean_accuracy_transferred=0.92,
            clean_accuracy_baseline=0.94,
            transferred_time_s=150.0,
            baseline_time_s=600.0,
        )
        self.assertIn("# Robustness Transfer Report", report)
        self.assertIn("## Summary", report)
        self.assertIn("## Detailed Metrics", report)
        self.assertIn("## Analysis", report)
        self.assertIn("| Robustness Score | 0.8500 | 1.0000 |", report)
        self.assertIn("| Clean Accuracy   | 0.9200 | 0.9400 |", report)
        self.assertIn("| Training Time    | 150.0s | 600.0s |", report)
        self.assertIn("Compute Savings**: 75.0%", report)
        self.assertIn("transfer ratio exceeds 0.6", report)

    def test_generate_transfer_report_low_ratio_and_zero_time(self) -> None:
        """Test markdown report when transfer ratio <= 0.6 and zero baseline time."""
        report = generate_transfer_report(
            transferred_robustness=0.4,
            baseline_robustness=1.0,
            clean_accuracy_transferred=0.75,
            clean_accuracy_baseline=0.90,
            transferred_time_s=300.0,
            baseline_time_s=0.0,
        )
        self.assertIn("transfer ratio is below 0.6", report)
        self.assertIn("Compute Savings**: 0.0%", report)

    def test_generate_transfer_report_negative_savings_clamping(self) -> None:
        """Test that compute savings does not drop below 0% if transferred time > baseline."""
        report = generate_transfer_report(
            transferred_robustness=0.7,
            baseline_robustness=1.0,
            clean_accuracy_transferred=0.8,
            clean_accuracy_baseline=0.8,
            transferred_time_s=800.0,
            baseline_time_s=400.0,
        )
        self.assertIn("Compute Savings**: 0.0%", report)


class TestFoundationRegistry(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @mock.patch("nightmarenet.transfer.registry.AutoModel.from_pretrained")
    @mock.patch("nightmarenet.transfer.registry.AutoTokenizer.from_pretrained")
    def test_register_and_load_model(
        self, mock_tokenizer_cls: mock.MagicMock, mock_automodel_cls: mock.MagicMock
    ) -> None:
        """Test model registration, metadata persistence, and loading."""
        dummy_model = DummyBackboneModule()
        dummy_tok = DummyTokenizer()
        mock_automodel_cls.return_value = dummy_model
        mock_tokenizer_cls.return_value = dummy_tok

        registry = FoundationRegistry(cache_dir=self.cache_path)
        dest = registry.register(
            "fake/path/model",
            name="hardened-bert",
            metadata={"robustness_score": 0.88, "task": "adversarial_pretraining"},
        )
        self.assertTrue(dest.exists())
        self.assertTrue((dest / "nightmarenet_meta.json").exists())

        loaded_model, loaded_tok, meta = registry.load("hardened-bert")
        self.assertEqual(meta["robustness_score"], 0.88)
        self.assertEqual(meta["task"], "adversarial_pretraining")

    def test_load_non_existent_model_raises_error(self) -> None:
        """Test FileNotFoundError is raised when loading an unknown foundation model."""
        registry = FoundationRegistry(cache_dir=self.cache_path)
        with self.assertRaises(FileNotFoundError) as ctx:
            registry.load("non_existent_backbone")
        self.assertIn("Foundation model 'non_existent_backbone' not found", str(ctx.exception))

    @mock.patch("nightmarenet.transfer.registry.AutoModel.from_pretrained")
    @mock.patch("nightmarenet.transfer.registry.AutoTokenizer.from_pretrained")
    def test_overwrite_existing_model_registration(
        self, mock_tokenizer_cls: mock.MagicMock, mock_automodel_cls: mock.MagicMock
    ) -> None:
        """Test registering with an existing model name overwrites previous metadata."""
        dummy_model = DummyBackboneModule()
        dummy_tok = DummyTokenizer()
        mock_automodel_cls.return_value = dummy_model
        mock_tokenizer_cls.return_value = dummy_tok

        registry = FoundationRegistry(cache_dir=self.cache_path)
        registry.register("fake/path/1", name="model-v1", metadata={"version": 1})
        registry.register("fake/path/2", name="model-v1", metadata={"version": 2})

        _, _, meta = registry.load("model-v1")
        self.assertEqual(meta["version"], 2)

    def test_list_models_filtering(self) -> None:
        """Test list_models returns only directories containing config.json."""
        registry = FoundationRegistry(cache_dir=self.cache_path)
        # Create valid model directory
        valid_model_dir = self.cache_path / "valid_model"
        valid_model_dir.mkdir()
        (valid_model_dir / "config.json").write_text("{}", encoding="utf-8")

        # Create invalid directory (no config.json)
        invalid_dir = self.cache_path / "scratch_folder"
        invalid_dir.mkdir()

        models = registry.list_models()
        self.assertEqual(models, ["valid_model"])

    def test_list_models_empty_cache(self) -> None:
        """Test list_models on empty cache."""
        registry = FoundationRegistry(cache_dir=self.cache_path / "sub_empty")
        self.assertEqual(registry.list_models(), [])

    def test_get_registry_singleton(self) -> None:
        """Test get_registry returns singleton instance for matching cache_dir."""
        reg1 = get_registry(self.cache_path)
        reg2 = get_registry(self.cache_path)
        self.assertIs(reg1, reg2)

        custom_path = self.cache_path / "custom"
        reg3 = get_registry(custom_path)
        self.assertIsNot(reg1, reg3)
        self.assertEqual(reg3.cache_dir, custom_path)


class TestTransferConfig(unittest.TestCase):
    def test_default_config_values(self) -> None:
        """Test default values of TransferConfig."""
        config = TransferConfig()
        self.assertEqual(config.task_type, "seq_classification")
        self.assertEqual(config.dataset, "sst2")
        self.assertEqual(config.num_labels, 2)
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.num_epochs, 3)
        self.assertEqual(config.learning_rate, 3e-5)
        self.assertFalse(config.strict_layer_freezing)

    def test_load_config_from_yaml(self) -> None:
        """Test load_config parses YAML file correctly."""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("task_type: token_classification\nnum_labels: 7\nbatch_size: 16\n")
            f_path = f.name

        try:
            config = load_config(f_path)
            self.assertEqual(config.task_type, "token_classification")
            self.assertEqual(config.num_labels, 7)
            self.assertEqual(config.batch_size, 16)
        finally:
            Path(f_path).unlink(missing_ok=True)

    def test_load_config_empty_yaml(self) -> None:
        """Test load_config returns default TransferConfig for empty YAML file."""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("")
            f_path = f.name

        try:
            config = load_config(f_path)
            self.assertEqual(config.task_type, "seq_classification")
            self.assertEqual(config.num_labels, 2)
        finally:
            Path(f_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
