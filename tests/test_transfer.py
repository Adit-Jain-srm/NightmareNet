"""Tests for robustness transfer learning modules."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from nightmarenet.transfer.measurement import calculate_transfer_ratio, evaluate_transfer_efficiency
from nightmarenet.transfer.registry import FoundationRegistry
from nightmarenet.transfer.report import generate_transfer_report


def test_calculate_transfer_ratio():
    assert calculate_transfer_ratio(0.8, 1.0) == 0.8
    assert calculate_transfer_ratio(0.5, 0.0) == 0.0  # Safe division by zero
    assert calculate_transfer_ratio(0.65, 0.8) == 0.8125


def test_evaluate_transfer_efficiency():
    assert evaluate_transfer_efficiency(0.8) == "Highly Efficient (Saves 70%+ of compute)"
    assert evaluate_transfer_efficiency(0.5) == "Moderately Efficient (Partial transfer)"
    assert evaluate_transfer_efficiency(0.2) == "Weak (Full cycle still needed)"


def test_generate_transfer_report():
    report = generate_transfer_report(0.7, 0.9, 0.85, 0.88, 120.0, 600.0)
    assert "Highly Efficient" in report
    assert "0.7778" in report  # ratio = 0.7/0.9
    assert "80.0%" in report  # savings = (600-120)/600
    assert "ratio exceeds 0.6" in report


def test_registry_initialization(tmp_path):
    registry = FoundationRegistry(cache_dir=tmp_path)
    assert registry.cache_dir == tmp_path
    assert tmp_path.exists()


@patch("nightmarenet.transfer.registry.AutoModel.from_pretrained")
@patch("nightmarenet.transfer.registry.AutoTokenizer.from_pretrained")
def test_registry_registration_and_load(mock_tokenizer, mock_automodel, tmp_path):
    mock_model = MagicMock()
    mock_automodel.return_value = mock_model
    mock_tok = MagicMock()
    mock_tokenizer.return_value = mock_tok

    registry = FoundationRegistry(cache_dir=tmp_path)

    # Test register
    dest = registry.register("dummy_path", "test_foundation", metadata={"robustness": 0.9})
    assert dest.exists()
    assert (dest / "nightmarenet_meta.json").exists()
    mock_model.save_pretrained.assert_called_once_with(dest)
    mock_tok.save_pretrained.assert_called_once_with(dest)

    # Touch a config.json to mock huggingface output so list_models finds it
    (dest / "config.json").touch()

    # Test list_models
    models = registry.list_models()
    assert "test_foundation" in models

    # Test load
    backbone, tokenizer, meta = registry.load("test_foundation")
    assert meta.get("robustness") == 0.9
    mock_automodel.assert_called_with(dest)
    mock_tokenizer.assert_called_with(dest)


@patch("nightmarenet.transfer.registry.AutoModel.from_pretrained")
@patch("nightmarenet.transfer.registry.AutoTokenizer.from_pretrained")
@patch("nightmarenet.transfer.head_factory.AutoModelForSequenceClassification.from_pretrained")
def test_transfer_pipeline_integration(mock_head_cls, mock_tokenizer, mock_automodel, tmp_path):
    import torch

    from nightmarenet.transfer.fine_tune import TransferFineTuner

    registry = FoundationRegistry(cache_dir=tmp_path)

    # 1. Register foundation
    mock_base = MagicMock()
    mock_automodel.return_value = mock_base
    dest = registry.register("dummy_path", "int_foundation")

    # 2. Load foundation via head_factory
    mock_head = MagicMock()
    mock_head_cls.return_value = mock_head

    from nightmarenet.transfer.head_factory import create_transfer_model

    model = create_transfer_model(str(dest), task_type="seq_classification", num_labels=2)
    assert model == mock_head
    mock_head_cls.assert_called_with(str(dest), num_labels=2)

    # 3. Fine Tune (mocked dataloader and optimizer)
    from torch.utils.data import DataLoader
    from transformers import default_data_collator

    dummy_data = [
        {
            "input_ids": torch.zeros((1, 128), dtype=torch.long),
            "attention_mask": torch.ones((1, 128), dtype=torch.long),
            "labels": torch.zeros(1, dtype=torch.long),
        }
    ]
    dataloader = DataLoader(dummy_data, batch_size=1, collate_fn=default_data_collator)

    # Mock model outputs to have a dummy loss
    mock_output = MagicMock()
    mock_output.loss = torch.tensor(0.5, requires_grad=True)
    mock_head.return_value = mock_output

    # Provide a simple architecture for freezing
    mock_layer = MagicMock()
    mock_layer.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
    mock_head.base_model.encoder.layer = [mock_layer, mock_layer]

    device = torch.device("cpu")
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=3e-5)
    tuner = TransferFineTuner(mock_head, optimizer, device)

    metrics = tuner.run(dataloader, num_epochs=1, freeze_bottom_n=1, strict_layer_freezing=True)

    assert "avg_loss" in metrics
    assert "per_epoch_losses" in metrics
    assert metrics["layers_frozen"] is True
