"""Tests for GLUE evaluation module."""

from unittest.mock import MagicMock, patch

import torch

from nightmarenet.evaluation.glue import (
    GLUE_TASKS,
    _tokenize_glue_task,
    evaluate_glue,
    evaluate_glue_task,
)


class TestGlueTasks:
    """Test GLUE task registry and metadata."""

    def test_all_tasks_have_required_keys(self):
        required = {"dataset", "subset", "input_columns", "label_column", "num_labels", "metric"}
        for name, cfg in GLUE_TASKS.items():
            assert required.issubset(cfg.keys()), f"Task '{name}' missing keys"

    def test_known_tasks_exist(self):
        assert "sst2" in GLUE_TASKS
        assert "mrpc" in GLUE_TASKS
        assert "qnli" in GLUE_TASKS
        assert "rte" in GLUE_TASKS
        assert "cola" in GLUE_TASKS

    def test_input_columns_are_lists(self):
        for name, cfg in GLUE_TASKS.items():
            assert isinstance(cfg["input_columns"], list), f"Task '{name}'"
            assert len(cfg["input_columns"]) in (1, 2), f"Task '{name}'"


class TestTokenizeGlueTask:
    """Test GLUE tokenization helper."""

    def test_single_sentence_task(self):
        from datasets import Dataset

        ds = Dataset.from_dict(
            {
                "sentence": ["Hello world", "Test sentence"],
                "label": [0, 1],
            }
        )
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "attention_mask": [[1, 1, 1], [1, 1, 1]],
        }
        # Mock the map to simulate tokenization
        result = _tokenize_glue_task(ds, tokenizer, ["sentence"], max_length=16)
        assert "labels" in result.column_names  # label→labels rename

    def test_sentence_pair_task(self):
        from datasets import Dataset

        ds = Dataset.from_dict(
            {
                "sentence1": ["Hello", "Test"],
                "sentence2": ["World", "Data"],
                "label": [0, 1],
            }
        )
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "attention_mask": [[1, 1, 1], [1, 1, 1]],
        }
        result = _tokenize_glue_task(ds, tokenizer, ["sentence1", "sentence2"], max_length=16)
        assert "labels" in result.column_names


class TestEvaluateGlueTask:
    """Test single-task GLUE evaluation."""

    def test_unknown_task_returns_error(self):
        model = MagicMock()
        tokenizer = MagicMock()
        result = evaluate_glue_task(model, tokenizer, "nonexistent_task")
        assert "error" in result

    @patch("datasets.load_dataset")
    def test_evaluate_sst2_mock(self, mock_load):
        """Mocked end-to-end for SST-2."""
        from datasets import Dataset

        val_ds = Dataset.from_dict(
            {
                "sentence": ["good movie", "bad movie", "great film", "terrible show"],
                "label": [1, 0, 1, 0],
            }
        )
        mock_load.return_value = {"validation": val_ds}

        # Create a mock model that returns logits
        model = MagicMock()
        logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.9, 0.1]])
        model.return_value = MagicMock(logits=logits)
        model.eval = MagicMock()

        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": [[1, 2]] * 4,
            "attention_mask": [[1, 1]] * 4,
        }
        tokenizer.pad_token = "[PAD]"

        result = evaluate_glue_task(
            model, tokenizer, "sst2", device="cpu", max_length=16, batch_size=4
        )
        assert result["task"] == "sst2"
        assert "accuracy" in result
        assert "f1_weighted" in result
        assert result["primary_metric"] == "accuracy"


class TestEvaluateGlue:
    """Test multi-task GLUE evaluation."""

    @patch("nightmarenet.evaluation.glue.evaluate_glue_task")
    def test_evaluates_all_tasks(self, mock_eval):
        mock_eval.return_value = {
            "task": "sst2",
            "accuracy": 0.85,
            "f1_weighted": 0.84,
            "primary_metric": "accuracy",
            "primary_score": 0.85,
            "num_samples": 100,
        }
        results = evaluate_glue(MagicMock(), MagicMock(), tasks=["sst2", "mrpc"])
        assert "sst2" in results
        assert "mrpc" in results
        assert "average" in results
        assert results["average"]["tasks_evaluated"] == 2

    @patch("nightmarenet.evaluation.glue.evaluate_glue_task")
    def test_handles_failed_tasks(self, mock_eval):
        def side_effect(model, tokenizer, task_name, **kwargs):
            if task_name == "mrpc":
                return {"task": "mrpc", "error": "some error"}
            return {
                "task": task_name,
                "accuracy": 0.9,
                "primary_score": 0.9,
                "primary_metric": "accuracy",
                "num_samples": 50,
            }

        mock_eval.side_effect = side_effect
        results = evaluate_glue(MagicMock(), MagicMock(), tasks=["sst2", "mrpc"])
        assert results["average"]["tasks_evaluated"] == 1
        assert results["average"]["tasks_failed"] == 1

    @patch("nightmarenet.evaluation.glue.evaluate_glue_task")
    def test_default_tasks(self, mock_eval):
        mock_eval.return_value = {
            "task": "x",
            "primary_score": 0.5,
            "num_samples": 10,
        }
        results = evaluate_glue(MagicMock(), MagicMock())
        assert results["average"]["tasks_evaluated"] == len(GLUE_TASKS)
