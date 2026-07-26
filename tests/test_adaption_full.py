"""Tests for Adaption Labs dataset optimization — full SDK surface."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nightmarenet.data.adaption import (
    AdaptionOptimizer,
    _generate_idempotency_key,
    _validate_brand_controls,
    _validate_column_mapping,
)

# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidateColumnMapping:
    def test_rejects_empty_mapping(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_column_mapping({})

    def test_accepts_simple_mapping(self):
        _validate_column_mapping({"prompt": "text"})

    def test_accepts_context_as_list(self):
        _validate_column_mapping({"prompt": "text", "context": ["a", "b"]})

    def test_rejects_context_list_with_empty_string(self):
        with pytest.raises(ValueError, match="context list entries"):
            _validate_column_mapping(
                {"prompt": "text", "context": ["a", ""]},
            )

    def test_validates_against_dataset_columns(self):
        with pytest.raises(ValueError, match="does not exist"):
            _validate_column_mapping({"prompt": "missing"}, dataset_columns=["text", "label"])

    def test_context_list_validated_against_columns(self):
        with pytest.raises(ValueError, match="does not exist"):
            _validate_column_mapping(
                {"prompt": "text", "context": ["missing"]},
                dataset_columns=["text", "label"],
            )


class TestValidateBrandControls:
    def test_none_is_valid(self):
        _validate_brand_controls(None)

    def test_empty_dict_is_valid(self):
        _validate_brand_controls({})

    def test_valid_length(self):
        _validate_brand_controls({"length": "concise"})

    def test_invalid_length(self):
        with pytest.raises(ValueError, match="length must be one of"):
            _validate_brand_controls({"length": "super_long"})

    def test_safety_categories_must_be_list(self):
        with pytest.raises(ValueError, match="must be a list"):
            _validate_brand_controls({"safety_categories": "hate"})

    def test_valid_safety_categories(self):
        _validate_brand_controls({"safety_categories": ["hate", "harassment"]})

    def test_blueprint_must_be_string(self):
        with pytest.raises(ValueError, match="must be a string"):
            _validate_brand_controls({"blueprint": 123})

    def test_valid_blueprint(self):
        _validate_brand_controls({"blueprint": "Answer in formal English."})


class TestIdempotencyKey:
    def test_generates_unique_keys(self):
        k1 = _generate_idempotency_key()
        k2 = _generate_idempotency_key()
        assert k1 != k2
        assert k1.startswith("nn-")
        assert len(k1) == 19  # "nn-" + 16 hex chars


# ---------------------------------------------------------------------------
# AdaptionOptimizer — unavailable SDK
# ---------------------------------------------------------------------------


class TestAdaptionOptimizerUnavailable:
    def test_optimize_returns_none_without_sdk(self, monkeypatch):
        monkeypatch.delenv("ADAPTION_API_KEY", raising=False)
        with patch("nightmarenet.data.adaption.Adaption", None):
            optimizer = AdaptionOptimizer()
            result = optimizer.optimize_dataset(
                MagicMock(column_names=["text"]),
                {"prompt": "text"},
            )
            assert result is None

    def test_estimate_returns_none_without_key(self, monkeypatch):
        monkeypatch.delenv("ADAPTION_API_KEY", raising=False)
        optimizer = AdaptionOptimizer(api_key=None)
        result = optimizer.estimate_cost(
            MagicMock(column_names=["text"]),
            {"prompt": "text"},
        )
        assert result is None

    def test_hf_import_returns_none_without_sdk(self, monkeypatch):
        monkeypatch.delenv("ADAPTION_API_KEY", raising=False)
        with patch("nightmarenet.data.adaption.Adaption", None):
            optimizer = AdaptionOptimizer()
            result = optimizer.optimize_from_huggingface(
                "https://huggingface.co/datasets/test/ds",
                ["train.csv"],
                {"prompt": "text"},
            )
            assert result is None

    def test_kaggle_import_returns_none_without_sdk(self, monkeypatch):
        monkeypatch.delenv("ADAPTION_API_KEY", raising=False)
        with patch("nightmarenet.data.adaption.Adaption", None):
            optimizer = AdaptionOptimizer()
            result = optimizer.optimize_from_kaggle(
                "https://www.kaggle.com/datasets/test/ds",
                ["data.csv"],
                {"prompt": "text"},
            )
            assert result is None


# ---------------------------------------------------------------------------
# AdaptionOptimizer — with mock client
# ---------------------------------------------------------------------------


class TestAdaptionOptimizerWithMockClient:
    @pytest.fixture
    def mock_dataset(self):
        ds = MagicMock()
        ds.__len__ = MagicMock(return_value=2)
        ds.column_names = ["text", "label"]
        ds.select.return_value = ds
        ds.__iter__ = MagicMock(
            return_value=iter([{"text": "hello", "label": "1"}, {"text": "world", "label": "0"}])
        )
        return ds

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        upload = MagicMock(dataset_id="ds-123")
        client.datasets.upload_file.return_value = upload
        client.datasets.run.return_value = MagicMock(
            run_id="run-1",
            estimated_credits_consumed=2.5,
            estimated_minutes=1.0,
        )
        client.datasets.wait_for_completion.return_value = MagicMock(status="succeeded")
        client.datasets.download.return_value = "https://example.com/out.csv"
        client.datasets.get_status.return_value = MagicMock(
            row_count=2, status="succeeded", quality_score=0.85
        )
        return client

    def test_optimize_passes_brand_controls(self, mock_dataset, mock_client, monkeypatch):
        monkeypatch.setenv("ADAPTION_API_KEY", "test-key")
        brand = {"length": "concise", "hallucination_mitigation": True}

        with patch("nightmarenet.data.adaption.Adaption", MagicMock(return_value=mock_client)):
            with patch("nightmarenet.data.adaption.HFDataset") as hf_ds:
                hf_ds.from_csv.return_value = MagicMock(
                    __len__=lambda _: 2, column_names=["text", "label"]
                )
                with patch("urllib.request.urlretrieve"):
                    optimizer = AdaptionOptimizer()
                    optimizer.optimize_dataset(
                        mock_dataset,
                        {"prompt": "text"},
                        brand_controls=brand,
                    )

        call_kwargs = mock_client.datasets.run.call_args
        assert call_kwargs[1].get("brand_controls") == brand

    def test_optimize_passes_recipe_specification(self, mock_dataset, mock_client, monkeypatch):
        monkeypatch.setenv("ADAPTION_API_KEY", "test-key")
        recipe = {"recipes": {"reasoning_traces": True, "deduplication": True}}

        with patch("nightmarenet.data.adaption.Adaption", MagicMock(return_value=mock_client)):
            with patch("nightmarenet.data.adaption.HFDataset") as hf_ds:
                hf_ds.from_csv.return_value = MagicMock(__len__=lambda _: 2, column_names=["text"])
                with patch("urllib.request.urlretrieve"):
                    optimizer = AdaptionOptimizer()
                    optimizer.optimize_dataset(
                        mock_dataset,
                        {"prompt": "text"},
                        recipe_specification=recipe,
                    )

        call_kwargs = mock_client.datasets.run.call_args
        assert call_kwargs[1].get("recipe_specification") == recipe

    def test_optimize_adds_idempotency_key(self, mock_dataset, mock_client, monkeypatch):
        monkeypatch.setenv("ADAPTION_API_KEY", "test-key")

        with patch("nightmarenet.data.adaption.Adaption", MagicMock(return_value=mock_client)):
            with patch("nightmarenet.data.adaption.HFDataset") as hf_ds:
                hf_ds.from_csv.return_value = MagicMock(__len__=lambda _: 2, column_names=["text"])
                with patch("urllib.request.urlretrieve"):
                    optimizer = AdaptionOptimizer()
                    optimizer.optimize_dataset(
                        mock_dataset,
                        {"prompt": "text"},
                    )

        call_kwargs = mock_client.datasets.run.call_args
        job_spec = call_kwargs[1].get("job_specification", {})
        assert "idempotency_key" in job_spec
        assert job_spec["idempotency_key"].startswith("nn-")

    def test_estimate_cost_with_brand_controls(self, mock_dataset, mock_client, monkeypatch):
        monkeypatch.setenv("ADAPTION_API_KEY", "test-key")
        brand = {"length": "detailed"}

        with patch("nightmarenet.data.adaption.Adaption", MagicMock(return_value=mock_client)):
            optimizer = AdaptionOptimizer()
            result = optimizer.estimate_cost(mock_dataset, {"prompt": "text"}, brand_controls=brand)

        assert result is not None
        assert "credits" in result
        call_kwargs = mock_client.datasets.run.call_args
        assert call_kwargs[1].get("estimate") is True
        assert call_kwargs[1].get("brand_controls") == brand

    def test_huggingface_import(self, mock_client, monkeypatch):
        monkeypatch.setenv("ADAPTION_API_KEY", "test-key")
        mock_client.datasets.create_from_huggingface.return_value = MagicMock(dataset_id="ds-hf-1")

        with patch("nightmarenet.data.adaption.Adaption", MagicMock(return_value=mock_client)):
            optimizer = AdaptionOptimizer()
            result = optimizer.optimize_from_huggingface(
                "https://huggingface.co/datasets/test/ds",
                ["train.csv"],
                {"prompt": "text"},
                brand_controls={"hallucination_mitigation": True},
            )

        assert result is not None
        dataset_id, quality = result
        assert dataset_id == "ds-hf-1"
        mock_client.datasets.create_from_huggingface.assert_called_once()

    def test_kaggle_import(self, mock_client, monkeypatch):
        monkeypatch.setenv("ADAPTION_API_KEY", "test-key")
        mock_client.datasets.create_from_kaggle.return_value = MagicMock(dataset_id="ds-kg-1")

        with patch("nightmarenet.data.adaption.Adaption", MagicMock(return_value=mock_client)):
            optimizer = AdaptionOptimizer()
            result = optimizer.optimize_from_kaggle(
                "https://www.kaggle.com/datasets/test/ds",
                ["data.csv"],
                {"prompt": "text"},
            )

        assert result is not None
        dataset_id, _ = result
        assert dataset_id == "ds-kg-1"
        mock_client.datasets.create_from_kaggle.assert_called_once()


# ---------------------------------------------------------------------------
# Pipeline phase-aware optimization
# ---------------------------------------------------------------------------


class TestPipelinePhaseAwareOptimize:
    @pytest.fixture
    def pipeline_config(self):
        return {
            "model": {"name": "gpt2", "type": "causal_lm", "max_length": 32},
            "dataset": {"text_column": "text", "max_samples": 100},
            "training": {"batch_size": 4, "num_cycles": 1},
            "seed": 42,
            "adaption": {
                "enabled": True,
                "estimate_first": True,
                "max_credits": 10,
                "max_rows": 100,
                "column_mapping": {"prompt": "text"},
                "wake_controls": {
                    "enabled": True,
                    "brand_controls": {"hallucination_mitigation": True},
                },
                "dream_controls": {
                    "enabled": True,
                    "brand_controls": {"length": "concise"},
                    "recipe_specification": {"recipes": {"prompt_rephrase": True}},
                },
                "nightmare_controls": {"enabled": False},
            },
        }

    def test_estimate_gating_skips_when_over_budget(self, pipeline_config, monkeypatch):
        monkeypatch.setenv("ADAPTION_API_KEY", "test-key")
        pipeline_config["adaption"]["max_credits"] = 1

        from nightmarenet.pipeline import Pipeline

        pipe = Pipeline(pipeline_config)
        pipe._dataset = MagicMock(column_names=["text"])
        pipe._dataset.__len__ = MagicMock(return_value=50)

        mock_optimizer = MagicMock()
        mock_optimizer.estimate_cost.return_value = {"credits": 99.0, "estimated_minutes": 5.0}

        with patch("nightmarenet.data.adaption.Adaption", MagicMock()):
            with patch("nightmarenet.data.adaption.AdaptionOptimizer", return_value=mock_optimizer):
                pipe.optimize()

        mock_optimizer.optimize_dataset.assert_not_called()

    def test_phase_controls_call_optimizer_per_phase(self, pipeline_config, monkeypatch):
        monkeypatch.setenv("ADAPTION_API_KEY", "test-key")
        pipeline_config["adaption"]["estimate_first"] = False

        from nightmarenet.pipeline import Pipeline

        pipe = Pipeline(pipeline_config)
        pipe._dataset = MagicMock(column_names=["text"])
        pipe._dataset.__len__ = MagicMock(return_value=50)

        mock_optimizer = MagicMock()
        mock_optimizer.optimize_dataset.return_value = (
            MagicMock(column_names=["text"]),
            {"quality_score": 0.9},
        )

        with patch("nightmarenet.data.adaption.Adaption", MagicMock()):
            with patch("nightmarenet.data.adaption.AdaptionOptimizer", return_value=mock_optimizer):
                pipe.optimize()

        assert mock_optimizer.optimize_dataset.call_count == 2  # wake + dream (nightmare disabled)

    def test_disabled_adaption_is_noop(self, pipeline_config):
        pipeline_config["adaption"]["enabled"] = False

        from nightmarenet.pipeline import Pipeline

        pipe = Pipeline(pipeline_config)
        pipe._dataset = MagicMock()
        pipe.optimize()
        assert pipe._context.wake_dataset is None
