"""Tests for nightmarenet.hub — NightmareNetHubMixin.

All Hub network calls are mocked; no real uploads/downloads occur.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import torch

from nightmarenet.hub import NightmareNetHubMixin

# ---------------------------------------------------------------------------
# Fixtures — minimal model that uses the mixin
# ---------------------------------------------------------------------------


class _DummyModel(NightmareNetHubMixin, torch.nn.Module):
    """Minimal nn.Module that uses the mixin for testing."""

    def __init__(self, input_dim: int = 8, output_dim: int = 4, **kwargs: Any):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self._extra_config = {
            k: v for k, v in kwargs.items() if k not in ("input_dim", "output_dim")
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def get_nightmarenet_config(self) -> Dict[str, Any]:
        return {
            "input_dim": self.linear.in_features,
            "output_dim": self.linear.out_features,
            "robustness_score": 0.95,
            "distortion_families": ["gaussian_noise", "adversarial"],
            **self._extra_config,
        }


@pytest.fixture()
def dummy_model() -> _DummyModel:
    return _DummyModel(input_dim=8, output_dim=4)


@pytest.fixture()
def tmp_hub_dir(tmp_path: Path) -> Path:
    return tmp_path / "hub_model"


# ---------------------------------------------------------------------------
# _save_pretrained tests
# ---------------------------------------------------------------------------


class TestSavePretrained:
    def test_creates_weights_and_config(self, dummy_model: _DummyModel, tmp_hub_dir: Path) -> None:
        dummy_model._save_pretrained(tmp_hub_dir)

        assert (tmp_hub_dir / "pytorch_model.bin").exists()
        assert (tmp_hub_dir / "nightmarenet_config.json").exists()

    def test_weights_are_loadable(self, dummy_model: _DummyModel, tmp_hub_dir: Path) -> None:
        dummy_model._save_pretrained(tmp_hub_dir)

        state = torch.load(tmp_hub_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
        assert "linear.weight" in state
        assert "linear.bias" in state

    def test_config_contents(self, dummy_model: _DummyModel, tmp_hub_dir: Path) -> None:
        dummy_model._save_pretrained(tmp_hub_dir)

        with open(tmp_hub_dir / "nightmarenet_config.json") as f:
            config = json.load(f)

        assert config["input_dim"] == 8
        assert config["output_dim"] == 4
        assert config["robustness_score"] == 0.95
        assert "gaussian_noise" in config["distortion_families"]

    def test_creates_parent_dirs(self, dummy_model: _DummyModel, tmp_path: Path) -> None:
        deep_dir = tmp_path / "a" / "b" / "c" / "model"
        dummy_model._save_pretrained(deep_dir)
        assert deep_dir.exists()


# ---------------------------------------------------------------------------
# _from_pretrained tests
# ---------------------------------------------------------------------------


class TestFromPretrained:
    def test_roundtrip_local(self, dummy_model: _DummyModel, tmp_hub_dir: Path) -> None:
        """Save → load → compare weights."""
        dummy_model._save_pretrained(tmp_hub_dir)

        loaded = _DummyModel._from_pretrained(
            model_id=str(tmp_hub_dir),
            map_location="cpu",
        )

        for key in dummy_model.state_dict():
            assert torch.equal(
                dummy_model.state_dict()[key],
                loaded.state_dict()[key],
            ), f"Mismatch in {key}"

    def test_config_restored(self, dummy_model: _DummyModel, tmp_hub_dir: Path) -> None:
        dummy_model._save_pretrained(tmp_hub_dir)

        loaded = _DummyModel._from_pretrained(
            model_id=str(tmp_hub_dir),
            map_location="cpu",
        )

        cfg = loaded.get_nightmarenet_config()
        assert cfg["robustness_score"] == 0.95
        assert cfg["distortion_families"] == ["gaussian_noise", "adversarial"]

    def test_model_is_callable(self, dummy_model: _DummyModel, tmp_hub_dir: Path) -> None:
        dummy_model._save_pretrained(tmp_hub_dir)

        loaded = _DummyModel._from_pretrained(
            model_id=str(tmp_hub_dir),
            map_location="cpu",
        )
        loaded.eval()
        x = torch.randn(2, 8)
        out = loaded(x)
        assert out.shape == (2, 4)

    @patch("huggingface_hub.hf_hub_download")
    def test_hub_download_path(
        self,
        mock_download: MagicMock,
        dummy_model: _DummyModel,
        tmp_hub_dir: Path,
    ) -> None:
        """When model_id is not a local dir, hf_hub_download is called for
        each file (config + weights)."""
        dummy_model._save_pretrained(tmp_hub_dir)

        config_file = tmp_hub_dir / "nightmarenet_config.json"
        weights_file = tmp_hub_dir / "pytorch_model.bin"

        # hf_hub_download is called for each file; map by filename arg
        def _side_effect(filename, **kwargs):
            if filename == "nightmarenet_config.json":
                return str(config_file)
            if filename == "pytorch_model.bin":
                return str(weights_file)
            raise FileNotFoundError(filename)

        mock_download.side_effect = _side_effect

        loaded = _DummyModel._from_pretrained(
            model_id="user/repo",
            map_location="cpu",
        )
        # At least config + weights
        assert mock_download.call_count >= 2
        assert torch.equal(
            dummy_model.state_dict()["linear.weight"],
            loaded.state_dict()["linear.weight"],
        )


# ---------------------------------------------------------------------------
# Import guard tests
# ---------------------------------------------------------------------------


class TestImportGuard:
    def test_hub_base_is_real(self) -> None:
        """_HubMixinBase should be the real mixin when huggingface_hub is present."""
        from nightmarenet.hub import _HubMixinBase

        assert _HubMixinBase is not object

    def test_push_to_hub_requires_hf(self, dummy_model: _DummyModel) -> None:
        """push_to_hub should raise ImportError if huggingface_hub missing."""
        with patch("nightmarenet.hub._HubMixinBase", object):
            with pytest.raises(ImportError, match="huggingface-hub"):
                dummy_model.push_to_hub("user/repo")


# ---------------------------------------------------------------------------
# get_nightmarenet_config tests
# ---------------------------------------------------------------------------


class TestGetConfig:
    def test_base_returns_empty(self) -> None:
        class _Raw(NightmareNetHubMixin, torch.nn.Module):
            def forward(self, x):
                return x

        m = _Raw()
        assert m.get_nightmarenet_config() == {}

    def test_subclass_override(self, dummy_model: _DummyModel) -> None:
        cfg = dummy_model.get_nightmarenet_config()
        assert "robustness_score" in cfg
        assert cfg["input_dim"] == 8
