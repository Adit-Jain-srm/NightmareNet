"""HuggingFace Hub integration for NightmareNet models.

Provides ``NightmareNetHubMixin`` — a drop-in mixin that adds
``push_to_hub`` / ``from_pretrained`` to any ``torch.nn.Module``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch

if TYPE_CHECKING:
    from huggingface_hub import PyTorchModelHubMixin as _HubMixinBase
else:
    try:
        from huggingface_hub import PyTorchModelHubMixin as _HubMixinBase
    except ImportError:  # pragma: no cover
        _HubMixinBase = object

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = "nightmarenet_config.json"
_WEIGHTS_FILENAME = "pytorch_model.bin"
_SAFETENSORS_FILENAME = "model.safetensors"
_HUB_FILES: List[str] = [_CONFIG_FILENAME, _WEIGHTS_FILENAME, _SAFETENSORS_FILENAME]


def _require_hf_hub() -> None:
    """Raise a clear error when huggingface_hub is not installed."""
    if _HubMixinBase is object:
        raise ImportError(
            "The 'huggingface-hub' package is required for Hub integration. "
            "Install it with:  pip install nightmarenet[hub]"
        )


class NightmareNetHubMixin(_HubMixinBase):  # type: ignore[misc]
    """Mixin that adds Hub push/pull with NightmareNet metadata.

    Subclasses **must** be ``torch.nn.Module`` subclasses.
    """

    def get_nightmarenet_config(self) -> Dict[str, Any]:
        """Return NightmareNet-specific metadata to persist."""
        return {}

    def _save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Save weights + NightmareNet config to *save_directory*."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        weights_path = save_directory / _WEIGHTS_FILENAME
        torch.save(self.state_dict(), weights_path)
        logger.info("Saved model weights to %s", weights_path)

        config = self.get_nightmarenet_config()
        config_path = save_directory / _CONFIG_FILENAME
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, ensure_ascii=False, default=str)
        logger.info("Saved NightmareNet config to %s", config_path)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        map_location: str = "cpu",
        strict: bool = True,
        **model_kwargs: Any,
    ) -> "NightmareNetHubMixin":
        """Load a NightmareNet model from the Hub or a local directory."""
        _require_hf_hub()
        from huggingface_hub import hf_hub_download

        if os.path.isdir(model_id):
            model_dir = Path(model_id)
        else:
            first_path: Optional[Path] = None
            for filename in _HUB_FILES:
                try:
                    path = hf_hub_download(
                        repo_id=model_id,
                        filename=filename,
                        revision=revision,
                        cache_dir=str(cache_dir) if cache_dir else None,
                        force_download=force_download,
                        local_files_only=local_files_only,
                        token=token,
                    )
                    if first_path is None:
                        first_path = Path(path)
                except Exception:
                    logger.debug("Could not download %s, skipping", filename)
            if first_path is None:
                raise FileNotFoundError(
                    f"No files found in Hub repo {model_id!r}. "
                    f"Expected at least {_CONFIG_FILENAME}."
                )
            model_dir = first_path.parent

        config: Dict[str, Any] = {}
        config_path = model_dir / _CONFIG_FILENAME
        if config_path.exists():
            with open(config_path, encoding="utf-8") as fh:
                config = json.load(fh)

        merged = {**config, **model_kwargs}
        instance = cls(**merged)

        weights_path = model_dir / _WEIGHTS_FILENAME
        if weights_path.exists():
            state = torch.load(
                weights_path, map_location=map_location, weights_only=True
            )
        else:
            safetensors_path = model_dir / _SAFETENSORS_FILENAME
            if safetensors_path.exists():
                from safetensors.torch import load_file

                state = load_file(str(safetensors_path))
            else:
                raise FileNotFoundError(
                    f"No model weights found in {model_dir}. "
                    f"Expected {_WEIGHTS_FILENAME} or {_SAFETENSORS_FILENAME}."
                )

        instance.load_state_dict(state, strict=strict)
        logger.info("Loaded NightmareNet model from %s", model_dir)
        return instance

    def push_to_hub(self, repo_id: str, **kwargs: Any) -> Any:
        """Push model to HuggingFace Hub."""
        _require_hf_hub()
        return super().push_to_hub(repo_id, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs: Any,
    ) -> "NightmareNetHubMixin":
        """Load a pretrained model from the Hub or local path."""
        _require_hf_hub()
        return super().from_pretrained(  # type: ignore[no-any-return]
            pretrained_model_name_or_path, **kwargs
        )


__all__ = ["NightmareNetHubMixin"]
