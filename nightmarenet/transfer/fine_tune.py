"""Transfer fine-tuning pipeline.

Fine-tunes a foundation model on a downstream task with optional layer freezing.
Supported architectures for layer freezing include:
- BERT-like models (e.g., BERT, DistilBERT, RoBERTa) via `encoder.layer`
- GPT-like models (e.g., GPT-2) via `transformer.h`
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class TransferFineTuner:
    """Manages the fine-tuning of a foundation model on a downstream task."""

    def __init__(
        self,
        model: PreTrainedModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler

    def _freeze_layers(self, freeze_bottom_n: int) -> bool:
        """Freeze the bottom N layers of the backbone.

        Supported architectures: BERT-like (encoder.layer) and GPT-like (transformer.h).
        Returns True if layers were successfully found and frozen, False otherwise.
        """
        if freeze_bottom_n <= 0:
            return True

        # Simple heuristic for transformers architectures
        if hasattr(self.model, "base_model"):
            base = self.model.base_model
        else:
            base = self.model

        # Attempt to find the encoder layers
        layers = None
        if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
            layers = base.encoder.layer
        elif hasattr(base, "transformer") and hasattr(base.transformer, "h"):
            layers = base.transformer.h

        if layers is not None:
            frozen = 0
            for i, layer in enumerate(layers):
                if i < freeze_bottom_n:
                    for param in layer.parameters():
                        param.requires_grad = False
                    frozen += 1
                else:
                    for param in layer.parameters():
                        param.requires_grad = True
            logger.info("Froze bottom %d layers of the backbone.", frozen)
            return True
        else:
            logger.warning("Could not identify layers to freeze. Layer freezing skipped.")
            return False

    def _unfreeze_all(self) -> None:
        """Unfreeze all parameters in the model."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("Unfroze all layers.")

    def run(
        self,
        dataloader: DataLoader,
        num_epochs: int,
        freeze_bottom_n: int = 0,
        unfreeze_after_epoch: int = 1,
        strict_layer_freezing: bool = False,
    ) -> dict[str, Any]:
        """Run the fine-tuning loop.

        Args:
            dataloader: DataLoader for the downstream task.
            num_epochs: Number of epochs to train.
            freeze_bottom_n: Number of bottom layers to freeze.
            unfreeze_after_epoch: Epoch number (1-indexed) after which to unfreeze all layers.
            strict_layer_freezing: If True, raise RuntimeError if layers to freeze are not found.

        Returns:
            Dictionary with final loss, per-epoch losses, and layer freezing info.
        """
        self.model.to(self.device)
        self.model.train()

        total_loss = 0.0
        steps = 0
        per_epoch_losses = []
        layers_frozen = False

        for epoch in range(1, num_epochs + 1):
            if epoch == 1 and freeze_bottom_n > 0:
                layers_frozen = self._freeze_layers(freeze_bottom_n)
                if strict_layer_freezing and not layers_frozen:
                    raise RuntimeError(
                        "strict_layer_freezing is True, but could not identify "
                        "layers to freeze for model architecture."
                    )
            elif epoch > unfreeze_after_epoch and freeze_bottom_n > 0 and layers_frozen:
                self._unfreeze_all()
                layers_frozen = False  # Only unfreeze once

            logger.info("Starting fine-tuning epoch %d/%d", epoch, num_epochs)
            epoch_loss = 0.0

            progress = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
            for batch in progress:
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                self.optimizer.zero_grad()

                if self.scaler is not None:
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                        outputs = self.model(**batch)
                        loss = outputs.loss

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    self.optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                steps += 1
                progress.set_postfix({"loss": f"{loss_val:.4f}"})

            avg_epoch_loss = epoch_loss / len(dataloader)
            logger.info("Epoch %d average loss: %.4f", epoch, avg_epoch_loss)
            per_epoch_losses.append(avg_epoch_loss)
            total_loss += epoch_loss

        avg_loss = total_loss / max(steps, 1)
        final_epoch_loss = per_epoch_losses[-1] if per_epoch_losses else 0.0

        return {
            "avg_loss": avg_loss,
            "final_epoch_loss": final_epoch_loss,
            "per_epoch_losses": per_epoch_losses,
            "layers_frozen": freeze_bottom_n > 0,
        }
