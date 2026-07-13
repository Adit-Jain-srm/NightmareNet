"""RSLAD-style adversarial robust distillation for the compression phase."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

def fgsm_perturb(model: nn.Module, batch: dict, epsilon: float = 0.01) -> dict:
    """Generate FGSM adversarial examples from a batch.

    Args:
        model: The student model (used to compute loss for grad).
        batch: Input batch dict with 'input_ids', 'attention_mask', etc.
        epsilon: Perturbation magnitude (applied to embeddings).

    Returns:
        Batch with 'inputs_embeds' replaced by adversarially perturbed embeddings.
    """
    model.eval()
    input_ids = batch.get("input_ids")
    attention_mask = batch.get("attention_mask")
    labels = batch.get("labels", input_ids)

    embedding_layer = model.get_input_embeddings()
    embeds = embedding_layer(input_ids).detach()
    embeds.requires_grad_(True)

    outputs = model(
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        labels=labels,
    )
    loss = outputs.loss

    # Use autograd.grad to isolate gradient w.r.t. embeds only,
    # avoiding side effects on model parameter gradients.
    (embeds_grad,) = torch.autograd.grad(loss, embeds)
    adv_embeds = (embeds + epsilon * embeds_grad.sign()).detach()

    adv_dict = {k: v for k, v in batch.items() if k not in ("input_ids", "labels")}
    adv_dict["inputs_embeds"] = adv_embeds
    return adv_dict


def run_distillation(
    teacher: nn.Module,
    student: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 1,
    temperature: float = 4.0,
    alpha: float = 0.7,
    epsilon: float = 0.01,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> dict:
    """Run RSLAD-style distillation: teacher → student on adversarial inputs.

    Args:
        teacher: Frozen pre-pruned model.
        student: Pruned model to train.
        dataloader: DataLoader for distillation data.
        optimizer: Optimizer for the student.
        device: Device to run on.
        epochs: Number of distillation epochs.
        temperature: Softmax temperature for KL divergence.
        alpha: Weight for distillation loss vs task loss (1.0 = pure distillation).
        epsilon: FGSM perturbation magnitude.
        scaler: Optional GradScaler for AMP.

    Returns:
        Dict with average distillation loss.
    """
    teacher.eval()
    student.train()

    total_loss = 0.0
    total_steps = 0

    for epoch in range(epochs):
        for batch in tqdm(dataloader, desc=f"Distillation Epoch {epoch + 1}"):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Generate adversarial inputs (RSLAD: distill on adversarial data)
            adv_batch = fgsm_perturb(student, batch, epsilon=epsilon)
            adv_batch = {k: v.to(device) for k, v in adv_batch.items()}

            use_amp = scaler is not None
            with torch.amp.autocast("cuda", enabled=use_amp):
                # Teacher logits (no grad)
                with torch.no_grad():
                    teacher_out = teacher(**adv_batch)
                    teacher_logits = teacher_out.logits  # (B, T, V)

                # Student logits
                student.train()
                student_out = student(**adv_batch, labels=batch.get("input_ids"))
                student_logits = student_out.logits  # (B, T, V)
                task_loss = student_out.loss

                # KL divergence loss with temperature scaling
                kl_loss = func.kl_div(
                    func.log_softmax(student_logits / temperature, dim=-1),
                    func.softmax(teacher_logits / temperature, dim=-1),
                    reduction="batchmean",
                ) * (temperature ** 2)

                loss = alpha * kl_loss + (1.0 - alpha) * task_loss

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("Distillation: NaN/Inf loss, skipping step.")
                optimizer.zero_grad()
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            total_steps += 1

    avg_loss = total_loss / max(total_steps, 1)
    logger.info("Distillation complete. Avg loss: %.4f", avg_loss)
    return {"distillation_loss": avg_loss}
