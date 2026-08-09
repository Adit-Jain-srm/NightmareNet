"""Nightmare-phase vision adversarial attack distortion engines."""

from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn

from nightmarenet.distortions.vision.base import ImageDistortion


class PixelPerturbation(ImageDistortion):
    """L-infinity bounded random pixel perturbation (model-free)."""

    name = "vision_pixel_perturbation"
    phase = "nightmare"
    description = "L-infinity bounded random pixel perturbation"

    def __init__(self, epsilon: float = 0.1) -> None:
        self.epsilon = epsilon

    def distort(
        self, image: torch.Tensor, strength: float, seed: Optional[int] = None
    ) -> torch.Tensor:
        if strength <= 0.0:
            return image.clone()

        eps = strength * self.epsilon

        gen = None
        if seed is not None:
            gen = torch.Generator(device=image.device)
            gen.manual_seed(seed)

        # Generate uniform random noise in [-eps, eps]
        if gen is not None:
            noise = (
                torch.rand(image.shape, generator=gen, device=image.device, dtype=image.dtype) * 2.0
                - 1.0
            ) * eps
        else:
            noise = (
                torch.rand(image.shape, device=image.device, dtype=image.dtype) * 2.0 - 1.0
            ) * eps

        perturbed = image + noise
        return torch.clamp(perturbed, 0.0, 1.0)


class FGSM(ImageDistortion):
    """Fast Gradient Sign Method (FGSM) single-step adversarial attack."""

    name = "vision_fgsm"
    phase = "nightmare"
    description = (
        "Fast Gradient Sign Method single-step adversarial attack. "
        "Falls back to pixel perturbation when no model is injected."
    )

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        criterion: Optional[Union[nn.Module, Callable]] = None,
        epsilon: float = 0.1,
    ) -> None:
        self.model = model
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.epsilon = epsilon
        self.fallback = PixelPerturbation(epsilon=epsilon)

    def distort(
        self, image: torch.Tensor, strength: float, seed: Optional[int] = None
    ) -> torch.Tensor:
        if strength <= 0.0:
            return image.clone()

        if self.model is None:
            return self.fallback.distort(image, strength, seed)

        original_mode = self.model.training
        self.model.eval()

        try:
            # We need to run with gradient enabled
            input_tensor = image.clone().detach().unsqueeze(0)
            input_tensor.requires_grad_(True)

            with torch.enable_grad():
                outputs = self.model(input_tensor)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                # Predict pseudo-label if true label not provided
                target = logits.max(1)[1]
                loss = self.criterion(logits, target)

                grad = torch.autograd.grad(
                    loss, input_tensor, retain_graph=False, create_graph=False
                )[0]

            eps = strength * self.epsilon
            grad_sign = grad.sign().squeeze(0).detach()
            perturbed = image + eps * grad_sign

            return torch.clamp(perturbed, 0.0, 1.0)
        finally:
            if original_mode:
                self.model.train()


class PGD(ImageDistortion):
    """Projected Gradient Descent (PGD) multi-step adversarial attack."""

    name = "vision_pgd"
    phase = "nightmare"
    description = (
        "Projected Gradient Descent multi-step adversarial attack. "
        "Falls back to pixel perturbation when no model is injected."
    )

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        criterion: Optional[Union[nn.Module, Callable]] = None,
        epsilon: float = 0.1,
        steps: int = 10,
        alpha: Optional[float] = None,
    ) -> None:
        self.model = model
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha
        self.fallback = PixelPerturbation(epsilon=epsilon)

    def distort(
        self, image: torch.Tensor, strength: float, seed: Optional[int] = None
    ) -> torch.Tensor:
        if strength <= 0.0:
            return image.clone()

        if self.model is None:
            return self.fallback.distort(image, strength, seed)

        original_mode = self.model.training
        self.model.eval()

        try:
            num_steps = max(1, int(strength * self.steps))
            eps = strength * self.epsilon

            # Step size per iteration
            if self.alpha is not None:
                step_size = strength * self.alpha
            else:
                step_size = eps / max(1, num_steps)

            # Predict targets using clean image
            input_tensor = image.clone().detach().unsqueeze(0)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                target = logits.max(1)[1]

            x_adv = image.clone().detach()

            for _ in range(num_steps):
                x_adv_batch = x_adv.clone().detach().unsqueeze(0)
                x_adv_batch.requires_grad_(True)

                with torch.enable_grad():
                    outputs = self.model(x_adv_batch)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    loss = self.criterion(logits, target)
                    grad = torch.autograd.grad(
                        loss, x_adv_batch, retain_graph=False, create_graph=False
                    )[0]

                # Update step
                grad_sign = grad.sign().squeeze(0).detach()
                x_adv = x_adv + step_size * grad_sign

                # Project back to epsilon-ball
                x_adv = torch.clamp(x_adv, image - eps, image + eps)

                # Clamp to valid image pixel range
                x_adv = torch.clamp(x_adv, 0.0, 1.0)

            return x_adv.detach()
        finally:
            if original_mode:
                self.model.train()


class AdversarialPatch(ImageDistortion):
    """Applies a random noise patch to the image at a random location."""

    name = "vision_adversarial_patch"
    phase = "nightmare"
    description = "Applies a random noise patch at a random location scaled by strength"

    def __init__(self, patch_size: Union[int, Tuple[int, int]] = 8) -> None:
        self.patch_size = patch_size

    def distort(
        self, image: torch.Tensor, strength: float, seed: Optional[int] = None
    ) -> torch.Tensor:
        if strength <= 0.0:
            return image.clone()

        c, h, w = image.shape

        if isinstance(self.patch_size, int):
            ph, pw = self.patch_size, self.patch_size
        else:
            ph, pw = self.patch_size

        # Clamp patch size to image dimensions
        ph = min(ph, h)
        pw = min(pw, w)

        if ph <= 0 or pw <= 0:
            return image.clone()

        gen = None
        if seed is not None:
            gen = torch.Generator(device=image.device)
            gen.manual_seed(seed)

        # Get random top-left coordinate (y, x)
        if h - ph > 0:
            if gen is not None:
                y = torch.randint(0, h - ph + 1, (1,), generator=gen, device=image.device).item()
            else:
                y = torch.randint(0, h - ph + 1, (1,), device=image.device).item()
        else:
            y = 0

        if w - pw > 0:
            if gen is not None:
                x = torch.randint(0, w - pw + 1, (1,), generator=gen, device=image.device).item()
            else:
                x = torch.randint(0, w - pw + 1, (1,), device=image.device).item()
        else:
            x = 0

        # Generate noise patch
        if gen is not None:
            noise = torch.rand((c, ph, pw), generator=gen, device=image.device, dtype=image.dtype)
        else:
            noise = torch.rand((c, ph, pw), device=image.device, dtype=image.dtype)

        # Blend original patch area with the noise patch
        distorted = image.clone()
        orig_patch = image[:, y : y + ph, x : x + pw]
        distorted[:, y : y + ph, x : x + pw] = (1.0 - strength) * orig_patch + strength * noise

        return torch.clamp(distorted, 0.0, 1.0)
