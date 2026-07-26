"""Gaussian noise vision distortion."""

from typing import Optional

import torch

from nightmarenet.distortions.vision.base import ImageDistortion


class GaussianNoise(ImageDistortion):
    """Adds Gaussian noise to an image tensor."""

    name = "vision_gaussian_noise"
    phase = "nightmare"
    description = "Adds Gaussian noise to the image scaled by strength"

    def distort(
        self, image: torch.Tensor, strength: float, seed: Optional[int] = None
    ) -> torch.Tensor:
        if strength <= 0.0:
            return image.clone()

        gen = None
        if seed is not None:
            gen = torch.Generator(device=image.device)
            gen.manual_seed(seed)

        # Scale noise intensity by strength.
        # A standard deviation of strength / 2.0 provides noticeable
        # but bounded noise at strength 1.0.
        std = strength / 2.0

        noise = (
            torch.randn(image.shape, generator=gen, device=image.device, dtype=image.dtype) * std
        )
        distorted = image + noise

        return torch.clamp(distorted, 0.0, 1.0)
