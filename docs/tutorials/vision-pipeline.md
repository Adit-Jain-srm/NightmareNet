# Tutorial 4: Image and Vision Distortion Pipeline

In addition to text models, NightmareNet provides full native support for computer vision model evaluation. In this tutorial, we will learn how to load images, apply built-in vision distortions, and evaluate vision classifier robustness.

---

## 1. Vision Distortion Registry

Just like text, image perturbations are managed via the `VisionDistortionRegistry`. You can access it using:

```python
from nightmarenet.distortions.registry import get_vision_registry
registry = get_vision_registry()
```

### Supported Image Distortions

NightmareNet supports two phases of vision distortions:

| Distortion Name | Phase | Mechanism | Description |
| :--- | :--- | :--- | :--- |
| `vision_color_jitter` | dream | Textural | Adjusts brightness, contrast, saturation, and hue. |
| `vision_geometric_transform` | dream | Spatial | Applies random rotation, scaling, and translation. |
| `vision_gaussian_blur` | dream | Resolution | Blurs the image using a Gaussian kernel. |
| `vision_jpeg_compression` | dream | Compression | Simulates JPEG compression artifacts. |
| `vision_gaussian_noise` | dream | Noise | Adds stochastic Gaussian noise. |
| `vision_pixel_perturbation` | nightmare | Adversarial | Perturbs pixel values bounded by an $L_\infty$ epsilon limit. |
| `vision_fgsm` | nightmare | Adversarial | Fast Gradient Sign Method (gradient-based single-step attack). |
| `vision_pgd` | nightmare | Adversarial | Projected Gradient Descent (iterative gradient-based attack). |
| `vision_adversarial_patch` | nightmare | Adversarial | Superimposes an adversarial patch on the image. |

---

## 2. Loading and Converting Images

All vision distortions operate on PyTorch tensors representing images scaled in the range $[0.0, 1.0]$ with shape `(channels, height, width)`.

NightmareNet provides utility functions in [utils.py](../../nightmarenet/distortions/vision/utils.py) for easy conversions:

```python
from PIL import Image
from nightmarenet.distortions.vision.utils import to_tensor, to_pil

# 1. Load image using PIL
image_path = "data/sample_image.jpg"  # Ensure your image exists
pil_img = Image.open(image_path).convert("RGB")

# 2. Convert to PyTorch Tensor [C, H, W] in [0.0, 1.0]
img_tensor = to_tensor(pil_img)
print(f"Tensor Shape: {img_tensor.shape}, Range: [{img_tensor.min():.2f}, {img_tensor.max():.2f}]")

# 3. Convert back to PIL Image (e.g. for displaying or saving)
pil_output = to_pil(img_tensor)
pil_output.save("results/processed.jpg")
```

---

## 3. Applying Distortions

Let's apply both a mild dream distortion and an aggressive nightmare distortion to an image tensor.

```python
import torch
from nightmarenet.distortions.registry import get_vision_registry

registry = get_vision_registry()

# Create a dummy image tensor (e.g. 3 channels, 224x224 pixels)
dummy_img = torch.rand(3, 224, 224)

# 1. Apply a dream distortion (Gaussian Noise) at strength 0.3
dream_img = registry.apply("vision_gaussian_noise", dummy_img, strength=0.3, seed=42)

# 2. Apply a nightmare distortion (Pixel Perturbation) at strength 0.8
nightmare_img = registry.apply("vision_pixel_perturbation", dummy_img, strength=0.8, seed=42)
```

---

## 4. Vision Robustness Evaluation

To evaluate a vision model's robustness profile, calculate how model classification confidence degrades under increasing distortion sweeps:

```python
import torch
from torchvision.models import resnet18, ResNet18_Weights
from nightmarenet.distortions.registry import get_vision_registry

# 1. Load target vision model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()

# 2. Prepare sample image
dummy_img = torch.rand(3, 224, 224)
preprocess = weights.transforms()
input_tensor = preprocess(dummy_img.unsqueeze(0))  # Add batch dimension

# 3. Evaluate baseline classification
with torch.no_grad():
    clean_logits = model(input_tensor)
    clean_probs = torch.softmax(clean_logits, dim=-1)
    clean_score, clean_class = clean_probs.max(dim=-1)

# 4. Sweep across distortion strengths
registry = get_vision_registry()
strengths = [0.1, 0.3, 0.5, 0.7, 0.9]

print(f"Clean Prediction Class: {clean_class.item()} (Conf: {clean_score.item():.2%})")
print("\nAdversarial Robustness Sweep:")

for strength in strengths:
    # Apply spatial/geometric transformation
    distorted_tensor = registry.apply("vision_geometric_transform", dummy_img, strength=strength, seed=42)
    distorted_input = preprocess(distorted_tensor.unsqueeze(0))
    
    with torch.no_grad():
        dist_logits = model(distorted_input)
        dist_probs = torch.softmax(dist_logits, dim=-1)
        dist_score, dist_class = dist_probs.max(dim=-1)
    
    print(f"Strength {strength:.1f}: Class = {dist_class.item()}, Conf = {dist_score.item():.2%}")
```
