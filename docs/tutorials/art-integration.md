# ART Integration Tutorial

This guide shows how to use IBM's [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) with NightmareNet to evaluate model robustness using standardized adversarial attacks.

## Installation

Install NightmareNet with the `art` optional extra:

```bash
pip install 'nightmarenet[art]'
```

Or add it to your existing installation:

```bash
pip install adversarial-robustness-toolbox>=1.15.0
```

## Quick Start

### 1. Wrap Your Model

```python
from nightmarenet.evaluation.art_adapter import NightmareNetARTClassifier

# Wrap any torch.nn.Module as an ART classifier
wrapper = NightmareNetARTClassifier(
    model=your_pytorch_model,
    nb_classes=10,
    input_shape=(3, 32, 32),      # C, H, W
    clip_values=(0.0, 1.0),       # input range
)

art_classifier = wrapper.classifier
```

### 2. Run a Single Attack

```python
import numpy as np
from nightmarenet.evaluation.art_adapter import run_art_attack

# Prepare test data (numpy arrays)
x_test = np.random.rand(100, 3, 32, 32).astype(np.float32)
y_test = np.random.randint(0, 10, 100)

# Run PGD attack
result = run_art_attack(
    art_classifier,
    attack_name="pgd",
    x=x_test,
    y=y_test,
    eps=0.3,          # perturbation budget
    max_iter=40,      # PGD iterations
)

print(f"Attack: {result.attack_name}")
print(f"Success Rate: {result.success_rate:.1%}")
print(f"Mean L2 Perturbation: {result.mean_perturbation:.4f}")
print(f"Query Count: {result.query_count}")
print(f"Time: {result.elapsed_seconds:.1f}s")
```

### 3. Run a Full Benchmark (PGD + FGSM + C&W)

```python
from nightmarenet.evaluation.art_adapter import run_art_benchmark

results = run_art_benchmark(
    art_classifier,
    x=x_test,
    y=y_test,
    attacks=["pgd", "fgsm", "cw"],
    eps=0.3,
)

for r in results:
    print(f"{r.attack_name:>6s} | success={r.success_rate:.1%} | L2={r.mean_perturbation:.4f}")
```

## Supported Attacks

| Key    | ART Class                    | Description                                      |
|--------|------------------------------|--------------------------------------------------|
| `pgd`  | `ProjectedGradientDescent`   | Iterative projected gradient descent (Madry et al.) |
| `fgsm` | `FastGradientMethod`         | Single-step fast gradient sign method (Goodfellow et al.) |
| `cw`   | `CarliniL2Method`            | Carlini & Wagner L2 attack                       |

## Metrics

Each `ARTAttackResult` contains:

- **`success_rate`** — Fraction of correctly-classified samples where the attack changed the prediction.
- **`mean_perturbation`** / **`median_perturbation`** — L2 norm of the adversarial perturbation.
- **`query_count`** — Estimated number of forward-pass queries made by the attack.
- **`elapsed_seconds`** — Wall-clock time for the attack.

## Graceful Degradation

If ART is not installed, importing the adapter raises a helpful error:

```python
>>> from nightmarenet.evaluation.art_adapter import NightmareNetARTClassifier
>>> NightmareNetARTClassifier(model=m, nb_classes=10, input_shape=(3,32,32))
ImportError: The Adversarial Robustness Toolbox (ART) is not installed.
Install it with:  pip install 'nightmarenet[art]'
```
