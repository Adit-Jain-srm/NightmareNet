# Tutorial 2: Creating and Registering Custom Distortions

NightmareNet is built to be modular and highly extensible. In this tutorial, we will explore the distortion engine architecture and walk step-by-step through implementing, registering, and testing a custom text distortion engine.

---

## 1. Distortion Engine Architecture

In NightmareNet, a **distortion** is a controlled perturbation applied to a piece of data (text or image) representing a potential real-world failure mode or adversarial attack.

All text distortions are registered with the **Distortion Registry**. The registry:
1. Coordinates built-in augmentations (like standard synonym replacements or typo injectors).
2. Performs entry-point discovery of third-party plugins.
3. Provides decorator-based APIs for fast custom script configuration.

---

## 2. Base Classes and Contracts

Every distortion engine in NightmareNet must conform to the contract defined by the `BaseDistortion` class in [base.py](../../nightmarenet/distortions/base.py).

### The Distortion Contract

If you write a class-based distortion, it should subclass `BaseDistortion` and satisfy the following constraints:
1. **Determinism**: The same combination of input text, strength, and seed must return identical results.
2. **Empty Safety**: An empty string input (`""`) must return an empty string (`""`) without throwing exceptions.
3. **No-op Bound**: When `strength=0.0`, the output must be approximately identical to the original input.
4. **Max Perturbation**: When `strength=1.0`, the engine applies maximum possible perturbation.

---

## 3. Implementing a Custom Distortion

Let's implement a custom class-based text distortion that swaps letters with a random character (e.g., replicating sensor noise).

### Option A: Class-Based Implementation

Create a file named `custom_swap.py`:

```python
import random
from typing import Optional
from nightmarenet.distortions.base import BaseDistortion

class RandomSwapDistortion(BaseDistortion):
    """Perturbs text by randomly replacing characters with '*' based on strength."""

    name: str = "random_star_swap"
    phase: str = "custom"  # can be 'dream', 'nightmare', or 'custom'
    description: str = "Replaces random letters with an asterisk (*)"

    def distort(self, text: str, strength: float, seed: Optional[int] = None) -> str:
        # Contract Check 1: Empty input returns empty
        if not text:
            return ""

        # Contract Check 2: Strength = 0.0 is a no-op
        if strength <= 0.0:
            return text

        if seed is not None:
            random.seed(seed)

        chars = list(text)
        # Determine how many characters to swap based on strength
        num_to_swap = int(len(chars) * strength)
        indices = random.sample(range(len(chars)), min(num_to_swap, len(chars)))

        for idx in indices:
            # We avoid altering whitespace to keep basic structure
            if not chars[idx].isspace():
                chars[idx] = "*"

        return "".join(chars)

    def validate(self) -> bool:
        # Self-validation
        return len(self.name) > 0 and self.phase in ["dream", "nightmare", "custom"]
```

### Option B: Decorator-Based Function Registration

If you don't need a full class structure, you can register a function directly with the registry using `@registry.register_decorator`:

```python
from nightmarenet.distortions.registry import get_registry

registry = get_registry()

@registry.register_decorator(
    name="leet_speak",
    phase="dream",
    description="Transforms characters to simple leetspeak equivalents."
)
def apply_leet_speak(text: str, strength: float, seed: int = None) -> str:
    if strength <= 0.0 or not text:
        return text

    leet_dict = {"e": "3", "a": "4", "o": "0", "i": "1", "t": "7"}
    chars = list(text.lower())
    
    # Randomly apply according to strength
    import random
    if seed is not None:
        random.seed(seed)

    for i, char in enumerate(chars):
        if char in leet_dict and random.random() < strength:
            chars[i] = leet_dict[char]

    return "".join(chars)
```

---

## 4. Registering Your Custom Class

If you write a class-based distortion (Option A), you must register it with the global `DistortionRegistry` instance:

```python
from nightmarenet.distortions.registry import get_registry
from custom_swap import RandomSwapDistortion

# Obtain global registry
registry = get_registry()

# Instantiate your class
swap_engine = RandomSwapDistortion()

# Register the distort method
registry.register(
    name=swap_engine.name,
    fn=swap_engine.distort,
    metadata={
        "phase": swap_engine.phase,
        "description": swap_engine.description,
        "source": "custom"
    }
)
```

---

## 5. Third-Party Plugin Distribution (Entry Points)

For distribution across python packages, NightmareNet auto-discovers plugins defined in your package's `pyproject.toml` or `setup.py` under the entry point group `nightmarenet.distortions`.

Add the following to your plugin package `pyproject.toml` file:

```toml
[project.entry-points."nightmarenet.distortions"]
random_star_swap = "my_plugin_module:RandomSwapDistortion"
```

When your package is installed in the environment, the `DistortionRegistry` will automatically discover and register `random_star_swap` on initialization.

---

## 6. Verification and Testing

Verify your custom engine functions correctly by writing unit tests mimicking the project structure.

Create `tests/test_custom_distortion.py`:

```python
import pytest
from custom_swap import RandomSwapDistortion
from nightmarenet.distortions.registry import get_registry

def test_custom_swap_contract():
    engine = RandomSwapDistortion()
    text = "Hello World"

    # Test empty string returns empty
    assert engine.distort("", strength=0.5) == ""

    # Test strength 0.0 is a no-op
    assert engine.distort(text, strength=0.0) == text

    # Test determinism
    res1 = engine.distort(text, strength=0.5, seed=42)
    res2 = engine.distort(text, strength=0.5, seed=42)
    assert res1 == res2

    # Test output bounds
    assert "*" in engine.distort(text, strength=0.8)

def test_registry_integration():
    registry = get_registry()
    engine = RandomSwapDistortion()
    
    registry.register(engine.name, engine.distort, {"source": "custom"})
    assert engine.name in registry.engine_names

    distorted = registry.apply(engine.name, "Unit Testing", strength=0.5, seed=10)
    assert "*" in distorted
```

---

## 7. Common Mistakes and Best Practices

### ❌ Modifying Whitespace and Control Characters
Avoid replacing whitespace (`\n`, `\t`, ` `) unless explicitly desired. Aligned sentences keep tokenization structures comparable.

### ❌ Side Effects on Input Reference
Python passes list references by reference, but strings are immutable. If you process mutable containers, copy them first using `copy.deepcopy` or `list(x)` to prevent side effects.

### ❌ Forgetting Seed Reset
Always honor the `seed` argument. If a seed is passed, seed the local generator (e.g. `random.seed(seed)`) to guarantee reproducibility across model evaluations.
