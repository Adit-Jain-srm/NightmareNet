# NightmareNet Plugin Development Guide

This guide explains how to create custom distortion plugins for NightmareNet using the plugin system.

## Overview

NightmareNet supports three types of custom distortion engines:

1. **Entry Point Plugins** - Third-party packages that register distortions via `pyproject.toml`
2. **Decorator-Based Plugins** - Single-file plugins using the registry decorator
3. **File-Based Custom Engines** - Load distortion functions from Python files at runtime

## Method 1: Entry Point Plugins (Recommended for Distribution)

Create a standalone Python package that registers distortion engines via entry points.

### Package Structure

```
nightmarenet-financial-distortions/
├── pyproject.toml
├── README.md
├── financial_distortions/
│   ├── __init__.py
│   ├── ticker.py
│   └── numbers.py
└── tests/
    └── test_financial_distortions.py
```

### pyproject.toml

```toml
[project]
name = "nightmarenet-financial-distortions"
version = "0.1.0"
description = "Financial domain distortion plugins for NightmareNet"
requires-python = ">=3.8"
dependencies = [
    "nightmarenet>=0.1.0",
]

[project.entry-points."nightmarenet.distortions"]
ticker_corrupt = "financial_distortions.ticker:TickerCorruption"
number_swap = "financial_distortions.numbers:NumberSwap"
```

### Plugin Implementation

```python
# financial_distortions/ticker.py
from nightmarenet.distortions.base import BaseDistortion
from typing import Optional

class TickerCorruption(BaseDistortion):
    """Corrupt stock ticker symbols in financial text."""
    
    name = "ticker_corrupt"
    phase = "nightmare"
    description = "Stock ticker symbol corruption"
    
    def distort(self, text: str, strength: float, seed: Optional[int] = None) -> str:
        import random
        if seed is not None:
            random.seed(seed)
        
        # Example: Replace AAPL with visually similar characters
        result = text
        if random.random() < strength:
            result = result.replace("AAPL", "AAPI")
        return result
```

### Installation and Usage

```bash
# Install the plugin package
pip install nightmarenet-financial-distortions

# Plugin engines are automatically available
nightmarenet distort --engine ticker_corrupt --strength 0.5 --text "AAPL rose 3%"

# List all engines (built-in + plugins)
nightmarenet distort --list-engines
```

## Method 2: Decorator-Based Plugins

For single-file plugins without creating a full package.

### Example: my_distortions.py

```python
from nightmarenet.distortions.registry import get_registry

registry = get_registry()

@registry.register_decorator('homoglyph', phase='nightmare', description='Latin to Cyrillic swap')
def homoglyph(text: str, strength: float, seed: int = None) -> str:
    """Replace Latin characters with visually similar Cyrillic characters."""
    import random
    if seed is not None:
        random.seed(seed)
    
    mapping = {'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р'}
    result = []
    for char in text:
        if random.random() < strength and char.lower() in mapping:
            result.append(mapping[char.lower()])
        else:
            result.append(char)
    return ''.join(result)
```

### Usage

```python
# Import the module to register the plugin
import my_distortions

# Use via CLI
nightmarenet distort --engine homoglyph --strength 0.3 --text "Hello world"
```

## Method 3: File-Based Custom Engines

Load distortion functions from Python files at runtime using the `custom:` prefix.

### Example: custom_distortions.py

```python
def reverse_text(text: str, strength: float, seed: int = None) -> str:
    """Reverse the input text."""
    return text[::-1]
```

### YAML Config Usage

```yaml
# configs/my_experiment.yaml
distortion:
  dream_strength: 0.25
  nightmare_strength: 0.8
  custom_engines:
    - engine: char_swap        # built-in
      strength: 0.3
    - engine: ticker_corrupt   # plugin (auto-discovered)
      strength: 0.5
    - engine: custom:./custom_distortions.py:reverse_text  # file-based custom
      strength: 0.7
```

## Plugin Validation

Use the provided validation helpers to ensure your plugin meets the contract.

### For BaseDistortion Classes

```python
from nightmarenet.distortions.testing import validate_distortion_plugin
from my_plugin import MyDistortion

failures = validate_distortion_plugin(MyDistortion)
if failures:
    print("Validation failed:")
    for f in failures:
        print(f"  - {f}")
else:
    print("Plugin is valid!")
```

### For Standalone Functions

```python
from nightmarenet.distortions.testing import validate_distortion_function

def my_distortion(text: str, strength: float, seed: int = None) -> str:
    return text.upper()

failures = validate_distortion_function(my_distortion)
if not failures:
    print("Function is valid!")
```

## Plugin Contract

All distortion engines must follow this contract:

- **Signature**: `(text: str, strength: float, seed: Optional[int] = None) -> str`
- **strength=0.0**: Should be approximately a no-op (return text unchanged)
- **strength=1.0**: Should produce maximum distortion
- **Determinism**: Same `(text, strength, seed)` must produce identical output
- **Empty Input**: Must return empty string without raising
- **Return Type**: Must return a string

## CLI Integration

### List Available Engines

```bash
nightmarenet distort --list-engines
```

Output:
```
Available distortion engines:

Built-in:
  dream (dream) - Mild stochastic augmentation
  nightmare (nightmare) - Adversarial perturbation

Plugins:
  ticker_corrupt (nightmare) [nightmarenet-financial-distortions] - Stock ticker corruption
  number_swap (dream) [nightmarenet-financial-distortions] - Numerical value perturbation

Custom:
  homoglyph (nightmare) - Latin to Cyrillic swap
```

### Use Plugin Engine

```bash
nightmarenet distort --engine ticker_corrupt --strength 0.5 --text "AAPL rose 3%"
```

## Error Handling

Plugin load failures are logged as warnings and do not crash the system. Check logs for:

```
WARNING: Failed to load distortion plugin 'my_plugin': ...
```

## Best Practices

1. **Use descriptive names** for your distortion engines
2. **Set appropriate phase** (`dream` for mild, `nightmare` for aggressive)
3. **Provide clear descriptions** for users
4. **Validate your plugin** before distribution
5. **Handle edge cases** (empty text, extreme strength values)
6. **Use seeds for reproducibility** when using random operations
7. **Document your distortion** behavior in docstrings

## Version Compatibility

Plugins should declare compatible NightmareNet versions in their metadata:

```toml
[project]
dependencies = [
    "nightmarenet>=0.1.0,<1.0.0",
]
```

## Example: Complete Plugin Package

See the `nightmarenet-financial-distortions` example structure:

```
nightmarenet-financial-distortions/
├── pyproject.toml
├── README.md
├── financial_distortions/
│   ├── __init__.py
│   ├── ticker.py          # TickerCorruption class
│   └── numbers.py         # NumberSwap class
└── tests/
    ├── __init__.py
    └── test_plugins.py    # Validation tests
```

For a complete working example, refer to the test suite in `tests/test_distortion_plugins.py`.
