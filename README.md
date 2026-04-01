# NightmareNet 🧠💤

**A Sleep-Inspired Training Paradigm for AI**

> *"We give AI a sleep cycle—so it learns what to forget, not just what to remember."*

---

## Overview

NightmareNet is a biologically inspired training framework that introduces **dream** and **nightmare** phases to improve model generalization and robustness. Instead of relying solely on scaling data and parameters, NightmareNet incorporates:

- **Synthetic distortion** (Dream Phase)
- **Controlled forgetting** (Compression Phase)
- **Adversarial stress testing** (Nightmare Phase)

This forces models to learn **invariant structures** rather than memorize patterns.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                Training Pipeline                 │
│                                                  │
│   ┌─────────┐   ┌─────────┐   ┌───────────┐    │
│   │  Wake   │──▶│  Dream  │──▶│ Nightmare │    │
│   │ Phase   │   │  Phase  │   │   Phase   │    │
│   └─────────┘   └─────────┘   └───────────┘    │
│       │                             │            │
│       │         ┌───────────┐       │            │
│       └────────▶│ Compress  │◀──────┘            │
│                 │   Phase   │                    │
│                 └───────────┘                    │
│                      │                           │
│                 ┌────▼─────┐                     │
│                 │ Evaluate  │                    │
│                 └──────────┘                     │
└─────────────────────────────────────────────────┘
```

### Training Phases

| Phase | Description | Data |
|-------|-------------|------|
| **Wake** | Standard supervised fine-tuning | Real-world data |
| **Dream** | Training on mildly distorted data | Synthetic dream data (strength 0.2–0.3) |
| **Nightmare** | Stress-testing on extreme perturbations | Adversarial nightmare data (strength 0.7–0.9) |
| **Compression** | Pruning & bottleneck to force abstraction | N/A (model surgery) |

### Distortion Types

- **Text-level**: character swaps, typos, word shuffling, token masking
- **Semantic-level**: synonym replacement, negation injection, topic splicing
- **Adversarial**: contradictory premises, ambiguous queries, cross-domain prompts

## Installation

```bash
# Clone the repository
git clone https://github.com/Adit-Jain-srm/NightmareNet.git
cd NightmareNet

# Install dependencies
pip install -r requirements.txt

# Install as editable package
pip install -e .
```

## Quick Start

### 1. Generate Dream & Nightmare Data

```bash
python scripts/generate_data.py --config configs/default.yaml --output data/generated/
```

### 2. Run Full Training Pipeline

```bash
python scripts/train.py --config configs/default.yaml
```

### 3. Evaluate a Checkpoint

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model --config configs/default.yaml
```

## Configuration

All hyperparameters are controlled via `configs/default.yaml`:

```yaml
model:
  name: gpt2
  max_length: 128

training:
  wake_epochs: 3
  dream_epochs: 2
  nightmare_epochs: 1
  num_cycles: 3
  learning_rate: 5.0e-5

distortion:
  dream_strength: 0.25
  nightmare_strength: 0.8

compression:
  pruning_ratio: 0.2
```

## Expected Outcomes

| Metric | Baseline Model | DreamPhase Model |
|--------|---------------|-----------------|
| Recall | High | Moderate |
| Generalization | Medium | High |
| Robustness | Low | High |
| Hallucination | High | Reduced |

## Project Structure

```
NightmareNet/
├── nightmarenet/          # Core library
│   ├── data/              # Dataset loading & generation
│   ├── distortions/       # Text, semantic, adversarial distortions
│   ├── training/          # Phase-based training pipeline
│   ├── compression/       # Pruning & bottleneck utilities
│   └── evaluation/        # Metrics & evaluation engine
├── configs/               # YAML configuration files
├── scripts/               # CLI entry points
├── tests/                 # Unit tests
├── notebooks/             # Demo notebooks
└── data/                  # Raw & generated datasets
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## License

MIT