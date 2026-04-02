# NightmareNet - Repository Knowledge

## Overview
Biologically-inspired training framework for autonomous AI self-improvement via Dream/Nightmare cycles.
- **Author**: Adit Jain | **License**: MIT | **Python**: ≥3.9 | **Version**: 0.2.0

## Architecture: 4-Phase Training Pipeline
1. **Wake** → standard supervised fine-tuning on clean data
2. **Dream** → training on mildly distorted data (strength ~0.25) + KL regularization
3. **Nightmare** → stress-testing with extreme distortions (strength ~0.8) + boosted LR
4. **Compression** → magnitude pruning or bottleneck projection

## Key Modules
- `nightmarenet/training/` — Trainer, phases (Wake/Dream/Nightmare/Compress), schedulers (Cyclic/Adaptive)
- `nightmarenet/distortions/` — text.py (char-level), semantic.py (meaning-level), adversarial.py (reasoning-level)
- `nightmarenet/data/` — HuggingFace loader, Dream/Nightmare dataset generators
- `nightmarenet/evaluation/` — recall, generalization, robustness curves, hallucination rate
- `nightmarenet/compression/` — MagnitudePruner, BottleneckWrapper
- `nightmarenet/api/` — FastAPI server (health, dream/nightmare distortion, robustness eval)
- `nightmarenet/utils/` — config loading/validation, logging

## Default Model: gpt2 (AutoModelForCausalLM)
## Default Dataset: wikitext / wikitext-2-raw-v1 from HuggingFace
## Config: configs/default.yaml (YAML-driven, schema-validated)

## CLI
- `scripts/train.py` — full training pipeline
- `scripts/generate_data.py` — distorted data generation
- `scripts/evaluate.py` — checkpoint evaluation vs baseline

## Tests: ~159 tests across 6 test files
