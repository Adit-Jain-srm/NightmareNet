# Model-aware learned adversarial distortion

## Overview

The learned adversarial generator supports two strategies:

- `attention`: the legacy DistilBERT-style masked-language-model path. It ranks
  tokens with fallback-model attention and replaces them with plausible masked-LM
  predictions.
- `gradient`: the model-aware path. It computes the current target model's loss
  gradient with respect to input embeddings, ranks words by the L2 norm of those
  gradients, and projects each selected gradient onto the target embedding matrix
  to choose an adversarial replacement.

When the gradient strategy has no target model or compatible tokenizer, it falls
back to the attention strategy. Existing configurations with `learned: 0.0` do not
instantiate or execute either learned generator.

## Configuration

```yaml
distortion:
  adversarial:
    learned: 0.3
    learned_model: "distilbert-base-uncased"
    learned_strategy: "gradient"  # "gradient" | "attention"
    learned_cache: true
    learned_seed: 42
```

`learned_model` remains the attention fallback model. `learned_cache` partitions
cached examples by cycle, allowing the current model to regenerate attacks at the
start of a later cycle while preserving examples generated during earlier cycles.

## Algorithm

For tokenized input embeddings \(E\), the generator performs one target-model
forward pass with gradients enabled and computes:

\[
I_i = \left\| \frac{\partial \mathcal{L}}{\partial E_i} \right\|_2
\]

Subword scores and gradients are averaged back to word positions. For a selected
word with gradient \(g_i\), candidate vocabulary tokens are ranked by:

\[
\operatorname*{argmax}_{v \in V} \langle W_v, g_i \rangle
\]

where \(W\) is the target model's input embedding matrix. The original token and
special tokens are excluded. `torch.autograd.grad(..., create_graph=False)` keeps
the attack graph isolated from model optimization and avoids parameter mutation.

## Cycle integration

`Pipeline.prepare()` creates the trainer before generating cycle-zero nightmare
data so the initial model and tokenizer can be passed to the dataset generator.
At the start of every later training cycle, `Trainer.train()` updates the target
model reference and regenerates nightmare data when gradient learned distortion
is enabled. Cache keys include the cycle identifier, input, strength, and effective
strategy.

## Verification

Run focused offline unit tests:

```bash
pytest tests/test_learned_adversarial.py -v
```

Run the existing distortion regression tests without network access:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
pytest tests/test_distortions.py -v
```

Run the real-model attack microbenchmark:

```bash
python scripts/benchmark_learned_adversarial.py \
  --model distilbert-base-uncased \
  --strength 0.7 \
  --repeats 3 \
  --output artifacts/learned-adversarial-benchmark.json \
  --enforce-slowdown
```

The benchmark uses identical samples and model compute for both strategies and
reports target reconstruction-loss increase, changed outputs, latency per sample,
and the gradient-to-attention slowdown ratio. Run it on the same hardware intended
for the PR evidence; do not copy results across machines.

## Full robustness comparison

The microbenchmark validates attack effectiveness and latency but does not replace
a training robustness benchmark. For PR evidence, run two otherwise identical
one-cycle experiments with `learned: 0.3`, changing only `learned_strategy` between
`attention` and `gradient`. Record:

- clean evaluation accuracy;
- average robustness across the configured distortion sweep;
- total training time and peak memory;
- seed, model, dataset subset, and hardware;
- gradient/attention runtime ratio.

Do not report projected values as measured results. Attach the generated config
files and raw metric artifacts so the experiment is reproducible.
