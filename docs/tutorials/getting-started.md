# Tutorial 1: Getting Started with NightmareNet

Welcome to NightmareNet! This tutorial will help you install NightmareNet, configure your first project, and run your first model robustness evaluation in under five minutes.

---

## 1. Introduction to NightmareNet

NightmareNet is an autonomous adversarial robustness platform designed to protect and harden language and vision models against silent performance degradation and distribution shifts. It utilizes a biologically-grounded learning cycle:

$$\text{Wake (Clean SFT)} \longrightarrow \text{Dream (Mild Augmentation)} \longrightarrow \text{Nightmare (Adversarial Stress)} \longrightarrow \text{Compress (Pruning & Distillation)}$$

This tutorial focuses on running your first evaluation cycle on a pre-trained model.

---

## 2. Installation and Environment Setup

Ensure you have Python 3.9 or higher. We recommend using a virtual environment.

### Step 1: Install from the Repository

Clone the repository and install the package with standard development and API dependencies:

```bash
# Clone the repository
git clone https://github.com/Adit-Jain-srm/NightmareNet.git
cd NightmareNet

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate       # On Windows (PowerShell)
source .venv/bin/activate    # On Unix/macOS

# Install with API and development dependencies
pip install -e ".[dev,api]"
```

### Step 2: Verify Installation

Ensure the Command Line Interface (CLI) is available on your path:

```bash
nightmarenet --help
```

You should see a list of subcommands: `train`, `evaluate`, `benchmark`, `distort`, `foundation`, and `transfer`.

---

## 3. Configuration

NightmareNet runs on declarative configuration files. A default configuration is located at `configs/default.yaml`.

Key config blocks include:
*   **model**: Specifies the target model name (e.g., `gpt2`), model type (e.g. `causal_lm`), and target hardware device.
*   **dataset**: Controls dataset source (e.g., Hugging Face `wikitext` or custom files).
*   **evaluation**: Dictates which metrics (`recall`, `generalization`, `robustness`, `hallucination`) are calculated.

---

## 4. Running Your First Evaluation

You can run an evaluation either using the CLI or directly in Python.

### Option A: Using the CLI (Fastest)

Run a quick robustness sweep over a text string at increasing distortion strengths (from 0.1 to 0.9):

```bash
nightmarenet evaluate --text "The quick brown fox jumps over the lazy dog." --strengths "0.1,0.3,0.5,0.7,0.9"
```

To output results in a structured format suitable for CI/CD gates:

```bash
nightmarenet evaluate --text "Robustness check." --strengths "0.1,0.5,0.9" --json
```

#### Expected CLI Output:
```json
{
  "model": "",
  "dataset": "sst2",
  "robustness_score": 0.654,
  "avg_dream_similarity": 0.852,
  "avg_nightmare_similarity": 0.456,
  "strengths": [
    {
      "strength": 0.1,
      "dream_similarity": 0.952,
      "nightmare_similarity": 0.892,
      "dream_sample": "The quick brown fox jumps...",
      "nightmare_sample": "The quick brown fox jumps..."
    }
  ]
}
```

### Option B: Using the Python API

For integration into python scripts or training loops, instantiate the `Evaluator`:

```python
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from nightmarenet.evaluation.evaluator import Evaluator
from nightmarenet.distortions.registry import get_registry

# 1. Setup model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 2. Setup a dummy dataset and loader
dataset = load_dataset("sst2", split="validation[:10]")
def tokenize_fn(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)
tokenized = dataset.map(tokenize_fn, batched=True)
tokenized.set_format("torch", columns=["input_ids", "attention_mask"])
dataloader = DataLoader(tokenized, batch_size=2)

# 3. Create Evaluator configuration
config = {
    "model": {"name": model_name, "max_length": 128},
    "dataset": {"text_column": "sentence"},
    "evaluation": {
        "metrics": ["recall", "robustness"],
        "robustness_strengths": [0.2, 0.5, 0.8]
    },
    "training": {"batch_size": 2}
}

# 4. Initialize and run
evaluator = Evaluator(model, tokenizer, config, device=device)
registry = get_registry()

results = evaluator.evaluate(
    clean_dataloader=dataloader,
    base_dataset=dataset,
    distortion_fn=registry.apply,
    label="baseline_evaluation"
)

print(f"Clean Recall (Token Accuracy): {results['recall']['token_accuracy']:.2%}")
print(f"Area Under Robustness Curve (AUC): {results['robustness']['auc_robustness']:.4f}")
```

---

## 5. Troubleshooting Common Issues

### Issue 1: `ModuleNotFoundError: No module named 'nightmarenet'`
*   **Cause**: You executed python/pytest from outside the virtual environment, or the package was not installed in editable mode.
*   **Solution**: Ensure your virtual environment is activated and run:
    ```bash
    pip install -e ".[dev,api]"
    ```
    If running scripts manually, set the python path:
    ```bash
    $env:PYTHONPATH="."  # Windows (PowerShell)
    export PYTHONPATH=.  # Unix/macOS
    ```

### Issue 2: `torch` or `transformers` Loading Errors / CUDA OOM
*   **Cause**: Heavy weights loaded on a low VRAM system.
*   **Solution**: Use a smaller model like `distilbert-base-uncased` or run evaluation on `"cpu"` by setting `device: "cpu"` in your script or configuration.

---

## 6. Next Steps

Now that you have run your first evaluation:
1. Learn how to write your own custom perturbations in [Tutorial 2: Custom Distortions](custom-distortions.md).
2. Deep dive into how metrics are calculated and how they apply to compliance frameworks in [Tutorial 3: Interpreting Results](interpreting-results.md).
3. Hardening a model? Check out [Tutorial 5: Deployment](deployment.md) to serve the model locally or run it via the docker container.
