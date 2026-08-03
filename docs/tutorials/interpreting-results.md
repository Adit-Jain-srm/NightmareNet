# Tutorial 3: Interpreting Results and Compliance Reports

NightmareNet outputs structured metrics to verify and report model quality under stress. In this tutorial, we will deep-dive into each metric produced, explain how to interpret robustness scores, and review how to read the generated EU AI Act compliance reports.

---

## 1. Core Metrics Reference

Here is a definition and breakdown of the five primary metrics calculated by the `Evaluator` class:

### 1. Recall Score
*   **Definition**: Measures the model's standard performance (accuracy) on clean, unperturbed in-distribution test datasets.
*   **Mechanism**: Computed using token-level accuracy (how often the model predicts the correct next token) and perplexity (predictive uncertainty).
*   **Ideal target**: Token accuracy $> 80\%$, low perplexity.

### 2. Generalization Score
*   **Definition**: Measures the model's ability to maintain performance on out-of-distribution (OOD) datasets.
*   **Mechanism**: Calculates the ratio between OOD perplexity and clean perplexity:
    
    $$\text{Generalization Ratio} = \frac{\text{OOD Perplexity}}{\text{Clean Perplexity}}$$
    
*   **Score Interpretation**: A ratio close to $1.0$ indicates perfect generalization (no performance drop). A larger ratio (e.g. $> 2.0$) indicates the model is highly sensitive to distribution shifts.

### 3. Robustness Score (AUC)
*   **Definition**: Compares the model's degradation profile across multiple distortion strengths.
*   **Mechanism**: Calculated as the Area Under the Curve (AUC) of the inverse perplexity values plotted against distortion strengths:
    
    $$\text{AUC Robustness} = \int_{0.1}^{0.9} \frac{1}{\text{Perplexity}(s)} \, ds$$
    
*   **Ideal target**: Closer to $1.0$ is better. An AUC increase indicates the model remains stable even under high-strength perturbations.

### 4. Hallucination Rate
*   **Definition**: Measures the proportion of generated tokens/responses that diverge semantically from reference contexts.
*   **Mechanism**: Evaluates semantic alignment, self-contradiction, and factual drift across generated outputs.

### 5. Classification Metrics
*   **Definition**: Evaluates standard sequence classification quality.
*   **Mechanism**: Produces Precision, Recall, Accuracy, and F1-scores.

---

## 2. EU AI Act Article 15 Compliance Reports

Article 15 of the **EU AI Act** mandates that high-risk AI systems demonstrate adequate robustness, accuracy, and cybersecurity. NightmareNet provides an automated reporting module ([report.py](../../nightmarenet/compliance/report.py)) that produces compliance reports as signed Markdown and JSON files.

### Enabling Compliance Reports

Set `compliance_report: true` in the `tracking` section of your configuration file:

```yaml
tracking:
  backend: none
  output_dir: "results"
  compliance_report: true
```

When you run a training or evaluation pipeline, NightmareNet automatically saves two files in your output directory:
1.  `<run_id>_compliance_report.json`
2.  `<run_id>_compliance_report.md`

---

## 3. Understanding the Compliance Report

Let's dissect a generated markdown compliance report.

### Section 1: Model & Dataset Metadata
Captures the lineage of the model and training set:
*   **Model Name**: The pre-trained base model name.
*   **Dataset Name & Path**: Tracks training data provenance to ensure reproducibility.

### Section 2: Artifact Integrity (Cybersecurity)
Computes cryptographic SHA-256 hashes of critical files:
*   `config_sha256`: Hash of the exact training configuration mapping.
*   `model_sha256`: Hash of the resulting model weights (e.g., `pytorch_model.bin` or `model.safetensors`).
*   **Why it matters**: Provides proof that the verified weights have not been altered prior to deployment.

### Section 3: Robustness & Accuracy Results
Summarizes metrics:
*   **Clean Accuracy**: Baseline performance.
*   **Distorted Accuracy**: Average accuracy under distortions.
*   **AUC Robustness**: Calculated degradation profile.
*   **Delta**: The difference in AUC between the trained model and baseline. A positive delta (e.g., $+0.14$) shows significant robustness gains.

### Section 4: Mappings to Standards
*   **EU AI Act Mapping**:
    *   *Accuracy*: Linked to Clean Recall/Perplexity logs.
    *   *Robustness*: Map to the AUC metrics table.
    *   *Cybersecurity*: Tied directly to file integrity hashes.
*   **NIST AI RMF Mapping**: Maps outputs to NIST functions: Govern (Lineage), Map (Metadata), Measure (Metrics), and Manage (Integrity).

---

## 4. Good vs. Poor Robustness

When analyzing degradation curves, look at how the model degrades across strengths:

| Degradation Profile | AUC Robustness | Interpretation | Recommendation |
| :--- | :--- | :--- | :--- |
| **Resilient** | $0.85 - 1.00$ | Stable perplexity even at high strengths ($0.7 - 0.9$). | Ready for production. |
| **Sensitive** | $0.50 - 0.84$ | Moderate degradation. Perplexity increases slowly. | Consider dream-phase SFT. |
| **Vulnerable** | $< 0.50$ | Severe degradation. Perplexity spikes instantly at low strength. | Run nightmare-phase adversarial training. |

### Misconception: "High Clean Accuracy Equals Robustness"
A model with $95\%$ clean accuracy can drop to $<10\%$ under a single-token adversarial distortion. Never rely solely on clean datasets; evaluate under increasing distortion sweeps using the `evaluate` tool.
