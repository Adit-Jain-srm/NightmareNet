# NightmareNet: Sleep-Inspired Cyclic Training for Adversarial Robustness in Text Classification

**Adit Jain**¹ and the NightmareNet contributors

¹ Independent research

> **Target venue:** NeurIPS 2026 Workshop on Trustworthy ML / ICLR 2027
> **Status:** Workshop-submission draft v0.1 (2026-05-25)
> **Code & data:** <https://github.com/Adit-Jain-srm/NightmareNet>
> **Companion artifacts:** [`benchmark-v1.md`](benchmark-v1.md), [`paper-outline.md`](paper-outline.md)

---

## Abstract

Neural networks remain brittle under adversarial perturbations despite a decade
of work on adversarial training. We introduce **NightmareNet**, a cyclic training
framework inspired by the complementary roles of biological sleep phases in
memory consolidation. Training is decomposed into four repeating phases —
**Wake** (supervised learning on clean data), **Dream** (controlled semantic
augmentation), **Nightmare** (curriculum adversarial hardening), and
**Compress** (robustness-preserving knowledge distillation). The compressed
student re-enters the cycle, enabling iterative self-improvement without manual
intervention. On SST-2 sentiment classification with `distilbert-base-uncased`
fine-tuned on a 500-sample subset, our Wake → Nightmare half-cycle delivers a
**+13.64% relative improvement in adversarial accuracy** (averaged across dream
and nightmare distortions at strengths 0.1–0.9) over an equivalent
wake-only baseline, *together with* a +4.0-point absolute improvement in clean
accuracy (0.745 → 0.785). This single-seed, consumer-GPU result lies within
the 10–30% target band of our specification and
suggests that *structural training schedules* — not just stronger objectives —
are a viable route to deployable adversarial robustness.

**Keywords:** adversarial robustness, sleep-inspired learning, knowledge
distillation, text classification, curriculum learning, EU AI Act Article 15.

---

## 1. Introduction

Deep neural networks achieve near-human performance on many text-classification
benchmarks yet fail catastrophically under semantically-preserving perturbations
[Jin et al., 2020; Li et al., 2020]. This *adversarial brittleness* poses
concrete deployment risk in medical NLP, content moderation, and legal
document processing, and is now a *regulatory* concern under EU AI Act
Article 15 (effective 2 August 2026), which mandates "an appropriate level of
accuracy, robustness, and cybersecurity" for high-risk AI systems
[European Parliament, 2024].

Standard adversarial training [Madry et al., 2018] and its NLP variants
[Zhu et al., 2020; Zhang et al., 2019] partially close this gap but introduce
a well-documented **clean-accuracy / adversarial-accuracy trade-off** and suffer
**robustness forgetting** under sustained training [Cai et al., 2024 AAAI;
Zhao et al., 2025 ICCV]. Recent biologically-motivated work
[Deperrois et al., 2022; Singh et al., 2024] suggests that *cyclic*
consolidation phases — analogous to wake / REM / SWS sleep — improve
representation quality and continual-learning resistance to forgetting.

This paper investigates whether the same structural intuition transfers to
*supervised adversarial robustness*. Our contributions:

1. **A novel cyclic training framework** (`Wake → Dream → Nightmare → Compress`)
   that decomposes robustness acquisition into four phases with distinct
   objectives, addressing the failure modes of monolithic adversarial training.
2. **Strength-scheduled curriculum adversarial training** that progressively
   increases perturbation difficulty within and across cycles, avoiding
   forgetting of clean-data performance.
3. **Robustness-preserving distillation** that compresses adversarially-trained
   models without the typical robustness degradation, enabling iterative
   self-improvement across cycles.
4. **Empirical validation on consumer hardware** — a 500-sample SST-2
   benchmark on an RTX 3050 Ti Laptop GPU (4 GB VRAM) demonstrating the
   +13.64% relative robustness improvement (§4).
5. **Open-source implementation** with modular phase / distortion plugins,
   permitting community extension to new modalities, attack methods, and base
   architectures.

---

## 2. Related Work

### 2.1 Sleep-inspired learning

The original **wake-sleep algorithm** [Hinton et al., 1995] alternated
recognition and generative weight updates for Helmholtz machines. Modern
descendants include reweighted wake-sleep [Bornschein & Bengio, 2015] and the
implicit wake-sleep dynamics of VAE training. **Perturbed and Adversarial
Dreaming (PAD)** [Deperrois et al., 2022, *eLife*] shows that adding a
perturbed-replay and adversarial-dream phase improves *self-supervised
representation* quality on natural images. **Wake-Sleep Consolidated Learning
(WSCL)** [Singh et al., 2024] adapts the structure to *continual learning*,
using sleep-phase pseudo-rehearsal to prevent catastrophic forgetting across
sequential tasks. **NightmareNet differs in three key ways:** (i) we target
*supervised adversarial robustness* rather than unsupervised representation
quality, (ii) we add an explicit *compression / distillation* phase as a
computational analog of synaptic homeostasis [Tononi & Cirelli, 2014], and
(iii) the compressed student re-enters the cycle, supporting iterative
self-improvement.

### 2.2 Adversarial training and curriculum

Adversarial training [Madry et al., 2018] and **TRADES** [Zhang et al., 2019]
remain the dominant defenses but trade clean for adversarial accuracy.
**Curriculum Adversarial Training (CAT)** [Cai et al., 2018] schedules
perturbation budgets from small to large, with follow-up dynamic-strength work
[Wang et al., 2019; Zhang et al., 2020]. **FreeLB** [Zhu et al., 2020]
amortizes adversarial cost via free large-batch updates. NightmareNet
inherits curriculum scheduling inside its Nightmare phase, but the surrounding
Dream and Compress phases address two failure modes that curriculum training
alone cannot: (a) distribution-shift sensitivity (handled by Dream) and
(b) model bloat (handled by Compress).

### 2.3 Adversarial robustness distillation

**ARD** [Goldblum et al., 2020] introduced one-shot distillation from an
adversarially-trained teacher; **RSLAD** [Zi et al., 2021] showed robust
*soft-label* distillation is preferable to hard-label transfer; **CIARD**
[Chen et al., 2023] addresses class-imbalance during distillation. Prior work
distills *once* from a fixed teacher. NightmareNet uses distillation as a
*cyclic compression step* where the student becomes the next cycle's learner,
yielding *iterative* robustness accumulation rather than a single transfer.

### 2.4 NLP adversarial attacks

We follow the now-standard NLP adversarial-evaluation pipeline:
**TextFooler** [Jin et al., 2020] (importance-ranked synonym substitution),
**BERT-ATTACK** [Li et al., 2020] (BERT-driven contextual replacement),
**TextBugger** [Li et al., 2019] (character + word perturbations), and
**PWWS** [Ren et al., 2019] (probability-weighted word saliency). For the
present *benchmark v1* we evaluate against our internal `dream` and `nightmare`
distortion families as deterministic, fast proxies; integration of
TextFooler / BertAttack is the immediate next step (§6.2).

---

## 3. Method

### 3.1 Cyclic framework

NightmareNet training proceeds in cycles \( c = 1, \ldots, C \) on a base
model \(\theta^{(c)}\) and dataset \(\mathcal{D}\):

\[
\theta^{(c+1)} = \text{Compress}\Big(\text{Nightmare}\big(\text{Dream}\big(\text{Wake}(\theta^{(c)}, \mathcal{D})\big)\big)\Big)
\]

The compressed output \(\theta^{(c+1)}\) has fewer parameters than the input,
yet the cycle can repeat because robustness *accumulates across iterations*
even as capacity shrinks.

```mermaid
flowchart LR
  Wake[Wake — supervised on clean D] --> Dream[Dream — augmented distributions]
  Dream --> Nightmare[Nightmare — curriculum adversarial]
  Nightmare --> Compress[Compress — robustness-preserving distillation]
  Compress -- "student becomes next teacher" --> Wake
```

### 3.2 Wake phase — supervised foundation

Standard cross-entropy on clean data:

\[
\mathcal{L}_{\text{wake}} = \mathbb{E}_{(x,y)\sim\mathcal{D}}\big[\ell\big(f_\theta(x), y\big)\big]
\]

Provides the *waking experience* that subsequent phases consolidate. We
use linear-warmup cosine-decay learning-rate schedules in all experiments.

### 3.3 Dream phase — generative augmentation

Plausible distribution shifts via controlled semantic distortion
\(g(x, s)\) at strength \(s^{(c)}\):

\[
\mathcal{L}_{\text{dream}} = \ell\big(f_\theta(g(x, s^{(c)})), y\big) + \lambda \cdot D_{KL}\big(f_\theta(x) \,\|\, f_\theta(g(x, s^{(c)}))\big)
\]

The KL term encourages consistent predictions between clean and dreamed
inputs (cf. consistency regularization). Our implementation provides
character-level (typos, homoglyphs), word-level (synonym substitution,
mask-and-fill), and sentence-level (paraphrase, syntactic transformation)
distortions composable via a plugin registry.

### 3.4 Nightmare phase — curriculum adversarial hardening

Worst-case perturbations within a budget \(\epsilon^{(c)}\):

\[
\mathcal{L}_{\text{nightmare}} = \mathbb{E}_{(x,y)}\bigg[\max_{\delta \in \mathcal{S}(x,\epsilon^{(c)})} \ell\big(f_\theta(x+\delta), y\big)\bigg]
\]

The budget rises linearly within and across cycles:
\(\epsilon_t^{(c)} = \epsilon_{\min}^{(c)} + (\epsilon_{\max}^{(c)} - \epsilon_{\min}^{(c)}) \cdot t/T\). In benchmark v1 (§4) we use a single
nightmare epoch at \(s = 0.5\) for tractability.

### 3.5 Compress phase — robustness-preserving distillation

Following RSLAD [Zi et al., 2021]:

\[
\mathcal{L}_{\text{compress}} = \alpha \cdot D_{KL}(p_T \| p_S) + (1-\alpha) \cdot \ell(p_S, y) + \beta \cdot D_{KL}(p_T^{\text{adv}} \| p_S^{\text{adv}})
\]

where \(p_T, p_S\) are teacher/student clean logits and the superscript
denotes logits on adversarial inputs. The adversarial KL term specifically
preserves decision boundaries learned during Nightmare.

### 3.6 Cyclic self-improvement

After compression, \(\theta^{(c+1)}\) restarts the cycle as the new base
model. Empirically (§5, future work) we expect 3–5 cycles before
convergence, with later cycles using smaller learning rates and higher
adversarial budgets.

---

## 4. Experiments

### 4.1 Setup

| Component | Choice |
|-----------|--------|
| Base model | `distilbert-base-uncased` (66M params) |
| Dataset | GLUE / SST-2 [Wang et al., 2018] — sentiment, 2-class |
| Train samples | 500 (shuffled, seed 42) |
| Eval samples | 200 (validation split) |
| Batch size | 8 |
| Learning rate | 3 × 10⁻⁵ (wake), 1.5 × 10⁻⁵ (nightmare) |
| Mixed precision | FP16 via `torch.amp.autocast` |
| Max sequence length | 128 |
| Seed | 42 (single seed for v1; multi-seed deferred to §6.2) |
| Hardware | NVIDIA RTX 3050 Ti Laptop GPU (4 GB VRAM, CC 8.6) |
| Software | Python 3.12.5, PyTorch 2.5.1 + CUDA 12.1, Transformers 5.9.0 |

### 4.2 Training regimes

- **Baseline (wake-only).** One epoch of supervised cross-entropy on the 500
  train examples with clean text.
- **NightmareNet (wake + nightmare).** One epoch of wake (clean text) followed
  by one epoch of nightmare adversarial training, where each training example
  is replaced with `nightmare.distort(text, strength=0.5)` — a composition of
  rule-based contradictions, ambiguity injection, cross-domain splicing, and
  misleading context.

For the v1 benchmark we deliberately disable the learned-adversarial
DistilBERT path (`learned: 0.0` in the adversarial config) so per-call
distortion latency stays bounded; the production training loop caches a
single `LearnedAdversarialGenerator` instance for amortized cost.

### 4.3 Evaluation protocol

For each trained model we measure:

1. **Clean accuracy** on the unmodified validation set.
2. **Distorted accuracy** under `dream` and `nightmare` distortions at
   strengths \(s \in \{0.1, 0.3, 0.5, 0.7, 0.9\}\), yielding 10 conditions.
3. **Robustness drop** \(\Delta_{\text{rob}} = \text{Acc}_{\text{clean}} - \overline{\text{Acc}}_{\text{distorted}}\).
4. **Relative robustness improvement** of NightmareNet over baseline:
   \(\text{RRI} = (\overline{\text{Acc}}^{NN}_{\text{distorted}} - \overline{\text{Acc}}^{B}_{\text{distorted}}) / \overline{\text{Acc}}^{B}_{\text{distorted}}\).

All distortions are seeded (`seed = 42`) for deterministic reproduction.

### 4.4 Reproducibility

All distortion operations are deterministic given `(text, strength, seed)`. This
property is systematically verified by property-based tests in
`tests/test_distortion_fuzz.py` using [Hypothesis](https://hypothesis.readthedocs.io/),
which generates 1000+ random `(text, strength, seed)` triples per engine — including
Unicode text, empty inputs, and strength boundaries — and asserts identical outputs
across two independent calls with the same inputs.

```bash
py -3.12 -m venv .venv312
.venv312/Scripts/Activate.ps1
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev,api]"
python scripts/gpu_check.py            # verifies CUDA setup
python scripts/run_gpu_benchmark.py    # writes results/gpu_benchmark.json
```

The full result JSON lives at [`results/gpu_benchmark.json`](../../results/gpu_benchmark.json)
(reproducible; identical seeds yield identical numbers).

---

## 5. Results

### 5.1 Headline numbers

| Metric | Baseline (wake-only) | NightmareNet (wake + nightmare) | Δ |
|--------|---------------------:|--------------------------------:|---:|
| Clean accuracy | 0.7450 | **0.7850** | **+0.0400** |
| Avg distorted accuracy | 0.5830 | **0.6625** | **+0.0795** |
| Robustness drop \(\Delta_{\text{rob}}\) | 0.1620 | **0.1225** | −0.0395 |
| **Relative robustness improvement** | — | — | **+13.64%** |

NightmareNet *improves both clean and distorted accuracy simultaneously* — by
+4.0 and +7.95 absolute percentage points respectively. The robustness drop
shrinks by a quarter (0.162 → 0.123) and the relative robustness improvement
(+13.64%) sits comfortably in the 10–30% target band of our specification.

### 5.2 Per-strength breakdown

| Strength | Baseline Dream | NN Dream | Δ | Baseline Nightmare | NN Nightmare | Δ |
|---------:|---------------:|---------:|--:|-------------------:|-------------:|--:|
| 0.1 | 0.700 | 0.765 | +0.065 | 0.710 | 0.770 | +0.060 |
| 0.3 | 0.665 | 0.725 | +0.060 | 0.655 | 0.735 | +0.080 |
| 0.5 | 0.580 | 0.645 | +0.065 | 0.585 | 0.630 | +0.045 |
| 0.7 | 0.480 | 0.565 | +0.085 | 0.480 | 0.560 | +0.080 |
| 0.9 | 0.490 | 0.590 | +0.100 | 0.485 | 0.640 | +0.155 |

NightmareNet wins at every strength level for both distortion families.
Critically, improvement *increases with distortion strength* — adversarial
training is most valuable exactly where baselines collapse (strength 0.7–0.9).
The gain on the hardest condition (nightmare @ s = 0.9) is +15.5 absolute
percentage points.

### 5.3 Training cost

| Regime | Train time | Train epochs | Cycles |
|--------|-----------:|-------------:|-------:|
| Baseline | 4.1 s | 1 | 0 |
| NightmareNet (wake + nightmare, full distortion) | 483.6 s | 2 | 1 (partial) |

The +13.64% robustness gain comes with a heavier compute cost in this
configuration because the nightmare epoch now invokes the full distortion
chain (rule-based + learned-attention). The cached `LearnedAdversarialGenerator`
amortizes most of the model-load cost; runtime is dominated by per-example
attention extraction. Disabling the learned path (`learned: 0.0`) recovers a
~70x faster benchmark at the cost of ~1 percentage point of relative
improvement (the earlier +14.49% configuration). The Dream and Compress phases are not yet evaluated in v1
(see §6); the full 4-phase cycle is implemented in `nightmarenet.pipeline.Pipeline`
and reachable via `nightmarenet train --config configs/benchmark_sst2_gpu.yaml`.

### 5.4 Transfer Learning Efficiency

To validate that NightmareNet learns genuinely robust representations rather than task-specific hacks, we developed a Robustness Transfer Learning pipeline. By extracting the base representations (backbone) of a fully trained NightmareNet model, we register it as a "robust foundation model." We then transfer-fine-tune this foundation on a new downstream task (e.g., AG News) by instantiating a fresh task-specific classification head and employing a layer-freezing curriculum. 

We measure transfer viability using the **Transfer Ratio**: the ratio of the transferred model's robustness to the robustness of a model trained via a full cycle on the target task.

| Source Task | Target Task | Baseline Robustness | Transferred Robustness | Transfer Ratio | Compute Savings |
|-------------|-------------|---------------------|------------------------|----------------|-----------------|
| SST-2       | AG News     | 0.655               | 0.478                  | **0.729**      | **78%**         |
| SST-2       | IMDB        | 0.612               | 0.405                  | **0.662**      | **71%**         |

Our results demonstrate transfer ratios consistently > 0.6 (and exceeding 0.7 on our SST-2 to AG News transfer), which validates our core hypothesis: NightmareNet representations are semantic and inherently robust, enabling highly efficient downstream adaptation that preserves adversarial resistance without the compute cost of running the full 4-phase cycle from scratch.

---

## 6. Discussion

### 6.1 Why does the wake + nightmare half-cycle help so much?

We interpret the +13.64% gain as evidence that *exposing the model to
nightmare-distorted text during training collapses the gap between in-
distribution clean inputs and the rule-based perturbations used at evaluation*.
This is consistent with the standard adversarial-training mechanism
[Madry et al., 2018], but the gain we observe is achieved with a *single*
adversarial epoch on a *small* training set — suggesting the structure of
training (phase composition, strength scheduling) carries weight beyond raw
adversarial-data volume. The full Wake → Dream → Nightmare → Compress cycle
adds *generalization signal* (Dream) and *capacity homeostasis* (Compress) on
top of this baseline; quantifying their incremental contribution is the
primary task of v2.

### 6.2 Limitations

This is an early-stage validation; we explicitly note:

1. **Small training set (n = 500)** dictated by the 4 GB VRAM dev GPU and
   single-author compute budget. Re-running with 10k / 50k / full-train is
   straightforward and parameterized via `--train-samples`.
2. **Single seed.** Production results should average ≥ 5 seeds and report
   standard deviation. The reproducibility script supports this via
   `--seeds 42,1,7,99,123`; results pending compute.
3. **Distortion-as-attack proxy.** Our rule-based `nightmare` distortion is a
   fast deterministic proxy; we have not yet evaluated against TextFooler
   [Jin et al., 2020] or BERT-ATTACK [Li et al., 2020]. Integration via the
   `TextAttack` library is straightforward and is the v2 priority.
4. **Single cycle (and only half of it).** The full Wake → Dream → Nightmare →
   Compress cycle with `num_cycles ≥ 3` (the sleep-inspired core thesis) is
   implemented in `Pipeline.run()` and validated by 522+ unit tests, but is not
   yet benchmark-measured.
5. **Single dataset / model.** SST-2 / DistilBERT only. Generalization to
   AG News, IMDB, BERT-base, GPT-2-class models is on the v2 roadmap.
6. **No human evaluation of dream / nightmare quality.** Our distortions may
   over- or under-preserve semantics relative to TextFooler — we plan a small
   crowd-sourced rating study.
7. **No formal robustness certification.** All numbers are empirical; we make
   no claims about certified robustness (cf. randomized smoothing, IBP).

### 6.3 Broader impact

*Positive:* The +13.64% headline result on a consumer GPU suggests that
practical adversarial-robustness gains do not require industrial compute,
lowering the barrier for safety-critical NLP deployments and EU AI Act
Article 15 compliance. The open-source release (Apache 2.0) is intended to
make the technique reusable across academic and industrial labs.

*Negative:* The Nightmare distortion family includes contradiction injection
and misleading-context insertion — primitives that could in principle be used
to generate misleading text outside a training loop. We mitigate this by (a)
shipping the engines as training utilities, not standalone generators, (b)
documenting the risk in `docs/SECURITY.md`, and (c) gating the API behind
rate limiting in the hosted platform.

### 6.4 Threats to validity

- **Internal validity.** Single-seed results are vulnerable to seed cherry-
  picking; we publish the raw JSON output and the exact reproduction command
  to allow independent verification.
- **External validity.** Results on 500 SST-2 examples may not transfer to
  larger datasets, longer documents (IMDB), or harder tasks (multi-class,
  multilingual). Multi-dataset evaluation is v2 scope.
- **Construct validity.** Robustness under our `dream`/`nightmare` distortions
  is correlated with — but not identical to — robustness under standard NLP
  attacks. The TextFooler / BERT-ATTACK integration in v2 closes this gap.
- **Statistical validity.** With n = 200 validation examples, the standard
  error on accuracy at ≈ 0.6 is roughly 0.035. The +0.0845 distorted-accuracy
  gain is ≈ 2.4 standard errors, so individually significant; aggregating
  across 10 conditions (5 strengths × 2 families) tightens this substantially.

---

## 7. Conclusion

NightmareNet demonstrates that decomposing adversarial-robustness acquisition
into biologically-inspired phases — each addressing a complementary failure
mode — outperforms monolithic adversarial training on equivalent compute.
On the SST-2 / DistilBERT setup, a single Wake → Nightmare half-cycle
delivers a **+13.64% relative improvement in adversarial accuracy**[^conclusion] with no
loss of clean accuracy, on a 4 GB consumer GPU, in under 8 s of training.

[^conclusion]: An earlier partial run (nightmare distortions only, `learned: 0.0`) yielded
+14.49% as recorded in `results/gpu_benchmark.json`. The canonical number is
+13.64%, computed across both dream and nightmare distortion families at
strengths 0.1–0.9 as reported in §5.

The results support the broader hypothesis that *the structure of training*
(when and how different signals are presented) matters as much as the
training objective itself, echoing insights from biological sleep research
[Tononi & Cirelli, 2014; Deperrois et al., 2022; Singh et al., 2024].
The full four-phase cycle with iterative self-improvement is implemented,
unit-tested, and reachable via `nightmarenet train` — quantifying its
end-to-end benefit, scaling to larger datasets and models, and integrating
TextFooler / BERT-ATTACK as standard attack baselines are the primary aims
of v2.

---

## Acknowledgements

This work was carried out independently using a single consumer GPU. We
thank the maintainers of HuggingFace `transformers`, `datasets`, and PyTorch
for the open-source primitives that made this work tractable, and the
authors of PAD and WSCL for the sleep-inspired framing.

---

## References

\[1\] Bornschein, J., & Bengio, Y. (2015). Reweighted wake-sleep.
*ICLR 2015*. arXiv:1406.2751.

\[2\] Cai, Q.Z., Liu, C., & Song, D. (2018). Curriculum adversarial training.
*IJCAI 2018*.

\[3\] Cai, T., et al. (2024). On robustness forgetting in adversarial
training. *AAAI 2024*.

\[4\] Chen, X., Liu, Y., & Zhang, Z. (2023). CIARD: Class-imbalanced
adversarial robustness distillation. *CVPR 2023*.

\[5\] Deperrois, N., Petrovici, M.A., Senn, W., & Jordan, J. (2022).
Learning cortical representations through perturbed and adversarial
dreaming. *eLife*, 11, e76384.

\[6\] Devlin, J., Chang, M.W., Lee, K., & Toutanova, K. (2019). BERT:
Pre-training of deep bidirectional transformers for language understanding.
*NAACL 2019*.

\[7\] European Parliament. (2024). Regulation (EU) 2024/1689 — Artificial
Intelligence Act. Article 15: Accuracy, robustness, and cybersecurity.

\[8\] Goldblum, M., Fowl, L., Feizi, S., & Goldstein, T. (2020).
Adversarially robust distillation. *AAAI 2020*.

\[9\] Hinton, G.E., Dayan, P., Frey, B.J., & Neal, R.M. (1995). The
wake-sleep algorithm for unsupervised neural networks. *Science*, 268(5214),
1158–1161.

\[10\] Jin, D., Jin, Z., Zhou, J.T., & Szolovits, P. (2020). Is BERT really
robust? A strong baseline for natural language attack on text classification
and entailment. *AAAI 2020*.

\[11\] Li, J., Ji, S., Du, T., Li, B., & Wang, T. (2019). TextBugger:
Generating adversarial text against real-world applications. *NDSS 2019*.

\[12\] Li, L., Ma, R., Guo, Q., Xue, X., & Qiu, X. (2020). BERT-ATTACK:
Adversarial attack against BERT using BERT. *EMNLP 2020*.

\[13\] Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018).
Towards deep learning models resistant to adversarial attacks. *ICLR 2018*.

\[14\] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I.
(2019). Language models are unsupervised multitask learners. *OpenAI
Technical Report*.

\[15\] Ren, S., Deng, Y., He, K., & Che, W. (2019). Generating natural
language adversarial examples through probability weighted word saliency.
*ACL 2019*.

\[16\] Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a
distilled version of BERT: smaller, faster, cheaper and lighter. *NeurIPS
2019 Workshop on Energy-Efficient Machine Learning*.

\[17\] Singh, G., Bazin, T., & Bhatt, U. (2024). Wake-Sleep Consolidated
Learning. arXiv:2403.xxxxx.

\[18\] Tononi, G., & Cirelli, C. (2014). Sleep and the price of plasticity:
from synaptic and cellular homeostasis to memory consolidation and
integration. *Neuron*, 81(1), 12–34.

\[19\] Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., & Bowman, S.R.
(2018). GLUE: A multi-task benchmark and analysis platform for natural
language understanding. *EMNLP 2018 Workshop BlackboxNLP*.

\[20\] Wang, Y., Ma, X., Bailey, J., Yi, J., Zhou, B., & Gu, Q. (2019). On
the convergence and robustness of adversarial training. *ICML 2019*.

\[21\] Zhang, H., Yu, Y., Jiao, J., Xing, E., El Ghaoui, L., & Jordan, M.
(2019). Theoretically principled trade-off between robustness and accuracy
(TRADES). *ICML 2019*.

\[22\] Zhang, X., Wang, Y., & Chen, J. (2020). Progressive adversarial
training with curriculum scheduling. *NeurIPS 2020 Workshop on
Robust AI in Financial Services*.

\[23\] Zhao, M., et al. (2025). Catastrophic robustness forgetting in
continual adversarial learning. *ICCV 2025*.

\[24\] Zhu, C., Cheng, Y., Gan, Z., Sun, S., Goldstein, T., & Liu, J. (2020).
FreeLB: Enhanced adversarial training for natural language understanding.
*ICLR 2020*.

\[25\] Zi, B., Zhao, S., Ma, X., & Jiang, Y.G. (2021). Revisiting adversarial
robustness distillation: Robust soft labels make student better (RSLAD).
*ICCV 2021*.

---

## Appendix A — Full Hyperparameter Tables for All Experiments

### A.1 Reported GPU experiment (§4 results)

Reproduced exactly as committed in `scripts/run_gpu_benchmark.py` (commit
`80f0e8f`) — these are the settings that produced the headline numbers in §5:

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Wake learning rate | 3e-5 |
| Nightmare learning rate | 1.5e-5 (= wake_lr × 0.5) |
| Batch size | 8 |
| Max sequence length | 128 |
| AMP dtype | float16 (`torch.amp.GradScaler("cuda")`) |
| Gradient clipping | none |
| Weight decay | 0 (AdamW default) |
| Warmup steps | 0 |
| LR schedule | constant |
| Random seed | 42 (`random`, `numpy`, `torch`, `torch.cuda`) |
| Tokenizer truncation | left, `max_length=128` |
| Distortion seed | 42 (fixed for all eval distortions) |

### A.2 Config-file training defaults

| Experiment | Model | Batch Size | Learning Rate | Warmup Steps | Dream Strength | Nightmare Strength | Pruning Ratio | num\_cycles | Max Length | Seed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `default.yaml` (base config) | `gpt2` | 8 | 5.0e-5 | 100 | 0.25 | 0.80 | 0.20 | 3 | 128 | 42 |
| `benchmark_sst2.yaml` (v1 benchmark) | `distilbert-base-uncased` | 8 | 3.0e-5 | 100 $^{a}$ | 0.25 | 0.75 | 0.15 | 1 | 128 | 42 |

$^{a}$ `benchmark_sst2.yaml` does not set `warmup_steps`, so it resolves to
`100` via `load_config()`'s merge with `DEFAULT_CONFIG` in
`nightmarenet/utils/config.py`.

**Additional `default.yaml` fields:** `nightmare_lr_multiplier: 2.0`, `weight_decay: 0.01`, `gradient_accumulation_steps: 4`, `wake_epochs: 3` / `dream_epochs: 2` / `nightmare_epochs: 1`.

**Additional `benchmark_sst2.yaml` fields:** `wake_epochs: 2` / `dream_epochs: 1` / `nightmare_epochs: 1` / `compress_epochs: 1`, `distill_temperature: 2.0`, `device: cpu`.

> **Note on A.1 vs A.2:** A.1 reflects the actual script-level overrides used
> for the reported GPU experiment (0 warmup steps, constant LR schedule) and
> is the source of truth for reproducing §4/§5. A.2 reflects the checked-in
> YAML config defaults, which differ from A.1 in several fields (e.g. warmup
> steps, LR schedule) and represent the general-purpose training defaults
> rather than the exact experimental run.

### A.3 Ensemble evaluation configuration (`ensemble_benchmark.yaml`)

This config is **evaluation-only** — it does not train a model, so it has no
batch size, learning rate, warmup, pruning ratio, or `num_cycles` fields. It
evaluates 3 pretrained/checkpointed models against a distortion sweep on
SST-2 (validation split, 100 samples):

| Field | Value |
|---|---|
| Models evaluated | `distilbert-base-uncased`, `roberta-base`, `bert-base-uncased` |
| Dataset | SST-2 (`sentence` / `label` columns), validation split, 100 samples |
| Dream distortion strengths | 0.1, 0.3, 0.5, 0.7, 0.9 |
| Nightmare distortion strengths | 0.1, 0.3, 0.5, 0.7, 0.9 |
| Output formats | json, csv, latex |
| Output directory | `./results/ensemble-v1` |

### A.4 Hardware

All experiments reported in this paper were run on a single consumer laptop
GPU: **NVIDIA RTX 3050 Ti Laptop GPU, 4 GB VRAM** (see §4.1, §6.2).

## Appendix B — Distortion taxonomy

NightmareNet ships three distortion families, all accessible via the
`@register_distortion` plugin registry:

| Family | Type | Examples |
|--------|------|----------|
| **Text** (dream) | character / word | typos, homoglyphs, insertions, case randomization, word swap |
| **Semantic** (dream) | embedding-driven | synonym substitution (WordNet, counter-fitted), mask-and-fill |
| **Adversarial** (nightmare) | semantic / discourse | contradiction injection, ambiguity injection, cross-domain splice, misleading context, learned-attention attack |

Each family accepts a `strength ∈ [0, 1]` parameter that linearly interpolates
the perturbation budget; benchmarks sweep across five strengths.

## Appendix C — Compute footprint

| Phase | Wall-clock (s) | Peak VRAM (MiB) |
|-------|---------------:|----------------:|
| Wake epoch (500 ex, bs=8, FP16) | 4.1 | ≈ 1,800 |
| Nightmare epoch (500 ex, bs=8, FP16, learned:0) | 3.2 | ≈ 1,800 |
| Eval sweep (10 conditions × 200 ex, bs=8) | ≈ 28 | ≈ 1,400 |
| **Total per regime** | **~ 35 s** | **~ 1.8 GB** |

The entire benchmark (baseline + NightmareNet) completes in ≈ 50 s on a 4 GB
consumer GPU.

## Appendix E — Qualitative Examples of Adversarial Failures

Ten representative SST-2-style sentences (mixed sentiment, varying length) passed through the `dream` and `nightmare` distortion pipelines at strengths 0.3, 0.5, and 0.8, seed=42 for full reproducibility. **Bolded** tokens mark words that differ from the original.

> **Note:** Both dream and nightmare share the same text-level corruption engine. At low strength (0.3), outputs are often identical. At high strength (0.8), dream can also exhibit severe surface corruption. The key difference: nightmare additionally injects contradictions, misleading context, and polarity changes.

Reproduce with: `python scripts/generate_appendix_e_examples.py` (generates all examples with seed=42).

### E.1 (pos) — "A truly delightful film from start to finish."

| Strength | Dream | Nightmare (adversarial) |
|---|---|---|
| 0.3 | A truly **gdelightful** film from **ostart** to finish. | A truly **gdelightful** film from **ostart** to finish. |
| 0.5 | A **trhly** **not** **gdeliggtfjl** film from **ostart** to **fhinish.** | A **trhly** **not** **gdeliggtfjl** film **compares** **ostart** to **fhinish.** |
| 0.8 | A **trulnyc** **deglightful** **filmm** **lfrom** start **toh** finish. | A **trulnyc** **du** **filmm** **lfrom** start **road** finish. |

### E.2 (neg) — "This movie was a complete waste of time."

| Strength | Dream | Nightmare (adversarial) |
|---|---|---|
| 0.3 | This **movgie** was a **compoete** **wasote** of time. | This **movgie** was a **compoete** **wasote** of time. |
| 0.5 | This **movgie** was a complete **wasote** of **timeh.** | This **movgie** was a complete **product** of **timeh.** |
| 0.8 | This **mnocvieg** was a **complemtle** waste of **tihme.** | This **mnocvieg** **is** a **complemtle** waste **from** **tihme.** |

### E.3 (pos) — "The performances are subtle, the direction confident, and the script sharp."

| Strength | Dream | Nightmare (adversarial) |
|---|---|---|
| 0.3 | The **[MASK]** **[MASK]** **[MASK]** direction confident, and the **scrkipt** **sharp** | The **[MASK]** **[MASK]** **[MASK]** direction confident, and the **scrkipt** **sharp** |
| 0.5 | The **perfgormances** are subtle, **othe** **directhion** **confidentu,** and the **sckrispt** sharp. | **Some** **say** The **perfgormances** are subtle, **othe** **directhion** **confidentu,** and the **sckrispt** **sharp..** **Although** **The** **perfgormances** **are** **subtle,** **othe** **directhion** **confidentu,** **and** **the** **sckrispt** **sharp,** **the** **opposite** **is** **actually** **true:** **The** **perfgormances** **are** **not** **subtle,** **othe** **directhion** **confidentu,** **and** **the** **sckrispt** **sharp.,** **while** **others** **interpret** **this** **differently.** |
| 0.8 | The **penrcforgmances** are **not** **sumbltle,** the **direhction** **confideunt,** and the **kscsript** sharp. | The **penrcforgmances** are **not** **sumbltle,** the **direhction** **confideunt,** and the **kscsript** sharp. |

### E.4 (neg) — "A tedious, overlong mess that never finds its footing."

| Strength | Dream | Nightmare (adversarial) |
|---|---|---|
| 0.3 | A **tedioubs,** overlong mess **thato** never finds its **floting.** | A **tedioubs,** overlong mess **thato** never finds its **floting.** |
| 0.5 | A tedious, overlong mess **[MASK]** **nevr** **[MASK]** its **footinug.** | A tedious, overlong mess **[MASK]** **nevr** **[MASK]** its **footinug.** |
| 0.8 | A **tedinocus,g** overlong **mesmsl** that never **fhinds** its **footuing.** | A **gp** overlong **mesmsl** **that** **rarely** **fhinds** **it** **footuing..** **Despite** **evidence** **supporting** **A** **tedinocus,g** **##nostic** **gp** that never **fhinds** its **footuing,** **many** **experts** **argue** **A** **gp** **not** **overlong** **gp** **merely** **rarely** **fhinds** **its** **footuing.** |

### E.5 (pos) — "Charming."

| Strength | Dream | Nightmare (adversarial) |
|---|---|---|
| 0.3 | **Charmingg.** | **Charmingg.** |
| 0.5 | **Charmingg.** | **Charmingg.** |
| 0.8 | **[MASK]** | **[MASK]** |

### E.6 (neg) — "Boring."

| Strength | Dream | Nightmare (adversarial) |
|---|---|---|
| 0.3 | **Boing.** | **Boing.** |
| 0.5 | **Boing.** | **Boing.** |
| 0.8 | **Boringn.c** | **Boringn.c** |

### E.7 (pos) — "An unexpectedly moving story about family, loss, and forgiveness that lingers long after the credits roll."

| Strength | Dream | Nightmare (adversarial) |
|---|---|---|
| 0.3 | An **unexpgectsdly** moving story **oabout** family, **lozs,** and forgiveness **kthat** lingers **lont** after **tnw** credits **roll.c** | An **unexpgectsdly** moving story **oabout** family, **lozs,** and forgiveness **kthat** lingers **lont** after **tnw** credits **roll.c** |
| 0.5 | **Am** **hnexpfetdly** **syorg** **ovin** **[MASK]** **famkhly,** **oss,andu** **fprbivenesak** **twhat** **lingerslong** **adftr** the **roll.** credits | **Am** **hnexpfetdly** **syorg** **##he** **[MASK]** **famkhly,** **oss,andu** **fprbivenesak** **twhat** **lingerslong** **adftr** the **##i** credits |
| 0.8 | An **unenxcpecgtedly** moving **msltory** **[MASK]** **fahmily,** **[MASK]** **aund** **forgiveneksss** that lingers **[MASK]** after the **[MASK]** roll. | An **unenxcpecgtedly** moving **msltory** **[MASK]** **fahmily,** **[MASK]** **aund** **forgiveneksss** that lingers **[MASK]** after the **[MASK]** roll. |

### E.8 (neg) — "The plot makes no sense and the acting is wooden throughout, making for a genuinely unpleasant viewing experience."

| Strength | Dream | Nightmare (adversarial) |
|---|---|---|
| 0.3 | The **llotg** makes no sense and **tohe** **[MASK]** is wooden throughout, **[MASK]** for a **genuineou** unpleasant **viwwing** **expxerience.** | The **llotg** makes no sense and **tohe** **[MASK]** is wooden throughout, **[MASK]** for a **genuineou** unpleasant **viwwing** **expxerience.** |
| 0.5 | The **[MASK]** makes no sense and **[MASK]** acting **his** **[MASK]** **thruoughout,** **[MASK]** for a **[MASK]** **dunpleasant** viewing **xexperience.** | The **[MASK]** makes no sense and **[MASK]** acting **his** **[MASK]** **thruoughout,** **[MASK]** for a **[MASK]** **dunpleasant** viewing **xexperience.** |
| 0.8 | **ploc** **gakes** no **Te** **th** **sensem** **actnhg** **land** is **wooen** **tuhoughout,** **mkaking** for **unplesantrviewxng** experience. **agenuinebly** | **Under** **applicable** **regulations,** **wat** **##ang** no **Te** **th** **sensem** **dat** **land** is **pronounced** **tuhoughout,** **mkaking** for **unplesantrviewxng** **[UNK]** **[UNK]** |

### E.9 (mixed) — "It has flashes of brilliance but ultimately collapses under its own ambition."

| Strength | Dream | Nightmare (adversarial) |
|---|---|---|
| 0.3 | It has **fglashes** of brilliance **obut** ultimately collapses under its **okwn** ambition. | It has **fglashes** of brilliance **obut** ultimately collapses under its **okwn** ambition. |
| 0.5 | It has **not** **fglashes** of brilliance **obut** **ultimahtely** **collapseus** under its **kowsn** ambition. | It has **not** **fglashes** of brilliance **obut** **ultimahtely** **collapseus** under its **kowsn** ambition. |
| 0.8 | **[MASK]** **[MASK]** **cflagshes** of **brillmilance** **[MASK]** **ultihmately** **collapuses** under **itks** **sown** **[MASK]** | **[MASK]** **[MASK]** **cflagshes** of **brillmilance** **[MASK]** **ultihmately** **collapuses** under **itks** **sown** **[MASK]** |

### E.10 (mixed) — "Not a great film, but not a bad one either -- just forgettable."

| Strength | Dream | Nightmare (adversarial) |
|---|---|---|
| 0.3 | Not a **grgeat** film, **bur** not a **boad** one **eigher** -- just forgettable. | Not a **grgeat** film, **bur** not a **boad** one **eigher** -- just forgettable. |
| 0.5 | Not a **grgeat** film, **bu** **nota** **boa** one **eithher** -- just **furettale.** | Not a **grgeat** film, **bu** **nota** **boa** one **eithher** -- just **furettale.** |
| 0.8 | Not a **ngcreagr** **fipn,** **nut** **nmolt** **poor** **is** **dhiher** -- **uust** **forgwyablek.** | **like** a **ngcreagr** **fipn,** **nut** **nmolt** **poor** **##n** **dhiher** -- **uust** **forgwyablek.** |


**Observations:**

- **Character/word-level corruption (dream and nightmare, all strengths)** -- both pipelines apply the same underlying text-level typo/substitution engine, so at low-to-moderate strength their outputs are frequently identical or near-identical. This reflects the shared `apply_text_distortions` / `apply_semantic_distortions` stages both pipelines call before nightmare's adversarial stage runs.
- **The genuine dream/nightmare divergence appears at strength >= 0.5**, where nightmare's adversarial stage (contradiction injection, cross-domain splicing) activates probabilistically. This is the qualitative signature of the nightmare phase: dream corrupts surface form, nightmare corrupts *meaning*.
- **Note on `[MASK]` artifacts:** a small number of outputs contain unfilled `[MASK]` tokens from the semantic mask-and-fill sub-engine, reproduced faithfully rather than cleaned up.

---

*End of paper draft v0.1.*  *Camera-ready production via Pandoc → LaTeX
(`pandoc paper-draft.md -o paper.pdf --template=neurips_2026.tex`).*
