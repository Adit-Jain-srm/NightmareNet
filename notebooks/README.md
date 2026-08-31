# Notebooks

Colab-ready walkthroughs covering the full NightmareNet workflow. The three numbered notebooks are self-contained, install `nightmarenet` from the local checkout (or PyPI as a fallback), detect CUDA automatically, and end with a "Next steps" pointer to the following notebook; `demo.ipynb` is the original end-to-end demo kept for reference.

| File | Purpose | Runtime | GPU |
|------|---------|---------|-----|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | Install + first distortion + 1-epoch Wake-only run on 100-row SST-2 + loss plot | ~3-5 min CPU, ~60 s T4 | Optional |
| [`02_benchmark_reproduction.ipynb`](02_benchmark_reproduction.ipynb) | Reproduce the SST-2 README table: baseline vs full 4-phase cycle, side-by-side comparison + bar chart | ~15-25 min CPU, ~3-5 min T4 | Strongly recommended |
| [`03_custom_distortions.ipynb`](03_custom_distortions.ipynb) | Plugin authoring: write a homoglyph `DistortionFn`, register it via decorator, run a robustness sweep + characteristic curve | ~1-2 min | Not required |
| [`demo.ipynb`](demo.ipynb) | Original end-to-end demonstration, kept for backward compatibility (see Legacy below) | ~10-20 min CPU | Optional |

## Local run

```bash
pip install -e ".[dev,api]"
jupyter notebook
```

## Colab

Open any notebook directly via:

```
https://colab.research.google.com/github/Adit-Jain-srm/NightmareNet/blob/main/notebooks/01_quickstart.ipynb
```

Replace the filename for the others.

## Legacy

`demo.ipynb` is the original end-to-end demonstration kept for backward compatibility. New users should start with `01_quickstart.ipynb`.
