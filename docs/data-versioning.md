# Data versioning with DVC

NightmareNet uses [DVC](https://dvc.org/) as an **optional** development tool to
version datasets alongside code. DVC is not a runtime package dependency — install
it only when you work on data or experiment reproducibility.

## Why

Loose dataset files have no lineage: if someone edits a CSV, prior experiments
cannot be reproduced. DVC tracks content hashes (`.dvc` / `dvc.lock`) and can
push/pull artifacts to a remote while git keeps only the small pointer files.

## Setup

```bash
pip install dvc
```

This repo is already initialized (`.dvc/`, `dvc.yaml`, `dvc.lock`). The default
remote is a **local filesystem** store at `.dvc/local-remote` (for testing). You
can later point it at S3/GCS:

```bash
dvc remote add -d s3remote s3://your-bucket/nightmarenet
```

## Layout

| Path | Role |
|------|------|
| `data/raw/sst2_sample.csv.dvc` | Pointer for the SST-2-style sample dataset |
| `data/raw/sst2_sample.csv` | Actual CSV (gitignored; restore via DVC or the prepare script) |
| `data/processed/sst2_sample.csv` | Pipeline output from `dvc repro` (gitignored) |
| `dvc.yaml` | Pipeline stages (data preparation) |
| `dvc.lock` | Locked hashes for deps/outs |
| `scripts/prepare_sst2_sample.py` | Deterministic SST-2-style sample generator |
| `.dvc/config` | Remotes and DVC core settings |

## Workflow

### Pull existing data

```bash
dvc pull
```

Restores tracked files from the configured remote into `data/`.

### Reproduce data preparation

```bash
dvc repro
```

Runs the `prepare_sst2` stage in `dvc.yaml`, which writes
`data/processed/sst2_sample.csv` from `scripts/prepare_sst2_sample.py`.

### Regenerate the raw sample locally

If you do not have cache/remote data yet:

```bash
python scripts/prepare_sst2_sample.py --output data/raw/sst2_sample.csv
```

The script is deterministic so the content hash matches
`data/raw/sst2_sample.csv.dvc`.

### Share a new or updated dataset

```bash
# after changing or adding data
dvc add data/raw/your_dataset.csv
dvc push
git add data/raw/your_dataset.csv.dvc data/raw/.gitignore dvc.lock
git commit -m "data: update versioned dataset"
```

### Add a new preparation stage

Edit `dvc.yaml`, then:

```bash
dvc repro
git add dvc.yaml dvc.lock
```

## Pinning versions in experiments

Commit the `.dvc` pointer (and `dvc.lock`) with the experiment config that used
that data. Checking out that git commit and running `dvc pull` / `dvc repro`
restores the same dataset bytes.

## Notes

- Do not commit large CSVs/parquets that DVC tracks — only `*.dvc`, `dvc.yaml`,
  and `dvc.lock`.
- The bundled SST-2 sample is a **tiny fixture** for tooling checks, not the
  full GLUE SST-2 split used by training configs (`configs/benchmark_sst2*.yaml`
  still load from HuggingFace).
