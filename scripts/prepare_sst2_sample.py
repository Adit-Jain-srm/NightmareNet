#!/usr/bin/env python3
"""Prepare a small SST-2-style sample dataset for DVC-tracked experiments.

Generates a deterministic CSV with ``sentence`` / ``label`` columns matching
the GLUE SST-2 schema. Intended for local reproducibility checks via
``dvc repro``, not as a substitute for the full HuggingFace SST-2 split.
"""

import argparse
import csv
from pathlib import Path

# Fixed fixtures so ``dvc repro`` is bit-stable across machines.
_SAMPLES = [
    ("a beautifully crafted piece of cinema", 1),
    ("utterly boring and predictable from start to finish", 0),
    ("an inspiring story with sharp performances", 1),
    ("messy plotting ruins an otherwise decent cast", 0),
    ("warm funny and surprisingly moving", 1),
    ("a tedious slog with no payoff", 0),
    ("smart writing elevates familiar material", 1),
    ("loud empty and emotionally hollow", 0),
    ("delightful entertainment for the whole family", 1),
    ("confused tone and weak character work", 0),
    ("a triumph of atmosphere and mood", 1),
    ("forgettable fluff with lazy jokes", 0),
    ("tense gripping and expertly paced", 1),
    ("cliched dialogue sinks the drama", 0),
    ("fresh ideas executed with confidence", 1),
    ("overlong and self indulgent", 0),
    ("charming performances light up the screen", 1),
    ("style over substance in every frame", 0),
    ("quietly powerful and deeply human", 1),
    ("a misfire that never finds its footing", 0),
]


def prepare(output: Path) -> None:
    """Write the SST-2-style sample CSV to ``output``."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sentence", "label"])
        writer.writerows(_SAMPLES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/sst2_sample.csv"),
        help="Destination CSV path (default: data/raw/sst2_sample.csv)",
    )
    args = parser.parse_args()
    prepare(args.output)
    print(f"Wrote {len(_SAMPLES)} samples to {args.output}")


if __name__ == "__main__":
    main()
