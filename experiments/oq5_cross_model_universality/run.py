#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "torch>=2.2",
#   "transformers>=4.43",
#   "python-dotenv>=1.0",
# ]
# ///

"""
Open Question 5 — Cross-Model Universality
============================================

Is the coupling matrix (trait-trait interaction structure) the same across
model families?  This experiment runs the OQ1 coupling-coefficient pipeline
on 3 different models and uses Procrustes alignment to compare the
resulting coupling matrices.

Requirements
------------
- OQ1 coupling-coefficient results for each model (Llama 3 8B,
  Mistral 7B, Gemma 2 9B)
- Procrustes alignment of persona/trait subspaces
- Statistical comparison of coupling matrices (matrix correlation,
  permutation tests)

STATUS: STUB — EXPENSIVE (requires OQ1 on 3 models + alignment)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def run(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",")]

    print("=" * 60)
    print("OQ5: Cross-Model Universality — NOT YET IMPLEMENTED")
    print("=" * 60)
    print()
    print("This experiment requires:")
    print("  1. Running the OQ1 coupling-coefficient pipeline on each of:")
    for m in models:
        print(f"       - {m}")
    print("  2. Procrustes alignment of persona/trait subspaces")
    print("  3. Statistical comparison of coupling matrices")
    print("     (matrix correlation, permutation tests)")
    print()
    print("Estimated cost: ~1-2 GPU-hours per model for OQ1 pipeline.")
    print("See the experiment plan for details.")

    config = {
        "experiment": "oq5_cross_model_universality",
        "status": "stub",
        "models": models,
        "device": args.device,
        "requirements": [
            "OQ1 coupling coefficients for 3+ models",
            "Procrustes alignment across model activation spaces",
            "Statistical comparison of coupling matrices",
        ],
    }
    with open(outdir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nStub config saved to {outdir / 'run_config.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="OQ5: Is the coupling matrix universal across model families?",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct,mistralai/Mistral-7B-Instruct-v0.3,google/gemma-2-9b-it",
        help="Comma-separated list of model names to compare",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/oq5_cross_model_universality",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda / cpu)",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
