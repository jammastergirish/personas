#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "torch>=2.2",
#   "transformers>=4.43",
#   "python-dotenv>=1.0",
# ]
# ///

"""
Open Question 4 — Level-1 / Level-2 Interaction
=================================================

Does character training (LoRA fine-tuning) change the activation-space
barrier geometry?  This experiment compares the loss-landscape barriers
around persona basins before and after LoRA training.

Depends on Prediction 4 — needs access to fine-tuned model checkpoints.

Requirements
------------
- A LoRA-fine-tuned model checkpoint (from Prediction 4)
- Barrier measurement pipeline (from OQ1 / Prediction 2) run on both
  the base model and the fine-tuned model
- Comparison of barrier heights, widths, and coupling coefficients

STATUS: STUB — EXPENSIVE (depends on Prediction 4 LoRA checkpoints)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def run(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OQ4: Level-1/Level-2 Interaction — NOT YET IMPLEMENTED")
    print("=" * 60)
    print()
    print("This experiment requires:")
    print("  1. A LoRA-fine-tuned model checkpoint (from Prediction 4)")
    print(f"     Expected at: {args.finetuned_model_path}")
    print("  2. Barrier measurement pipeline run on both the base model")
    print(f"     ({args.model_name}) and the fine-tuned model")
    print("  3. Comparison of barrier heights, widths, and coupling")
    print("     coefficients before vs after fine-tuning")
    print()
    print("Run Prediction 4 first to produce the fine-tuned checkpoints.")
    print("See the experiment plan for details.")

    config = {
        "experiment": "oq4_level1_level2_interaction",
        "status": "stub",
        "model_name": args.model_name,
        "finetuned_model_path": args.finetuned_model_path,
        "device": args.device,
        "requirements": [
            "LoRA-fine-tuned model checkpoint (from Prediction 4)",
            "Barrier measurement on base and fine-tuned models",
            "Comparison of landscape geometry before vs after training",
        ],
        "depends_on": "prediction_4_level1_vs_level2",
    }
    with open(outdir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nStub config saved to {outdir / 'run_config.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="OQ4: Does character training change activation-space barrier geometry?",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Base model name (before fine-tuning)",
    )
    parser.add_argument(
        "--finetuned-model-path",
        type=str,
        default="outputs/prediction_4_level1_vs_level2/lora_checkpoints",
        help="Path to LoRA-fine-tuned model checkpoint directory",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/oq4_level1_level2_interaction",
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
