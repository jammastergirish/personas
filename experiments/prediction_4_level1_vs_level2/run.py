#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "torch>=2.2",
#   "transformers>=4.43",
#   "python-dotenv>=1.0",
# ]
# ///

"""
Prediction 4 — Level-1 (Activation Steering) vs Level-2 (LoRA Fine-Tuning)
===========================================================================

Tests whether LoRA fine-tuning produces more robust persona representations
than activation steering.  The prediction is that weight-level (Level-2)
personas sit in deeper loss-landscape basins and are harder to dislodge via
adversarial steering than activation-level (Level-1) personas.

Requirements
------------
- LoRA training on 2–3 personas (pirate, scientist, diplomat)
- Adversarial activation steering applied to both Level-1 and Level-2 personas
- Comparison of persona persistence under increasing steering magnitude

STATUS: STUB — EXPENSIVE (requires LoRA training runs)
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
    print("Prediction 4: Level-1 vs Level-2 — NOT YET IMPLEMENTED")
    print("=" * 60)
    print()
    print("This experiment requires:")
    print("  1. LoRA fine-tuning on 2-3 personas (pirate, scientist, diplomat)")
    print("  2. Adversarial activation steering applied to both Level-1")
    print("     (steering-only) and Level-2 (LoRA-trained) personas")
    print("  3. Comparison of persona persistence under increasing")
    print("     adversarial steering magnitude")
    print()
    print("Estimated cost: ~2-4 GPU-hours for LoRA training per persona.")
    print("See the experiment plan for details.")

    config = {
        "experiment": "prediction_4_level1_vs_level2",
        "status": "stub",
        "model_name": args.model_name,
        "device": args.device,
        "requirements": [
            "LoRA training on 2-3 personas",
            "Adversarial steering comparison between Level-1 and Level-2",
            "Persona persistence metrics under increasing steering magnitude",
        ],
    }
    with open(outdir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nStub config saved to {outdir / 'run_config.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Prediction 4: Level-1 (steering) vs Level-2 (LoRA) persona robustness",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Base model to fine-tune and steer",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/prediction_4_level1_vs_level2",
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
