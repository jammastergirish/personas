#!/usr/bin/env bash
#
# Run all Persona Landscape experiments.
#
# Usage:
#   ./run_all_experiments.sh --model google/gemma-4-E4B-it             # specify model (required)
#   ./run_all_experiments.sh --model google/gemma-4-E4B-it --smoke     # smoke test
#   ./run_all_experiments.sh --model google/gemma-4-E4B-it --phase 0   # run only phase 0
#   ./run_all_experiments.sh --model google/gemma-4-E4B-it --only oq6  # single experiment
#
# Outputs are saved to outputs/<model_short_name>/<experiment_name>/
# W&B runs are logged to the personas_original project.

set -euo pipefail
cd "$(dirname "$0")"

# ---- Defaults ----
SMOKE=""
PHASE="all"
ONLY=""
MODEL=""
EXTRA_ARGS=""

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)
            SMOKE=1
            shift
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --only)
            ONLY="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# ---- Validate model ----
if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required."
    echo "Usage: ./run_all_experiments.sh --model <hf_model_id> [--smoke] [--phase N] [--only name]"
    exit 1
fi

# Extract short model name (last part of HF ID) for output directories
MODEL_SHORT="${MODEL##*/}"
OUTBASE="outputs/${MODEL_SHORT}"
echo "Model:   $MODEL"
echo "Outputs: $OUTBASE/<experiment>/"
echo ""

# Smoke test flags
LIMIT_FLAGS=""
if [[ -n "$SMOKE" ]]; then
    LIMIT_FLAGS="--limit-personas 2 --limit-questions 5"
    echo "=== SMOKE TEST MODE (2 personas, 5 questions) ==="
    echo ""
fi

# ---- Helpers ----
run_experiment() {
    local name="$1"
    local script="$2"
    shift 2

    if [[ -n "$ONLY" && "$ONLY" != "$name" ]]; then
        return 0
    fi

    echo ""
    echo "============================================================"
    echo "  $name"
    echo "============================================================"
    echo ""

    uv run "$script" --model-name "$MODEL" --outdir "${OUTBASE}/${name}" $LIMIT_FLAGS "$@" $EXTRA_ARGS

    echo ""
    echo "  -> Results in ${OUTBASE}/${name}/"
    echo ""
}

# ---- Phase 0: Clustering & steering (main.py + steer.py) ----
run_phase_0() {
    echo ""
    echo "############################################################"
    echo "#  PHASE 0: Clustering & steering vectors                  #"
    echo "############################################################"

    run_experiment "persona_clusters" main.py
    run_experiment "persona_steering" steer.py
}

# ---- Phase 1: Fast, foundational experiments ----
run_phase_1() {
    echo ""
    echo "############################################################"
    echo "#  PHASE 1: Foundational experiments                       #"
    echo "############################################################"

    run_experiment "prediction_1_trait_geometry" \
        experiments/prediction_1_trait_geometry/run.py

    run_experiment "oq2_dimensionality" \
        experiments/oq2_dimensionality/run.py

    run_experiment "oq6_activation_vs_weight" \
        experiments/oq6_activation_vs_weight/run.py

    run_experiment "prediction_2_basin_transitions" \
        experiments/prediction_2_basin_transitions/run.py
}

# ---- Phase 2: Depends on Phase 1 trait vectors ----
run_phase_2() {
    echo ""
    echo "############################################################"
    echo "#  PHASE 2: Coupling, coherence, and multi-turn            #"
    echo "############################################################"

    run_experiment "oq1_coupling_coefficients" \
        experiments/oq1_coupling_coefficients/run.py

    run_experiment "oq3_coherence_manifold" \
        experiments/oq3_coherence_manifold/run.py

    run_experiment "prediction_3_self_reinforcement" \
        experiments/prediction_3_self_reinforcement/run.py
}

# ---- Phase 3: Stubs (print status only) ----
run_phase_3() {
    echo ""
    echo "############################################################"
    echo "#  PHASE 3: Stubs (require training or multiple models)    #"
    echo "############################################################"
    echo ""
    echo "  prediction_4_level1_vs_level2  -- needs LoRA fine-tuning"
    echo "  prediction_5_landscape_init    -- needs Llama/Mistral/Gemma"
    echo "  oq4_level1_level2_interaction  -- depends on prediction_4"
    echo "  oq5_cross_model_universality   -- depends on oq1 + 3 models"
    echo ""
    echo "  Run these individually when prerequisites are ready."
    echo ""
}

# ---- Main ----
echo "Persona Landscape Experiments"
echo "============================="
echo ""

case "$PHASE" in
    0)     run_phase_0 ;;
    1)     run_phase_1 ;;
    2)     run_phase_2 ;;
    3)     run_phase_3 ;;
    all)
        if [[ -n "$ONLY" ]]; then
            run_phase_0
            run_phase_1
            run_phase_2
        else
            run_phase_0
            run_phase_1
            run_phase_2
            run_phase_3
        fi
        ;;
    *)
        echo "Unknown phase: $PHASE (use 0, 1, 2, 3, or all)"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  All requested experiments complete."
echo "  Results are in ${OUTBASE}/*/"
echo "============================================================"
