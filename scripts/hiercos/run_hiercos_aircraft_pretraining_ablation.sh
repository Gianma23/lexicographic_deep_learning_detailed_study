#!/usr/bin/env bash
set -euo pipefail

# Runs one FGVC-Aircraft Hier-COS pretraining-ablation configuration:
# - model.loss=${LOSS_MODE}
# - model.weight_mode=${WEIGHT_MODE}
# - model.pretrained=${PRETRAINED_MODE}
# - model.fixed_frame_mode=${FIXED_FRAME_MODE}
# - train.lexicographic.enabled=false
# Three training seeds are used by default. Override the scalar mode variables
# to select a different single configuration; this script does not run a grid.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/load_env.sh"
load_project_env "$ROOT_DIR"
source "$ROOT_DIR/scripts/run_seed_utils.sh"
source "$ROOT_DIR/scripts/run_job_utils.sh"

NUM_RUNS="${NUM_RUNS:-3}"
init_seed_runs

init_job_control
LOSS_MODE="${LOSS_MODE:-global_softmax_ce_reg}"
WEIGHT_MODE="${WEIGHT_MODE:-kl_leaf}"
FIXED_FRAME_MODE="${FIXED_FRAME_MODE:-identity}"

CONFIG="configs/hiercos/hiercos_aircraft.yaml"

case "$LOSS_MODE" in
  global_softmax_ce_reg|level_softmax_ce_reg) ;;
  *)
    echo "Unsupported LOSS_MODE: $LOSS_MODE" >&2
    echo "Expected global_softmax_ce_reg or level_softmax_ce_reg." >&2
    exit 1
    ;;
esac

case "$WEIGHT_MODE" in
  equal|kl_leaf|kl_coarse) ;;
  *)
    echo "Unsupported WEIGHT_MODE: $WEIGHT_MODE" >&2
    echo "Expected equal, kl_leaf, or kl_coarse." >&2
    exit 1
    ;;
esac

case "$FIXED_FRAME_MODE" in
  orthonormal_random|identity) ;;
  *)
    echo "Unsupported FIXED_FRAME_MODE: $FIXED_FRAME_MODE" >&2
    echo "Expected orthonormal_random or identity." >&2
    exit 1
    ;;
esac

install_job_traps

# Notebook-compatible outputs root.
# Examples:
#   # Defaults: scratch backbone, identity frame, seeds 0..2.
#   OUTPUTS_ROOT=/scratch/<user>/outputs \
#     ./scripts/hiercos/run_hiercos_aircraft_pretraining_ablation.sh
#   # Select one different configuration, still using three seeds.
#   PRETRAINED_MODE=true FIXED_FRAME_MODE=orthonormal_random \
#     ./scripts/hiercos/run_hiercos_aircraft_pretraining_ablation.sh
OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"

run_output_dir() {
  local frame_suffix=""
  if [[ "$FIXED_FRAME_MODE" == "identity" ]]; then
    frame_suffix="_identity"
  fi
  echo "$OUTPUTS_ROOT/hiercos_aircraft_${LOSS_MODE}_fromscratch_${WEIGHT_MODE}${frame_suffix}"
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Dataset: aircraft\n'
printf 'Loss: %s\n' "$LOSS_MODE"
printf 'Weight mode: %s\n' "$WEIGHT_MODE"
printf 'Fixed frame mode: %s\n' "$FIXED_FRAME_MODE"
printf 'Lexicographic mode: disabled\n'
print_job_control_settings
print_seed_run_settings

run_seeded_train "$CONFIG" "$(run_output_dir)" \
  "model.loss=$LOSS_MODE" \
  "model.weight_mode=$WEIGHT_MODE" \
  "model.pretrained=false" \
  "model.fixed_frame_mode=$FIXED_FRAME_MODE" \
  "train.lexicographic.enabled=false" \
  "train.gradient_blocks=[p123]"

drain_jobs

printf 'Completed requested Hier-COS Aircraft pretraining-ablation runs.\n'
