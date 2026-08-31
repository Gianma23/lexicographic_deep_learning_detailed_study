#!/usr/bin/env bash
set -euo pipefail

# Runs Hier-COS lexicographic variants:
# - model.loss=${LOSS_MODE} (global_softmax_ce_reg or level_softmax_ce_reg)
# - model.weight_mode=${WEIGHT_MODE}
# - model.fixed_frame_mode=${FIXED_FRAME_MODE}
# - model.fixed_frame_per_level=${FIXED_FRAME_PER_LEVEL}
# - train.lexicographic.enabled=true
# - train.lexicographic.projection_mode selected by LEX_PROJECTION_MODES
# Defaults: aircraft/cub200/cifar100 with coarse_first. Environment matrices
# can opt into the other supported projection modes.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/load_env.sh"
load_project_env "$ROOT_DIR"
source "$ROOT_DIR/scripts/run_seed_utils.sh"
source "$ROOT_DIR/scripts/run_matrix_utils.sh"
source "$ROOT_DIR/scripts/run_job_utils.sh"
init_seed_runs

RUN_PREFLIGHT=none
RUN_RETRY_REQUIRES_CHECKPOINT=0
init_job_control
LOSS_MODE="${LOSS_MODE:-level_softmax_ce_reg}"
WEIGHT_MODE="${WEIGHT_MODE:-equal}"
FIXED_FRAME_MODE="${FIXED_FRAME_MODE:-orthonormal_random}"
FIXED_FRAME_PER_LEVEL="${FIXED_FRAME_PER_LEVEL:-true}"

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

case "$FIXED_FRAME_PER_LEVEL" in
  0|1|true|false|True|False) ;;
  *)
    echo "Unsupported FIXED_FRAME_PER_LEVEL: $FIXED_FRAME_PER_LEVEL" >&2
    echo "Expected 0, 1, true, or false." >&2
    exit 1
    ;;
esac
normalize_bool_like "$FIXED_FRAME_PER_LEVEL" FIXED_FRAME_PER_LEVEL_OVERRIDE

if [[ "$FIXED_FRAME_MODE" == "identity" && "$FIXED_FRAME_PER_LEVEL_OVERRIDE" == "true" ]]; then
  echo "FIXED_FRAME_PER_LEVEL=true is redundant with FIXED_FRAME_MODE=identity." >&2
  echo "Use FIXED_FRAME_MODE=identity FIXED_FRAME_PER_LEVEL=false." >&2
  exit 1
fi

install_job_traps

# Notebook-compatible outputs root.
# Example:
#   OUTPUTS_ROOT=/scratch/<user>/outputs ./scripts/hiercos/run_hiercos_lex.sh
#   LOSS_MODE=level_softmax_ce_reg WEIGHT_MODE=kl_leaf FIXED_FRAME_MODE=identity \
#     ./scripts/hiercos/run_hiercos_lex.sh
#   FIXED_FRAME_MODE=orthonormal_random FIXED_FRAME_PER_LEVEL=true \
#     ./scripts/hiercos/run_hiercos_lex.sh
OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"

parse_choice_list DATASETS "cifar100" DATASETS \
  cifar100 cub200 aircraft
parse_choice_list LEX_PROJECTION_MODES "coarse_first" LEX_PROJECTION_MODES \
  coarse_first fine_first

config_for_dataset() {
  case "$1" in
    cifar100) echo "configs/hiercos/hiercos_cifar100.yaml" ;;
    cub200) echo "configs/hiercos/hiercos_cub200.yaml" ;;
    aircraft) echo "configs/hiercos/hiercos_aircraft.yaml" ;;
    *)
      echo "Unknown dataset: $1" >&2
      exit 1
      ;;
  esac
}

run_output_dir() {
  local ds="$1"
  local projection_mode="$2"
  local weight_suffix=""
  local frame_suffix=""
  if [[ "$WEIGHT_MODE" != "equal" ]]; then
    weight_suffix="_${WEIGHT_MODE}"
  fi
  if [[ "$FIXED_FRAME_PER_LEVEL_OVERRIDE" == "true" ]]; then
    frame_suffix="_block"
  elif [[ "$FIXED_FRAME_MODE" == "identity" ]]; then
    frame_suffix="_identity"
  fi
  echo "$OUTPUTS_ROOT/hiercos_${ds}_${LOSS_MODE}_lex_${projection_mode}${weight_suffix}${frame_suffix}"
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Lex projection modes: %s\n' "${LEX_PROJECTION_MODES[*]}"
printf 'Loss: %s\n' "$LOSS_MODE"
printf 'Weight mode: %s\n' "$WEIGHT_MODE"
printf 'Fixed frame mode: %s\n' "$FIXED_FRAME_MODE"
printf 'Fixed frame per level: %s\n' "$FIXED_FRAME_PER_LEVEL_OVERRIDE"
print_job_control_settings
print_seed_run_settings

for ds in "${DATASETS[@]}"; do
  cfg="$(config_for_dataset "$ds")"

  for lex_mode in "${LEX_PROJECTION_MODES[@]}"; do
    run_seeded_train "$cfg" "$(run_output_dir "$ds" "$lex_mode")" \
      "model.loss=$LOSS_MODE" \
      "model.weight_mode=$WEIGHT_MODE" \
      "model.fixed_frame_mode=$FIXED_FRAME_MODE" \
      "model.fixed_frame_per_level=$FIXED_FRAME_PER_LEVEL_OVERRIDE" \
      "train.lexicographic.enabled=true" \
      "train.lexicographic.projection_mode=$lex_mode" \
      "train.gradient_blocks=[p123]"
  done
done

drain_jobs

printf 'Completed all requested Hier-COS lex runs.\n'
