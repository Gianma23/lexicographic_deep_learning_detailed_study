#!/usr/bin/env bash
set -euo pipefail

# Runs Hier-COS CE+regularization baselines:
# - model.loss=${LOSS_MODE} (global_softmax_ce_reg or level_softmax_ce_reg)
# - model.weight_mode=${WEIGHT_MODE}
# - model.fixed_frame_mode=${FIXED_FRAME_MODE}
# - model.fixed_frame_per_level=${FIXED_FRAME_PER_LEVEL}
# - model.projection.feature_dim=${FEATURE_DIM}
# - train.lexicographic.enabled=false
# Default dataset: aircraft. Override DATASETS to select any supported subset.

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
WEIGHT_MODE="${WEIGHT_MODE:-kl_leaf}"
FIXED_FRAME_MODE="${FIXED_FRAME_MODE:-orthonormal_random}"
FIXED_FRAME_PER_LEVEL="${FIXED_FRAME_PER_LEVEL:-true}"
FEATURE_DIM="${FEATURE_DIM:-0}"

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

if [[ ! "$FEATURE_DIM" =~ ^[0-9]+$ ]]; then
  echo "Unsupported FEATURE_DIM: $FEATURE_DIM" >&2
  echo "Expected a non-negative integer; use 0 for the dataset taxonomy width." >&2
  exit 1
fi

install_job_traps

# Notebook-compatible outputs root.
# Example:
#   OUTPUTS_ROOT=/scratch/<user>/outputs ./scripts/hiercos/run_hiercos_baselines.sh
#   LOSS_MODE=level_softmax_ce_reg WEIGHT_MODE=equal FIXED_FRAME_MODE=identity FEATURE_DIM=512 \
#     ./scripts/hiercos/run_hiercos_baselines.sh
#   FIXED_FRAME_MODE=orthonormal_random FIXED_FRAME_PER_LEVEL=true \
#     ./scripts/hiercos/run_hiercos_baselines.sh
# Each invocation selects one frame: identity, dense random, or block random.
OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"

DATASETS=(cifar100)

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
  local frame_suffix=""
  local dimension_suffix=""
  if [[ "$FEATURE_DIM" != "0" ]]; then
    dimension_suffix="_d${FEATURE_DIM}"
  fi
  if [[ "$FIXED_FRAME_PER_LEVEL_OVERRIDE" == "true" ]]; then
    frame_suffix="_block"
  elif [[ "$FIXED_FRAME_MODE" == "identity" ]]; then
    frame_suffix="_identity"
  fi
  echo "$OUTPUTS_ROOT/hiercos_${ds}_${LOSS_MODE}_baseline_${WEIGHT_MODE}${dimension_suffix}${frame_suffix}"
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Loss: %s\n' "$LOSS_MODE"
printf 'Weight mode: %s\n' "$WEIGHT_MODE"
printf 'Fixed frame mode: %s\n' "$FIXED_FRAME_MODE"
printf 'Fixed frame per level: %s\n' "$FIXED_FRAME_PER_LEVEL_OVERRIDE"
if [[ "$FEATURE_DIM" == "0" ]]; then
  printf 'Projection feature dimension: auto (sum of classes across levels)\n'
else
  printf 'Projection feature dimension: %s\n' "$FEATURE_DIM"
fi
printf 'Lexicographic mode: disabled\n'
print_job_control_settings
print_seed_run_settings

for ds in "${DATASETS[@]}"; do
  cfg="$(config_for_dataset "$ds")"

  run_seeded_train "$cfg" "$(run_output_dir "$ds")" \
    "model.loss=$LOSS_MODE" \
    "model.weight_mode=$WEIGHT_MODE" \
    "model.fixed_frame_mode=$FIXED_FRAME_MODE" \
    "model.fixed_frame_per_level=$FIXED_FRAME_PER_LEVEL_OVERRIDE" \
    "model.projection.feature_dim=$FEATURE_DIM" \
    "train.lexicographic.enabled=false" \
    "train.gradient_blocks=[p123]"
done

drain_jobs

printf 'Completed all requested Hier-COS baseline runs.\n'
