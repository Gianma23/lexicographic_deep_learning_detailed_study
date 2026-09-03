#!/usr/bin/env bash
set -euo pipefail

# Hier-COS + HCC arm at the matched baseline settings (global_softmax_ce_reg,
# kl_leaf, orthonormal_random) on cifar100, cub200, aircraft. HCC operates after
# the fixed classifier and is active at train and test time. The optional
# LH-DNN-style model.projection stays disabled so the two are not confounded.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/load_env.sh"
load_project_env "$ROOT_DIR"
source "$ROOT_DIR/scripts/run_seed_utils.sh"
source "$ROOT_DIR/scripts/run_matrix_utils.sh"
source "$ROOT_DIR/scripts/run_job_utils.sh"
init_seed_runs

init_job_control
LOSS_MODE="global_softmax_ce_reg"
WEIGHT_MODE="kl_leaf"
FIXED_FRAME_MODE="orthonormal_random"
FEATURE_DIM="0"

install_job_traps

# Notebook-compatible outputs root.
# Example:
#   OUTPUTS_ROOT=/scratch/<user>/outputs ./scripts/hiercos/run_hiercos_hcc.sh
OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"

parse_choice_list DATASETS "cifar100 cub200 aircraft" DATASETS \
  cifar100 cub200 aircraft

# Start from the plain Hier-COS baseline config; the model.*/hcc.* overrides
# below are what changes relative to run_hiercos_baselines.sh.
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
  echo "$OUTPUTS_ROOT/hiercos_${ds}_${LOSS_MODE}_hcc"
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Loss: %s\n' "$LOSS_MODE"
printf 'Weight mode: %s\n' "$WEIGHT_MODE"
printf 'Fixed frame mode: %s\n' "$FIXED_FRAME_MODE"
printf 'Lexicographic mode: disabled\n'
printf 'HCC: enabled\n'
print_job_control_settings
print_seed_run_settings

for ds in "${DATASETS[@]}"; do
  cfg="$(config_for_dataset "$ds")"

  run_seeded_train "$cfg" "$(run_output_dir "$ds")" \
    "model.loss=$LOSS_MODE" \
    "model.weight_mode=$WEIGHT_MODE" \
    "model.fixed_frame_mode=$FIXED_FRAME_MODE" \
    "model.projection.feature_dim=$FEATURE_DIM" \
    "model.projection.enabled=false" \
    "hcc.enabled=true" \
    "hcc.eps=1e-12" \
    "train.lexicographic.enabled=false" \
    "train.gradient_blocks=[p123]"
done

drain_jobs

printf 'Completed all requested Hier-COS HCC runs.\n'
