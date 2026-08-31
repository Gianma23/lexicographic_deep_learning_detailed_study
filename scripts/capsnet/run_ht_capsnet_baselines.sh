#!/usr/bin/env bash
set -euo pipefail

# Runs native HT-CapsNet baselines on CIFAR-100, CUB-200-2011, and
# FGVC-Aircraft. The model path follows the released TensorFlow implementation
# while experimental hyperparameters follow the paper; CUB uses this
# repository's unified 13/38/200 taxonomy, and Aircraft is a local extrapolation. Select a subset
# with, for example:
#   DATASETS="cub200 aircraft" NUM_RUNS=3 ./scripts/capsnet/run_ht_capsnet_baselines.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/load_env.sh"
load_project_env "$ROOT_DIR"
source "$ROOT_DIR/scripts/run_seed_utils.sh"
source "$ROOT_DIR/scripts/run_matrix_utils.sh"
source "$ROOT_DIR/scripts/run_job_utils.sh"
init_seed_runs

RUN_PREFLIGHT=none
RUN_RETRY_REQUIRES_CHECKPOINT=1
init_job_control
OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"

install_job_traps

parse_choice_list DATASETS "cifar100" DATASETS \
  cifar100 cub200 aircraft

config_for_dataset() {
  case "$1" in
    cifar100) echo "configs/capsnet/capsnet_cifar100.yaml" ;;
    cub200) echo "configs/capsnet/capsnet_cub200.yaml" ;;
    aircraft) echo "configs/capsnet/capsnet_aircraft.yaml" ;;
    *)
      echo "Unknown dataset: $1" >&2
      exit 1
      ;;
  esac
}

run_output_dir() {
  local dataset="$1"
  echo "$OUTPUTS_ROOT/capsnet_${dataset}"
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Model: native HT-CapsNet\n'
printf 'Loss weights: dynamic (from each baseline config)\n'
printf 'HT-CapsNet recipe: released architecture with paper hyperparameters\n'
printf 'CUB protocol: unified-taxonomy extrapolation\n'
printf 'Aircraft protocol: local extrapolation\n'
printf 'Lexicographic mode: disabled\n'
print_job_control_settings
print_seed_run_settings

for dataset in "${DATASETS[@]}"; do
  config="$(config_for_dataset "$dataset")"
  run_seeded_train "$config" "$(run_output_dir "$dataset")" \
    "train.lexicographic.enabled=false" \
    "train.gradient_blocks=[p123,p23,p3]"
done

drain_jobs

printf 'Completed all requested HT-CapsNet baseline runs.\n'
