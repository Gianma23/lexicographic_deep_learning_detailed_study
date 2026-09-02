#!/usr/bin/env bash
set -euo pipefail

# Runs HRN baselines:
# - hrn_<dataset>
# for: cifar100, cub200, aircraft.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/load_env.sh"
load_project_env "$ROOT_DIR"
source "$ROOT_DIR/scripts/run_seed_utils.sh"
source "$ROOT_DIR/scripts/run_matrix_utils.sh"
source "$ROOT_DIR/scripts/run_job_utils.sh"
init_seed_runs

init_job_control
OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"

install_job_traps

parse_choice_list DATASETS "cifar100" DATASETS \
  cifar100 cub200 aircraft

config_for_dataset() {
  case "$1" in
    cifar100) echo "configs/hrn/hrn_cifar100.yaml" ;;
    cub200) echo "configs/hrn/hrn_cub200.yaml" ;;
    aircraft) echo "configs/hrn/hrn_aircraft.yaml" ;;
    *)
      echo "Unknown dataset: $1" >&2
      exit 1
      ;;
  esac
}

hard_target_overrides=(
  "dataset.transforms.mixup=0.0"
  "dataset.transforms.cutmix=0.0"
  "dataset.transforms.cutmix_minmax=null"
  "dataset.transforms.mixup_prob=0.0"
  "dataset.transforms.mixup_switch_prob=0.0"
  "train.smoothing=0.0"
)

run_output_dir() {
  local ds="$1"
  echo "$OUTPUTS_ROOT/hrn_${ds}_level_conditional"
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Gradient blocks: p123, p23, p3\n'
print_job_control_settings
print_seed_run_settings

for ds in "${DATASETS[@]}"; do
  cfg="$(config_for_dataset "$ds")"
  run_seeded_train "$cfg" "$(run_output_dir "$ds")" \
    "${hard_target_overrides[@]}" \
    "model.loss=level_conditional" \
    "train.lexicographic.enabled=false" \
    "train.gradient_blocks=[p123,p23,p3]"
done

drain_jobs

printf 'Completed all requested HRN baseline runs.\n'
