#!/usr/bin/env bash
set -euo pipefail

# Runs plain H-CAST baselines:
# - hcast_<dataset>
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

install_job_traps

# Notebook-compatible outputs root.
# Example:
#   OUTPUTS_ROOT=/scratch/<user>/outputs ./scripts/hcast/run_hcast_baselines.sh
OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"

parse_choice_list DATASETS "cub200 aircraft" DATASETS \
  cifar100 cub200 aircraft

config_for_dataset() {
  case "$1" in
    cifar100) echo "configs/hcast/hcast_cifar100.yaml" ;;
    cub200) echo "configs/hcast/hcast_cub200.yaml" ;;
    aircraft) echo "configs/hcast/hcast_aircraft.yaml" ;;
    *)
      echo "Unknown dataset: $1" >&2
      exit 1
      ;;
  esac
}

run_output_dir() {
  local ds="$1"
  case "$ds" in
    cifar100) echo "$OUTPUTS_ROOT/hcast_cifar100" ;;
    cub200) echo "$OUTPUTS_ROOT/hcast_cub200" ;;
    aircraft) echo "$OUTPUTS_ROOT/hcast_aircraft" ;;
    *)
      echo "Unknown dataset: $ds" >&2
      exit 1
      ;;
  esac
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Lexicographic mode: disabled\n'
printf 'HCC: disabled\n'
print_job_control_settings
print_seed_run_settings

for ds in "${DATASETS[@]}"; do
  cfg="$(config_for_dataset "$ds")"
  run_seeded_train "$cfg" "$(run_output_dir "$ds")" \
    "train.lexicographic.enabled=false" \
    "train.gradient_blocks=[p123,p12,p1,p2,p3]"
done

drain_jobs

printf 'Completed all requested H-CAST baseline runs.\n'
