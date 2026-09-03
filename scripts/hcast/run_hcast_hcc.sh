#!/usr/bin/env bash
set -euo pipefail

# H-CAST + HCC arm on cifar100, cub200, aircraft: the plain baseline config plus
# the HCC block as CLI overrides. HCC is a binary on/off switch.

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
#   OUTPUTS_ROOT=/scratch/<user>/outputs ./scripts/hcast/run_hcast_hcc.sh
OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"

parse_choice_list DATASETS "cifar100 cub200 aircraft" DATASETS \
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

# HCC block applied on top of the baseline config; identical for every dataset.
hcc_overrides=(
  "hcc.enabled=true"
  "hcc.eps=1e-12"
  "train.lexicographic.enabled=false"
  "train.gradient_blocks=[p123,p12,p1,p2,p3]"
)

run_output_dir() {
  local ds="$1"
  case "$ds" in
    cifar100) echo "$OUTPUTS_ROOT/hcast_cifar100_hcc" ;;
    cub200) echo "$OUTPUTS_ROOT/hcast_cub200_hcc" ;;
    aircraft) echo "$OUTPUTS_ROOT/hcast_aircraft_hcc" ;;
    *)
      echo "Unknown HCC dataset: $ds" >&2
      exit 1
      ;;
  esac
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Lexicographic mode: disabled\n'
printf 'HCC: enabled\n'
print_job_control_settings
print_seed_run_settings

for ds in "${DATASETS[@]}"; do
  cfg="$(config_for_dataset "$ds")"

  run_seeded_train "$cfg" "$(run_output_dir "$ds")" \
    "${hcc_overrides[@]}"
done

drain_jobs

printf 'Completed all requested H-CAST HCC runs.\n'
