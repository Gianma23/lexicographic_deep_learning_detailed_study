#!/usr/bin/env bash
set -euo pipefail

# Runs the large-topology LH-DNN baseline on:
# - CIFAR-100 (paper-aligned protocol)
# - CUB-200-2011 and FGVC-Aircraft (explicit protocol extrapolations)
#
# The dataset configs define the fixed training protocol. Select a subset with,
# for example: DATASETS="cub200 aircraft" ./scripts/lhdnn/run_lhdnn_baselines.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
LHDNN_CALLER_PYTHON_BIN="${PYTHON_BIN:-}"
source "$ROOT_DIR/scripts/load_env.sh"
load_project_env "$ROOT_DIR"
source "$ROOT_DIR/scripts/run_seed_utils.sh"
source "$ROOT_DIR/scripts/run_matrix_utils.sh"
source "$ROOT_DIR/scripts/run_job_utils.sh"
init_seed_runs

if [[ -n "$LHDNN_CALLER_PYTHON_BIN" ]]; then
  PYTHON_BIN="$LHDNN_CALLER_PYTHON_BIN"
elif [[ -n "${PYTHON_BIN:-}" && "$PYTHON_BIN" != "python" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
init_job_control
OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"

install_job_traps

parse_choice_list DATASETS "cifar100 cub200 aircraft" DATASETS \
  cifar100 cub200 aircraft

config_for_dataset() {
  case "$1" in
    cifar100) echo "configs/lhdnn/lhdnn_cifar100.yaml" ;;
    cub200) echo "configs/lhdnn/lhdnn_cub200.yaml" ;;
    aircraft) echo "configs/lhdnn/lhdnn_aircraft.yaml" ;;
    *)
      echo "Unknown dataset: $1" >&2
      exit 1
      ;;
  esac
}

run_output_dir() {
  local ds="$1"
  echo "$OUTPUTS_ROOT/lhdnn_${ds}"
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Model: large LH-DNN with projection and advantage topology\n'
printf 'CIFAR-100 protocol: paper-aligned\n'
printf 'CUB/Aircraft protocol: explicit extrapolation\n'
print_job_control_settings
print_seed_run_settings

for ds in "${DATASETS[@]}"; do
  cfg="$(config_for_dataset "$ds")"
  run_seeded_train "$cfg" "$(run_output_dir "$ds")"
done

drain_jobs

printf 'Completed all requested LH-DNN baseline runs.\n'
