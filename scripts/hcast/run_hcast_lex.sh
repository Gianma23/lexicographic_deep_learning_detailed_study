#!/usr/bin/env bash
set -euo pipefail

# Runs the selected H-CAST lexicographic matrix. Lexicographic projection is
# always active for the whole run. Override with whitespace-separated DATASETS
# or with LEX_PROJECTION_MODE to select another priority order.
#
# Starts from the plain H-CAST baseline config for each dataset and adds the
# lexicographic block as CLI overrides, so `configs/hcast/` keeps only base
# presets. `model.loss.globalkl=false` is required by lexicographic H-CAST.

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

# Notebook-compatible outputs root (run dir names match notebooks/hcast_analysis.ipynb).
# Example:
#   OUTPUTS_ROOT=/scratch/<user>/outputs ./scripts/hcast/run_hcast_lex.sh
OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"
LEX_PROJECTION_MODE="${LEX_PROJECTION_MODE:-fine_first}"

case "$LEX_PROJECTION_MODE" in
  coarse_first|fine_first) ;;
  *)
    echo "Unsupported LEX_PROJECTION_MODE: $LEX_PROJECTION_MODE" >&2
    echo "Expected coarse_first or fine_first." >&2
    exit 2
    ;;
esac
parse_choice_list DATASETS "cub200 aircraft cifar100" DATASETS cifar100 cub200 aircraft

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
  echo "$OUTPUTS_ROOT/hcast_${ds}_lex_${LEX_PROJECTION_MODE}"
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Lex projection mode: %s\n' "$LEX_PROJECTION_MODE"
print_job_control_settings
print_seed_run_settings

for ds in "${DATASETS[@]}"; do
  cfg="$(config_for_dataset "$ds")"

  run_seeded_train "$cfg" "$(run_output_dir "$ds")" \
    "model.loss.globalkl=false" \
    "train.lexicographic.enabled=true" \
    "train.lexicographic.projection_mode=$LEX_PROJECTION_MODE" \
    "train.gradient_blocks=[p123,p12]" \
    "train.lexicographic.eps=1.0e-12" \
    "train.lexicographic.log_metrics=true"
done

drain_jobs

printf 'Completed all requested lex runs.\n'
