#!/usr/bin/env bash
set -euo pipefail

# HRN + HCC arm on top of the plain HRN baseline configs (cifar100, cub200,
# aircraft). HCC constrains HRN's emitted score triple, at train and test time.
# model.loss=level_conditional telescopes to the `native` objective, so it
# changes nothing about what is optimised and only adds the per-level terms.

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
#   OUTPUTS_ROOT=/scratch/<user>/outputs ./scripts/hrn/run_hrn_hcc.sh
OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"

parse_choice_list DATASETS "cifar100" DATASETS \
  cifar100 cub200 aircraft

# Start from the plain HRN baseline config; the model.loss/hcc.* overrides
# below are what changes relative to run_hrn_baselines.sh.
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

run_output_dir() {
  local ds="$1"
  echo "$OUTPUTS_ROOT/hrn_${ds}_level_conditional_hcc"
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
    "model.loss=level_conditional" \
    "hcc.enabled=true" \
    "hcc.eps=1e-12" \
    "train.lexicographic.enabled=false" \
    "train.gradient_blocks=[p123,p23,p3]"
done

drain_jobs

printf 'Completed all requested HRN HCC runs.\n'
