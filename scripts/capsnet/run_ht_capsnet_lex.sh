#!/usr/bin/env bash
set -euo pipefail

# Runs native HT-CapsNet lexicographic training from the paper-baseline configs.
# Defaults: all datasets, 100 epochs, start@0, coarse-first, and unit
# per-level margin-loss weights (`model.loss.weight_mode=none`). The baseline
# configs now use the same 100-epoch budget, so the `train.epochs=100` override
# below is redundant and is kept only as an explicit guard; baseline-versus-lex
# is therefore epoch-matched, though still not level-weight matched.

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
LEX_PROJECTION_MODE="${LEX_PROJECTION_MODE:-coarse_first}"

case "$LEX_PROJECTION_MODE" in
  coarse_first|fine_first) ;;
  *)
    echo "Unsupported LEX_PROJECTION_MODE: $LEX_PROJECTION_MODE" >&2
    echo "Expected coarse_first or fine_first." >&2
    exit 2
    ;;
esac

install_job_traps

parse_choice_list DATASETS "cub200 aircraft cifar100" DATASETS \
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
  echo "$OUTPUTS_ROOT/ht_capsnet_${dataset}_lex_${LEX_PROJECTION_MODE}"
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Lex projection mode: %s\n' "$LEX_PROJECTION_MODE"
printf 'HT-CapsNet lex training epochs: 100\n'
print_job_control_settings
print_seed_run_settings

for dataset in "${DATASETS[@]}"; do
  config="$(config_for_dataset "$dataset")"
  run_seeded_train "$config" "$(run_output_dir "$dataset")" \
    "train.epochs=100" \
    "train.lexicographic.enabled=true" \
    "train.lexicographic.projection_mode=$LEX_PROJECTION_MODE" \
    "train.gradient_blocks=[p123,p23]" \
    "train.lexicographic.eps=1.0e-12" \
    "train.lexicographic.log_metrics=true"
done

drain_jobs

printf 'Completed all requested native HT-CapsNet lex runs.\n'
