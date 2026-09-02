#!/usr/bin/env bash
set -euo pipefail

# Hier-COS training with tempered soft CE on taxonomy-subspace scores against
# the attainable path-energy profile.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/load_env.sh"
load_project_env "$ROOT_DIR"
source "$ROOT_DIR/scripts/run_seed_utils.sh"
source "$ROOT_DIR/scripts/run_matrix_utils.sh"
source "$ROOT_DIR/scripts/run_job_utils.sh"
init_seed_runs

init_job_control
SUBSPACE_TAU="${SUBSPACE_TAU:-0.1}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"

parse_choice_list DATASETS "cifar100" DATASETS cifar100 cub200 aircraft

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
  local dataset="$1"
  echo "$OUTPUTS_ROOT/hiercos_${dataset}_subspace"
}

install_job_traps

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Training: soft cross-entropy on subspace scores vs the induced path-energy profile\n'
printf 'Subspace tau: %s\n' "$SUBSPACE_TAU"
printf 'Subspace level weighting: model.weight_mode weights the level losses (target is weight-free)\n'
printf 'Hier-COS LH projection: disabled\n'
printf 'HCC: disabled\n'
printf 'Lexicographic training: disabled\n'
print_job_control_settings
print_seed_run_settings

for dataset in "${DATASETS[@]}"; do
  config="$(config_for_dataset "$dataset")"
  # model.loss and model.alpha are inert here: enabling subspace supervision
  # bypasses the native Hier-COS loss entirely. They are pinned so
  # config_resolved.yaml records a known state. model.weight_mode is NOT inert:
  # it sets the per-level coefficients of the scalarisation, the same role it
  # has in the native softmax losses, which is what keeps the arm comparable to
  # the kl_leaf baselines. The target geometry carries no weighting.
  run_seeded_train "$config" "$(run_output_dir "$dataset")" \
    "model.loss=kl_reg" \
    "model.weight_mode=kl_leaf" \
    "model.alpha=0.0" \
    "model.fixed_frame_mode=orthonormal_random" \
    "model.fixed_frame_per_level=false" \
    "model.projection.enabled=false" \
    "model.projection.advantage_enabled=false" \
    "train.lexicographic.enabled=false" \
    "train.gradient_blocks=[p123]" \
    "hcc.enabled=false" \
    "train.smoothing=0.0" \
    "dataset.transforms.mixup=0.0" \
    "dataset.transforms.cutmix=0.0" \
    "dataset.transforms.cutmix_minmax=null" \
    "train.subspace_supervision.enabled=true" \
    "train.subspace_supervision.tau=$SUBSPACE_TAU"
done

drain_jobs

printf 'Completed all requested Hier-COS subspace runs.\n'
