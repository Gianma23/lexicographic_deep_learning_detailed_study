#!/usr/bin/env bash
set -euo pipefail

# Hier-COS training with hard-label CE on taxonomy-subspace scores.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/load_env.sh"
load_project_env "$ROOT_DIR"
source "$ROOT_DIR/scripts/run_seed_utils.sh"
source "$ROOT_DIR/scripts/run_matrix_utils.sh"
init_seed_runs

PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
MAX_RESUME_RETRIES="${MAX_RESUME_RETRIES:-1}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"

parse_choice_list DATASETS "cifar100 cub200 aircraft" DATASETS cifar100 cub200 aircraft

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
  echo "$OUTPUTS_ROOT/hiercos_${dataset}_subspace_cross_entropy"
}

kill_running_jobs() {
  jobs -pr | xargs -r kill 2>/dev/null || true
}

handle_interrupt() {
  echo "[INTERRUPT] Received signal, stopping running jobs..." >&2
  kill_running_jobs
  wait || true
  exit 130
}

handle_exit() {
  local rc=$?
  if (( rc != 0 )); then
    kill_running_jobs
    wait || true
  fi
}

trap handle_interrupt INT TERM
trap handle_exit EXIT

run_train() {
  local config="$1"
  local run_dir="$2"
  shift 2

  local -a cmd=(
    "$PYTHON_BIN" -m train.train
    --config "$config"
    "train.output_dir=$run_dir"
    "$@"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY-RUN] '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if (( MAX_RESUME_RETRIES > 0 )); then
      printf '[DRY-RUN][RETRY x%s] ' "$MAX_RESUME_RETRIES"
      printf '%q ' "${cmd[@]}" "train.resume=$run_dir/latest.pt"
      printf '\n'
    fi
    return
  fi

  while (( "$(jobs -pr | wc -l)" >= MAX_PARALLEL )); do
    set +e
    wait -n
    local rc=$?
    set -e
    if (( rc != 0 )); then
      echo "[ERROR] One run failed (exit=${rc}); stopping remaining jobs." >&2
      kill_running_jobs
      exit "$rc"
    fi
  done

  printf '[RUN] '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  (
    set +e
    "${cmd[@]}"
    rc=$?
    attempt=0
    while (( rc != 0 && attempt < MAX_RESUME_RETRIES )); do
      attempt=$((attempt + 1))
      echo "[RETRY ${attempt}/${MAX_RESUME_RETRIES}] run_dir=$run_dir resume=$run_dir/latest.pt" >&2
      "${cmd[@]}" "train.resume=$run_dir/latest.pt"
      rc=$?
    done
    exit "$rc"
  ) &
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Training: hard-label cross-entropy on raw subspace scores, alpha=0\n'
printf 'Hier-COS LH projection: disabled\n'
printf 'HCC: disabled\n'
printf 'Lexicographic training: disabled\n'
printf 'Dry run: %s\n' "$DRY_RUN"
printf 'Max parallel: %s\n' "$MAX_PARALLEL"
printf 'Max resume retries on failure: %s\n' "$MAX_RESUME_RETRIES"
print_seed_run_settings

for dataset in "${DATASETS[@]}"; do
  config="$(config_for_dataset "$dataset")"
  run_seeded_train "$config" "$(run_output_dir "$dataset")" \
    "model.loss=kl_reg" \
    "model.weight_mode=kl_leaf" \
    "model.alpha=0.0" \
    "model.fixed_frame_mode=orthonormal_random" \
    "model.fixed_frame_per_level=false" \
    "model.projection.enabled=false" \
    "model.projection.advantage_enabled=false" \
    "train.lexicographic.enabled=false" \
    "hcc.enabled=false" \
    "train.smoothing=0.0" \
    "dataset.transforms.mixup=0.0" \
    "dataset.transforms.cutmix=0.0" \
    "dataset.transforms.cutmix_minmax=null" \
    "train.subspace_supervision.enabled=true" \
    "train.subspace_supervision.loss=cross_entropy"
done

if [[ "$DRY_RUN" != "1" ]]; then
  while (( "$(jobs -pr | wc -l)" > 0 )); do
    set +e
    wait -n
    rc=$?
    set -e
    if (( rc != 0 )); then
      echo "[ERROR] One run failed (exit=${rc}); stopping remaining jobs." >&2
      kill_running_jobs
      exit "$rc"
    fi
  done
fi

printf 'Completed all requested Hier-COS subspace runs.\n'
