#!/usr/bin/env bash
set -euo pipefail

# Runs native HT-CapsNet lexicographic training from the paper-baseline configs.
# Defaults: all datasets, start@0, coarse-first, orthogonalize-all, and unit
# per-level margin-loss weights (`model.loss.weight_mode=none`).

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
LEX_PROJECTION_MODE="${LEX_PROJECTION_MODE:-coarse_first}"
LEX_PROJECTION_RULE="${LEX_PROJECTION_RULE:-orthogonalize_all}"

case "$LEX_PROJECTION_MODE" in
  coarse_first|fine_first) ;;
  *)
    echo "Unsupported LEX_PROJECTION_MODE: $LEX_PROJECTION_MODE" >&2
    echo "Expected coarse_first or fine_first." >&2
    exit 2
    ;;
esac
case "$LEX_PROJECTION_RULE" in
  orthogonalize_all|conflict_only) ;;
  *)
    echo "Unsupported LEX_PROJECTION_RULE: $LEX_PROJECTION_RULE" >&2
    echo "Expected orthogonalize_all or conflict_only." >&2
    exit 2
    ;;
esac

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

parse_choice_list DATASETS "cifar100 cub200 aircraft" DATASETS \
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

run_train() {
  local config="$1"
  local run_dir="$2"
  shift 2

  local cmd=(
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
  else
    while (( "$(jobs -pr | wc -l)" >= MAX_PARALLEL )); do
      if ! wait -n; then
        local rc=$?
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
  fi
}

run_output_dir() {
  local dataset="$1"
  echo "$OUTPUTS_ROOT/ht_capsnet_${dataset}_lex_${LEX_PROJECTION_RULE}_${LEX_PROJECTION_MODE}"
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Lex projection mode: %s\n' "$LEX_PROJECTION_MODE"
printf 'Lex projection rule: %s\n' "$LEX_PROJECTION_RULE"
printf 'HT-CapsNet level weights: unit (weight_mode=none)\n'
printf 'Dry run: %s\n' "$DRY_RUN"
printf 'Max parallel: %s\n' "$MAX_PARALLEL"
printf 'Max resume retries on failure: %s\n' "$MAX_RESUME_RETRIES"
print_seed_run_settings

for dataset in "${DATASETS[@]}"; do
  config="$(config_for_dataset "$dataset")"
  run_seeded_train "$config" "$(run_output_dir "$dataset")" \
    "orthonormal_plugin.enabled=false" \
    "model.loss.weight_mode=none" \
    "train.lexicographic.enabled=true" \
    "train.lexicographic.projection_mode=$LEX_PROJECTION_MODE" \
    "train.lexicographic.projection_rule=$LEX_PROJECTION_RULE" \
    "train.lexicographic.eps=1.0e-12" \
    "train.lexicographic.log_metrics=true"
done

if [[ "$DRY_RUN" != "1" ]]; then
  while (( "$(jobs -pr | wc -l)" > 0 )); do
    if ! wait -n; then
      rc=$?
      echo "[ERROR] One run failed (exit=${rc}); stopping remaining jobs." >&2
      kill_running_jobs
      exit "$rc"
    fi
  done
fi

printf 'Completed all requested native HT-CapsNet lex runs.\n'
