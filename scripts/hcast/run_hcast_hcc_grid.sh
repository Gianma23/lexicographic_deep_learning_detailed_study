#!/usr/bin/env bash
set -euo pipefail

# Runs H-CAST + HCC variants on all supported HCC datasets:
# - hcast_hcc_<dataset>_step_0epoch
# - hcast_hcc_<dataset>_step_80epoch
# for: cifar100, cub200, aircraft.

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

# Notebook-compatible outputs root.
# Example:
#   OUTPUTS_ROOT=/scratch/<user>/outputs ./scripts/hcast/run_hcast_hcc_grid.sh
OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"

parse_choice_list DATASETS "cifar100 cub200 aircraft" DATASETS \
  cifar100 cub200 aircraft

hcc_config_for_dataset() {
  case "$1" in
    cifar100) echo "configs/hcast/hcast_hcc_cifar100.yaml" ;;
    cub200) echo "configs/hcast/hcast_hcc_cub200.yaml" ;;
    aircraft) echo "configs/hcast/hcast_hcc_aircraft.yaml" ;;
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
        rc=$?
        echo "[ERROR] One run failed (exit=${rc}); stopping remaining jobs." >&2
        jobs -pr | xargs -r kill 2>/dev/null || true
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
  local ds="$1"
  local epoch_tag="$2"
  case "$ds:$epoch_tag" in
    cifar100:e0) echo "$OUTPUTS_ROOT/hcast_hcc_cifar100_step_0epoch" ;;
    cifar100:e80) echo "$OUTPUTS_ROOT/hcast_hcc_cifar100_step_80epoch" ;;
    cub200:e0) echo "$OUTPUTS_ROOT/hcast_hcc_cub200_step_0epoch" ;;
    cub200:e80) echo "$OUTPUTS_ROOT/hcast_hcc_cub200_step_80epoch" ;;
    aircraft:e0) echo "$OUTPUTS_ROOT/hcast_hcc_aircraft_step_0epoch_nokl" ;;
    aircraft:e80) echo "$OUTPUTS_ROOT/hcast_hcc_aircraft_step_80epoch_nokl" ;;
    *)
      echo "Unknown HCC run tuple: $ds $epoch_tag" >&2
      exit 1
      ;;
  esac
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Lexicographic mode: disabled\n'
printf 'HCC: enabled\n'
printf 'HCC alpha schedule: step\n'
printf 'Dry run: %s\n' "$DRY_RUN"
printf 'Max parallel: %s\n' "$MAX_PARALLEL"
printf 'Max resume retries on failure: %s\n' "$MAX_RESUME_RETRIES"
print_seed_run_settings

for ds in "${DATASETS[@]}"; do
  hcc_cfg="$(hcc_config_for_dataset "$ds")"

  run_seeded_train "$hcc_cfg" "$(run_output_dir "$ds" e0)" \
    "hcc.enabled=true" \
    "hcc.alpha_start_epoch=0" \
    "train.lexicographic.enabled=false"

  run_seeded_train "$hcc_cfg" "$(run_output_dir "$ds" e80)" \
    "hcc.enabled=true" \
    "hcc.alpha_start_epoch=80" \
    "train.lexicographic.enabled=false"
done

if [[ "$DRY_RUN" != "1" ]]; then
  while (( "$(jobs -pr | wc -l)" > 0 )); do
    if ! wait -n; then
      rc=$?
      echo "[ERROR] One run failed (exit=${rc}); stopping remaining jobs." >&2
      jobs -pr | xargs -r kill 2>/dev/null || true
      exit "$rc"
    fi
  done
fi

printf 'Completed all requested H-CAST HCC runs.\n'
