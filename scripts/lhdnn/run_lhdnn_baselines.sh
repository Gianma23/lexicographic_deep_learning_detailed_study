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
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
MAX_RESUME_RETRIES="${MAX_RESUME_RETRIES:-1}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"

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
    cifar100) echo "configs/lhdnn/lhdnn_cifar100.yaml" ;;
    cub200) echo "configs/lhdnn/lhdnn_cub200.yaml" ;;
    aircraft) echo "configs/lhdnn/lhdnn_aircraft.yaml" ;;
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
      printf '[DRY-RUN][RETRY x%s if latest.pt exists] ' "$MAX_RESUME_RETRIES"
      printf '%q ' "${cmd[@]}" "train.resume=$run_dir/latest.pt"
      printf '\n'
    fi
  else
    while (( "$(jobs -pr | wc -l)" >= MAX_PARALLEL )); do
      if wait -n; then
        :
      else
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
        if [[ ! -f "$run_dir/latest.pt" ]]; then
          echo "[NO RETRY] No checkpoint at $run_dir/latest.pt; preserving the original failure." >&2
          break
        fi
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
  echo "$OUTPUTS_ROOT/lhdnn_${ds}"
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Model: large LH-DNN with projection and advantage topology\n'
printf 'CIFAR-100 protocol: paper-aligned\n'
printf 'CUB/Aircraft protocol: explicit extrapolation\n'
printf 'Dry run: %s\n' "$DRY_RUN"
printf 'Max parallel: %s\n' "$MAX_PARALLEL"
printf 'Max resume retries on failure: %s\n' "$MAX_RESUME_RETRIES"
print_seed_run_settings

for ds in "${DATASETS[@]}"; do
  cfg="$(config_for_dataset "$ds")"
  run_seeded_train "$cfg" "$(run_output_dir "$ds")"
done

if [[ "$DRY_RUN" != "1" ]]; then
  while (( "$(jobs -pr | wc -l)" > 0 )); do
    if wait -n; then
      :
    else
      rc=$?
      echo "[ERROR] One run failed (exit=${rc}); stopping remaining jobs." >&2
      jobs -pr | xargs -r kill 2>/dev/null || true
      exit "$rc"
    fi
  done
fi

printf 'Completed all requested LH-DNN baseline runs.\n'
