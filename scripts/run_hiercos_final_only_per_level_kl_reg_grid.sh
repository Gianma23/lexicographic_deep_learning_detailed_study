#!/usr/bin/env bash
set -euo pipefail

# Runs three explicit Hier-COS final-only ablations:
# - model.loss=per_level_kl_reg
# - model.transform_mode=final_only
# - train.lexicographic.enabled=true, start_epoch=0
# - model.weight_mode:
#   - cub200: kl_leaf
#   - cifar100/aircraft: equal

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

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
#   OUTPUTS_ROOT=/scratch/<user>/outputs ./scripts/run_hiercos_final_only_per_level_kl_reg_grid.sh
OUTPUTS_ROOT="${OUTPUTS_ROOT:-/scratch/g.saggini1/outputs}"

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

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Dry run: %s\n' "$DRY_RUN"
printf 'Max parallel: %s\n' "$MAX_PARALLEL"
printf 'Max resume retries on failure: %s\n' "$MAX_RESUME_RETRIES"

# 1) CIFAR-100 (equal weights)
run_train "configs/hiercos/hiercos_cifar100.yaml" \
  "$OUTPUTS_ROOT/hiercos_cifar100_per_level_kl_reg_equal_final_only" \
  "model.loss=per_level_kl_reg" \
  "model.weight_mode=equal" \
  "model.transform_mode=final_only" \
  "train.lexicographic.enabled=true" \
  "train.lexicographic.start_epoch=0"

# 2) CUB-200 (kl_leaf weights)
run_train "configs/hiercos/hiercos_cub200.yaml" \
  "$OUTPUTS_ROOT/hiercos_cub200_per_level_kl_reg_kl_leaf_final_only" \
  "model.loss=per_level_kl_reg" \
  "model.weight_mode=kl_leaf" \
  "model.transform_mode=final_only" \
  "train.lexicographic.enabled=true" \
  "train.lexicographic.start_epoch=0"

# 3) FGVC-Aircraft (equal weights)
run_train "configs/hiercos/hiercos_aircraft.yaml" \
  "$OUTPUTS_ROOT/hiercos_aircraft_per_level_kl_reg_equal_final_only" \
  "model.loss=per_level_kl_reg" \
  "model.weight_mode=equal" \
  "model.transform_mode=final_only" \
  "train.lexicographic.enabled=true" \
  "train.lexicographic.start_epoch=0"

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

printf 'Completed all requested Hier-COS final-only runs.\n'
