#!/usr/bin/env bash
set -euo pipefail

# Runs the full Hier-COS loss grid requested:
# - model.loss=kl_reg
# - model.loss=per_level_ce with ce_weight_mode in {equal, kl_leaf, kl_coarse}
#   and lexicographic mode enabled
# for all datasets: cifar100, cub200, aircraft.

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
#   OUTPUTS_ROOT=/scratch/<user>/outputs ./scripts/run_hiercos_full_grid.sh
OUTPUTS_ROOT="${OUTPUTS_ROOT:-/scratch/g.saggini1/outputs}"

DATASETS=(cifar100 cub200 aircraft)

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
  local loss_mode="$2"
  local ce_mode="$3"

  case "$loss_mode:$ce_mode" in
    kl_reg:na) echo "$OUTPUTS_ROOT/hiercos_${ds}" ;;
    per_level_ce:equal) echo "$OUTPUTS_ROOT/hiercos_${ds}_ce_equal" ;;
    per_level_ce:kl_leaf) echo "$OUTPUTS_ROOT/hiercos_${ds}_ce_kl_leaf" ;;
    per_level_ce:kl_coarse) echo "$OUTPUTS_ROOT/hiercos_${ds}_ce_kl_coarse" ;;
    *)
      echo "Unknown run naming tuple: $loss_mode $ce_mode" >&2
      exit 1
      ;;
  esac
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Dry run: %s\n' "$DRY_RUN"
printf 'Max parallel: %s\n' "$MAX_PARALLEL"
printf 'Max resume retries on failure: %s\n' "$MAX_RESUME_RETRIES"

for ds in "${DATASETS[@]}"; do
  cfg="$(config_for_dataset "$ds")"

  # 1) Paper-aligned Hier-COS KL + regularization
  run_train "$cfg" "$(run_output_dir "$ds" kl_reg na)" \
    "model.loss=kl_reg"

  # 2) Local CE ablation with all supported weight modes + lexicographic mode
  run_train "$cfg" "$(run_output_dir "$ds" per_level_ce equal)" \
    "model.loss=per_level_ce" \
    "model.ce_weight_mode=equal" \
    "train.lexicographic.enabled=true" \
    "train.lexicographic.start_epoch=0"

  run_train "$cfg" "$(run_output_dir "$ds" per_level_ce kl_leaf)" \
    "model.loss=per_level_ce" \
    "model.ce_weight_mode=kl_leaf" \
    "train.lexicographic.enabled=true" \
    "train.lexicographic.start_epoch=0"

  run_train "$cfg" "$(run_output_dir "$ds" per_level_ce kl_coarse)" \
    "model.loss=per_level_ce" \
    "model.ce_weight_mode=kl_coarse" \
    "train.lexicographic.enabled=true" \
    "train.lexicographic.start_epoch=0"
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

printf 'Completed all requested Hier-COS runs.\n'
