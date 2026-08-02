#!/usr/bin/env bash
set -euo pipefail

# Runs Hier-COS + HCC variants (HCC generalized from H-CAST, see
# models/common/hcc.py) on top of the plain Hier-COS baseline configs:
# - hiercos_hcc_<dataset>_step_0epoch
# - hiercos_hcc_<dataset>_step_80epoch
# for: cifar100, cub200, aircraft.
#
# HCC needs a specific config combination forced on top of each baseline,
# deviating from the paper-faithful dense/global fixed-frame setting:
# - model.fixed_frame_per_level=true so `node_logits_per_level` exists as
#   independent per-level linear blocks (HCC's linear parent-sums-children
#   constraint has no meaning on the dense/global frame's mixed node logits).
# - model.fixed_frame_mode=identity: the simplest, most literal analogue of
#   H-CAST's plain learnable per-level heads (no confound from an arbitrary
#   frozen in-block rotation). orthonormal_random (still per-level) is a
#   legitimate secondary ablation -- swap the override below to try it.
# - model.loss=level_softmax_ce_reg so per-level node logits feed independent
#   per-level CE instead of a shared global softmax over all levels.
# - model.projection.enabled is left at its baseline default (false): Hier-COS's
#   own `projection` block is an embedded LH-DNN-style mechanism; combining it
#   with HCC would confound two different consistency corrections in one run.

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
#   OUTPUTS_ROOT=/scratch/<user>/outputs ./scripts/hiercos/run_hiercos_hcc_grid.sh
OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"

parse_choice_list DATASETS "cifar100 cub200 aircraft" DATASETS \
  cifar100 cub200 aircraft

# Start from the plain Hier-COS baseline config; the model.*/hcc.* overrides
# below are what changes relative to run_hiercos_baselines.sh.
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
  local epoch_tag="$2"
  echo "$OUTPUTS_ROOT/hiercos_hcc_${ds}_level_softmax_ce_reg_identity_per_level_step_${epoch_tag}"
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
  cfg="$(config_for_dataset "$ds")"

  run_seeded_train "$cfg" "$(run_output_dir "$ds" 0epoch)" \
    "model.loss=level_softmax_ce_reg" \
    "model.fixed_frame_mode=identity" \
    "model.fixed_frame_per_level=true" \
    "hcc.enabled=true" \
    "hcc.temperature=10" \
    "hcc.eps=1e-12" \
    "hcc.alpha_schedule=step" \
    "hcc.alpha_start_epoch=0" \
    "train.lexicographic.enabled=false"

  # 80% of train.epochs (100), matching H-CAST's own 80/100 convention.
  run_seeded_train "$cfg" "$(run_output_dir "$ds" 80epoch)" \
    "model.loss=level_softmax_ce_reg" \
    "model.fixed_frame_mode=identity" \
    "model.fixed_frame_per_level=true" \
    "hcc.enabled=true" \
    "hcc.temperature=10" \
    "hcc.eps=1e-12" \
    "hcc.alpha_schedule=step" \
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

printf 'Completed all requested Hier-COS HCC runs.\n'
