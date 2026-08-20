#!/usr/bin/env bash
set -euo pipefail

# Runs the HRN + HCC arm (HCC generalized from H-CAST, see
# models/common/hcc.py) on top of the plain HRN baseline configs:
# - hrn_<dataset>_level_marginal_hcc
# for: cifar100, cub200, aircraft.
#
# HCC is a binary on/off switch; there is no onset/alpha/temperature ablation.
# HRN's real hierarchical training signal is a combinatorial marginal loss
# over sigmoid tree scores (_hierarchical_loss/_level_marginal_tree_losses),
# not the `logits_per_level` field. HCC intercepts the pre-sigmoid tree
# logits (classifier_1/2/3) and re-sigmoids the corrected values before they
# reach that loss; the separate leaf-only species_ce_logits head is
# untouched. model.loss=level_marginal is required so the loss decomposes
# into 3 per-level terms matching HCC's 3-level design (the same requirement
# already used by train.lexicographic for HRN, for the same reason).

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
ACTIVE_JOBS=0

if [[ ! "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer, got: $MAX_PARALLEL" >&2
  exit 2
fi
if [[ ! "$MAX_RESUME_RETRIES" =~ ^[0-9]+$ ]]; then
  echo "MAX_RESUME_RETRIES must be a non-negative integer, got: $MAX_RESUME_RETRIES" >&2
  exit 2
fi

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

wait_for_one_job() {
  local rc
  if wait -n; then
    ACTIVE_JOBS=$((ACTIVE_JOBS - 1))
    return 0
  else
    rc=$?
    echo "[ERROR] One run failed (exit=${rc}); stopping remaining jobs." >&2
    exit "$rc"
  fi
}

# Notebook-compatible outputs root.
# Example:
#   OUTPUTS_ROOT=/scratch/<user>/outputs ./scripts/hrn/run_hrn_hcc.sh
OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"

parse_choice_list DATASETS "cifar100 cub200 aircraft" DATASETS \
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

run_train() {
  local config="$1"
  local run_dir="$2"
  shift 2

  if [[ -f "$run_dir/test_metrics.yaml" ]]; then
    echo "[SKIP] Completed run already exists: $run_dir"
    return 0
  fi

  local resume_arg=""
  if [[ -f "$run_dir/latest.pt" ]]; then
    resume_arg="train.resume=$run_dir/latest.pt"
    echo "[RESUME] Found checkpoint: $run_dir/latest.pt"
  elif [[ -e "$run_dir/config_resolved.yaml" || -e "$run_dir/run_log.jsonl" || \
          -e "$run_dir/best_topdown.pt" || -e "$run_dir/best_independent.pt" ]]; then
    echo "[ERROR] Existing run artifacts have no resumable latest.pt: $run_dir" >&2
    return 1
  fi

  local cmd=(
    "$PYTHON_BIN" -m train.train
    --config "$config"
    "train.output_dir=$run_dir"
    "$@"
  )
  if [[ -n "$resume_arg" ]]; then
    cmd+=("$resume_arg")
  fi

  local retry_cmd=("${cmd[@]}")
  if [[ -z "$resume_arg" ]]; then
    retry_cmd+=("train.resume=$run_dir/latest.pt")
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY-RUN] '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if (( MAX_RESUME_RETRIES > 0 )); then
      printf '[DRY-RUN][RETRY x%s if latest.pt exists] ' "$MAX_RESUME_RETRIES"
      printf '%q ' "${retry_cmd[@]}"
      printf '\n'
    fi
  else
    while (( ACTIVE_JOBS >= MAX_PARALLEL )); do
      wait_for_one_job
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
        "${retry_cmd[@]}"
        rc=$?
      done
      exit "$rc"
    ) &
    ACTIVE_JOBS=$((ACTIVE_JOBS + 1))
  fi
}

run_output_dir() {
  local ds="$1"
  echo "$OUTPUTS_ROOT/hrn_${ds}_level_marginal_hcc"
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Lexicographic mode: disabled\n'
printf 'HCC: enabled\n'
printf 'Dry run: %s\n' "$DRY_RUN"
printf 'Max parallel: %s\n' "$MAX_PARALLEL"
printf 'Max resume retries on failure: %s\n' "$MAX_RESUME_RETRIES"
print_seed_run_settings

for ds in "${DATASETS[@]}"; do
  cfg="$(config_for_dataset "$ds")"

  run_seeded_train "$cfg" "$(run_output_dir "$ds")" \
    "model.loss=level_marginal" \
    "hcc.enabled=true" \
    "hcc.eps=1e-12" \
    "train.lexicographic.enabled=false" \
    "train.gradient_blocks=[p123]"
done

if [[ "$DRY_RUN" != "1" ]]; then
  while (( ACTIVE_JOBS > 0 )); do
    wait_for_one_job
  done
fi

printf 'Completed all requested HRN HCC runs.\n'
