#!/usr/bin/env bash
set -euo pipefail

# Runs one FGVC-Aircraft Hier-COS pretraining-ablation configuration:
# - model.loss=${LOSS_MODE}
# - model.weight_mode=${WEIGHT_MODE}
# - model.pretrained=${PRETRAINED_MODE}
# - model.fixed_frame_mode=${FIXED_FRAME_MODE}
# - model.transform_mode=full
# - train.lexicographic.enabled=false
# Three training seeds are used by default. Override the scalar mode variables
# to select a different single configuration; this script does not run a grid.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/load_env.sh"
load_project_env "$ROOT_DIR"
source "$ROOT_DIR/scripts/run_seed_utils.sh"

NUM_RUNS="${NUM_RUNS:-3}"
init_seed_runs

PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
MAX_RESUME_RETRIES="${MAX_RESUME_RETRIES:-1}"
LOSS_MODE="${LOSS_MODE:-global_softmax_ce_reg}"
WEIGHT_MODE="${WEIGHT_MODE:-kl_leaf}"
FIXED_FRAME_MODE="${FIXED_FRAME_MODE:-identity}"

CONFIG="configs/hiercos/hiercos_aircraft.yaml"

case "$LOSS_MODE" in
  global_softmax_ce_reg|level_softmax_ce_reg) ;;
  *)
    echo "Unsupported LOSS_MODE: $LOSS_MODE" >&2
    echo "Expected global_softmax_ce_reg or level_softmax_ce_reg." >&2
    exit 1
    ;;
esac

case "$WEIGHT_MODE" in
  equal|kl_leaf|kl_coarse) ;;
  *)
    echo "Unsupported WEIGHT_MODE: $WEIGHT_MODE" >&2
    echo "Expected equal, kl_leaf, or kl_coarse." >&2
    exit 1
    ;;
esac

case "$PRETRAINED_MODE" in
  true|false) ;;
  *)
    echo "Unsupported PRETRAINED_MODE: $PRETRAINED_MODE" >&2
    echo "Expected true or false." >&2
    exit 1
    ;;
esac

case "$FIXED_FRAME_MODE" in
  orthonormal_random|identity) ;;
  *)
    echo "Unsupported FIXED_FRAME_MODE: $FIXED_FRAME_MODE" >&2
    echo "Expected orthonormal_random or identity." >&2
    exit 1
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

# Notebook-compatible outputs root.
# Examples:
#   # Defaults: scratch backbone, identity frame, seeds 0..2.
#   OUTPUTS_ROOT=/scratch/<user>/outputs \
#     ./scripts/hiercos/run_hiercos_aircraft_pretraining_ablation.sh
#   # Select one different configuration, still using three seeds.
#   PRETRAINED_MODE=true FIXED_FRAME_MODE=orthonormal_random \
#     ./scripts/hiercos/run_hiercos_aircraft_pretraining_ablation.sh
OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"

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
  local frame_suffix=""
  if [[ "$FIXED_FRAME_MODE" == "identity" ]]; then
    frame_suffix="_identity"
  fi
  echo "$OUTPUTS_ROOT/hiercos_aircraft_${LOSS_MODE}_fromscratch_${WEIGHT_MODE}${frame_suffix}"
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Dataset: aircraft\n'
printf 'Loss: %s\n' "$LOSS_MODE"
printf 'Weight mode: %s\n' "$WEIGHT_MODE"
printf 'Pretrained: %s\n' "$PRETRAINED_MODE"
printf 'Fixed frame mode: %s\n' "$FIXED_FRAME_MODE"
printf 'Transform mode: full\n'
printf 'Lexicographic mode: disabled\n'
printf 'Dry run: %s\n' "$DRY_RUN"
printf 'Max parallel: %s\n' "$MAX_PARALLEL"
printf 'Max resume retries on failure: %s\n' "$MAX_RESUME_RETRIES"
print_seed_run_settings

run_seeded_train "$CONFIG" "$(run_output_dir)" \
  "model.loss=$LOSS_MODE" \
  "model.weight_mode=$WEIGHT_MODE" \
  "model.pretrained=$PRETRAINED_MODE" \
  "model.fixed_frame_mode=$FIXED_FRAME_MODE" \
  "model.transform_mode=full" \
  "train.lexicographic.enabled=false"

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

printf 'Completed requested Hier-COS Aircraft pretraining-ablation runs.\n'
