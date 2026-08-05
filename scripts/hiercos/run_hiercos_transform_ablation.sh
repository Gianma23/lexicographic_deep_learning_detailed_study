#!/usr/bin/env bash
set -euo pipefail

# Runs the selected Hier-COS transformation-layer ablation matrix:
# - model.loss=${LOSS_MODE} (global_softmax_ce_reg or level_softmax_ce_reg)
# - model.weight_mode=${WEIGHT_MODE}
# - model.fixed_frame_mode=${FIXED_FRAME_MODE}
# - model.fixed_frame_per_level=${FIXED_FRAME_PER_LEVEL}
# - train.lexicographic.enabled=true
# - train.lexicographic.projection_mode=coarse_first
# - train.lexicographic.projection_rule=orthogonalize_all
# - model.transform_mode selected by TRANSFORM_MODES
# The matching full-transform reference is produced by
# scripts/hiercos/run_hiercos_lex_orthogonalize_all.sh.
# Defaults: cub200/aircraft/cifar100 with final_only.

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
LOSS_MODE="${LOSS_MODE:-level_softmax_ce_reg}"
WEIGHT_MODE="${WEIGHT_MODE:-equal}"
FIXED_FRAME_MODE="${FIXED_FRAME_MODE:-orthonormal_random}"
FIXED_FRAME_PER_LEVEL="${FIXED_FRAME_PER_LEVEL:-0}"

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

case "$FIXED_FRAME_MODE" in
  orthonormal_random|orthonormal_block_random|identity) ;;
  *)
    echo "Unsupported FIXED_FRAME_MODE: $FIXED_FRAME_MODE" >&2
    echo "Expected orthonormal_random, orthonormal_block_random, or identity." >&2
    exit 1
    ;;
esac

case "$FIXED_FRAME_PER_LEVEL" in
  0|1|true|false|True|False) ;;
  *)
    echo "Unsupported FIXED_FRAME_PER_LEVEL: $FIXED_FRAME_PER_LEVEL" >&2
    echo "Expected 0, 1, true, or false." >&2
    exit 1
    ;;
esac
normalize_bool_like "$FIXED_FRAME_PER_LEVEL" FIXED_FRAME_PER_LEVEL_OVERRIDE

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
#   OUTPUTS_ROOT=/scratch/<user>/outputs ./scripts/hiercos/run_hiercos_transform_ablation.sh
#   LOSS_MODE=level_softmax_ce_reg WEIGHT_MODE=equal FIXED_FRAME_MODE=identity \
#     ./scripts/hiercos/run_hiercos_transform_ablation.sh
OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"

parse_choice_list DATASETS "cifar100" DATASETS \
  cifar100 cub200 aircraft
parse_choice_list TRANSFORM_MODES "bn_linear" TRANSFORM_MODES \
  full bn_linear final_only

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
  local transform_mode="$2"
  local weight_suffix=""
  local frame_suffix=""
  local per_level="$FIXED_FRAME_PER_LEVEL"
  if [[ "$WEIGHT_MODE" != "equal" ]]; then
    weight_suffix="_${WEIGHT_MODE}"
  fi
  if [[ "$FIXED_FRAME_MODE" == "identity" && "$per_level" =~ ^(1|true|True)$ ]]; then
    frame_suffix="_identity_per_level"
  elif [[ "$FIXED_FRAME_MODE" == "identity" ]]; then
    frame_suffix="_identity"
  fi
  echo "$OUTPUTS_ROOT/hiercos_${ds}_${LOSS_MODE}_lex_orthogonalize_all_coarse_first_${transform_mode}${weight_suffix}${frame_suffix}"
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Transform modes: %s\n' "${TRANSFORM_MODES[*]}"
printf 'Loss: %s\n' "$LOSS_MODE"
printf 'Weight mode: %s\n' "$WEIGHT_MODE"
printf 'Fixed frame mode: %s\n' "$FIXED_FRAME_MODE"
printf 'Fixed frame per level: %s\n' "$FIXED_FRAME_PER_LEVEL"
printf 'Projection mode: coarse_first\n'
printf 'Projection rule: orthogonalize_all\n'
printf 'Dry run: %s\n' "$DRY_RUN"
printf 'Max parallel: %s\n' "$MAX_PARALLEL"
printf 'Max resume retries on failure: %s\n' "$MAX_RESUME_RETRIES"
print_seed_run_settings

for ds in "${DATASETS[@]}"; do
  cfg="$(config_for_dataset "$ds")"

  for transform_mode in "${TRANSFORM_MODES[@]}"; do
    run_seeded_train "$cfg" "$(run_output_dir "$ds" "$transform_mode")" \
      "model.loss=$LOSS_MODE" \
      "model.weight_mode=$WEIGHT_MODE" \
      "model.transform_mode=$transform_mode" \
      "model.fixed_frame_mode=$FIXED_FRAME_MODE" \
      "model.fixed_frame_per_level=$FIXED_FRAME_PER_LEVEL_OVERRIDE" \
      "train.lexicographic.enabled=true" \
      "train.lexicographic.projection_mode=coarse_first" \
      "train.lexicographic.projection_rule=orthogonalize_all"
  done
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

printf 'Completed all requested Hier-COS transform ablation runs.\n'
