#!/usr/bin/env bash
set -euo pipefail

# Runs Hier-COS with an LH-style projection after the complete transform:
# - full transform mode retains its final residual skip
# - projection uses A=[W_1; ...; W_(l-1)] with no activation derivative
# - each projected level uses a learnable FC head
# - the three FC outputs are concatenated before the global fixed frame
# - LH-DNN advantage baselines are selected by ADVANTAGE_ENABLED
# - model.loss=level_softmax_ce_reg
# - model.projection.enabled=true
# - model.projection.eps=${PROJECTION_EPS}
# - model.transform_mode selected by TRANSFORM_MODES
# - model.fixed_frame_mode=${FIXED_FRAME_MODE}
# - train.lexicographic.enabled=false
#
# Examples:
#   ./scripts/hiercos/run_hiercos_lhdnn_projection.sh
#   DATASETS="cub200 aircraft" NUM_RUNS=3 ./scripts/hiercos/run_hiercos_lhdnn_projection.sh
#   ADVANTAGE_ENABLED=true DATASETS=cifar100 ./scripts/hiercos/run_hiercos_lhdnn_projection.sh
#   FIXED_FRAME_MODE=identity DATASETS=cub200 ./scripts/hiercos/run_hiercos_lhdnn_projection.sh

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
WEIGHT_MODE="${WEIGHT_MODE:-equal}"
FIXED_FRAME_MODE="${FIXED_FRAME_MODE:-orthonormal_random}"
PROJECTION_EPS="${PROJECTION_EPS:-1.0e-6}"
ADVANTAGE_ENABLED="${ADVANTAGE_ENABLED:-false}"

case "$WEIGHT_MODE" in
  equal|kl_leaf|kl_coarse) ;;
  *)
    echo "Unsupported WEIGHT_MODE: $WEIGHT_MODE" >&2
    echo "Expected equal, kl_leaf, or kl_coarse." >&2
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

case "$ADVANTAGE_ENABLED" in
  true|false) ;;
  *)
    echo "Unsupported ADVANTAGE_ENABLED: $ADVANTAGE_ENABLED" >&2
    echo "Expected true or false." >&2
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

OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"

parse_choice_list DATASETS "aircraft cub200" DATASETS \
  cifar100 cub200 aircraft inat19
parse_choice_list TRANSFORM_MODES "full" TRANSFORM_MODES \
  full bn_linear final_only

config_for_dataset() {
  case "$1" in
    cifar100) echo "configs/hiercos/hiercos_cifar100.yaml" ;;
    cub200) echo "configs/hiercos/hiercos_cub200.yaml" ;;
    aircraft) echo "configs/hiercos/hiercos_aircraft.yaml" ;;
    inat19) echo "configs/hiercos/hiercos_inat19.yaml" ;;
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
  local transform_suffix=""
  local frame_suffix=""
  local advantage_suffix=""
  if [[ "$WEIGHT_MODE" != "equal" ]]; then
    weight_suffix="_${WEIGHT_MODE}"
  fi
  if [[ "$transform_mode" != "full" ]]; then
    transform_suffix="_${transform_mode}"
  fi
  if [[ "$FIXED_FRAME_MODE" == "identity" ]]; then
    frame_suffix="_identity"
  fi
  if [[ "$ADVANTAGE_ENABLED" == "true" ]]; then
    advantage_suffix="_advantage"
  fi
  echo "$OUTPUTS_ROOT/hiercos_${ds}_level_softmax_ce_reg_projection${weight_suffix}${transform_suffix}${frame_suffix}${advantage_suffix}"
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Transform modes: %s\n' "${TRANSFORM_MODES[*]}"
printf 'Loss: level_softmax_ce_reg\n'
printf 'Weight mode: %s\n' "$WEIGHT_MODE"
printf 'Fixed frame mode: %s\n' "$FIXED_FRAME_MODE"
printf 'LH-style stacked-weight projection: enabled\n'
printf 'LH-DNN advantage baselines: %s\n' "$ADVANTAGE_ENABLED"
printf 'Projection epsilon: %s\n' "$PROJECTION_EPS"
printf 'Lexicographic mode: disabled\n'
printf 'Dry run: %s\n' "$DRY_RUN"
printf 'Max parallel: %s\n' "$MAX_PARALLEL"
printf 'Max resume retries on failure: %s\n' "$MAX_RESUME_RETRIES"
print_seed_run_settings

for ds in "${DATASETS[@]}"; do
  cfg="$(config_for_dataset "$ds")"

  for transform_mode in "${TRANSFORM_MODES[@]}"; do
    run_seeded_train "$cfg" "$(run_output_dir "$ds" "$transform_mode")" \
      "model.loss=level_softmax_ce_reg" \
      "model.weight_mode=$WEIGHT_MODE" \
      "model.transform_mode=$transform_mode" \
      "model.fixed_frame_mode=$FIXED_FRAME_MODE" \
      "model.fixed_frame_per_level=false" \
      "model.projection.enabled=true" \
      "model.projection.advantage_enabled=$ADVANTAGE_ENABLED" \
      "model.projection.eps=$PROJECTION_EPS" \
      "train.lexicographic.enabled=false"
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

printf 'Completed all requested Hier-COS LH-DNN projection runs.\n'
