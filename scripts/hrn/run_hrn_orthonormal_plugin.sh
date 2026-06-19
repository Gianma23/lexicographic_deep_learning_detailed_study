#!/usr/bin/env bash
set -euo pipefail

# Runs HRN with the shared orthonormal taxonomy-frame plugin.
# Defaults:
# - orthonormal_plugin.loss=global_softmax_ce_reg
# - orthonormal_plugin.weight_mode=equal
# - hard targets: MixUp/CutMix disabled and smoothing set to 0
# - output dirs: hrn_<dataset>_orthonormal_plugin_<loss>_baseline_<weight>
# for: cifar100, cub200, aircraft.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/load_env.sh"
load_project_env "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
MAX_RESUME_RETRIES="${MAX_RESUME_RETRIES:-1}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"

LOSS_MODE="${LOSS_MODE:-global_softmax_ce_reg}"
WEIGHT_MODE="${WEIGHT_MODE:-equal}"
PLUGIN_ALPHA="${PLUGIN_ALPHA:-0.05}"
PLUGIN_TRANSFORM_MODE="${PLUGIN_TRANSFORM_MODE:-full}"
PLUGIN_FIXED_FRAME_MODE="${PLUGIN_FIXED_FRAME_MODE:-orthonormal_random}"
PLUGIN_TRANSFORM_LR_SCALE="${PLUGIN_TRANSFORM_LR_SCALE:-1.0}"

case "$LOSS_MODE" in
  kl_reg|global_softmax_ce_reg|level_softmax_ce_reg) ;;
  *)
    echo "Unsupported LOSS_MODE: $LOSS_MODE" >&2
    echo "Expected kl_reg, global_softmax_ce_reg, or level_softmax_ce_reg." >&2
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

case "$PLUGIN_TRANSFORM_MODE" in
  full|bn_linear|final_only) ;;
  *)
    echo "Unsupported PLUGIN_TRANSFORM_MODE: $PLUGIN_TRANSFORM_MODE" >&2
    echo "Expected full, bn_linear, or final_only." >&2
    exit 1
    ;;
esac

case "$PLUGIN_FIXED_FRAME_MODE" in
  orthonormal_random|identity) ;;
  *)
    echo "Unsupported PLUGIN_FIXED_FRAME_MODE: $PLUGIN_FIXED_FRAME_MODE" >&2
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

DATASETS=(cifar100 cub200 aircraft)

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

hard_target_overrides=(
  "dataset.transforms.mixup=0.0"
  "dataset.transforms.cutmix=0.0"
  "dataset.transforms.cutmix_minmax=null"
  "dataset.transforms.mixup_prob=0.0"
  "dataset.transforms.mixup_switch_prob=0.0"
  "train.smoothing=0.0"
)

plugin_overrides=(
  "orthonormal_plugin.enabled=true"
  "orthonormal_plugin.loss=$LOSS_MODE"
  "orthonormal_plugin.weight_mode=$WEIGHT_MODE"
  "orthonormal_plugin.alpha=$PLUGIN_ALPHA"
  "orthonormal_plugin.transform_mode=$PLUGIN_TRANSFORM_MODE"
  "orthonormal_plugin.fixed_frame_mode=$PLUGIN_FIXED_FRAME_MODE"
  "orthonormal_plugin.transform_lr_scale=$PLUGIN_TRANSFORM_LR_SCALE"
)

run_output_dir() {
  local ds="$1"
  echo "$OUTPUTS_ROOT/hrn_${ds}_orthonormal_plugin_${LOSS_MODE}_baseline_${WEIGHT_MODE}"
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Plugin loss: %s\n' "$LOSS_MODE"
printf 'Plugin weight mode: %s\n' "$WEIGHT_MODE"
printf 'Plugin alpha: %s\n' "$PLUGIN_ALPHA"
printf 'Plugin transform mode: %s\n' "$PLUGIN_TRANSFORM_MODE"
printf 'Plugin fixed frame: %s\n' "$PLUGIN_FIXED_FRAME_MODE"
printf 'Dry run: %s\n' "$DRY_RUN"
printf 'Max parallel: %s\n' "$MAX_PARALLEL"
printf 'Max resume retries on failure: %s\n' "$MAX_RESUME_RETRIES"

for ds in "${DATASETS[@]}"; do
  cfg="$(config_for_dataset "$ds")"
  run_train "$cfg" "$(run_output_dir "$ds")" \
    "${hard_target_overrides[@]}" \
    "${plugin_overrides[@]}" \
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

printf 'Completed all requested HRN orthonormal-plugin runs.\n'
