#!/usr/bin/env bash
set -euo pipefail

# Runs H-CAST with the shared orthonormal taxonomy-frame plugin.
# Defaults:
# - orthonormal_plugin.loss=global_softmax_ce_reg
# - orthonormal_plugin.weight_mode=equal
# - hard targets: MixUp/CutMix disabled and smoothing set to 0
# - plugin baselines plus lexicographic orthogonalize-all variants
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
# Empty means: match the alpha in the corresponding Hier-COS dataset config.
# Set PLUGIN_ALPHA explicitly only to force one value across every dataset.
PLUGIN_ALPHA="${PLUGIN_ALPHA:-}"
PLUGIN_TRANSFORM_MODE="${PLUGIN_TRANSFORM_MODE:-full}"
PLUGIN_FIXED_FRAME_MODE="${PLUGIN_FIXED_FRAME_MODE:-orthonormal_random}"
PLUGIN_TRANSFORM_LR_SCALE="${PLUGIN_TRANSFORM_LR_SCALE:-1.0}"

RUN_BASELINES="${RUN_BASELINES:-0}"
RUN_LEX="${RUN_LEX:-1}"
LEX_START_EPOCH="${LEX_START_EPOCH:-0}"
LEX_PROJECTION_RULE="${LEX_PROJECTION_RULE:-orthogonalize_all}"

case "$LOSS_MODE" in
  kl_reg|global_softmax_ce_reg|level_softmax_ce_reg) ;;
  *)
    echo "Unsupported LOSS_MODE for this runner: $LOSS_MODE" >&2
    echo "Expected kl_reg, global_softmax_ce_reg, or level_softmax_ce_reg." >&2
    exit 1
    ;;
esac

if [[ "$LOSS_MODE" == "kl_reg" && "$RUN_LEX" == "1" ]]; then
  echo "LOSS_MODE=kl_reg does not expose per-level losses for lexicographic mode." >&2
  echo "Use RUN_LEX=0 or choose global_softmax_ce_reg/level_softmax_ce_reg." >&2
  exit 1
fi

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

case "$LEX_PROJECTION_RULE" in
  orthogonalize_all|conflict_only) ;;
  *)
    echo "Unsupported LEX_PROJECTION_RULE: $LEX_PROJECTION_RULE" >&2
    echo "Expected orthogonalize_all or conflict_only." >&2
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
LEX_PROJECTION_MODES=(coarse_first fine_first)

config_for_dataset() {
  case "$1" in
    cifar100) echo "configs/hcast/hcast_cifar100.yaml" ;;
    cub200) echo "configs/hcast/hcast_cub200.yaml" ;;
    aircraft) echo "configs/hcast/hcast_aircraft.yaml" ;;
    *)
      echo "Unknown dataset: $1" >&2
      exit 1
      ;;
  esac
}

plugin_alpha_for_dataset() {
  if [[ -n "$PLUGIN_ALPHA" ]]; then
    echo "$PLUGIN_ALPHA"
    return
  fi

  case "$1" in
    cifar100) echo "0.05" ;;
    cub200) echo "0.15" ;;
    aircraft) echo "0.1" ;;
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

plugin_overrides=(
  "orthonormal_plugin.enabled=true"
  "orthonormal_plugin.loss=$LOSS_MODE"
  "orthonormal_plugin.weight_mode=$WEIGHT_MODE"
  "orthonormal_plugin.transform_mode=$PLUGIN_TRANSFORM_MODE"
  "orthonormal_plugin.fixed_frame_mode=$PLUGIN_FIXED_FRAME_MODE"
  "orthonormal_plugin.transform_lr_scale=$PLUGIN_TRANSFORM_LR_SCALE"
  "model.loss.globalkl=false"
  "dataset.transforms.mixup=0.0"
  "dataset.transforms.cutmix=0.0"
  "dataset.transforms.cutmix_minmax=null"
  "dataset.transforms.mixup_prob=0.0"
  "dataset.transforms.mixup_switch_prob=0.0"
  "train.smoothing=0.0"
)

baseline_output_dir() {
  local ds="$1"
  echo "$OUTPUTS_ROOT/hcast_${ds}_orthonormal_plugin_${LOSS_MODE}_baseline_${WEIGHT_MODE}"
}

lex_output_dir() {
  local ds="$1"
  local projection_mode="$2"
  echo "$OUTPUTS_ROOT/hcast_${ds}_orthonormal_plugin_${LOSS_MODE}_lex_${LEX_PROJECTION_RULE}_${projection_mode}_${WEIGHT_MODE}"
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Plugin loss: %s\n' "$LOSS_MODE"
printf 'Plugin weight mode: %s\n' "$WEIGHT_MODE"
if [[ -n "$PLUGIN_ALPHA" ]]; then
  printf 'Plugin alpha override (all datasets): %s\n' "$PLUGIN_ALPHA"
else
  printf 'Plugin alpha: dataset-specific (matching Hier-COS configs)\n'
fi
printf 'Plugin transform mode: %s\n' "$PLUGIN_TRANSFORM_MODE"
printf 'Plugin fixed frame: %s\n' "$PLUGIN_FIXED_FRAME_MODE"
printf 'Run baselines: %s\n' "$RUN_BASELINES"
printf 'Run lex: %s\n' "$RUN_LEX"
printf 'Lex start epoch: %s\n' "$LEX_START_EPOCH"
printf 'Lex projection rule: %s\n' "$LEX_PROJECTION_RULE"
printf 'Dry run: %s\n' "$DRY_RUN"
printf 'Max parallel: %s\n' "$MAX_PARALLEL"
printf 'Max resume retries on failure: %s\n' "$MAX_RESUME_RETRIES"

for ds in "${DATASETS[@]}"; do
  cfg="$(config_for_dataset "$ds")"
  plugin_alpha="$(plugin_alpha_for_dataset "$ds")"
  printf 'Dataset %s plugin alpha: %s\n' "$ds" "$plugin_alpha"

  if [[ "$RUN_BASELINES" == "1" ]]; then
    run_train "$cfg" "$(baseline_output_dir "$ds")" \
      "${plugin_overrides[@]}" \
      "orthonormal_plugin.alpha=$plugin_alpha" \
      "train.lexicographic.enabled=false"
  fi

  if [[ "$RUN_LEX" == "1" ]]; then
    for projection_mode in "${LEX_PROJECTION_MODES[@]}"; do
      run_train "$cfg" "$(lex_output_dir "$ds" "$projection_mode")" \
        "${plugin_overrides[@]}" \
        "orthonormal_plugin.alpha=$plugin_alpha" \
        "train.lexicographic.enabled=true" \
        "train.lexicographic.start_epoch=$LEX_START_EPOCH" \
        "train.lexicographic.projection_mode=$projection_mode" \
        "train.lexicographic.projection_rule=$LEX_PROJECTION_RULE"
    done
  fi
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

printf 'Completed all requested H-CAST orthonormal-plugin runs.\n'
