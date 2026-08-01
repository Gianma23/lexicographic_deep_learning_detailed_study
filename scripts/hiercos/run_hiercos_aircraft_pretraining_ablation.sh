#!/usr/bin/env bash
set -euo pipefail

# FGVC-Aircraft backbone-pretraining x fixed-frame ablation.
#
# Research question: identity beats orthonormal_random on Aircraft/CUB but loses
# on CIFAR-100. The candidate explanation is that identity trades training-fit
# capacity for a smaller generalization gap, and that the trade is free only when
# the backbone is pretrained and the training set is already fit to ~100%.
# On the current runs that hypothesis is confounded, because pretraining moves
# together with dataset size, images-per-class, resolution, backbone family and
# hierarchy construction.
#
# This script isolates the single variable by toggling `model.pretrained` on one
# fixed dataset, holding dataset, taxonomy, resolution, architecture and schedule
# constant:
# - model.pretrained=${PRETRAINED_MODES}          (true = ImageNet, false = scratch)
# - model.fixed_frame_mode=${FIXED_FRAME_MODES}   (orthonormal_random vs identity)
# - model.loss=${LOSS_MODE}, model.weight_mode=${WEIGHT_MODE}
# - model.transform_mode=full
# - train.lexicographic per ${LEX_MODE}
#
# Readout: the sign and size of (identity - orthonormal_random) test FPA within
# each pretraining arm. The prediction is that the identity advantage seen with
# pretrained=true (+1.07 pp top-down FPA on the existing baseline runs) shrinks
# toward zero or reverses with pretrained=false, and that train FPA stops
# saturating at ~1.0 in the scratch arm. If the identity advantage survives
# intact while train FPA still saturates, pretraining is not the driver.
#
# Comparison baseline: the existing pretrained runs
#   $OUTPUTS_ROOT/hiercos_aircraft_global_softmax_ce_reg_baseline_kl_leaf[_identity]
# This script writes to a separate `pretrainablation` namespace and never
# resumes or overwrites those directories.

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
LOSS_MODE="${LOSS_MODE:-global_softmax_ce_reg}"
WEIGHT_MODE="${WEIGHT_MODE:-kl_leaf}"
LEX_MODE="${LEX_MODE:-off}"
# Empty means "use the value in configs/hiercos/hiercos_aircraft.yaml" (0.1).
# When set, it is applied to BOTH pretraining arms so the ablation stays
# single-variable. A from-scratch ResNet-50 trains at base_lr * this scale.
BACKBONE_LR_SCALE="${BACKBONE_LR_SCALE:-}"
# Empty means "use the config value" (100). Lower it only for pilot runs.
EPOCHS="${EPOCHS:-}"

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

case "$LEX_MODE" in
  off|coarse_first) ;;
  *)
    echo "Unsupported LEX_MODE: $LEX_MODE" >&2
    echo "Expected off or coarse_first." >&2
    exit 1
    ;;
esac

if [[ -n "$BACKBONE_LR_SCALE" && ! "$BACKBONE_LR_SCALE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "Unsupported BACKBONE_LR_SCALE: $BACKBONE_LR_SCALE" >&2
  echo "Expected a non-negative decimal, or empty to keep the config value." >&2
  exit 1
fi

if [[ -n "$EPOCHS" && ! "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Unsupported EPOCHS: $EPOCHS" >&2
  echo "Expected a positive integer, or empty to keep the config value." >&2
  exit 1
fi

parse_choice_list PRETRAINED_MODES "true false" PRETRAINED_MODES \
  true false
parse_choice_list FIXED_FRAME_MODES "orthonormal_random identity" FIXED_FRAME_MODES \
  orthonormal_random identity

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
#   OUTPUTS_ROOT=/scratch/<user>/outputs NUM_RUNS=3 \
#     ./scripts/hiercos/run_hiercos_aircraft_pretraining_ablation.sh
#   # scratch arm only, comparing against the existing pretrained baselines:
#   PRETRAINED_MODES=false NUM_RUNS=3 \
#     ./scripts/hiercos/run_hiercos_aircraft_pretraining_ablation.sh
#   # also test whether coarse_first amplifies the frame effect:
#   LEX_MODE=coarse_first NUM_RUNS=3 \
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
  local pretrained="$1"
  local frame_mode="$2"
  local backbone_tag="scratch"
  local frame_suffix=""
  local lex_suffix=""
  local blr_suffix=""
  local epochs_suffix=""
  if [[ "$pretrained" == "true" ]]; then
    backbone_tag="pretrained"
  fi
  if [[ "$frame_mode" == "identity" ]]; then
    frame_suffix="_identity"
  fi
  if [[ "$LEX_MODE" != "off" ]]; then
    lex_suffix="_lex_${LEX_MODE}"
  fi
  if [[ -n "$BACKBONE_LR_SCALE" ]]; then
    blr_suffix="_blr${BACKBONE_LR_SCALE//./p}"
  fi
  if [[ -n "$EPOCHS" ]]; then
    epochs_suffix="_ep${EPOCHS}"
  fi
  echo "$OUTPUTS_ROOT/hiercos_aircraft_${LOSS_MODE}_pretrainablation_${WEIGHT_MODE}_${backbone_tag}${frame_suffix}${lex_suffix}${blr_suffix}${epochs_suffix}"
}

lex_overrides() {
  if [[ "$LEX_MODE" == "off" ]]; then
    printf '%s\n' "train.lexicographic.enabled=false"
  else
    printf '%s\n' \
      "train.lexicographic.enabled=true" \
      "train.lexicographic.start_epoch=0" \
      "train.lexicographic.projection_mode=coarse_first" \
      "train.lexicographic.projection_rule=orthogonalize_all"
  fi
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Dataset: aircraft (%s)\n' "$CONFIG"
printf 'Pretraining modes: %s\n' "${PRETRAINED_MODES[*]}"
printf 'Fixed frame modes: %s\n' "${FIXED_FRAME_MODES[*]}"
printf 'Loss: %s\n' "$LOSS_MODE"
printf 'Weight mode: %s\n' "$WEIGHT_MODE"
printf 'Transform mode: full\n'
printf 'Lexicographic mode: %s\n' "$LEX_MODE"
if [[ -n "$BACKBONE_LR_SCALE" ]]; then
  printf 'Backbone lr scale: %s (override, applied to both arms)\n' "$BACKBONE_LR_SCALE"
else
  printf 'Backbone lr scale: config default (0.1)\n'
fi
if [[ -n "$EPOCHS" ]]; then
  printf 'Epochs: %s (override)\n' "$EPOCHS"
else
  printf 'Epochs: config default (100)\n'
fi
printf 'Dry run: %s\n' "$DRY_RUN"
printf 'Max parallel: %s\n' "$MAX_PARALLEL"
printf 'Max resume retries on failure: %s\n' "$MAX_RESUME_RETRIES"
print_seed_run_settings
printf 'Total runs: %s\n' "$(( ${#PRETRAINED_MODES[@]} * ${#FIXED_FRAME_MODES[@]} * NUM_RUNS ))"

mapfile -t LEX_ARGS < <(lex_overrides)

for pretrained in "${PRETRAINED_MODES[@]}"; do
  for frame_mode in "${FIXED_FRAME_MODES[@]}"; do
    extra_args=()
    if [[ -n "$BACKBONE_LR_SCALE" ]]; then
      extra_args+=("model.backbone_lr_scale=$BACKBONE_LR_SCALE")
    fi
    if [[ -n "$EPOCHS" ]]; then
      extra_args+=("train.epochs=$EPOCHS")
    fi

    run_seeded_train "$CONFIG" "$(run_output_dir "$pretrained" "$frame_mode")" \
      "model.loss=$LOSS_MODE" \
      "model.weight_mode=$WEIGHT_MODE" \
      "model.transform_mode=full" \
      "model.fixed_frame_mode=$frame_mode" \
      "model.pretrained=$pretrained" \
      "${LEX_ARGS[@]}" \
      ${extra_args[@]+"${extra_args[@]}"}
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

printf 'Completed all requested Hier-COS Aircraft pretraining ablation runs.\n'
