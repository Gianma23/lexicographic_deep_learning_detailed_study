#!/usr/bin/env bash
set -euo pipefail

# Runs the CIFAR-100 Hier-COS backbone-capacity ladder: the same WideResNet
# trunk at reduced depth/width, holding everything else at the settings of the
# existing WRN-28-8 anchor runs.
# - model.variant stays haframe_wide_resnet (the CIFAR-native from-scratch trunk)
# - model.wide_depth / model.wide_widen_factor selected by WRN_SIZES
# - model.loss=${LOSS_MODE}, model.weight_mode=${WEIGHT_MODE}
# - model.fixed_frame_mode=${FIXED_FRAME_MODE}, per-level=${FIXED_FRAME_PER_LEVEL}
# - one arm per entry in MECHANISMS (baseline / lex_coarse_first / lex_fine_first)
# Three training seeds by default, matching the anchors' seed coverage.
#
# Purpose: RQ8 in docs/experiment_matrix.md asks whether the frame/lex/weight
# conclusions are conditional on the feature extractor. The haframe_resnet50 arm
# answers that badly, because it changes architecture, ImageNet pretraining, and
# input resolution at once. Shrinking the WideResNet changes capacity alone:
# the trunk implementation, the 32 px native resolution, the head, the frame,
# and the loss are all untouched.
#
# Why depth and width are separate knobs (models/common/cifar_wide_resnet.py):
# - wide_depth d sets blocks per stage as n = (d - 4) / 6, hence the
#   (d - 4) % 6 == 0 constraint. Parameters scale roughly linearly in d.
# - wide_widen_factor k sets stage channels [16, 16k, 32k, 64k]. Parameters
#   scale quadratically in k.
# WRN-16-8 (10.96M) and WRN-28-4 (5.85M) therefore sit in a similar size band
# but get there by different mechanisms (fewer layers vs narrower layers), which
# is the point of running both against the 23.35M WRN-28-8 anchor.
#
# Geometry note: WRN strides are fixed at 1, 2, 2 regardless of depth and width,
# so a 32 px input always ends on an 8 x 8 map. The hard-coded avg_pool2d(out, 8)
# in the Hier-COS WideResNet wrapper stays correct at every rung and the node
# embedding dimension never changes. Do NOT combine this script with the 224 px
# override from run_hiercos_cifar100_resnet50_pretrained.sh: at 224 px the same
# pool yields a 7 x 7 map and the embedding becomes out_dim * 49.
#
# Comparability with the existing WRN-28-8 anchors, verified before writing this:
# - The anchors ran with model.fixed_frame_per_level=false and
#   model.projection.feature_dim=0; both are reproduced here.
# - The lex anchors (2026-08-05) recorded projection_rule=orthogonalize_all,
#   which no longer exists as a knob. The two commits that removed it (b8f4c41,
#   3a81048) deleted only the unused conflict_only rule and the
#   pairwise_orthogonal mode; the coarse_first / orthogonalize_all computation
#   was not modified.
# - The anchors predate train.gradient_blocks and so used the p123/p12/p1
#   default, while this script passes [p123] like the current launchers. For
#   Hier-COS that is inert: every trunk parameter receives gradient from all
#   three levels, so p12/p1 are empty. The anchor logs confirm it
#   (cos_t2_mid_coarse = 0.0, and the t2t1/t3t2t1 norms equal the t1 norms).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/load_env.sh"
load_project_env "$ROOT_DIR"
source "$ROOT_DIR/scripts/run_seed_utils.sh"
source "$ROOT_DIR/scripts/run_matrix_utils.sh"

NUM_RUNS="${NUM_RUNS:-3}"
init_seed_runs

PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
MAX_RESUME_RETRIES="${MAX_RESUME_RETRIES:-1}"
LOSS_MODE="${LOSS_MODE:-global_softmax_ce_reg}"
WEIGHT_MODE="${WEIGHT_MODE:-kl_leaf}"
FIXED_FRAME_MODE="${FIXED_FRAME_MODE:-orthonormal_random}"
FIXED_FRAME_PER_LEVEL="${FIXED_FRAME_PER_LEVEL:-false}"

CONFIG="configs/hiercos/hiercos_cifar100.yaml"

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
  orthonormal_random|identity) ;;
  *)
    echo "Unsupported FIXED_FRAME_MODE: $FIXED_FRAME_MODE" >&2
    echo "Expected orthonormal_random or identity." >&2
    exit 1
    ;;
esac

normalize_bool_like "$FIXED_FRAME_PER_LEVEL" FIXED_FRAME_PER_LEVEL

parse_choice_list MECHANISMS "baseline lex_coarse_first" MECHANISMS \
  baseline lex_coarse_first lex_fine_first

# Backbone rungs as "<depth>-<widen_factor>". Both numbers are validated here so
# a typo fails before any GPU time is spent; train/config_validation.py enforces
# the same depth constraint.
WRN_SIZES_RAW="${WRN_SIZES:-16-8 28-4}"
read -r -a WRN_SIZES_LIST <<< "$WRN_SIZES_RAW"
if (( ${#WRN_SIZES_LIST[@]} == 0 )); then
  echo "WRN_SIZES must contain at least one <depth>-<widen_factor> entry." >&2
  exit 1
fi

for size in "${WRN_SIZES_LIST[@]}"; do
  if [[ ! "$size" =~ ^([1-9][0-9]*)-([1-9][0-9]*)$ ]]; then
    echo "Unsupported WRN_SIZES entry '$size'. Expected <depth>-<widen_factor>, e.g. 16-8." >&2
    exit 1
  fi
  depth="${BASH_REMATCH[1]}"
  widen="${BASH_REMATCH[2]}"
  if (( (depth - 4) % 6 != 0 )); then
    echo "Unsupported WideResNet depth '$depth' in WRN_SIZES entry '$size'." >&2
    echo "Depth must satisfy (depth - 4) % 6 == 0, e.g. 16, 22, 28, 40." >&2
    exit 1
  fi
  if (( depth == 28 && widen == 8 )); then
    echo "[WARN] WRN_SIZES contains 28-8, the anchor configuration." >&2
    echo "[WARN] The finished anchor seeds live in the untagged run directories;" >&2
    echo "[WARN] this script would write separate _wrn28_8 directories instead." >&2
  fi
done

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
#   # Defaults: WRN-16-8 and WRN-28-4, baseline vs lex coarse-first, seeds 0..2.
#   OUTPUTS_ROOT=/scratch/<user>/outputs \
#     ./scripts/hiercos/run_hiercos_cifar100_backbone_ladder.sh
#   # Inspect the full plan without training.
#   DRY_RUN=1 ./scripts/hiercos/run_hiercos_cifar100_backbone_ladder.sh
#   # Single-seed pilot of one rung before committing the full ladder.
#   NUM_RUNS=1 WRN_SIZES=16-8 \
#     ./scripts/hiercos/run_hiercos_cifar100_backbone_ladder.sh
#   # Capacity curve only, no mechanism arm.
#   MECHANISMS=baseline ./scripts/hiercos/run_hiercos_cifar100_backbone_ladder.sh
#   # Denser ladder.
#   WRN_SIZES="16-8 28-4 16-4 28-2" \
#     ./scripts/hiercos/run_hiercos_cifar100_backbone_ladder.sh
#
# Cost: the existing WRN-28-8 CIFAR-100 runs measure ~1.78 it/s at batch_size=64
# for 100 epochs. Compute scales roughly with parameter count, so WRN-16-8
# (0.47x) and WRN-28-4 (0.25x) should come in well under the anchor's wall-clock
# per seed. The default plan is 2 rungs x 2 arms x 3 seeds = 12 runs; check GPU
# occupancy before launching, since MAX_PARALLEL=1 only serializes this script's
# own jobs and not anything already training.
OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"

run_train() {
  local config="$1"
  local run_dir="$2"
  shift 2

  local base_cmd=(
    "$PYTHON_BIN" -m train.train
    --config "$config"
    "train.output_dir=$run_dir"
    "$@"
  )
  local cmd=("${base_cmd[@]}")
  if [[ -f "$run_dir/latest.pt" ]]; then
    cmd+=("train.resume=$run_dir/latest.pt")
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY-RUN] '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if (( MAX_RESUME_RETRIES > 0 )); then
      printf '[DRY-RUN][RETRY x%s] ' "$MAX_RESUME_RETRIES"
      printf '%q ' "${base_cmd[@]}" "train.resume=$run_dir/latest.pt"
      printf '\n'
    fi
  else
    while (( "$(jobs -pr | wc -l)" >= MAX_PARALLEL )); do
      set +e
      wait -n
      rc=$?
      set -e
      if (( rc != 0 )); then
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
        "${base_cmd[@]}" "train.resume=$run_dir/latest.pt"
        rc=$?
      done
      exit "$rc"
    ) &
  fi
}

# Anchor naming plus a mandatory _wrn<depth>_<widen> tag. The tag is never
# omitted: without it a shrunken run would resolve to the same directory as the
# finished WRN-28-8 seeds and resume from their latest.pt, destroying them.
run_output_dir() {
  local mechanism="$1"
  local depth="$2"
  local widen="$3"
  local arm_tag=""
  local frame_suffix=""

  case "$mechanism" in
    baseline) arm_tag="baseline_${WEIGHT_MODE}" ;;
    lex_coarse_first) arm_tag="lex_coarse_first_${WEIGHT_MODE}" ;;
    lex_fine_first) arm_tag="lex_fine_first_${WEIGHT_MODE}" ;;
    *)
      echo "Unknown mechanism: $mechanism" >&2
      exit 1
      ;;
  esac

  if [[ "$FIXED_FRAME_MODE" == "identity" ]]; then
    frame_suffix="_identity"
  elif [[ "$FIXED_FRAME_PER_LEVEL" == "true" ]]; then
    frame_suffix="_block"
  fi

  echo "$OUTPUTS_ROOT/hiercos_cifar100_${LOSS_MODE}_${arm_tag}${frame_suffix}_wrn${depth}_${widen}"
}

run_mechanism() {
  local mechanism="$1"
  local depth="$2"
  local widen="$3"
  local run_dir
  run_dir="$(run_output_dir "$mechanism" "$depth" "$widen")"

  local -a common_args=(
    "model.variant=haframe_wide_resnet"
    "model.loss=$LOSS_MODE"
    "model.weight_mode=$WEIGHT_MODE"
    "model.fixed_frame_mode=$FIXED_FRAME_MODE"
    "model.fixed_frame_per_level=$FIXED_FRAME_PER_LEVEL"
    "model.wide_depth=$depth"
    "model.wide_widen_factor=$widen"
    "train.gradient_blocks=[p123]"
  )

  case "$mechanism" in
    baseline)
      run_seeded_train "$CONFIG" "$run_dir" \
        "${common_args[@]}" \
        "model.projection.feature_dim=0" \
        "train.lexicographic.enabled=false"
      ;;
    lex_coarse_first|lex_fine_first)
      run_seeded_train "$CONFIG" "$run_dir" \
        "${common_args[@]}" \
        "train.lexicographic.enabled=true" \
        "train.lexicographic.projection_mode=${mechanism#lex_}"
      ;;
  esac
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Dataset: cifar100\n'
printf 'Backbone: haframe_wide_resnet (from scratch, 32 px native)\n'
printf 'Backbone rungs: %s\n' "${WRN_SIZES_LIST[*]}"
printf 'Mechanisms: %s\n' "${MECHANISMS[*]}"
printf 'Loss: %s\n' "$LOSS_MODE"
printf 'Weight mode: %s\n' "$WEIGHT_MODE"
printf 'Fixed frame mode: %s\n' "$FIXED_FRAME_MODE"
printf 'Fixed frame per level: %s\n' "$FIXED_FRAME_PER_LEVEL"
printf 'Dry run: %s\n' "$DRY_RUN"
printf 'Max parallel: %s\n' "$MAX_PARALLEL"
printf 'Max resume retries on failure: %s\n' "$MAX_RESUME_RETRIES"
print_seed_run_settings
printf 'Total runs: %s\n' \
  "$(( ${#WRN_SIZES_LIST[@]} * ${#MECHANISMS[@]} * NUM_RUNS ))"

for size in "${WRN_SIZES_LIST[@]}"; do
  [[ "$size" =~ ^([1-9][0-9]*)-([1-9][0-9]*)$ ]]
  depth="${BASH_REMATCH[1]}"
  widen="${BASH_REMATCH[2]}"

  for mechanism in "${MECHANISMS[@]}"; do
    printf '\n[PLAN] WRN-%s-%s / %s -> %s\n' \
      "$depth" "$widen" "$mechanism" "$(run_output_dir "$mechanism" "$depth" "$widen")"
    run_mechanism "$mechanism" "$depth" "$widen"
  done
done

if [[ "$DRY_RUN" != "1" ]]; then
  while (( "$(jobs -pr | wc -l)" > 0 )); do
    set +e
    wait -n
    rc=$?
    set -e
    if (( rc != 0 )); then
      echo "[ERROR] One run failed (exit=${rc}); stopping remaining jobs." >&2
      jobs -pr | xargs -r kill 2>/dev/null || true
      exit "$rc"
    fi
  done
fi

printf '\nCompleted requested Hier-COS CIFAR-100 backbone ladder runs.\n'
