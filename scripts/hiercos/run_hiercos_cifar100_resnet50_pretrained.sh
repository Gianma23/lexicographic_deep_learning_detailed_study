#!/usr/bin/env bash
set -euo pipefail

# Runs one CIFAR-100 Hier-COS configuration on a ResNet-50 instead of the
# from-scratch WideResNet-28-8 used by the CIFAR-100 config:
# - model.loss=${LOSS_MODE}
# - model.weight_mode=${WEIGHT_MODE}
# - model.variant=haframe_resnet50
# - model.pretrained=${PRETRAINED_MODE}
# - model.pool=average
# - model.fixed_frame_mode=${FIXED_FRAME_MODE}
# - dataset.image_size=${IMAGE_SIZE}
# - train.lexicographic.enabled=false
# Three training seeds are used by default. Override the scalar mode variables
# to select a different single configuration; this script does not run a grid.
#
# Purpose: CIFAR-100 is the only dataset whose backbone is trained from scratch
# (`model.pretrained` is ignored for haframe_wide_resnet, which never loads
# weights), so pretraining cannot be varied there without swapping the
# architecture. Holding the architecture at haframe_resnet50 and toggling
# PRETRAINED_MODE makes pretraining a single controlled variable on CIFAR-100.
# Run the four combinations of PRETRAINED_MODE x FIXED_FRAME_MODE to get the
# pretraining-by-frame matrix that the FGVC-Aircraft attempt could not deliver
# (a from-scratch ResNet-50 on Aircraft's 3,334 training images collapsed to
# chance, ~1% train FPA; CIFAR-100 has 45,000, so its scratch arm should learn).
#
# Note for PRETRAINED_MODE=false: configs/hiercos/hiercos_cifar100.yaml sets
# model.backbone_lr_scale=0.1, so a from-scratch backbone trains at 0.1 * 0.1 =
# 0.01. That is inherited unchanged, which keeps the pretrained and scratch arms
# differing in exactly one setting. If the scratch arm underfits, raise it with
# an explicit `model.backbone_lr_scale=` override and re-run BOTH arms, so the
# comparison stays single-variable.
#
# Two settings are load-bearing and must not be dropped:
# - model.pool=average. The factory defaults to pool='max', which builds
#   MaxPool2d(kernel_size=7, stride=7); ResNet-50 downsamples 32x, so a 32x32
#   input yields a 1x1 feature map and the run dies on the first forward pass
#   with "Calculated output size: (2048x0x0)". This is what killed the earlier
#   hand-launched hiercos_cifar100_..._identity_resnet50 attempt.
# - IMAGE_SIZE. At 32 the ResNet-50 stem collapses the image immediately and the
#   ImageNet features are largely wasted, which defeats the point of the run.
#   The default upsamples to 224 (matching the CUB-200/FGVC-Aircraft configs),
#   at a real cost in wall-clock; see the note below.

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
PRETRAINED_MODE="${PRETRAINED_MODE:-false}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"

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

case "$PRETRAINED_MODE" in
  true|false) ;;
  *)
    echo "Unsupported PRETRAINED_MODE: $PRETRAINED_MODE" >&2
    echo "Expected true or false." >&2
    exit 1
    ;;
esac

if [[ ! "$IMAGE_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "Unsupported IMAGE_SIZE: $IMAGE_SIZE" >&2
  echo "Expected a positive integer (224 recommended, 32 reproduces the raw CIFAR resolution)." >&2
  exit 1
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

# Notebook-compatible outputs root.
# Examples:
#   # Defaults: pretrained ResNet-50, identity frame, 224px, seeds 0..2.
#   OUTPUTS_ROOT=/scratch/<user>/outputs \
#     ./scripts/hiercos/run_hiercos_cifar100_resnet50_pretrained.sh
#   # Matching orthonormal arm, needed for any frame comparison.
#   FIXED_FRAME_MODE=orthonormal_random \
#     ./scripts/hiercos/run_hiercos_cifar100_resnet50_pretrained.sh
#   # From-scratch ResNet-50, same architecture, for the pretraining contrast.
#   PRETRAINED_MODE=false FIXED_FRAME_MODE=identity \
#     ./scripts/hiercos/run_hiercos_cifar100_resnet50_pretrained.sh
#   # Cheap single-seed pilot at reduced resolution.
#   NUM_RUNS=1 IMAGE_SIZE=112 \
#     ./scripts/hiercos/run_hiercos_cifar100_resnet50_pretrained.sh
#
# Cost: at IMAGE_SIZE=224 this pushes 45,000 train + 5,000 val + 10,000 test
# images through a ResNet-50 each epoch. Measured on this machine at
# batch_size=64: 1.76 it/s over 703 train batches, so ~7 min per epoch and
# ~12 h per seed for the default 100 epochs. That is roughly the same wall-clock
# as the existing from-scratch WideResNet CIFAR-100 runs, which measure 1.78
# it/s. Both frames at three seeds is therefore ~70 h; a single-seed pilot of
# both arms is ~24 h. A frame comparison needs both arms.
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

# Deliberately a fresh namespace: it does not touch the dead hand-launched
# hiercos_cifar100_global_softmax_ce_reg_baseline_kl_leaf_identity_resnet50
# directory, which holds a failed resume record and no checkpoints.
# PRETRAINED_MODE=true keeps the historical `resnet50pretrained` tag so the runs
# already on disk stay addressable and resumable; the scratch arm gets its own
# `resnet50scratch` tag and can never collide with them.
run_output_dir() {
  local backbone_tag="resnet50pretrained"
  local frame_suffix=""
  local image_suffix=""
  if [[ "$PRETRAINED_MODE" == "false" ]]; then
    backbone_tag="resnet50scratch"
  fi
  if [[ "$FIXED_FRAME_MODE" == "identity" ]]; then
    frame_suffix="_identity"
  fi
  if [[ "$IMAGE_SIZE" != "224" ]]; then
    image_suffix="_img${IMAGE_SIZE}"
  fi
  echo "$OUTPUTS_ROOT/hiercos_cifar100_${LOSS_MODE}_${backbone_tag}_${WEIGHT_MODE}${frame_suffix}${image_suffix}"
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Dataset: cifar100\n'
printf 'Loss: %s\n' "$LOSS_MODE"
printf 'Weight mode: %s\n' "$WEIGHT_MODE"
if [[ "$PRETRAINED_MODE" == "true" ]]; then
  printf 'Backbone: haframe_resnet50 (ImageNet pretrained, average pool)\n'
else
  printf 'Backbone: haframe_resnet50 (random init / from scratch, average pool)\n'
fi
printf 'Pretrained: %s\n' "$PRETRAINED_MODE"
printf 'Fixed frame mode: %s\n' "$FIXED_FRAME_MODE"
printf 'Image size: %s\n' "$IMAGE_SIZE"
printf 'Lexicographic mode: disabled\n'
printf 'Dry run: %s\n' "$DRY_RUN"
printf 'Max parallel: %s\n' "$MAX_PARALLEL"
printf 'Max resume retries on failure: %s\n' "$MAX_RESUME_RETRIES"
print_seed_run_settings

run_seeded_train "$CONFIG" "$(run_output_dir)" \
  "model.loss=$LOSS_MODE" \
  "model.weight_mode=$WEIGHT_MODE" \
  "model.variant=haframe_resnet50" \
  "model.pretrained=$PRETRAINED_MODE" \
  "model.pool=average" \
  "model.fixed_frame_mode=$FIXED_FRAME_MODE" \
  "dataset.image_size=$IMAGE_SIZE" \
  "dataset.transforms.manual.resize_before_crop=true" \
  "dataset.transforms.manual.resize_before_crop_size=$IMAGE_SIZE" \
  "dataset.transforms.eval.resize_mode=resize" \
  "dataset.transforms.eval.resize_size=$IMAGE_SIZE" \
  "train.lexicographic.enabled=false" \
  "train.gradient_blocks=[p123]"

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

printf 'Completed requested Hier-COS CIFAR-100 pretrained ResNet-50 runs.\n'
