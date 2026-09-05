#!/usr/bin/env bash
set -euo pipefail

# CIFAR-100 Hier-COS backbone-capacity ladder on the LH-projection arm: the same
# WideResNet trunk at reduced depth/width (WRN_SIZES), with every LH setting
# held at the corrected WRN-28-8/d512 anchor. All rungs use three seeds.
#
# Do not combine this script with a 224 px override: the Hier-COS WideResNet
# wrapper's hard-coded avg_pool2d(out, 8) assumes the 32 px native resolution.
# Rationale and anchor comparability: docs/experiment_matrix.md.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/load_env.sh"
load_project_env "$ROOT_DIR"
source "$ROOT_DIR/scripts/run_seed_utils.sh"
source "$ROOT_DIR/scripts/run_matrix_utils.sh"
source "$ROOT_DIR/scripts/run_job_utils.sh"

NUM_RUNS="${NUM_RUNS:-3}"
init_seed_runs

init_job_control
CONFIG="configs/hiercos/hiercos_cifar100.yaml"

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

install_job_traps

# Notebook-compatible outputs root.
# Examples:
#   # Planned matrix: WRN-16-8 and WRN-28-4 on LH-projection, seeds 0..1.
#   NUM_RUNS=3 OUTPUTS_ROOT=/scratch/<user>/outputs \
#     ./scripts/hiercos/run_hiercos_cifar100_backbone_ladder.sh
#   # Inspect the full plan without training.
#   DRY_RUN=1 ./scripts/hiercos/run_hiercos_cifar100_backbone_ladder.sh
#   # Single-seed pilot of one rung before committing the full ladder.
#   NUM_RUNS=1 WRN_SIZES=16-8 \
#     ./scripts/hiercos/run_hiercos_cifar100_backbone_ladder.sh
#   # Denser ladder.
#   WRN_SIZES="16-8 28-4 16-4 28-2" \
#     ./scripts/hiercos/run_hiercos_cifar100_backbone_ladder.sh
#
# Cost: the existing WRN-28-8 CIFAR-100 runs measure ~1.78 it/s at batch_size=64
# for 100 epochs. Compute scales roughly with parameter count, so WRN-16-8
# (0.47x) and WRN-28-4 (0.25x) should come in well under the anchor's wall-clock
# per seed. With NUM_RUNS=3 the plan is 2 rungs x 1 LH arm x 3 seeds = 6 runs;
# check GPU occupancy before launching, since MAX_PARALLEL=1 only serializes
# this script's own jobs and not anything already training.
OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"

# Anchor naming plus a mandatory _wrn<depth>_<widen> tag. The tag is never
# omitted: without it a shrunken run would resolve to the same directory as the
# finished WRN-28-8 seeds and resume from their latest.pt, destroying them.
run_output_dir() {
  local depth="$1"
  local widen="$2"
  echo "$OUTPUTS_ROOT/hiercos_cifar100_level_softmax_ce_reg_projection_d512_kl_leaf_block_wrn${depth}_${widen}"
}

run_lh_projection() {
  local depth="$1"
  local widen="$2"
  local run_dir
  run_dir="$(run_output_dir "$depth" "$widen")"

  local -a common_args=(
    "model.variant=haframe_wide_resnet"
    "model.loss=level_softmax_ce_reg"
    "model.weight_mode=kl_leaf"
    "model.fixed_frame_mode=orthonormal_random"
    "model.fixed_frame_per_level=true"
    "model.wide_depth=$depth"
    "model.wide_widen_factor=$widen"
    "model.projection.enabled=true"
    "model.projection.advantage_enabled=false"
    "model.projection.feature_dim=512"
    "model.projection.eps=1.0e-6"
    "train.lexicographic.enabled=false"
    "train.gradient_blocks=[p123]"
  )

  run_seeded_train "$CONFIG" "$run_dir" "${common_args[@]}"
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Dataset: cifar100\n'
printf 'Backbone: haframe_wide_resnet (from scratch, 32 px native)\n'
printf 'Backbone rungs: %s\n' "${WRN_SIZES_LIST[*]}"
printf 'Mechanism: LH-projection (advantage disabled, explicit lex disabled)\n'
printf 'Loss: level_softmax_ce_reg\n'
printf 'Weight mode: kl_leaf\n'
printf 'Fixed frame: orthonormal_random, per-level block\n'
printf 'Projection feature dimension: 512\n'
print_job_control_settings
print_seed_run_settings
printf 'Total runs: %s\n' "$(( ${#WRN_SIZES_LIST[@]} * NUM_RUNS ))"

for size in "${WRN_SIZES_LIST[@]}"; do
  [[ "$size" =~ ^([1-9][0-9]*)-([1-9][0-9]*)$ ]]
  depth="${BASH_REMATCH[1]}"
  widen="${BASH_REMATCH[2]}"

  printf '\n[PLAN] WRN-%s-%s / LH-projection -> %s\n' \
    "$depth" "$widen" "$(run_output_dir "$depth" "$widen")"
  run_lh_projection "$depth" "$widen"
done

drain_jobs

printf '\nCompleted requested Hier-COS CIFAR-100 backbone ladder runs.\n'
