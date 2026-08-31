#!/usr/bin/env bash
set -euo pipefail

# Runs Hier-COS with an LH-style projection after the complete transform:
# - the original PReLU activations are retained
# - the residual transform retains both skips
# - a shared terminal PReLU is always inserted and the projection uses the
#   LH-DNN form A[b]=[W_1; ...; W_(l-1)] * rho'(k[b])
# - each projected level uses a learnable FC head
# - the three FC outputs remain independent under an identity or per-level
#   block-random fixed frame
# - LH-DNN advantage baselines are selected by ADVANTAGE_ENABLED
# - model.loss=level_softmax_ce_reg
# - model.projection.enabled=true
# - model.weight_mode=${WEIGHT_MODE}
# - model.weight_beta=${WEIGHT_BETA} when cumulative_branching is selected
# - model.projection.feature_dim=${FEATURE_DIM}
# - model.projection.eps=${PROJECTION_EPS}
# - model.fixed_frame_mode=${FIXED_FRAME_MODE}
# - train.lexicographic.enabled=false
#
# Examples:
#   ./scripts/hiercos/run_hiercos_lhdnn_projection.sh
#   DATASETS="cub200 aircraft" NUM_RUNS=3 ./scripts/hiercos/run_hiercos_lhdnn_projection.sh
#   ADVANTAGE_ENABLED=true DATASETS=cifar100 ./scripts/hiercos/run_hiercos_lhdnn_projection.sh
#   FIXED_FRAME_MODE=identity DATASETS=cub200 ./scripts/hiercos/run_hiercos_lhdnn_projection.sh
#   FIXED_FRAME_MODE=orthonormal_random FIXED_FRAME_PER_LEVEL=true DATASETS=cub200 \
#     ./scripts/hiercos/run_hiercos_lhdnn_projection.sh
#   WEIGHT_MODE=cumulative_branching WEIGHT_BETA=0.5 DATASETS="cub200 aircraft" \
#     ./scripts/hiercos/run_hiercos_lhdnn_projection.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/load_env.sh"
load_project_env "$ROOT_DIR"
source "$ROOT_DIR/scripts/run_seed_utils.sh"
source "$ROOT_DIR/scripts/run_matrix_utils.sh"
source "$ROOT_DIR/scripts/run_job_utils.sh"
init_seed_runs

RUN_PREFLIGHT=none
RUN_RETRY_REQUIRES_CHECKPOINT=0
init_job_control
WEIGHT_MODE="${WEIGHT_MODE:-kl_leaf}"
WEIGHT_BETA="${WEIGHT_BETA:-0.5}"
FIXED_FRAME_MODE="${FIXED_FRAME_MODE:-orthonormal_random}"
FIXED_FRAME_PER_LEVEL="${FIXED_FRAME_PER_LEVEL:-true}"
FEATURE_DIM="${FEATURE_DIM:-512}"
PROJECTION_EPS="${PROJECTION_EPS:-1.0e-6}"
ADVANTAGE_ENABLED="${ADVANTAGE_ENABLED:-false}"

case "$WEIGHT_MODE" in
  equal|kl_leaf|kl_coarse|cumulative_branching|marginal_branching) ;;
  *)
    echo "Unsupported WEIGHT_MODE: $WEIGHT_MODE" >&2
    echo "Expected equal, kl_leaf, kl_coarse, cumulative_branching, or marginal_branching." >&2
    exit 1
    ;;
esac

if [[ "$WEIGHT_MODE" == "cumulative_branching" && ! "$WEIGHT_BETA" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "Unsupported WEIGHT_BETA: $WEIGHT_BETA" >&2
  echo "Expected a non-negative decimal number for cumulative_branching." >&2
  exit 1
fi

case "$FIXED_FRAME_MODE" in
  orthonormal_random|identity) ;;
  *)
    echo "Unsupported FIXED_FRAME_MODE: $FIXED_FRAME_MODE" >&2
    echo "Expected orthonormal_random or identity." >&2
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

if [[ "$FIXED_FRAME_MODE" == "identity" && "$FIXED_FRAME_PER_LEVEL_OVERRIDE" == "true" ]]; then
  echo "FIXED_FRAME_PER_LEVEL=true is redundant with FIXED_FRAME_MODE=identity." >&2
  echo "Use FIXED_FRAME_MODE=identity FIXED_FRAME_PER_LEVEL=false." >&2
  exit 1
fi

if [[ "$FIXED_FRAME_MODE" == "orthonormal_random" && "$FIXED_FRAME_PER_LEVEL_OVERRIDE" != "true" ]]; then
  echo "FIXED_FRAME_MODE=orthonormal_random requires FIXED_FRAME_PER_LEVEL=true in this independent-head LH-DNN runner." >&2
  exit 1
fi

if [[ ! "$FEATURE_DIM" =~ ^[0-9]+$ ]]; then
  echo "Unsupported FEATURE_DIM: $FEATURE_DIM" >&2
  echo "Expected a non-negative integer; use 0 for the dataset taxonomy width." >&2
  exit 1
fi

case "$ADVANTAGE_ENABLED" in
  true|false) ;;
  *)
    echo "Unsupported ADVANTAGE_ENABLED: $ADVANTAGE_ENABLED" >&2
    echo "Expected true or false." >&2
    exit 1
    ;;
esac

if [[ -n "${PROJECTION_RHO_ENABLED:-}" ]]; then
  echo "PROJECTION_RHO_ENABLED is no longer supported." >&2
  echo "The shared terminal PReLU and its rho' derivative are always applied." >&2
  exit 1
fi

install_job_traps

OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"

parse_choice_list DATASETS "cifar100" DATASETS \
  cifar100 cub200 aircraft

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

run_output_dir() {
  local ds="$1"
  local weight_suffix=""
  local frame_suffix=""
  local advantage_suffix=""
  local dimension_suffix=""
  if [[ "$FEATURE_DIM" != "0" ]]; then
    dimension_suffix="_d${FEATURE_DIM}"
  fi
  case "$WEIGHT_MODE" in
    equal) ;;
    marginal_branching) weight_suffix="_mb" ;;
    cumulative_branching)
      local beta_tag="${WEIGHT_BETA//./p}"
      weight_suffix="_cb${beta_tag}"
      ;;
    *) weight_suffix="_${WEIGHT_MODE}" ;;
  esac
  if [[ "$FIXED_FRAME_PER_LEVEL_OVERRIDE" == "true" ]]; then
    frame_suffix="_block"
  elif [[ "$FIXED_FRAME_MODE" == "identity" ]]; then
    frame_suffix="_identity"
  fi
  if [[ "$ADVANTAGE_ENABLED" == "true" ]]; then
    advantage_suffix="_advantage"
  fi
  echo "$OUTPUTS_ROOT/hiercos_${ds}_level_softmax_ce_reg_projection${dimension_suffix}${weight_suffix}${frame_suffix}${advantage_suffix}"
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Loss: level_softmax_ce_reg\n'
printf 'Weight mode: %s\n' "$WEIGHT_MODE"
if [[ "$WEIGHT_MODE" == "cumulative_branching" ]]; then
  printf 'Cumulative branching beta: %s\n' "$WEIGHT_BETA"
fi
printf 'Fixed frame mode: %s\n' "$FIXED_FRAME_MODE"
printf 'Fixed frame per level: %s\n' "$FIXED_FRAME_PER_LEVEL_OVERRIDE"
printf 'LH-style stacked-weight projection: enabled\n'
printf 'Projected transform: original PReLU activations and residual skips\n'
printf 'Shared terminal PReLU/rho derivative: always applied\n'
printf 'Hier-COS recursive post-abs advantage: %s\n' "$ADVANTAGE_ENABLED"
if [[ "$FEATURE_DIM" == "0" ]]; then
  printf 'Projection feature dimension: auto (sum of classes across levels)\n'
else
  printf 'Projection feature dimension: %s\n' "$FEATURE_DIM"
fi
printf 'Projection epsilon: %s\n' "$PROJECTION_EPS"
printf 'Lexicographic mode: disabled\n'
print_job_control_settings
print_seed_run_settings

weight_beta_args=()
if [[ "$WEIGHT_MODE" == "cumulative_branching" ]]; then
  weight_beta_args=("model.weight_beta=$WEIGHT_BETA")
fi

for ds in "${DATASETS[@]}"; do
  cfg="$(config_for_dataset "$ds")"

  run_seeded_train "$cfg" "$(run_output_dir "$ds")" \
    "model.loss=level_softmax_ce_reg" \
    "model.weight_mode=$WEIGHT_MODE" \
    "${weight_beta_args[@]}" \
    "model.fixed_frame_mode=$FIXED_FRAME_MODE" \
    "model.fixed_frame_per_level=$FIXED_FRAME_PER_LEVEL_OVERRIDE" \
    "model.projection.enabled=true" \
    "model.projection.advantage_enabled=$ADVANTAGE_ENABLED" \
    "model.projection.feature_dim=$FEATURE_DIM" \
    "model.projection.eps=$PROJECTION_EPS" \
    "train.lexicographic.enabled=false" \
    "train.gradient_blocks=[p123]"
done

drain_jobs

printf 'Completed all requested Hier-COS LH-DNN projection runs.\n'
