#!/usr/bin/env bash
set -euo pipefail

# HRN + LH-projection arm on top of the plain HRN baseline configs (cifar100,
# cub200, aircraft). `model.projection.enabled=true` also replaces HRN's output
# branches (one shared pool, a shared_linear/ReLU branch point, one Linear head
# per level, and a score-space detached parent advantage); those capacity
# changes stay confounded with the projection. The projection is backward-only,
# so this arm is a gradient-space method and is inactive at inference.
# model.loss=level_conditional telescopes to `native`, as in run_hrn_hcc.sh.
# Mutually exclusive with HCC and with train.lexicographic.
# Knobs: DATASETS, PROJECTION_EPS. Details: docs/PRESETS.md.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/load_env.sh"
load_project_env "$ROOT_DIR"
source "$ROOT_DIR/scripts/run_seed_utils.sh"
source "$ROOT_DIR/scripts/run_matrix_utils.sh"
source "$ROOT_DIR/scripts/run_job_utils.sh"
init_seed_runs

init_job_control
OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"
# Numerical regularisation of the projector's Gram solve. Positive by contract;
# it makes the projector stable but not exact, which is why the mechanism is
# read as reducing cross-level interference rather than enforcing an exact
# ordering. Matches the model default.
PROJECTION_EPS="${PROJECTION_EPS:-1.0e-6}"

if [[ ! "$PROJECTION_EPS" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
  echo "Unsupported PROJECTION_EPS: $PROJECTION_EPS" >&2
  echo "Expected a positive decimal number, optionally in exponent form." >&2
  exit 2
fi

install_job_traps

parse_choice_list DATASETS "cifar100" DATASETS \
  cifar100 cub200 aircraft

# Start from the plain HRN baseline config; the model.loss/model.projection.*
# overrides below are what changes relative to run_hrn_baselines.sh.
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

run_output_dir() {
  local ds="$1"
  echo "$OUTPUTS_ROOT/hrn_${ds}_level_conditional_projection"
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'HRN loss mode: level_conditional\n'
printf 'LH-projection: enabled (pooled branch point, shared linear + ReLU, direct level heads, detached score advantage)\n'
printf 'Projection epsilon: %s\n' "$PROJECTION_EPS"
printf 'Gradient blocks: p123, p23, p3\n'
printf 'Lexicographic mode: disabled\n'
printf 'HCC: disabled\n'
print_job_control_settings
print_seed_run_settings

for ds in "${DATASETS[@]}"; do
  cfg="$(config_for_dataset "$ds")"

  run_seeded_train "$cfg" "$(run_output_dir "$ds")" \
    "model.loss=level_conditional" \
    "model.projection.enabled=true" \
    "model.projection.eps=$PROJECTION_EPS" \
    "hcc.enabled=false" \
    "train.lexicographic.enabled=false" \
    "train.gradient_blocks=[p123,p23,p3]"
done

drain_jobs

printf 'Completed all requested HRN LH-projection runs.\n'
