#!/usr/bin/env bash
set -euo pipefail

# Runs the HT-CapsNet + HCC arm (HCC generalized from H-CAST, see
# models/common/hcc.py) on top of the plain HT-CapsNet baseline configs:
# - capsnet_<dataset>_hcc
# for: cifar100, cub200, aircraft.
#
# HCC is a binary on/off switch; there is no onset/alpha/temperature ablation.
# Caveat: HT-CapsNet's `logits_per_level` are capsule-length margins
# (safe_norm, in [0, inf)), not free real-valued classifier logits, and the
# margin loss compares them against fixed thresholds (margin_m_pos/
# margin_m_neg). HCC's unconstrained linear correction can push corrected
# values outside what the margin loss was designed for; treat results here as
# an exploratory ablation, not evidence with the same strength as H-CAST/HRN.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/load_env.sh"
load_project_env "$ROOT_DIR"
source "$ROOT_DIR/scripts/run_seed_utils.sh"
source "$ROOT_DIR/scripts/run_matrix_utils.sh"
source "$ROOT_DIR/scripts/run_job_utils.sh"
init_seed_runs

init_job_control

install_job_traps

# Notebook-compatible outputs root.
# Example:
#   OUTPUTS_ROOT=/scratch/<user>/outputs ./scripts/capsnet/run_ht_capsnet_hcc.sh
OUTPUTS_ROOT="${OUTPUTS_ROOT:?Set OUTPUTS_ROOT in .env or the process environment}"

parse_choice_list DATASETS "cifar100 cub200 aircraft" DATASETS \
  cifar100 cub200 aircraft

# Start from the plain HT-CapsNet baseline config; hcc.* overrides below are
# the only thing that changes relative to run_ht_capsnet_baselines.sh.
config_for_dataset() {
  case "$1" in
    cifar100) echo "configs/capsnet/capsnet_cifar100.yaml" ;;
    cub200) echo "configs/capsnet/capsnet_cub200.yaml" ;;
    aircraft) echo "configs/capsnet/capsnet_aircraft.yaml" ;;
    *)
      echo "Unknown dataset: $1" >&2
      exit 1
      ;;
  esac
}

run_output_dir() {
  local ds="$1"
  echo "$OUTPUTS_ROOT/capsnet_${ds}_hcc"
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Lexicographic mode: disabled\n'
printf 'HCC: enabled\n'
print_job_control_settings
print_seed_run_settings

for ds in "${DATASETS[@]}"; do
  cfg="$(config_for_dataset "$ds")"

  run_seeded_train "$cfg" "$(run_output_dir "$ds")" \
    "hcc.enabled=true" \
    "hcc.eps=1e-12" \
    "train.lexicographic.enabled=false" \
    "train.gradient_blocks=[p123,p23,p3]"
done

drain_jobs

printf 'Completed all requested HT-CapsNet HCC runs.\n'
