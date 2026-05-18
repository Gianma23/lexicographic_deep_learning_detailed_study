#!/usr/bin/env bash
set -euo pipefail

# Runs the full H-CAST ablation grid requested:
# - H-CAST with/without global KL
# - H-CAST + HCC (alpha start epoch 0) with/without global KL
# - H-CAST + HCC (alpha start epoch 80) with/without global KL
# - H-CAST + lexicographic (start epoch 0)
# - H-CAST + lexicographic (start epoch 80)
# for all datasets: cifar100, cub200, aircraft.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
MAX_RESUME_RETRIES="${MAX_RESUME_RETRIES:-1}"

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

# Notebook-compatible outputs root (run dir names match notebooks/hcast_analysis.ipynb).
# Example:
#   OUTPUTS_ROOT=/scratch/<user>/outputs ./scripts/run_hcast_full_grid.sh
OUTPUTS_ROOT="${OUTPUTS_ROOT:-/scratch/g.saggini1/outputs}"

# Dataset -> config mapping
DATASETS=(cifar100 cub200 aircraft)

base_config_for_dataset() {
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

hcc_config_for_dataset() {
  case "$1" in
    cifar100) echo "configs/hcast/hcast_hcc_cifar100.yaml" ;;
    cub200) echo "configs/hcast/hcast_hcc_cub200.yaml" ;;
    aircraft) echo "configs/hcast/hcast_hcc_aircraft.yaml" ;;
    *)
      echo "Unknown dataset: $1" >&2
      exit 1
      ;;
  esac
}

lex_config_for_dataset() {
  case "$1" in
    cifar100) echo "configs/hcast/hcast_lex_cifar100.yaml" ;;
    cub200) echo "configs/hcast/hcast_lex_cub200.yaml" ;;
    aircraft) echo "configs/hcast/hcast_lex_aircraft.yaml" ;;
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
  local family="$2"
  local epoch_tag="$3"
  local kl_tag="$4"
  case "$family:$epoch_tag:$kl_tag" in
    hcast:na:true) echo "$OUTPUTS_ROOT/hcast_${ds}" ;;
    hcast:na:false) echo "$OUTPUTS_ROOT/hcast_${ds}_nokl" ;;

    hcc:e0:true) echo "$OUTPUTS_ROOT/hcast_hcc_${ds}_step_0epoch" ;;
    hcc:e0:false) echo "$OUTPUTS_ROOT/hcast_hcc_${ds}_step_0epoch_nokl" ;;
    hcc:e80:true) echo "$OUTPUTS_ROOT/hcast_hcc_${ds}_step_80epoch" ;;
    hcc:e80:false) echo "$OUTPUTS_ROOT/hcast_hcc_${ds}_step_80epoch_nokl" ;;

    lex:e0:na) echo "$OUTPUTS_ROOT/hcast_lex_${ds}" ;;
    lex:e80:na) echo "$OUTPUTS_ROOT/hcast_lex_${ds}_step_80epoch" ;;
    *)
      echo "Unknown run naming tuple: $family $epoch_tag $kl_tag" >&2
      exit 1
      ;;
  esac
}

printf 'Outputs root: %s\n' "$OUTPUTS_ROOT"
printf 'Dry run: %s\n' "$DRY_RUN"
printf 'Max parallel: %s\n' "$MAX_PARALLEL"
printf 'Max resume retries on failure: %s\n' "$MAX_RESUME_RETRIES"

for ds in "${DATASETS[@]}"; do
  base_cfg="$(base_config_for_dataset "$ds")"
  hcc_cfg="$(hcc_config_for_dataset "$ds")"
  lex_cfg="$(lex_config_for_dataset "$ds")"

  # 1) H-CAST with/without KL
  for kl in false true; do
    run_train "$base_cfg" "$(run_output_dir "$ds" hcast na "$kl")" \
      "model.loss.globalkl=$kl"
  done

  # 2) H-CAST + HCC epoch 0 with/without KL
  for kl in false true; do
    run_train "$hcc_cfg" "$(run_output_dir "$ds" hcc e0 "$kl")" \
      "hcc.enabled=true" \
      "hcc.alpha_start_epoch=0" \
      "model.loss.globalkl=$kl"
  done

  # 3) H-CAST + HCC epoch 80 with/without KL
  for kl in false true; do
    run_train "$hcc_cfg" "$(run_output_dir "$ds" hcc e80 "$kl")" \
      "hcc.enabled=true" \
      "hcc.alpha_start_epoch=80" \
      "model.loss.globalkl=$kl"
  done

  # 4) H-CAST + lexicographic epoch 0
  # Lexicographic mode requires globalkl=false by design.
  run_train "$lex_cfg" "$(run_output_dir "$ds" lex e0 na)" \
    "train.lexicographic.enabled=true" \
    "train.lexicographic.start_epoch=0" \
    "model.loss.globalkl=false"

  # 5) H-CAST + lexicographic epoch 80
  # Lexicographic mode requires globalkl=false by design.
  run_train "$lex_cfg" "$(run_output_dir "$ds" lex e80 na)" \
    "train.lexicographic.enabled=true" \
    "train.lexicographic.start_epoch=80" \
    "model.loss.globalkl=false"
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

printf 'Completed all requested runs.\n'
