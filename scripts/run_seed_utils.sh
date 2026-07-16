#!/usr/bin/env bash

init_seed_runs() {
  NUM_RUNS="${NUM_RUNS:-1}"
  BASE_SEED="${BASE_SEED:-0}"
  SPLIT_SEED="${SPLIT_SEED:-0}"

  if [[ ! "$NUM_RUNS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'NUM_RUNS must be a positive integer, got: %s\n' "$NUM_RUNS" >&2
    return 1
  fi
  if [[ ! "$BASE_SEED" =~ ^[0-9]+$ ]]; then
    printf 'BASE_SEED must be a non-negative integer, got: %s\n' "$BASE_SEED" >&2
    return 1
  fi
  if [[ ! "$SPLIT_SEED" =~ ^[0-9]+$ ]]; then
    printf 'SPLIT_SEED must be a non-negative integer, got: %s\n' "$SPLIT_SEED" >&2
    return 1
  fi

  export NUM_RUNS BASE_SEED SPLIT_SEED
}

seed_output_dir() {
  local experiment_dir="$1"
  local seed="$2"
  printf '%s/seed_%s\n' "$experiment_dir" "$seed"
}

run_seeded_train() {
  local config="$1"
  local experiment_dir="$2"
  shift 2

  local run_index seed
  for ((run_index = 0; run_index < NUM_RUNS; run_index++)); do
    seed=$((BASE_SEED + run_index))
    run_train "$config" "$(seed_output_dir "$experiment_dir" "$seed")" \
      "$@" \
      "dataset.split_seed=$SPLIT_SEED" \
      "train.seed=$seed"
  done
}

print_seed_run_settings() {
  printf 'Runs per experiment: %s\n' "$NUM_RUNS"
  printf 'Training seeds: %s..%s\n' "$BASE_SEED" "$((BASE_SEED + NUM_RUNS - 1))"
  printf 'Fixed dataset split seed: %s\n' "$SPLIT_SEED"
}
