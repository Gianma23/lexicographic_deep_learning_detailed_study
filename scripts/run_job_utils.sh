#!/usr/bin/env bash

# Shared job-control machinery for the training launchers.
#
# A launcher sources this file, calls init_job_control (which reads the common
# environment knobs and validates them) and install_job_traps, then launches
# work through run_train / run_seeded_train and closes with drain_jobs.
#
# Every launcher shares one resume policy. The two knobs below are read by
# init_job_control from the environment; no launcher sets them, so re-entering a
# partly finished campaign behaves the same way whichever script is invoked.
#
#   RUN_PREFLIGHT           (default: strict)
#     strict  Skip a run_dir that already holds test_metrics.yaml, announce and
#             attach a resume when latest.pt exists, and refuse to start a
#             run_dir that holds other artifacts but no resumable latest.pt.
#     resume  If run_dir/latest.pt exists, attach train.resume to the first
#             attempt. Silent.
#     none    Launch whatever it is asked to launch. No inspection of run_dir.
#
#   RUN_RETRY_REQUIRES_CHECKPOINT   (default: 1)
#     1       Only retry when latest.pt exists, so a run that died before its
#             first checkpoint keeps its original failure.
#     0       Retry a failed run unconditionally.
#
# The defaults make a re-run idempotent: completed seeds are skipped, interrupted
# ones resume, and a directory that cannot be resumed safely stops the campaign
# instead of being overwritten. Override RUN_PREFLIGHT=none for a deliberate
# retrain, having removed the old run_dir first. Both knobs change which commands
# get launched, so set them consciously.

init_job_control() {
  PYTHON_BIN="${PYTHON_BIN:-python}"
  DRY_RUN="${DRY_RUN:-0}"
  MAX_PARALLEL="${MAX_PARALLEL:-1}"
  MAX_RESUME_RETRIES="${MAX_RESUME_RETRIES:-3}"
  RUN_PREFLIGHT="${RUN_PREFLIGHT:-strict}"
  RUN_RETRY_REQUIRES_CHECKPOINT="${RUN_RETRY_REQUIRES_CHECKPOINT:-1}"

  if [[ ! "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_PARALLEL must be a positive integer, got: $MAX_PARALLEL" >&2
    exit 2
  fi
  if [[ ! "$MAX_RESUME_RETRIES" =~ ^[0-9]+$ ]]; then
    echo "MAX_RESUME_RETRIES must be a non-negative integer, got: $MAX_RESUME_RETRIES" >&2
    exit 2
  fi
  case "$RUN_PREFLIGHT" in
    none|resume|strict) ;;
    *)
      echo "RUN_PREFLIGHT must be one of: none, resume, strict. Got: $RUN_PREFLIGHT" >&2
      exit 2
      ;;
  esac
  case "$RUN_RETRY_REQUIRES_CHECKPOINT" in
    0|1) ;;
    *)
      echo "RUN_RETRY_REQUIRES_CHECKPOINT must be 0 or 1, got: $RUN_RETRY_REQUIRES_CHECKPOINT" >&2
      exit 2
      ;;
  esac
}

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

install_job_traps() {
  trap handle_interrupt INT TERM
  trap handle_exit EXIT
}

# Wait for one background job to finish. On failure, report the real exit
# status and stop the whole campaign with it.
#
# `wait -n` has to stay inside an `if` condition: these launchers run under
# `set -e`, and a bare `wait -n` that fails would abort before its status could
# be read. Note also that `if ! wait -n; then rc=$?` does NOT work -- $? is then
# the status of the negation, which is always 0.
wait_for_one_job() {
  local rc
  if wait -n; then
    return 0
  else
    rc=$?
    echo "[ERROR] One run failed (exit=${rc}); stopping remaining jobs." >&2
    kill_running_jobs
    exit "$rc"
  fi
}

# Block until a parallel slot is free.
wait_for_job_slot() {
  while (( "$(jobs -pr | wc -l)" >= MAX_PARALLEL )); do
    wait_for_one_job
  done
}

# Wait for every outstanding job, propagating the first failure.
drain_jobs() {
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  while (( "$(jobs -pr | wc -l)" > 0 )); do
    wait_for_one_job
  done
}

# run_train <config> <run_dir> [extra train.* overrides...]
#
# Launches one training run in the background, honouring MAX_PARALLEL, and
# retries it from latest.pt up to MAX_RESUME_RETRIES times on failure.
run_train() {
  local config="$1"
  local run_dir="$2"
  shift 2

  local resume_arg=""
  if [[ "$RUN_PREFLIGHT" == "strict" ]]; then
    if [[ -f "$run_dir/test_metrics.yaml" ]]; then
      echo "[SKIP] Completed run already exists: $run_dir"
      return 0
    fi
    if [[ -f "$run_dir/latest.pt" ]]; then
      resume_arg="train.resume=$run_dir/latest.pt"
      echo "[RESUME] Found checkpoint: $run_dir/latest.pt"
    elif [[ -e "$run_dir/config_resolved.yaml" || -e "$run_dir/run_log.jsonl" || \
            -e "$run_dir/best_topdown.pt" || -e "$run_dir/best_independent.pt" ]]; then
      echo "[ERROR] Existing run artifacts have no resumable latest.pt: $run_dir" >&2
      return 1
    fi
  elif [[ "$RUN_PREFLIGHT" == "resume" ]]; then
    if [[ -f "$run_dir/latest.pt" ]]; then
      resume_arg="train.resume=$run_dir/latest.pt"
    fi
  fi

  local -a cmd=(
    "$PYTHON_BIN" -m train.train
    --config "$config"
    "train.output_dir=$run_dir"
    "$@"
  )
  if [[ -n "$resume_arg" ]]; then
    cmd+=("$resume_arg")
  fi

  # The command used for every retry always carries a resume override.
  local -a retry_cmd=("${cmd[@]}")
  if [[ -z "$resume_arg" ]]; then
    retry_cmd+=("train.resume=$run_dir/latest.pt")
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY-RUN] '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if (( MAX_RESUME_RETRIES > 0 )); then
      if (( RUN_RETRY_REQUIRES_CHECKPOINT == 1 )); then
        printf '[DRY-RUN][RETRY x%s if latest.pt exists] ' "$MAX_RESUME_RETRIES"
      else
        printf '[DRY-RUN][RETRY x%s] ' "$MAX_RESUME_RETRIES"
      fi
      printf '%q ' "${retry_cmd[@]}"
      printf '\n'
    fi
    return 0
  fi

  wait_for_job_slot

  printf '[RUN] '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  (
    set +e
    "${cmd[@]}"
    rc=$?
    attempt=0
    while (( rc != 0 && attempt < MAX_RESUME_RETRIES )); do
      if (( RUN_RETRY_REQUIRES_CHECKPOINT == 1 )) && [[ ! -f "$run_dir/latest.pt" ]]; then
        echo "[NO RETRY] No checkpoint at $run_dir/latest.pt; preserving the original failure." >&2
        break
      fi
      attempt=$((attempt + 1))
      echo "[RETRY ${attempt}/${MAX_RESUME_RETRIES}] run_dir=$run_dir resume=$run_dir/latest.pt" >&2
      "${retry_cmd[@]}"
      rc=$?
    done
    exit "$rc"
  ) &
}

print_job_control_settings() {
  printf 'Dry run: %s\n' "$DRY_RUN"
  printf 'Max parallel: %s\n' "$MAX_PARALLEL"
  printf 'Max resume retries on failure: %s\n' "$MAX_RESUME_RETRIES"
  printf 'Preflight: %s (retry requires checkpoint: %s)\n' \
    "$RUN_PREFLIGHT" "$RUN_RETRY_REQUIRES_CHECKPOINT"
}
