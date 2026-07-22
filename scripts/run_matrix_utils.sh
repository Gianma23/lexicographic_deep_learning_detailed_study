#!/usr/bin/env bash

# Parse a whitespace-separated environment variable into a validated Bash array.
# Usage:
#   parse_choice_list DATASETS "cub200 aircraft" DATASETS cifar100 cub200 aircraft
parse_choice_list() {
  if (( $# < 4 )); then
    echo "parse_choice_list requires ENV_NAME DEFAULT OUTPUT_ARRAY ALLOWED..." >&2
    return 2
  fi

  local env_name="$1"
  local default_value="$2"
  local output_name="$3"
  shift 3
  local -a allowed=("$@")
  local raw_value="${!env_name:-$default_value}"
  local -a parsed=()
  read -r -a parsed <<< "$raw_value"
  if (( ${#parsed[@]} == 0 )); then
    echo "$env_name must contain at least one value." >&2
    return 2
  fi

  local value candidate matched
  for value in "${parsed[@]}"; do
    matched=0
    for candidate in "${allowed[@]}"; do
      if [[ "$value" == "$candidate" ]]; then
        matched=1
        break
      fi
    done
    if (( matched == 0 )); then
      echo "Unsupported $env_name value '$value'. Expected one of: ${allowed[*]}." >&2
      return 2
    fi
  done

  local -n output_ref="$output_name"
  output_ref=("${parsed[@]}")
}

normalize_bool_like() {
  local raw_value="$1"
  local output_name="$2"
  case "$raw_value" in
    1|true|True) printf -v "$output_name" '%s' "true" ;;
    0|false|False) printf -v "$output_name" '%s' "false" ;;
    *)
      printf 'Expected a boolean-like value (0, 1, true, false), got: %s\n' "$raw_value" >&2
      return 1
      ;;
  esac
}
