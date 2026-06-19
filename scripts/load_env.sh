#!/usr/bin/env bash

# Source repository-local KEY=VALUE settings while preserving exported overrides.
load_project_env() {
  local repo_root="$1"
  local env_file="${PROJECT_ENV_FILE:-$repo_root/.env}"
  local line key value

  [[ -f "$env_file" ]] || return 0

  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ "$line" =~ ^[[:space:]]*$ || "$line" =~ ^[[:space:]]*# ]] && continue
    line="${line#export }"
    if [[ "$line" != *=* ]]; then
      printf 'Invalid .env entry in %s: %s\n' "$env_file" "$line" >&2
      return 1
    fi

    key="${line%%=*}"
    value="${line#*=}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"

    if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      printf 'Invalid .env key in %s: %s\n' "$env_file" "$key" >&2
      return 1
    fi
    if [[ ( "$value" == \"*\" && "$value" == *\" ) || ( "$value" == \'*\' && "$value" == *\' ) ]]; then
      value="${value:1:${#value}-2}"
    fi
    if [[ ! -v "$key" ]]; then
      printf -v "$key" '%s' "$value"
      export "$key"
    fi
  done < "$env_file"
}

