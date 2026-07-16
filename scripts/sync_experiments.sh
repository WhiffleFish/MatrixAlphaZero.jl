#!/usr/bin/env bash
# Pull generated experiment artifacts from the server down into their local-equivalent
# paths, since the repo is checked out at the same relative location on both sides.
#
# Usage:
#   scripts/sync_experiments.sh              # sync current dir (if inside the repo), else everything under experiments/
#   scripts/sync_experiments.sh some/subdir   # sync just that subdir (relative to repo root)
#   scripts/sync_experiments.sh --dry-run     # preview without copying
#   scripts/sync_experiments.sh --delete      # also remove local files gone on remote

set -euo pipefail

REMOTE_HOST="omak.colorado.edu"
REMOTE_ROOT="~/code/MatrixAlphaZero.jl"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXTRA_ARGS=()
SUBDIR=""

for arg in "$@"; do
  case "$arg" in
    --dry-run) EXTRA_ARGS+=(--dry-run) ;;
    --delete)  EXTRA_ARGS+=(--delete) ;;
    *)         SUBDIR="$arg" ;;
  esac
done

if [[ -z "$SUBDIR" ]]; then
  # Default to wherever the caller is currently sitting, if that's inside the repo.
  case "$PWD" in
    "$LOCAL_ROOT"/*) SUBDIR="${PWD#"$LOCAL_ROOT"/}/" ;;
    "$LOCAL_ROOT")   SUBDIR="experiments/" ;;
    *)               SUBDIR="experiments/" ;;
  esac
fi

RSYNC_ARGS=(-avzm --exclude='.CondaPkg/' --exclude='*.jl')
if (( ${#EXTRA_ARGS[@]} )); then
  RSYNC_ARGS+=("${EXTRA_ARGS[@]}")
fi

# A trailing slash on both paths syncs the directory contents, avoiding a
# nested <subdir>/<subdir> directory when the local destination already exists.
rsync "${RSYNC_ARGS[@]}" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/${SUBDIR%/}/" \
  "${LOCAL_ROOT}/${SUBDIR%/}/"
