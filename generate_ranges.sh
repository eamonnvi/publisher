#!/usr/bin/env bash
# generate_ranges.sh
# Usage:
#   generate_ranges.sh DRAFT.md [--chunk N]
# or
#   generate_ranges.sh --count N [--chunk N]
# Prints lines: "START END" for each chunk

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 DRAFT.md [--chunk N]  |  $0 --count N [--chunk N]" >&2
  exit 1
fi

CHUNK=10
COUNT=""
DRAFT=""

# Parse args
if [[ "$1" == "--count" ]]; then
  shift
  COUNT="${1:?Missing count after --count}"; shift
else
  DRAFT="$1"; shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --chunk) CHUNK="${2:?}"; shift 2 ;;
    *) echo "[warn] Unknown arg: $1" >&2; shift ;;
  esac
done

# Derive COUNT if we got a draft
if [[ -n "${DRAFT}" ]]; then
  if [[ ! -f "$DRAFT" ]]; then
    echo "[error] File not found: $DRAFT" >&2
    exit 1
  fi
  # use your CLI to list sections and count lines
  COUNT=$(python3 novel_toolkit_cli.py "$DRAFT" --mode concise --list-sections | wc -l | awk '{print $1}')
fi

# Validate
if [[ -z "${COUNT}" ]]; then
  echo "[error] Could not determine section count" >&2
  exit 1
fi
if ! [[ "$COUNT" =~ ^[0-9]+$ && "$CHUNK" =~ ^[0-9]+$ ]]; then
  echo "[error] COUNT and CHUNK must be integers" >&2
  exit 1
fi
if [[ "$COUNT" -eq 0 ]]; then
  exit 0
fi

# Emit ranges
START=1
while [[ $START -le $COUNT ]]; do
  END=$(( START + CHUNK - 1 ))
  if [[ $END -gt $COUNT ]]; then END=$COUNT; fi
  echo "$START $END"
  START=$(( END + 1 ))
done
