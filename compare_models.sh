#!/usr/bin/env bash
# compare_models.sh
# One-button compare between two models with tone controls.
# Submits, fetches, diffs, and records the chosen settings.

set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 DRAFT.md MODE MODEL_A MODEL_B [SLICE=10] [--poll] [--wait SECS] [--every SECS] [tone-controls...]" >&2
  exit 1
fi

DRAFT="$1"
MODE="$2"
MODEL_A="$3"
MODEL_B="$4"
SLICE="${5:-10}"
shift $(( $# >= 5 ? 5 : 4 ))

# Optional flags
POLL=false
WAIT=0
EVERY=5
EXTRA_ARGS=()
TONE_SUMMARY=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --poll)  POLL=true; shift ;;
    --wait)  WAIT="${2:-0}"; shift 2 ;;
    --every) EVERY="${2:-5}"; shift 2 ;;
    --english-variant|--style|--emphasis)
      EXTRA_ARGS+=("$1" "$2"); TONE_SUMMARY+=("$1=$2"); shift 2 ;;
    *) echo "[warn] Ignoring unknown arg: $1" >&2; shift ;;
  esac
done

STAMP="$(date +%Y%m%d_%H%M%S)"
echo "[info] Comparing ${DRAFT} (${MODE}) → ${MODEL_A} vs ${MODEL_B} [slice=${SLICE}]"
if [[ ${#TONE_SUMMARY[@]} -gt 0 ]]; then
  echo "[info] Tone controls: ${TONE_SUMMARY[*]}"
fi

# Submit batches
TSV_A="$(/usr/bin/env bash ./submit_batches_auto.sh "$DRAFT" "$MODE" "$MODEL_A" "$SLICE" "${EXTRA_ARGS[@]}" | tail -n1)"
TSV_B="$(/usr/bin/env bash ./submit_batches_auto.sh "$DRAFT" "$MODE" "$MODEL_B" "$SLICE" "${EXTRA_ARGS[@]}" | tail -n1)"

MERGED="compare_merged_${STAMP}.tsv"
{
  head -n 5 "$TSV_A"
  awk 'BEGIN{FS="\t"} !/^#/{print $0}' "$TSV_A" | sed '1d'
  awk 'BEGIN{FS="\t"} !/^#/{print $0}' "$TSV_B" | sed '1d'
} > "$MERGED"

echo "[ok] Merged TSV → ${MERGED}"

# Fetch and compare
FETCH_ARGS=( "$MERGED" --mode "$MODE" "${EXTRA_ARGS[@]}" )
if $POLL; then
  FETCH_ARGS+=( --poll --wait "$WAIT" --every "$EVERY" )
fi

COMPARE_DIR="$(/usr/bin/env bash ./fetch_compare_press.sh "${FETCH_ARGS[@]}")"

# Append tone controls to README
if [[ -n "${COMPARE_DIR:-}" && -d "$COMPARE_DIR" && ${#TONE_SUMMARY[@]} -gt 0 ]]; then
  {
    echo
    echo "Tone controls:"
    for kv in "${TONE_SUMMARY[@]}"; do
      echo "  - $kv"
    done
  } >> "$COMPARE_DIR/README.txt"
fi

echo
echo "[ok] Compare bundle: ${COMPARE_DIR}"
echo "    Open ${COMPARE_DIR}/README.txt"