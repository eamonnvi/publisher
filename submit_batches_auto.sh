#!/usr/bin/env bash
# submit_batches_auto.sh
# Auto-slice a draft into N-sized ranges and submit batches for each.

set -euo pipefail

DRAFT="${1:?Usage: $0 DRAFT.md [MODE] [MODEL] [CHUNK] }"
MODE="${2:-concise}"
MODEL="${3:-gpt-4.1-mini}"
CHUNK="${4:-10}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[error] OPENAI_API_KEY is not set" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTFILE="batch_ids_${STAMP}.tsv"

echo -e "# draft\t${DRAFT}" > "$OUTFILE"
echo -e "# mode\t${MODE}"  >> "$OUTFILE"
echo -e "# model\t${MODEL}" >> "$OUTFILE"
echo -e "# generated\t${STAMP}" >> "$OUTFILE"
echo -e "range\tmodel\tbatch_id\toutdir" >> "$OUTFILE"

# Generate ranges
if ! RANGES=$(./generate_ranges.sh "$DRAFT" --chunk "$CHUNK"); then
  echo "[error] Failed to generate ranges" >&2
  exit 1
fi

while IFS= read -r RANGE; do
  [[ -z "$RANGE" ]] && continue
  echo "[info] Submitting range $RANGE ..."
  LOG=$(python3 novel_toolkit_cli.py "$DRAFT" \
    --mode "$MODE" \
    --sections-range $RANGE \
    --batch --model "$MODEL" --verbose 2>&1)

  echo "$LOG"

  BATCH_ID=$(echo "$LOG" | grep -o 'batch_[a-z0-9]\{32\}' | tail -n 1 || true)
  OUTDIR=$(echo "$LOG" | awk '/^\[ok\] submitted batch:/ {printed=1} printed && /\[ok\] submitted batch:/ {printed=0} {print}' | grep -o 'outputs/[^ ]\+' | tail -n 1 || true)
  # Robust fallback: read the last submitted.json written during this submit
  if [[ -z "$OUTDIR" ]]; then
    OUTDIR=$(ls -td outputs/* 2>/dev/null | head -n1 || true)
  fi

  if [[ -n "$BATCH_ID" ]]; then
    echo -e "${RANGE}\t${MODEL}\t${BATCH_ID}\t${OUTDIR}" | tee -a "$OUTFILE"
  else
    echo "[warn] No batch id detected for range $RANGE" | tee -a "$OUTFILE"
  fi
done <<< "$RANGES"

echo "[ok] All batch IDs saved to $OUTFILE"
echo "$OUTFILE"
