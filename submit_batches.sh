#!/usr/bin/env bash
# submit_batches.sh
# Slice a draft into ranges, submit batches, and record (range, batch_id, outdir) as TSV.

set -euo pipefail

DRAFT="${DRAFT:-TPV-28aug2025.md}"
MODE="${MODE:-copyedit_suggestions}"
MODEL="${MODEL:-gpt-4.1}"
# Edit these to taste
RANGES=("1 10" "11 20" "21 30" "31 40" "41 46")

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTFILE="batch_ids_${STAMP}.tsv"

{
  echo -e "# draft\t${DRAFT}"
  echo -e "# mode\t${MODE}"
  echo -e "# model\t${MODEL}"
  echo -e "# generated\t${STAMP}"
  echo -e "range\tbatch_id\toutdir"
} > "$OUTFILE"

for RANGE in "${RANGES[@]}"; do
  echo "[info] Submitting range $RANGE ..."
  # Capture CLI stderr (verbose log) and stdout
  LOG="$(python3 novel_toolkit_cli.py "$DRAFT" \
    --mode "$MODE" \
    --sections-range $RANGE \
    --retries 4 \
    --batch --model "$MODEL" --verbose 2>&1)"

  echo "$LOG"

  # 1) Extract batch_id
  BATCH_ID="$(echo "$LOG" | grep -o 'batch_[a-z0-9]\{32\}' | tail -n 1 || true)"

  # 2) Extract outdir by pulling the parent of .../batch_input.jsonl
  #    Example line: [batch] wrote 10 lines -> outputs/TPV-…/batch_input.jsonl
  OUTDIR="$(echo "$LOG" \
    | sed -n 's/.*-> \(.*\)\/batch_input\.jsonl.*/\1/p' \
    | tail -n 1 || true)"

  # Fallback: if OUTDIR empty, try to grab the “wrote:” list and strip filenames to dir
  if [[ -z "$OUTDIR" ]]; then
    OUTDIR="$(echo "$LOG" \
      | sed -n 's/^[[:space:]]*•[[:space:]]*\(outputs\/.*\)\/.*/\1/p' \
      | head -n 1 || true)"
  fi

  if [[ -n "$BATCH_ID" ]]; then
    echo -e "${RANGE}\t${BATCH_ID}\t${OUTDIR}" | tee -a "$OUTFILE"
  else
    echo -e "${RANGE}\t[NO_BATCH_ID]\t${OUTDIR}" | tee -a "$OUTFILE"
    echo "[warn] No batch id detected for range $RANGE"
  fi
done

echo "[ok] All batch IDs saved to $OUTFILE"
