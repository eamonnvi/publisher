#!/usr/bin/env bash
# submit_auto.sh
set -euo pipefail

DRAFT="${1:?Usage: $0 DRAFT.md [MODE] [MODEL] [CHUNK]}"
MODE="${2:-concise}"
MODEL="${3:-gpt-4.1-mini}"
CHUNK="${4:-10}"

[[ -z "${OPENAI_API_KEY:-}" ]] && { echo "[error] OPENAI_API_KEY is not set" >&2; exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
DATE_ONLY="$(date +%Y%m%d)"
OUTFILE="batch_ids_${STAMP}.tsv"
DRAFT_STEM="$(basename "${DRAFT%.*}")"

{
  printf '# draft	%s
' "$DRAFT"
  printf '# mode	%s
'  "$MODE"
  printf '# model	%s
' "$MODEL"
  printf '# generated	%s
' "$STAMP"
  printf 'range	model	batch_id	outdir
'
} > "$OUTFILE"

# Each line: "A B"
if ! RANGES=$(./generate_ranges.sh "$DRAFT" --chunk "$CHUNK"); then
  echo "[error] Failed to generate ranges" >&2
  exit 1
fi

while IFS= read -r RANGE; do
  [[ -z "$RANGE" ]] && continue
  read -r A B <<<"$RANGE"

  BASENAME="${DRAFT_STEM}.${MODE}.${MODEL}.r${A}-${B}"

  echo "[info] Submitting $A..$B for $MODEL ..."
  LOG=$(
    python3 novel_toolkit_cli.py "$DRAFT"       --mode "$MODE"       --sections-range "$A" "$B"       --batch       --model "$MODEL"       --outdir "outputs-${DATE_ONLY}/PENDING"       --basename "$BASENAME"       --verbose 2>&1
  ) || true
  echo "$LOG"

  BATCH_ID=$(printf '%s
' "$LOG" | grep -o 'batch_[a-f0-9]\{32\}' | tail -n 1 || true)
  if [[ -n "$BATCH_ID" ]]; then
    RUN_BASE="outputs-${DATE_ONLY}/${BATCH_ID}"
    FINAL_OUTDIR="${RUN_BASE}/outputs"
    mkdir -p "$FINAL_OUTDIR"

    # If the CLI wrote into the PENDING path, move it under the batch base
    PENDING_DIR=$(find "outputs-${DATE_ONLY}/PENDING" -maxdepth 1 -type d -name "${BASENAME}.*" -print -quit 2>/dev/null || true)
    if [[ -n "${PENDING_DIR:-}" ]]; then
      mkdir -p "$FINAL_OUTDIR"
      mv "$PENDING_DIR" "$FINAL_OUTDIR"/ 2>/dev/null || true
      rmdir "outputs-${DATE_ONLY}/PENDING" 2>/dev/null || true
    fi

    printf '%s %s	%s	%s	%s
' "$A" "$B" "$MODEL" "$BATCH_ID" "$RUN_BASE" | tee -a "$OUTFILE"
  else
    echo "[warn] No batch id detected for $MODEL $A..$B" | tee -a "$OUTFILE"
  fi
done <<< "$RANGES"

echo "[ok] All batch IDs saved to $OUTFILE"
echo "$OUTFILE"
