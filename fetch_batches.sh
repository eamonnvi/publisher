#!/usr/bin/env bash
# fetch_batches.sh
# Read a TSV of: range<TAB>batch_id<TAB>outdir and fetch each batch via the CLI

set -euo pipefail

TSV="${1:-}"
if [[ -z "$TSV" || ! -f "$TSV" ]]; then
  echo "usage: $0 path/to/batch_ids_YYYYMMDD_HHMMSS.tsv" >&2
  exit 1
fi

# You can override these at runtime: EVERY=5 WAIT=300 ./fetch_batches.sh file.tsv
EVERY="${EVERY:-5}"
WAIT="${WAIT:-300}"

# Required by the CLI even on fetch (used for mode validation + filenames)
DRAFT="${DRAFT:-TPV-13aug2025.md}"
MODE="${MODE:-concise}"

echo "[info] Fetching batches listed in: $TSV"
echo "[info] DRAFT=$DRAFT MODE=$MODE EVERY=${EVERY}s WAIT=${WAIT}s"

# Skip header lines starting with '#', then process rows
# Format: range<TAB>batch_id<TAB>outdir
tail -n +1 "$TSV" | while IFS=$'\t' read -r RANGE BATCH_ID OUTDIR; do
  # skip header row that starts with 'range' or comment lines
  [[ "$RANGE" == "range" ]] && continue
  [[ "$RANGE" =~ ^# ]] && continue
  [[ -z "$BATCH_ID" || "$BATCH_ID" == "[NO_BATCH_ID]" ]] && {
    echo "[warn] Skipping row with empty/invalid batch id (range='$RANGE')"
    continue
  }
  if [[ -z "$OUTDIR" ]]; then
    echo "[warn] No outdir recorded for $BATCH_ID (range '$RANGE'); skipping"
    continue
  fi

  echo "[info] Fetching $BATCH_ID -> $OUTDIR (range $RANGE)…"
  mkdir -p "$OUTDIR"

  # The CLI writes markdown/jsonl/meta into --outdir
  python3 novel_toolkit_cli.py "$DRAFT" \
    --mode "$MODE" \
    --fetch-id "$BATCH_ID" \
    --poll --every "$EVERY" --wait "$WAIT" \
    --outdir "$OUTDIR" \
    --verbose

  echo "[ok] Fetched $BATCH_ID into $OUTDIR"
done

echo "[done] All batches fetched."
