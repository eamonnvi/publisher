#!/usr/bin/env bash
# fetch_batches.sh
# Read a TSV of: range [model] batch_id outdir and fetch each batch via the CLI
set -euo pipefail

usage() {
  echo "usage: $0 path/to/batch_ids_YYYYMMDD_HHMMSS.tsv" >&2
  echo "env:   EVERY=5 WAIT=300 DRAFT=... MODE=... REBATCH=0" >&2
}

TSV="${1:-}"
if [[ -z "$TSV" || ! -f "$TSV" ]]; then
  usage; exit 1
fi

# Defaults (overridable via env)
EVERY="${EVERY:-5}"
WAIT="${WAIT:-300}"
DRAFT="${DRAFT:-TPV-13aug2025.md}"
MODE="${MODE:-concise}"

# Try to pick DRAFT and MODE from TSV header unless env overrides them
TSV_DRAFT="$(awk -F'\t' '$1=="# draft"{print $2; exit}' "$TSV" || true)"
TSV_MODE="$(awk -F'\t'  '$1=="# mode"{print $2; exit}' "$TSV" || true)"
if [[ -n "${TSV_DRAFT:-}" && "$DRAFT" == "TPV-13aug2025.md" ]]; then
  DRAFT="$TSV_DRAFT"
fi
if [[ -n "${TSV_MODE:-}" && "$MODE" == "concise" ]]; then
  MODE="$TSV_MODE"
fi

echo "[info] Fetching batches listed in: $TSV"
echo "[info] DRAFT=$DRAFT MODE=$MODE EVERY=${EVERY}s WAIT=${WAIT}s"
echo "[info] REBATCH=${REBATCH:-0} (set REBATCH=1 to auto-retry empties)"

# Python normaliser: emits exactly 4 tab-separated fields per data row:
#   range\tmodel\tbatch_id\trun_base
canonical_stream() {
  python3 - "$TSV" <<'PY'
import sys, re, pathlib
p = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
re_hdr = re.compile(r'^\s*#')
re_col = re.compile(r'^\s*range(\s|$)', re.I)

# 4-field rows: "A B   MODEL   BATCH   OUTDIR"
pat4 = re.compile(r'^\s*([0-9]+)\s+([0-9]+)\s+(\S+)\s+(\S+)\s+(.+)$')
# 3-field rows (legacy): "A B   BATCH   OUTDIR"  (synthesise MODEL as "-")
pat3 = re.compile(r'^\s*([0-9]+)\s+([0-9]+)\s+(\S+)\s+(.+)$')

for line in p:
    if re_hdr.match(line):      # skip "# ..." headers
        continue
    if re_col.match(line):      # skip "range ..." header row
        continue
    line = line.rstrip()
    if not line:
        continue

    m = pat4.match(line)
    if m:
        a, b, model, batch, base = m.groups()
        print(f"{a} {b}\t{model}\t{batch}\t{base}")
        continue

    m = pat3.match(line)
    if m:
        a, b, batch, base = m.groups()
        print(f"{a} {b}\t-\t{batch}\t{base}")
        continue

    # silently ignore malformed rows
PY
}

DRAFT_STEM="$(basename "${DRAFT%.*}")"

# shellcheck disable=SC2046
while IFS=$'\t' read -r RANGE MODEL BATCH_ID RUN_BASE; do
  [[ -z "${BATCH_ID:-}" ]] && continue

  # Validate batch id shape early
  if [[ ! "$BATCH_ID" =~ ^batch_[A-Za-z0-9_-]+$ ]]; then
    echo "[warn] Skipping malformed row: bad batch_id='$BATCH_ID' (range='$RANGE')" >&2
    continue
  fi

  OUTDIR="${RUN_BASE%/}/outputs"
  mkdir -p "$OUTDIR"

  # Split "A B" -> A / B for a stable BASENAME
  read -r A B <<<"$RANGE"

  # Build BASENAME (omit model if legacy 3-col row)
  if [[ "${MODEL:-}" == "-" || -z "${MODEL:-}" ]]; then
    BASENAME="${DRAFT_STEM}.${MODE}.r${A}-${B}"
  else
    BASENAME="${DRAFT_STEM}.${MODE}.${MODEL}.r${A}-${B}"
  fi

  # Surface submit-time batch_map.json (if the stamped submit dir exists)
  submit_dir="$(find "$OUTDIR" -maxdepth 1 -type d -name "${BASENAME}.*" -print -quit 2>/dev/null || true)"
  if [[ -n "$submit_dir" && -f "$submit_dir/batch_map.json" ]]; then
    cp -f "$submit_dir/batch_map.json" "$OUTDIR/batch_map.json"
  else
    echo "[warn] batch_map.json not found; headings will fall back to custom_id"
  fi

  echo "[info] Fetching ${BATCH_ID} -> ${OUTDIR} (basename=${BASENAME}; range ${RANGE}${MODEL:+, model ${MODEL}})…"

  # Your CLI syntax: requires --mode and the draft positional arg
  python3 novel_toolkit_cli.py \
    --mode "$MODE" \
    --fetch-id "$BATCH_ID" \
    --outdir "$OUTDIR" \
    --basename "$BASENAME" \
    ${MODEL:+--model "$MODEL"} \
    --every "$EVERY" \
    --wait "$WAIT" \
    "$DRAFT"

  echo "[ok] Fetched ${BATCH_ID} into ${OUTDIR}"

  # --- Optional auto-rebatch path (opt-in with REBATCH=1) ---
  if [[ "${REBATCH:-0}" == "1" ]]; then
    if grep -Rqs "\[empty or unparsable output;" "$OUTDIR"; then
      echo "[rebatch] Empties detected for $BATCH_ID; preparing retry…" >&2

      # 1) Retrieve output_file_id for this batch
      OUTPUT_FILE_ID="$(
        python3 - "$BATCH_ID" <<'PY' || true
import os, sys
from openai import OpenAI
bid = sys.argv[1]
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
b = client.batches.retrieve(bid)
ofid = None
# Try both object and dict styles
ofid = getattr(b, "output_file_id", None) or (b.model_dump().get("output_file_id") if hasattr(b, "model_dump") else None)
if not ofid and isinstance(b, dict):
    ofid = b.get("output_file_id")
if not ofid:
    raise SystemExit("[error] no output_file_id on batch")
print(ofid)
PY
      )"
      if [[ -n "$OUTPUT_FILE_ID" && "$OUTPUT_FILE_ID" != \[error* ]]; then
        # 2) Rebatch just the missing items (requires ntl_rebatch_missing.py)
        RETRY_LOG="$(python3 ntl_rebatch_missing.py "${MODEL:-gpt-4.1-mini}" "$MODE" "$OUTDIR" "$OUTPUT_FILE_ID" "$OUTDIR/sections.json" 2>&1 || true)"
        echo "$RETRY_LOG"
        RETRY_ID="$(echo "$RETRY_LOG" | grep -o 'batch_[A-Za-z0-9_-]*' | tail -n1 || true)"
        if [[ -n "$RETRY_ID" ]]; then
          RETRY_TSV="retry_ids_$(date +%Y%m%d_%H%M%S).tsv"
          printf 'range\tmodel\tbatch_id\toutdir\n' > "$RETRY_TSV"
          printf '%s\t%s\t%s\t%s\n' "$RANGE" "${MODEL:-gpt-4.1-mini}" "$RETRY_ID" "$RUN_BASE" >> "$RETRY_TSV"
          echo "[rebatch] Submitted retry batch: $RETRY_ID → recorded in $RETRY_TSV"
        else
          echo "[rebatch] Retry submission did not return a batch id." >&2
        fi
      else
        echo "[rebatch] Could not obtain output_file_id for $BATCH_ID" >&2
      fi
    fi
  fi
done < <(canonical_stream)

echo "[done] All batches fetched."

