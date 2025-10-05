#!/usr/bin/env bash
# submit_ranges.sh
set -euo pipefail

usage(){ cat <<EOF
Usage: $0 --draft FILE.md --mode MODE --models m1[,m2,...] --ranges "A B;C D;..."
EOF
}

DRAFT=""; MODE="concise"; MODELS=""; RANGES_SPEC=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --draft)   DRAFT="${2:-}"; shift 2;;
    --mode)    MODE="${2:-}"; shift 2;;
    --models)  MODELS="${2:-}"; shift 2;;
    --ranges)  RANGES_SPEC="${2:-}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "[error] Unknown arg: $1" >&2; usage; exit 1;;
  esac
done

[[ -z "$DRAFT" || -z "$MODELS" || -z "$RANGES_SPEC" ]] && { echo "[error] Missing required args"; usage; exit 1; }
[[ -z "${OPENAI_API_KEY:-}" ]] && { echo "[error] OPENAI_API_KEY is not set" >&2; exit 1; }

IFS=',' read -r -a MODEL_ARR <<<"$MODELS"
IFS=';' read -r -a RANGE_ARR <<<"$RANGES_SPEC"

STAMP="$(date +%Y%m%d_%H%M%S)"
DATE_ONLY="$(date +%Y%m%d)"
OUTFILE="batch_ids_${STAMP}.tsv"
DRAFT_STEM="$(basename "${DRAFT%.*}")"

{
  printf '# draft	%s
' "$DRAFT"
  printf '# mode	%s
'  "$MODE"
  printf '# models	%s
' "$MODELS"
  printf '# generated	%s
' "$STAMP"
  printf 'range	model	batch_id	outdir
'
} > "$OUTFILE"

trim(){ sed -E 's/^[[:space:]]+|[[:space:]]+$//g'; }

for RANGE in "${RANGE_ARR[@]}"; do
  RANGE="$(printf '%s' "$RANGE" | trim)"
  [[ -z "$RANGE" ]] && continue
  read -r A B <<<"$RANGE" || true
  if [[ -z "${A:-}" || -z "${B:-}" || ! "$A" =~ ^[0-9]+$ || ! "$B" =~ ^[0-9]+$ ]]; then
    echo "[warn] Skipping invalid range spec: '$RANGE'"
    continue
  fi

  for MODEL in "${MODEL_ARR[@]}"; do
    MODEL="$(printf '%s' "$MODEL" | trim)"
    [[ -z "$MODEL" ]] && continue

    BASENAME="${DRAFT_STEM}.${MODE}.${MODEL}.r${A}-${B}"

    echo "[info] Submitting $A..$B for ${MODEL} ..."
    LOG=$(
      python3 novel_toolkit_cli.py "$DRAFT"         --mode "$MODE"         --sections-range "$A" "$B"         --batch         --model "$MODEL"         --outdir "outputs-${DATE_ONLY}/PENDING"         --basename "$BASENAME"         --verbose 2>&1
    ) || true
    echo "$LOG"

    BATCH_ID=$(printf '%s\n' "$LOG" | grep -o 'batch_[a-f0-9]\{32\}' | tail -n1 || true)
    if [[ -n "$BATCH_ID" ]]; then
      RUN_BASE="outputs-${DATE_ONLY}/${BATCH_ID}"
      FINAL_OUTDIR="${RUN_BASE}/outputs"
      mkdir -p "$FINAL_OUTDIR"

      # Move stamped run dir from PENDING to FINAL_OUTDIR
      PENDING_DIR=$(find "outputs-${DATE_ONLY}/PENDING" -maxdepth 1 -type d -name "${BASENAME}.*" -print -quit 2>/dev/null || true)
      if [[ -n "${PENDING_DIR:-}" ]]; then
        mv "$PENDING_DIR" "$FINAL_OUTDIR"/ 2>/dev/null || true
        rmdir "outputs-${DATE_ONLY}/PENDING" 2>/dev/null || true
      fi

      printf '%s %s\t%s\t%s\t%s\n' "$A" "$B" "$MODEL" "$BATCH_ID" "$RUN_BASE" | tee -a "$OUTFILE"
    else
      echo "[warn] No batch id detected for ${MODEL} $A..$B" | tee -a "$OUTFILE"
    fi
  done
done

echo "[ok] All batch IDs saved to $OUTFILE"
echo "$OUTFILE"
