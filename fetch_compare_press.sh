#!/usr/bin/env bash
# fetch_compare_press.sh
# Fetches batches listed in a TSV and writes per-range comparison diffs.
# Expected columns (tab or space separated, header lines allowed):
#   range    model    batch_id    outdir
#
# Usage:
#   fetch_compare_press.sh SUBMIT_TSV [--mode MODE] [--poll] [--wait SECONDS] [--every SECONDS]
#                                    [--english-variant "British English"] [--style STYLE] [--emphasis EMPHASIS]
#
# Improvements in this version:
# - Auto-retry if the fetch yields no outputs yet (even after --poll) — configurable via RETRY_WAIT/RETRY_SLEEP
# - Auto-basename per range/model/mode: R<range>.<mode>.<model>.md

set -euo pipefail

TSV="${1:?Usage: $0 SUBMIT_TSV [--mode MODE] [--poll] [--wait SECONDS] [--every SECONDS] [--english-variant ..] [--style ..] [--emphasis ..] }"
shift || true

trap 'echo; echo "[info] interrupted"; exit 130' INT

# Defaults
POLL=false
WAIT=0
EVERY=5
MODE="press_synopsis"
EXTRA_ARGS=()
TONE_SUMMARY=()

# Auto-retry controls (in addition to --poll): 0 = infinite
RETRY_WAIT="${RETRY_WAIT:-0}"
RETRY_SLEEP="${RETRY_SLEEP:-10}"

# Parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --poll)  POLL=true; shift ;;
    --wait)  WAIT="${2:-0}"; shift 2 ;;
    --every) EVERY="${2:-5}"; shift 2 ;;
    --mode)  MODE="${2:-press_synopsis}"; shift 2 ;;
    --english-variant|--style|--emphasis)
      EXTRA_ARGS+=("$1" "$2")
      TONE_SUMMARY+=("$1=$2")
      shift 2
      ;;
    *)
      echo "[warn] Unknown arg: $1" >&2
      shift
      ;;
  esac
done

# Sanity: OpenAI key
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[error] OPENAI_API_KEY is not set" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
COMPARE_DIR="press_compare_${STAMP}"
mkdir -p "$COMPARE_DIR"

TRIPLE_LIST="${COMPARE_DIR}/md_paths.tsv"
echo -e "range\tmodel\tmd_path" > "$TRIPLE_LIST"

if [[ ${#TONE_SUMMARY[@]} -gt 0 ]]; then
  echo "[info] Tone controls: ${TONE_SUMMARY[*]}"
fi

# Helper: check for a markdown artifact inside OUTDIR
_find_md() {
  local OUTDIR="$1"
  local MD=""
  local META_CAND
  META_CAND="$(ls -1 "$OUTDIR"/*.meta.json 2>/dev/null | head -n1 || true)"
  if [[ -n "$META_CAND" ]] && command -v jq >/dev/null 2>&1; then
    MD="$(jq -r '.outputs.markdown // empty' "$META_CAND" 2>/dev/null || true)"
    if [[ -n "$MD" && -f "$OUTDIR/$MD" ]]; then
      printf "%s\n" "$OUTDIR/$MD"
      return 0
    fi
  fi
  # Fallback: first .md
  MD="$(ls -1 "$OUTDIR"/*.md 2>/dev/null | head -n1 || true)"
  [[ -n "$MD" ]] && printf "%s\n" "$MD" || true
  return 0
}

# Helper: fetch one batch id into its outdir and record md path (with auto-retry & auto-basename)
fetch_one() {
  local RANGE="$1"
  local MODEL="$2"
  local BATCH_ID="$3"
  local OUTDIR_IN="$4"

  # Normalize OUTDIR
  local OUTDIR="$OUTDIR_IN"
  if [[ "$OUTDIR" == */batch_input.jsonl ]]; then
    OUTDIR="$(dirname "$OUTDIR")"
  fi
  mkdir -p "$OUTDIR"

  # Auto-basename: R<range>.<mode>.<model>
  local SAFE_RANGE="${RANGE// /-}"
  local BASENAME="R${SAFE_RANGE}.${MODE}.${MODEL}"

  # Build fetch args
  local FETCH_ARGS=( --mode "$MODE" --outdir "$OUTDIR" --fetch-id "$BATCH_ID" --model "$MODEL" --basename "$BASENAME" )
  if $POLL; then
    FETCH_ARGS+=( --poll --wait "$WAIT" --every "$EVERY" )
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    FETCH_ARGS+=("${EXTRA_ARGS[@]}")
  fi

  echo "[info] fetching ${RANGE} | ${MODEL} | ${BATCH_ID} → ${OUTDIR} (mode=${MODE})" >&2

  # Attempt 1
  python3 novel_toolkit_cli.py DUMMY.md "${FETCH_ARGS[@]}" 1>/dev/null || true

  # Check for md
  local MD_PATH
  MD_PATH="$(_find_md "$OUTDIR")"
  if [[ -n "$MD_PATH" ]]; then
    echo -e "${RANGE}\t${MODEL}\t${MD_PATH}" >> "$TRIPLE_LIST"
    return 0
  fi

  # Auto-retry loop (in addition to --poll). RETRY_WAIT=0 => infinite.
  local start now elapsed
  start="$(date +%s)"
  while : ; do
    # Decide if we should keep trying
    now="$(date +%s)"; elapsed=$(( now - start ))
    if [[ "$RETRY_WAIT" -ne 0 && "$elapsed" -ge "$RETRY_WAIT" ]]; then
      break
    fi
    sleep "$RETRY_SLEEP"
    python3 novel_toolkit_cli.py DUMMY.md "${FETCH_ARGS[@]}" 1>/dev/null || true
    MD_PATH="$(_find_md "$OUTDIR")"
    if [[ -n "$MD_PATH" ]]; then
      echo -e "${RANGE}\t${MODEL}\t${MD_PATH}" >> "$TRIPLE_LIST"
      return 0
    fi
  done

  echo "[warn] Could not locate a markdown output in ${OUTDIR}" >&2
  return 0
}

# -------- Robust TSV parsing --------
# Accept tabs or spaces. Skip comments (#...), blank lines, and header lines (where any column is 'batch_id').
# Only keep rows where some field matches /^batch_/.
readarray -t ROWS < <(
  awk '
    BEGIN{FS="[ \t]+"}
    { gsub(/\r$/,"") }          # strip CR on CRLF files
    /^[ \t]*#/ {next}           # comments
    NF==0 {next}                # blank
    {
      # header if any field equals batch_id (case-insensitive)
      hdr=0
      for(i=1;i<=NF;i++){t=$i; gsub(/[ \t]/,"",t); if(tolower(t)=="batch_id"){hdr=1}}
      if(hdr){next}

      # Determine range & model positions:
      # if first two tokens are ints => "start end" range; model at $3; batch id somewhere from $4+
      isint1 = ($1 ~ /^[0-9]+$/)
      isint2 = ($2 ~ /^[0-9]+$/)
      if(isint1 && isint2 && NF>=3){
        rng=$1" "$2; mdl=$3; k=4
      } else {
        rng=$1; mdl=(NF>=2?$2:""); k=3
      }

      # find batch id in the remainder
      bid=""
      for(i=k;i<=NF;i++){ if($i ~ /^batch_[A-Za-z0-9]+$/){ bid=$i; break } }
      if(bid==""){ next }

      outdir=""
      if(i<NF){ outdir=$(i+1) }   # take next token if present

      print rng "\t" mdl "\t" bid "\t" outdir
    }
  ' "$TSV"
)

if [[ ${#ROWS[@]} -eq 0 ]]; then
  echo "[error] No data rows with batch ids found in $TSV" >&2
  exit 1
fi

echo "[info] fetching range | model | batch_id → outdir (mode=${MODE})"

# Process rows
for ROW in "${ROWS[@]}"; do
  IFS=$'\t' read -r RANGE MODEL BATCH_ID OUTDIR <<<"$ROW"
  if [[ -z "$BATCH_ID" || -z "$OUTDIR" ]]; then
    echo "[warn] Skipping row (missing batch_id or outdir): $ROW" >&2
    continue
  fi
  fetch_one "$RANGE" "$MODEL" "$BATCH_ID" "$OUTDIR"
done

# Build comparisons per range (preserve "1 8" as a single range key)
RANGES=()
while IFS= read -r rng; do
  [[ -n "$rng" ]] && RANGES+=("$rng")
done < <(awk -F'\t' 'NR>1 {print $1}' "$TRIPLE_LIST" | sort -u)

if [[ ${#RANGES[@]} -eq 0 ]]; then
  echo "[warn] No md files recorded; nothing to compare." >&2
  echo "$COMPARE_DIR"
  exit 0
fi

for R in "${RANGES[@]}"; do
  mapfile -t LINES < <(awk -F'\t' -v r="$R" 'NR>1 && $1==r {print $0}' "$TRIPLE_LIST")
  if [[ ${#LINES[@]} -lt 2 ]]; then
    echo "[note] Range '$R' has <2 outputs; skipping diff." >&2
    continue
  fi

  M1=$(echo "${LINES[0]}" | awk -F'\t' '{print $2}')
  P1=$(echo "${LINES[0]}" | awk -F'\t' '{print $3}')
  M2=$(echo "${LINES[1]}" | awk -F'\t' '{print $2}')
  P2=$(echo "${LINES[1]}" | awk -F'\t' '{print $3}')

  SAFE_R=$(echo "$R" | tr ' ' '-' | tr '/' '-')
  DIFF_OUT="${COMPARE_DIR}/diff_${SAFE_R}_${M1}_vs_${M2}.diff"

  if command -v colordiff >/dev/null 2>&1; then
    colordiff -u "$P1" "$P2" > "$DIFF_OUT" || true
  else
    diff -u "$P1" "$P2" > "$DIFF_OUT" || true
  fi
  echo "[ok] wrote ${DIFF_OUT}" >&2
done

# Index
INDEX="${COMPARE_DIR}/README.txt"
{
  echo "Model comparisons"
  echo "Generated: ${STAMP}"
  echo "Mode: ${MODE}"
  if [[ ${#TONE_SUMMARY[@]} -gt 0 ]]; then
    echo "Tone controls:"
    for kv in "${TONE_SUMMARY[@]}"; do
      echo "  - $kv"
    done
  fi
  echo
  echo "MD paths:"
  if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "$TRIPLE_LIST"
  else
    cat "$TRIPLE_LIST"
  fi
  echo
  echo "Diffs:"
  ls -1 "${COMPARE_DIR}"/diff_*.diff 2>/dev/null || echo "(none)"
} > "$INDEX"

echo "[ok] Compare bundle in: ${COMPARE_DIR}"
echo "${COMPARE_DIR}"