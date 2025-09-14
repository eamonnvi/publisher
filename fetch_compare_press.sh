#!/usr/bin/env bash
# fetch_compare_press.sh
# Fetches batches listed in a TSV and writes per-range comparison diffs.
# Expected columns (tab OR space separated):
#   range   model   batch_id   outdir
# Lines starting with '#' and header rows are ignored.
#
# Usage:
#   fetch_compare_press.sh SUBMIT_TSV [--mode MODE] [--poll] [--wait SECONDS] [--every SECONDS] \
#                                    [--english-variant "British English"] [--style STYLE] [--emphasis EMPHASIS]

set -euo pipefail
trap 'echo; echo "[info] interrupted"; exit 130' INT

TSV="${1:?Usage: $0 SUBMIT_TSV [--mode MODE] [--poll] [--wait SECONDS] [--every SECONDS] [--english-variant ..] [--style ..] [--emphasis ..] }"
shift || true

# ---------- defaults ----------
POLL=false
WAIT=0
EVERY=5
MODE="press_synopsis"
EXTRA_ARGS=()
TONE_SUMMARY=()

# ---------- optional venv auto-activation ----------
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  VENV="$HOME/Projects/Palace/palace_env/bin/activate"
  [[ -f "$VENV" ]] && source "$VENV"
fi

# ---------- parse flags ----------
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

# ---------- sanity ----------
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

# ---------- helpers ----------
fetch_one() {
  local RANGE="$1"
  local MODEL="$2"
  local BATCH_ID="$3"
  local OUTDIR="$4"

  # If TSV recorded ".../batch_input.jsonl", strip it to get the folder.
  if [[ "$OUTDIR" == */batch_input.jsonl ]]; then
    OUTDIR="$(dirname "$OUTDIR")"
  fi
  mkdir -p "$OUTDIR"

  # Nice basename for outputs (avoids DUMMY.*)
  local BASENAME="R${RANGE// /-}.${MODE}.${MODEL}"

  # Build CLI args
  local FETCH_ARGS=( --mode "$MODE" --outdir "$OUTDIR" --fetch-id "$BATCH_ID" --model "$MODEL" --basename "$BASENAME" )
  if $POLL; then
    FETCH_ARGS+=( --poll --wait "$WAIT" --every "$EVERY" )
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    FETCH_ARGS+=("${EXTRA_ARGS[@]}")
  fi

  echo "[info] fetching ${RANGE} | ${MODEL} | ${BATCH_ID} → ${OUTDIR} (mode=${MODE})" >&2
  if ! python3 novel_toolkit_cli.py DUMMY.md "${FETCH_ARGS[@]}" 1>/dev/null; then
    echo "[warn] fetch failed for ${BATCH_ID}; continuing to look for artifacts…" >&2
  fi

  # Prefer meta.json → outputs.markdown, fall back to first *.md
  local META_CAND MD
  META_CAND="$(ls -1 "$OUTDIR"/*.meta.json 2>/dev/null | head -n1 || true)"
  if [[ -n "$META_CAND" ]] && command -v jq >/dev/null 2>&1; then
    MD="$(jq -r '.outputs.markdown // empty' "$META_CAND" 2>/dev/null || true)"
    if [[ -n "$MD" && -f "$OUTDIR/$MD" ]]; then
      echo -e "${RANGE}\t${MODEL}\t${OUTDIR}/${MD}" >> "$TRIPLE_LIST"
      return 0
    fi
  fi
  MD="$(ls -1 "$OUTDIR"/*.md 2>/dev/null | head -n1 || true)"
  if [[ -n "$MD" ]]; then
    echo -e "${RANGE}\t${MODEL}\t${MD}" >> "$TRIPLE_LIST"
    return 0
  fi

  echo "[warn] Could not locate a markdown output in ${OUTDIR}" >&2
  return 0
}

# ---------- robust TSV parsing ----------
# Accept tabs or spaces, skip comments/blank lines and header rows anywhere.
# Only keep rows that contain a "batch_*" token (in any column).
# Print: <range>\t<model>\t<batch_id>\t<outdir>
mapfile -t ROWS < <(
  awk '
    BEGIN{OFS="\t"}
    { sub(/\r$/,"") }                   # strip CR
    /^[ \t]*#/ {next}                   # comments
    NF==0 {next}                        # blanks
    {
      # Split on runs of tabs/spaces to normalize
      n=split($0, a, /[ \t]+/)
      # Detect header if any field (case-insensitive) equals "batch_id"
      hdr=0
      for(i=1;i<=n;i++){
        t=a[i]; gsub(/[ \t]/,"",t); tl=toupper(t)
        if(tl=="BATCH_ID"){hdr=1;break}
      }
      if(hdr){next}

      # Find batch_id token position and build fields around it.
      bi=0
      for(i=1;i<=n;i++){ if(a[i] ~ /^batch_[A-Za-z0-9]+$/){ bi=i; break } }
      if(bi==0){ next }                 # skip if no batch id token

      # RANGE: if first two fields are integers → "X Y", else just field 1
      isint1=(a[1] ~ /^[0-9]+$/)
      isint2=(a[2] ~ /^[0-9]+$/)
      if(isint1 && isint2){ range=a[1] " " a[2]; model=a[3] }
      else { range=a[1]; model=(n>=2?a[2]:"") }

      batch=a[bi]
      outdir=(bi<n ? a[bi+1] : "")

      print range, model, batch, outdir
    }
  ' "$TSV"
)

if [[ ${#ROWS[@]} -eq 0 ]]; then
  echo "[error] No data rows with batch ids found in $TSV" >&2
  exit 1
fi

echo "[info] fetching range | model | batch_id → outdir (mode=${MODE})"

# ---------- fetch all rows ----------
for ROW in "${ROWS[@]}"; do
  # Split the normalized row (always tab-separated now)
  IFS=$'\t' read -r RANGE MODEL BATCH_ID OUTDIR <<<"$ROW"
  if [[ -z "${BATCH_ID:-}" || -z "${OUTDIR:-}" ]]; then
    echo "[warn] Skipping row (missing batch_id or outdir): $ROW" >&2
    continue
  fi
  # Normalize outdir by stripping trailing /batch_input.jsonl if present
  if [[ "$OUTDIR" == */batch_input.jsonl ]]; then
    OUTDIR="${OUTDIR%/batch_input.jsonl}"
  fi
  fetch_one "$RANGE" "$MODEL" "$BATCH_ID" "$OUTDIR"
done

# ---------- build comparisons per range ----------
# Keep multi-word ranges intact
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
  # Collect all (model, path) rows for this range
  mapfile -t LINES < <(awk -F'\t' -v r="$R" 'NR>1 && $1==r {print $0}' "$TRIPLE_LIST")

  # Need at least two different models to compare
  if [[ ${#LINES[@]} -lt 2 ]]; then
    echo "[note] Range '$R' has <2 outputs; skipping diff." >&2
    continue
  fi

  # Pick first two rows (distinct models) for a single diff
  M1=$(echo "${LINES[0]}" | awk -F'\t' '{print $2}')
  P1=$(echo "${LINES[0]}" | awk -F'\t' '{print $3}')
  M2=""
  P2=""
  for i in "${!LINES[@]}"; do
    mm=$(echo "${LINES[$i]}" | awk -F'\t' '{print $2}')
    pp=$(echo "${LINES[$i]}" | awk -F'\t' '{print $3}')
    if [[ "$mm" != "$M1" ]]; then M2="$mm"; P2="$pp"; break; fi
  done
  if [[ -z "$M2" ]]; then
    echo "[note] Range '$R' only produced one model; skipping diff." >&2
    continue
  fi

  SAFE_R=$(echo "$R" | tr ' ' '-' | tr '/' '-')
  DIFF_OUT="${COMPARE_DIR}/diff_${SAFE_R}_${M1}_vs_${M2}.diff"

  if command -v colordiff >/dev/null 2>&1; then
    colordiff -u "$P1" "$P2" > "$DIFF_OUT" || true
  else
    diff -u "$P1" "$P2" > "$DIFF_OUT" || true
  fi
  echo "[ok] wrote ${DIFF_OUT}" >&2
done

# ---------- index ----------
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