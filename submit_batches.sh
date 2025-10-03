#!/usr/bin/env bash
# submit_batches.sh — DEPRECATED shim → submit_ranges.sh
# Deprecated: 2025-10-03
# DEPRECATED: use submit_ranges.sh instead (this shim will be removed in a future release)

set -u

# Resolve the directory of this script so we call the sibling script reliably
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TARGET="$SCRIPT_DIR/submit_ranges.sh"

# Help passthrough
if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  cat <<'USAGE'
[DEPRECATED] submit_batches.sh → submit_ranges.sh

Use:
  ./submit_ranges.sh --draft FILE --mode MODE --models m1[,m2] --ranges "A B;C D"

All other flags are forwarded unchanged.
USAGE
  # Don’t exit yet: still forward so users see the canonical help
fi

# Guard: target must exist and be executable
if [[ ! -x "$TARGET" ]]; then
  echo "[error] submit_ranges.sh not found or not executable at: $TARGET" >&2
  exit 127
fi

echo "[warn] submit_batches.sh is deprecated. Forwarding to:"
echo "       $TARGET $*"

# Hand off control; final exit code will be that of submit_ranges.sh
exec "$TARGET" "$@"
