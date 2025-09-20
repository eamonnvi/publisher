#!/usr/bin/env bash
echo "[warn] submit_batches.sh is deprecated. Use: ./submit_ranges.sh --draft FILE --mode MODE --models m1[,m2] --ranges \"A B;C D\""
exec ./submit_ranges.sh "$@"
