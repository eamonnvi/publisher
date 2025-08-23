#!/usr/bin/env python3
"""
Fetch a completed OpenAI Batch, parse outputs, and write Markdown/JSONL.

Usage:
  python3 fetch_batch.py \
    --batch-id BATCH_... \
    --run-dir /path/to/outputs/TPV-... \
    --wait 120 --every 5 \
    --format both \
    --title "Copyedit Report" \
    --verbose
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI

# Reuse your toolkit helpers
from ntk_batch import poll_batch, get_batch_status, _download_file_bytes, parse_batch_output_ndjson
from ntk_core import write_markdown, write_jsonl


def _load_batch_map(path: Path) -> Dict[str, str]:
    """
    Load custom_id -> heading map if present.
    Return {} if missing or unreadable.
    """
    try:
        import json
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        sys.stderr.write(f"[warn] failed to load batch_map.json: {e}\n")
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch and render OpenAI Batch results.")
    ap.add_argument("--batch-id", required=True, help="The batch id to fetch (e.g., batch_abc123...)")
    ap.add_argument("--run-dir", required=True, type=Path, help="Directory where batch_input.jsonl and batch_map.json live")
    ap.add_argument("--wait", type=int, default=0, help="Poll for up to N seconds until completion")
    ap.add_argument("--every", type=int, default=5, help="Polling interval (seconds) when --wait is set")
    ap.add_argument("--format", choices=["md", "json", "both"], default="both", help="Output format(s) to write")
    ap.add_argument("--title", default="Batch Results", help="Report title for Markdown")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging to stderr")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # Client
    client = OpenAI()

    # Load mapping (if present)
    batch_map = _load_batch_map(run_dir / "batch_map.json")

    # Get current status (and optionally poll)
    info = get_batch_status(client, args.batch_id)
    status = info.get("status")

    if args.verbose:
        sys.stderr.write(f"[status] {args.batch_id}: {status}\n")

    if status != "completed" and args.wait:
        info = poll_batch(client, args.batch_id, timeout_s=args.wait, every_s=max(1, args.every), verbose=args.verbose)
        status = info.get("status")

    if status != "completed":
        sys.stderr.write(f"[warn] batch status is '{status}'. Try again later or increase --wait.\n")
        sys.exit(2)

    # Download raw NDJSON
    out_id = info.get("output_file_id")
    if not out_id:
        sys.stderr.write("[error] completed but no output_file_id in batch status.\n")
        sys.exit(2)

    ndjson_bytes = _download_file_bytes(client, out_id)
    raw_path = run_dir / "batch_output.raw.ndjson"
    raw_path.write_bytes(ndjson_bytes)
    if args.verbose:
        sys.stderr.write(f"[ok] saved raw NDJSON -> {raw_path}\n")

    # Parse into triples: [(heading, text, model)]
    triples = parse_batch_output_ndjson(ndjson_bytes, batch_map)

    # Write outputs
    # For md/json convenience, we drop the model field when writing
    md_pairs: List[Tuple[str, str]] = [(h, t) for (h, t, _m) in triples]

    wrote = []

    if args.format in {"md", "both"}:
        md_path = run_dir / "results.md"
        write_markdown(md_pairs, md_path, args.title)
        wrote.append(str(md_path))

    if args.format in {"json", "both"}:
        jsonl_path = run_dir / "results.jsonl"
        write_jsonl(md_pairs, jsonl_path)
        wrote.append(str(jsonl_path))

    if wrote:
        sys.stderr.write("[ok] wrote:\n")
        for p in wrote:
            sys.stderr.write(f"  • {p}\n")


if __name__ == "__main__":
    main()
