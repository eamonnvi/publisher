#!/usr/bin/env python3
# novel_toolkit_cli.py
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

from ntk_core import (
    iter_sections,
    run_sync_collect,
    write_markdown,
    write_jsonl,
)
from ntk_prompts import PROMPTS  # derive valid modes dynamically


def parse_args() -> argparse.Namespace:
    modes = sorted(PROMPTS.keys())

    p = argparse.ArgumentParser(description="Arbuthnot Books — Novel Toolkit (sync CLI)")
    p.add_argument("draft", help="Input .md or .txt file")

    p.add_argument("--mode", required=True, choices=modes,
                   help="Processing mode (from ntk_prompts.PROMPTS)")
    p.add_argument("--model", default="gpt-4.1",
                   choices=["gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4.1-mini"])
    p.add_argument("--max-output-tokens", type=int, default=2000)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--retries", type=int, default=2)

    # Slicing / splitting
    p.add_argument("--heading-regex", default=r"(?m)^\s*#{3,6}\s+(.+?)\s*$",
                   help="Regex that captures ATX headings in group(1). Default matches ### .. ######")
    p.add_argument("--sections-first", type=int, metavar="N",
                   help="SYNC ONLY: process only the first N sections")
    p.add_argument("--whole", action="store_true",
                   help="Treat the entire file as a single section (bypass heading splitting)")

    # Output directorying
    p.add_argument("--outdir", default="outputs",
                   help="Base directory to save outputs (default: ./outputs)")
    p.add_argument("--basename", default=None,
                   help="Base filename (default: draft stem + .mode.model)")
    p.add_argument("--run-tag", default=None,
                   help="Optional short tag included in the run subfolder name (e.g. 'pass1')")
    p.add_argument("--flat", action="store_true",
                   help="Do NOT create a timestamped subfolder; write directly into --outdir")

    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Read input
    draft_path = Path(args.draft)
    text = draft_path.read_text(encoding="utf-8")

    # Build sections
    if args.whole:
        sections = [("Full Manuscript", text.strip())]
        if args.sections_first and args.verbose:
            print("[note] --whole ignores --sections-first", file=sys.stderr)
    else:
        sections = iter_sections(text, args.heading_regex)
        if args.sections_first is not None:
            n = max(0, int(args.sections_first))
            sections = sections[:n]
            if args.verbose:
                print(f"[sync] selection: first {n} of {len(sections)} total", file=sys.stderr)

    # Run sync (collect results)
    triples = run_sync_collect(
        sections=sections,
        mode=args.mode,
        model=args.model,
        timeout=args.timeout,
        max_out=args.max_output_tokens,
        retries=args.retries,
        verbose=args.verbose,
    )

    # Naming
    stem = args.basename or f"{draft_path.stem}.{args.mode}.{args.model}"

    # Decide final directory
    base_out = Path(args.outdir)
    if args.flat:
        run_dir = base_out
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f".{args.run_tag}" if args.run_tag else ""
        run_dir = base_out / f"{stem}{tag}.{ts}"

    run_dir.mkdir(parents=True, exist_ok=True)

    md_path = run_dir / f"{stem}.md"
    jsonl_path = run_dir / f"{stem}.jsonl"
    meta_path = run_dir / f"{stem}.meta.json"

    # Title by mode
    title = ("Copyedit Report" if args.mode.startswith("copyedit")
             else "Concise Report" if args.mode.startswith("concise")
             else args.mode.replace("_", " ").title())

    # Write files
    write_markdown(triples, md_path, title)
    write_jsonl(triples, jsonl_path)

    meta = {
        "draft": str(draft_path),
        "mode": args.mode,
        "model": args.model,
        "max_output_tokens": args.max_output_tokens,
        "timeout": args.timeout,
        "retries": args.retries,
        "whole": bool(args.whole),
        "sections_first": args.sections_first,
        "num_sections": len(sections),
        "outdir": str(run_dir.resolve()),
        "outputs": {
            "markdown": md_path.name,
            "jsonl": jsonl_path.name,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.verbose:
        print(f"[ok] wrote:", file=sys.stderr)
        print(f"  • {md_path}", file=sys.stderr)
        print(f"  • {jsonl_path}", file=sys.stderr)
        print(f"  • {meta_path}", file=sys.stderr)


if __name__ == "__main__":
    main()