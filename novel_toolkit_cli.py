#!/usr/bin/env python3
# novel_toolkit_cli.py
import argparse
import sys
from pathlib import Path

from ntk_core import iter_sections, run_sync
from ntk_prompts import PROMPTS  # to derive valid modes dynamically


def parse_args() -> argparse.Namespace:
    modes = sorted(PROMPTS.keys())  # e.g., concise, concise_para, concise_telegraphic, copyedit, critique, …

    p = argparse.ArgumentParser(description="Arbuthnot Books — Novel Toolkit (sync CLI)")
    p.add_argument("draft", help="Input .md or .txt file")
    p.add_argument("--mode", required=True, choices=modes, help="Processing mode (from ntk_prompts.PROMPTS)")
    p.add_argument("--sections-first", type=int, metavar="N",
                   help="SYNC ONLY: process only the first N sections")
    p.add_argument("--model", default="gpt-4.1",
                   choices=["gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4.1-mini"])
    p.add_argument("--max-output-tokens", type=int, default=2000)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--heading-regex", default=r"(?m)^\s*#{3,6}\s+(.+?)\s*$",
                   help="Regex that captures ATX headings in group(1). Default matches ### .. ######")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Read file right here (keeps ntk_core focused on API + logic)
    text = Path(args.draft).read_text(encoding="utf-8")

    # Slice into sections
    sections = iter_sections(text, args.heading_regex)

    # Optional sync-only slicer
    if args.sections_first is not None:
        n = max(0, int(args.sections_first))
        sections = sections[:n]
        if args.verbose:
            print(f"[sync] selection: first {n} of {len(sections)} total", file=sys.stderr)

    # Run synchronously (no batch here)
    run_sync(
        sections=sections,
        mode=args.mode,
        model=args.model,
        timeout=args.timeout,
        max_out=args.max_output_tokens,  # ntk_core.run_sync expects max_out
        retries=args.retries,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()