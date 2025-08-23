#!/usr/bin/env python3
# novel_toolkit_cli.py
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from ntk_core import iter_sections, run_sync_collect, write_markdown, write_jsonl
from ntk_prompts import PROMPTS


def parse_args() -> argparse.Namespace:
    modes = sorted(PROMPTS.keys())

    p = argparse.ArgumentParser(description="Arbuthnot Books — Novel Toolkit (sync CLI)")
    p.add_argument("draft", help="Input .md or .txt file")
    p.add_argument("--mode", required=True, choices=modes,
                   help="Processing mode (from ntk_prompts.PROMPTS)")
    p.add_argument("--whole", action="store_true",
                   help="Treat entire file as a single section (global prompt)")
    p.add_argument("--sections-first", type=int, metavar="N",
                   help="SYNC ONLY: process only the first N sections (ignored with --whole)")
    p.add_argument("--list-sections", action="store_true",
                   help="List detected sections and exit (ignored with --whole)")

    p.add_argument("--model", default="gpt-4.1",
                   choices=["gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4.1-mini"])
    p.add_argument("--max-output-tokens", type=int, default=2000)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--heading-regex", default=r"(?m)^\s*#{3,6}\s+(.+?)\s*$",
                   help="Regex that captures ATX headings in group(1). Default matches ###..######")

    p.add_argument("--out-dir", type=Path, default=Path("outputs"),
                   help="Base output directory (a timestamped subfolder will be created)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def build_out_dir(base: Path, draft_path: Path, mode: str, model: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    leaf = f"{draft_path.stem}.{mode}.{model}.{stamp}"
    out_dir = base / leaf
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main():
    args = parse_args()
    draft_path = Path(args.draft)
    text = draft_path.read_text(encoding="utf-8")

    # ---- sectioning (no accidental re-assignment below this point) ----------
    if args.whole:
        sections = [("Full Manuscript", text.strip())]
        if args.sections_first:
            if args.verbose:
                print("[note] --whole ignores --sections-first", file=sys.stderr)
    else:
        sections = iter_sections(text, args.heading_regex)
        if args.list_sections:
            for i, (h, b) in enumerate(sections, 1):
                print(f"{i:03d} | {h} | {len((b or '').strip())} chars")
            return
        if args.sections_first is not None:
            n = max(0, int(args.sections_first))
            sections = sections[:n]
            if args.verbose:
                print(f"[sync] selection: first {n} of {len(iter_sections(text, args.heading_regex))} total",
                      file=sys.stderr)

    # ---- run sync and write outputs -----------------------------------------
    if args.verbose:
        total = len(sections)
        print(f"[sync] selection: first {total} of {total} total", file=sys.stderr)

    triples = run_sync_collect(
        sections=sections,
        mode=args.mode,
        model=args.model,
        timeout=args.timeout,
        max_out=args.max_output_tokens,
        retries=args.retries,
        verbose=args.verbose,
    )

    out_dir = build_out_dir(args.out_dir, draft_path, args.mode, args.model)

    # Title + filenames
    title = ("Copyedit Report" if args.mode.startswith("copyedit")
             else "Concise Report")
    md_name = f"{draft_path.stem}.{args.mode}.{args.model}.md"
    jl_name = f"{draft_path.stem}.{args.mode}.{args.model}.jsonl"
    meta_name = f"{draft_path.stem}.{args.mode}.{args.model}.meta.json"

    # Write files
    write_markdown(triples, out_dir / md_name, title)
    write_jsonl(triples, out_dir / jl_name)

    meta = {
        "draft": str(draft_path),
        "mode": args.mode,
        "model": args.model,
        "max_output_tokens": args.max_output_tokens,
        "timeout": args.timeout,
        "retries": args.retries,
        "whole": args.whole,
        "sections_first": args.sections_first,
        "count": len(triples),
    }
    (out_dir / meta_name).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.verbose:
        print("[ok] wrote:", file=sys.stderr)
        print(f"  • {out_dir / md_name}", file=sys.stderr)
        print(f"  • {out_dir / jl_name}", file=sys.stderr)
        print(f"  • {out_dir / meta_name}", file=sys.stderr)


if __name__ == "__main__":
    main()
