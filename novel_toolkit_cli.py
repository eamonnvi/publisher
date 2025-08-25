#!/usr/bin/env python3
# novel_toolkit_cli.py
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from ntk_prompts import PROMPTS  # derive valid modes dynamically
from ntk_core import (
    iter_sections,
    run_sync,
    run_sync_collect,
    write_markdown,
    write_jsonl,
)
from ntk_batch import (
    run_batch,
    get_batch_status,
    poll_batch,
    _download_file_bytes,
    parse_batch_output_ndjson,
)
from openai import OpenAI


def parse_args() -> argparse.Namespace:
    modes = sorted(PROMPTS.keys())

    p = argparse.ArgumentParser(description="Arbuthnot Books — Novel Toolkit (sync + batch CLI)")
    p.add_argument("draft", help="Input .md or .txt file")
    p.add_argument("--mode", required=True, choices=modes, help="Processing mode (from ntk_prompts.PROMPTS)")
    p.add_argument("--list-sections", action="store_true", help="List detected sections and exit")

    # Slicers (ignored with --whole). Priority: --section > --sections-range > --sections-first
    p.add_argument("--section", type=int, metavar="N",
                   help="Process only section N (1-based). Ignored with --whole.")
    p.add_argument("--sections-range", nargs=2, type=int, metavar=("START", "END"),
                   help="Process inclusive range START..END (1-based). Ignored with --whole.")
    p.add_argument("--sections-first", type=int, metavar="N",
                   help="Process only the first N sections. Ignored with --whole.")

    p.add_argument("--model", default="gpt-4.1",
                   choices=["gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4.1-mini"])
    p.add_argument("--max-output-tokens", type=int, default=2000)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--heading-regex", default=r"(?m)^\s*#{3,6}\s+(.+?)\s*$",
                   help="Regex that captures ATX headings in group(1). Default matches ### .. ######")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--whole", action="store_true",
                   help="Treat the entire file as a single section (bypass heading splitting)")

    # Output destinations
    p.add_argument("--outdir", default="outputs",
                   help="Base directory to save outputs (default: ./outputs)")
    p.add_argument("--basename", default=None,
                   help="Base filename (default: draft stem + .mode.model)")
    p.add_argument("--run-tag", default=None,
                   help="Optional tag included in the run subfolder name (e.g. 'pass1')")
    p.add_argument("--flat", action="store_true",
                   help="Do NOT create a timestamped subfolder; write directly into --outdir")

    # Batch controls
    p.add_argument("--batch", action="store_true", help="Submit work via OpenAI Batch API")
    p.add_argument("--fetch-id", help="Fetch a previously-submitted batch by id (skips submit)")
    p.add_argument("--poll", action="store_true", help="Poll until the fetched batch completes")
    p.add_argument("--every", type=int, default=5, help="Polling interval seconds")
    p.add_argument("--wait", type=int, help="Max seconds to wait when --poll is set")

    return p.parse_args()


def _apply_section_slicer(
    all_sections: List[Tuple[str, str]],
    args: argparse.Namespace,
    verbose: bool = False
) -> List[Tuple[str, str]]:
    """
    Apply exactly one slicer in priority order:
      --section > --sections-range > --sections-first
    """
    total = len(all_sections)
    if args.section is not None:
        n = max(1, int(args.section))
        if n > total:
            print(f"[error] --section {n} > total sections {total}", file=sys.stderr)
            return []
        out = [all_sections[n-1]]
        if verbose:
            print(f"[sync] selection: section {n} of {total} total", file=sys.stderr)
        return out

    if args.sections_range:
        a, b = args.sections_range
        a = max(1, int(a)); b = max(a, int(b))
        if a > total:
            print(f"[error] --sections-range start {a} > total sections {total}", file=sys.stderr)
            return []
        out = all_sections[a-1:b]
        if verbose:
            print(f"[sync] selection: range {a}..{a+len(out)-1} of {total} total", file=sys.stderr)
        return out

    if args.sections_first is not None:
        n = max(0, int(args.sections_first))
        out = all_sections[:n]
        if verbose:
            print(f"[sync] selection: first {len(out)} of {total} total", file=sys.stderr)
        return out

    # default: everything
    if verbose:
        print(f"[sync] selection: first {total} of {total} total", file=sys.stderr)
    return all_sections


def _compute_run_dir(draft_path: Path, args: argparse.Namespace) -> Path:
    """Create (or return) the run directory for outputs."""
    stem = args.basename or f"{draft_path.stem}.{args.mode}.{args.model}"
    base_out = Path(args.outdir)

    if args.flat:
        run_dir = base_out
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f".{args.run_tag}" if args.run_tag else ""
        run_dir = base_out / f"{stem}{tag}.{ts}"

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    args = parse_args()
    draft_path = Path(args.draft)

    # -------------------------------
    # FETCH-ONLY PATH (no splitting)
    # -------------------------------
    if args.fetch_id:
        # Use --outdir as the destination (recommended to point to the original submit folder)
        run_dir = Path(args.outdir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Load optional id→heading map
        map_path = run_dir / "batch_map.json"
        batch_map = {}
        if map_path.exists():
            try:
                batch_map = json.loads(map_path.read_text(encoding="utf-8"))
            except Exception:
                batch_map = {}
        else:
            if args.verbose:
                print("[warn] batch_map.json not found; headings will fall back to custom_id", file=sys.stderr)

        client = OpenAI()

        if args.poll:
            if args.verbose:
                print(f"[batch] polling {args.fetch_id} until completion…", file=sys.stderr)
            info = poll_batch(client, args.fetch_id, timeout_s=args.wait or 0, every_s=max(1, args.every), verbose=args.verbose)
        else:
            info = get_batch_status(client, args.fetch_id)

        st = info.get("status")
        # Always write a submitted.json (or update it) in the fetch dir for traceability
        (run_dir / "submitted.json").write_text(
            json.dumps({
                "draft": str(draft_path.name),
                "mode": args.mode,
                "model": args.model,
                "max_output_tokens": args.max_output_tokens,
                "whole": bool(args.whole),
                "batch_id": args.fetch_id,
                "batch_status": st,
            }, indent=2),
            encoding="utf-8",
        )

        if st != "completed":
            print(f"[warn] batch status = {st}. Try again later or use --poll.", file=sys.stderr)
            return

        out_id = info.get("output_file_id")
        if not out_id:
            print("[error] completed but no output_file_id", file=sys.stderr)
            return

        raw = _download_file_bytes(client, out_id)
        (run_dir / "batch_output.raw.ndjson").write_bytes(raw)

        triples = parse_batch_output_ndjson(raw, batch_map)

        # Build output filenames and title
        stem = args.basename or f"{draft_path.stem}.{args.mode}.{args.model}"
        md_path = run_dir / f"{stem}.md"
        jsonl_path = run_dir / f"{stem}.jsonl"
        meta_path = run_dir / f"{stem}.meta.json"
        title = ("Copyedit Report" if args.mode.startswith("copyedit")
                 else "Concise Report" if args.mode.startswith("concise")
                 else args.mode.replace("_", " ").title())

        write_markdown([(h, t) for (h, t, _) in triples], md_path, title)
        write_jsonl([(h, t) for (h, t, _) in triples], jsonl_path)

        meta = {
            "draft": str(draft_path.name),
            "mode": args.mode,
            "model": args.model,
            "max_output_tokens": args.max_output_tokens,
            "batch_id": args.fetch_id,
            "batch_status": st,
            "outdir": str(run_dir.resolve()),
            "num_items": len(triples),
            "outputs": {
                "markdown": md_path.name,
                "jsonl": jsonl_path.name,
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if args.verbose:
            print(f"[ok] fetched batch into {run_dir}", file=sys.stderr)
        return

    # --------------------------------
    # NORMAL (SYNC or SUBMIT) PATHS
    # --------------------------------

    # Read input (needed for list/sync/submit)
    text = draft_path.read_text(encoding="utf-8")

    # --list-sections (fast path) — always based on actual headings
    if args.list_sections:
        sections_all = iter_sections(text, args.heading_regex)
        for i, (h, b) in enumerate(sections_all, 1):
            print(f"{i:03d} | {h} | {len((b or '').strip())} chars")
        return

    # Build the section list
    if args.whole:
        sections = [("Full Manuscript", text.strip())]
        if args.sections_first or args.section or args.sections_range:
            if args.verbose:
                print("[note] --whole ignores --sections-first/--section/--sections-range", file=sys.stderr)
    else:
        all_sections = iter_sections(text, args.heading_regex)
        sections = _apply_section_slicer(all_sections, args, verbose=args.verbose)

    # Submit via Batch API
    if args.batch:
        # Compute run_dir first; we’ll write into it and save submitted.json here
        run_dir = _compute_run_dir(draft_path, args)

        # Submit the batch (run_batch writes batch_input.jsonl and batch_map.json)
        batch_id = run_batch(
            sections=sections,
            mode=args.mode,
            model=args.model,
            out_dir=run_dir,
            max_output_tokens=args.max_output_tokens,
            verbose=args.verbose,
        )
        # Persist submit metadata
        (run_dir / "submitted.json").write_text(
            json.dumps({
                "draft": str(draft_path.name),
                "mode": args.mode,
                "model": args.model,
                "max_output_tokens": args.max_output_tokens,
                "whole": bool(args.whole),
                "num_sections": len(sections),
                "batch_id": batch_id,
            }, indent=2),
            encoding="utf-8",
        )

        # Don’t fetch here; user can run a separate fetch with --fetch-id and the same --outdir
        if args.verbose:
            print(f"[warn] batch status = validating. Fetch later with --fetch-id {batch_id}.", file=sys.stderr)
        return

    # Synchronous mode (no batch)
    triples = run_sync_collect(
        sections=sections,
        mode=args.mode,
        model=args.model,
        timeout=args.timeout,
        max_out=args.max_output_tokens,
        retries=args.retries,
        verbose=args.verbose,
    )

    run_dir = _compute_run_dir(draft_path, args)
    stem = args.basename or f"{draft_path.stem}.{args.mode}.{args.model}"
    md_path = run_dir / f"{stem}.md"
    jsonl_path = run_dir / f"{stem}.jsonl"
    meta_path = run_dir / f"{stem}.meta.json"

    title = ("Copyedit Report" if args.mode.startswith("copyedit")
             else "Concise Report" if args.mode.startswith("concise")
             else args.mode.replace("_", " ").title())

    write_markdown(triples, md_path, title)
    write_jsonl(triples, jsonl_path)

    meta = {
        "draft": str(draft_path.name),
        "mode": args.mode,
        "model": args.model,
        "max_output_tokens": args.max_output_tokens,
        "whole": bool(args.whole),
        "num_sections": len(triples),
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