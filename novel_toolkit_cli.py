#!/usr/bin/env python3
# novel_toolkit_cli.py
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from openai import OpenAI

from ntk_core import iter_sections, run_sync_collect, write_markdown, write_jsonl
from ntk_prompts import PROMPTS  # derive valid modes dynamically
from ntk_batch import (
    run_batch,
    get_batch_status,
    poll_batch,
    _download_file_bytes,
    parse_batch_output_ndjson,
)


def parse_args() -> argparse.Namespace:
    modes = sorted(PROMPTS.keys())

    p = argparse.ArgumentParser(description="Arbuthnot Books — Novel Toolkit (sync + batch)")
    p.add_argument("draft", help="Input .md or .txt file")

    # What to do
    p.add_argument("--mode", required=True, choices=modes,
                   help="Processing mode (from ntk_prompts.PROMPTS)")
    p.add_argument("--list-sections", action="store_true", help="List detected sections and exit")

    # Slicing / splitting
    p.add_argument("--heading-regex", default=r"(?m)^\s*#{3,6}\s+(.+?)\s*$",
                   help="Regex that captures ATX headings in group(1). Default matches ### .. ######")
    p.add_argument("--sections-first", type=int, metavar="N",
                   help="SYNC ONLY: process only the first N sections")
    p.add_argument("--whole", action="store_true",
                   help="Treat the entire file as a single section (bypass heading splitting)")

    # Models / generation
    p.add_argument("--model", default="gpt-4.1",
                   choices=["gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4.1-mini"])
    p.add_argument("--max-output-tokens", type=int, default=2000)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--retries", type=int, default=2)

    # Output directorying
    p.add_argument("--outdir", default="outputs",
                   help="Base directory to save outputs (default: ./outputs)")
    p.add_argument("--basename", default=None,
                   help="Base filename (default: draft stem + .mode.model)")
    p.add_argument("--run-tag", default=None,
                   help="Optional short tag included in the run subfolder name (e.g. 'pass1')")
    p.add_argument("--flat", action="store_true",
                   help="Do NOT create a timestamped subfolder; write directly into --outdir")

    # Batch controls
    p.add_argument("--batch", action="store_true", help="Submit as OpenAI Batch job instead of sync")
    p.add_argument("--fetch-id", default=None, help="Fetch output for a previously submitted batch id")
    p.add_argument("--poll", action="store_true", help="Poll the batch until completion before fetching")
    p.add_argument("--wait", type=int, default=300, help="Total seconds to wait when --poll is used")
    p.add_argument("--every", type=int, default=5, help="Polling interval seconds")

    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _build_run_dir(base_out: Path, stem: str, flat: bool, run_tag: str | None) -> Path:
    if flat:
        run_dir = base_out
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f".{run_tag}" if run_tag else ""
        run_dir = base_out / f"{stem}{tag}.{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    args = parse_args()
    draft_path = Path(args.draft)
    text = draft_path.read_text(encoding="utf-8")

    # Quick list
    if args.list_sections:
        secs = iter_sections(text, args.heading_regex)
        for i, (h, b) in enumerate(secs, 1):
            print(f"{i:03d} | {h} | {len((b or '').strip())} chars")
        return

    # Sections (whole vs split)
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

    # Names & output paths
    base_out = Path(args.outdir)
    stem = args.basename or f"{draft_path.stem}.{args.mode}.{args.model}"
    run_dir = _build_run_dir(base_out, stem, args.flat, args.run_tag)
    md_path = run_dir / f"{stem}.md"
    jsonl_path = run_dir / f"{stem}.jsonl"
    meta_path = run_dir / f"{stem}.meta.json"
    submitted_path = run_dir / "submitted.json"
    batch_map_path = run_dir / "batch_map.json"

    # Batch fetch path (if the user provided a batch id to fetch)
    if args.fetch_id:
        client = OpenAI()
        if args.poll:
            info = poll_batch(client, args.fetch_id, timeout_s=args.wait, every_s=args.every, verbose=args.verbose)
        else:
            info = get_batch_status(client, args.fetch_id)
        st = info.get("status")
        if st != "completed":
            print(f"[warn] batch status = {st}. Try again later.", file=sys.stderr)
            return
        out_id = info.get("output_file_id")
        if not out_id:
            print("[error] completed but no output_file_id", file=sys.stderr)
            return

        # Load id→heading map if present
        cid_to_heading = {}
        if batch_map_path.exists():
            try:
                cid_to_heading = json.loads(batch_map_path.read_text(encoding="utf-8"))
            except Exception:
                cid_to_heading = {}

        raw = _download_file_bytes(client, out_id)
        (run_dir / "batch_output.raw.ndjson").write_bytes(raw)
        triples = parse_batch_output_ndjson(raw, cid_to_heading or None)

        title = ("Copyedit Report" if args.mode.startswith("copyedit")
                 else "Concise Report" if args.mode.startswith("concise")
                 else args.mode.replace("_", " ").title())
        # Write files
        write_markdown([(h, t) for (h, t, _) in triples], md_path, title)
        write_jsonl([(h, t) for (h, t, _) in triples], jsonl_path)
        meta = {
            "draft": str(args.draft),
            "mode": args.mode,
            "model": args.model,
            "max_output_tokens": args.max_output_tokens,
            "whole": bool(args.whole),
            "fetch_id": args.fetch_id,
            "num_sections": len(triples),
            "outdir": str(run_dir.resolve()),
            "outputs": {"markdown": md_path.name, "jsonl": jsonl_path.name},
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if args.verbose:
            print(f"[ok] wrote:", file=sys.stderr)
            print(f"  • {md_path}", file=sys.stderr)
            print(f"  • {jsonl_path}", file=sys.stderr)
            print(f"  • {meta_path}", file=sys.stderr)
        return

    # Batch submit path
    if args.batch:
        b_id = run_batch(
            sections=sections,
            mode=args.mode,
            model=args.model,
            out_dir=run_dir,
            max_output_tokens=args.max_output_tokens,
            verbose=args.verbose,
        )
        # Sidecar submission info for later fetch
        submitted = {
            "draft": args.draft,
            "mode": args.mode,
            "model": args.model,
            "max_output_tokens": args.max_output_tokens,
            "whole": bool(args.whole),
            "num_sections": len(sections),
            "batch_id": b_id,
        }
        submitted_path.write_text(json.dumps(submitted, indent=2), encoding="utf-8")
        if args.verbose:
            print(f"[ok] submitted batch: {b_id}", file=sys.stderr)
            print(f"[info] wrote:", file=sys.stderr)
            print(f"  • {submitted_path}", file=sys.stderr)
            if (run_dir / "batch_input.jsonl").exists():
                print(f"  • {run_dir / 'batch_input.jsonl'}", file=sys.stderr)
            if (run_dir / "batch_map.json").exists():
                print(f"  • {run_dir / 'batch_map.json'}", file=sys.stderr)
        return

    # Sync path
    triples = run_sync_collect(
        sections=sections,
        mode=args.mode,
        model=args.model,
        timeout=args.timeout,
        max_out=args.max_output_tokens,
        retries=args.retries,
        verbose=args.verbose,
    )

    # Title by mode
    title = ("Copyedit Report" if args.mode.startswith("copyedit")
             else "Concise Report" if args.mode.startswith("concise")
             else args.mode.replace("_", " ").title())

    # Write files
    write_markdown(triples, md_path, title)
    write_jsonl(triples, jsonl_path)
    meta = {
        "draft": str(args.draft),
        "mode": args.mode,
        "model": args.model,
        "max_output_tokens": args.max_output_tokens,
        "whole": bool(args.whole),
        "num_sections": len(triples),
        "outdir": str(run_dir.resolve()),
        "outputs": {"markdown": md_path.name, "jsonl": jsonl_path.name},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.verbose:
        print(f"[ok] wrote:", file=sys.stderr)
        print(f"  • {md_path}", file=sys.stderr)
        print(f"  • {jsonl_path}", file=sys.stderr)
        print(f"  • {meta_path}", file=sys.stderr)


if __name__ == "__main__":
    main()