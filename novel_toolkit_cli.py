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
    run_sync_collect,
    write_markdown,
    write_jsonl,
)
from ntk_batch import (
    run_batch,                # returns: (ndjson_path, batch_map, endpoint, batch_id)
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
    p.add_argument("--mode", required=True, choices=modes,
                   help="Processing mode (from ntk_prompts.PROMPTS)")

    # Discovery
    p.add_argument("--list-sections", action="store_true",
                   help="List detected sections and exit")

    # Slicers (ignored with --whole). Priority: --section > --sections-range > --sections-first
    p.add_argument("--section", type=int, metavar="N",
                   help="Process only section N (1-based). Ignored with --whole.")
    p.add_argument("--sections-range", nargs=2, type=int, metavar=("START", "END"),
                   help="Process inclusive range START..END (1-based). Ignored with --whole.")
    p.add_argument("--sections-first", type=int, metavar="N",
                   help="Process only the first N sections. Ignored with --whole.")
    p.add_argument("--whole", action="store_true",
                   help="Treat the entire file as a single section (bypass heading splitting)")

    # Model/runtime
    p.add_argument("--model", default="gpt-4.1",
                   choices=["gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4.1-mini"])
    p.add_argument("--max-output-tokens", type=int, default=2000)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--heading-regex", default=r"(?m)^\s*#{3,6}\s+(.+?)\s*$",
                   help="Regex that captures ATX headings in group(1). Default matches ### .. ######")
    p.add_argument("--verbose", action="store_true")

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


def _build_run_dir(draft_path: Path, args: argparse.Namespace) -> Path:
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
    text = draft_path.read_text(encoding="utf-8")

    # List sections fast path
    if args.list_sections:
        sections_all = iter_sections(text, args.heading_regex)
        for i, (h, b) in enumerate(sections_all, 1):
            print(f"{i:03d} | {h} | {len((b or '').strip())} chars")
        return

    # Build sections
    if args.whole:
        sections_all = [("Full Manuscript", text.strip())]
        if (args.section is not None) or args.sections_range or args.sections_first:
            if args.verbose:
                print("[note] --whole ignores --section/--sections-range/--sections-first", file=sys.stderr)
        sections = sections_all
    else:
        sections_all = iter_sections(text, args.heading_regex)
        sections = _apply_section_slicer(sections_all, args, verbose=args.verbose)

    # Output folder
    run_dir = _build_run_dir(draft_path, args)
    stem = args.basename or f"{draft_path.stem}.{args.mode}.{args.model}"
    md_path = run_dir / f"{stem}.md"
    jsonl_path = run_dir / f"{stem}.jsonl"
    meta_path = run_dir / f"{stem}.meta.json"

    # Batch fetch path (skip submit)
    if args.fetch_id:
        client = OpenAI()
        batch_id = args.fetch_id
        if args.poll:
            info = poll_batch(client, batch_id, timeout_s=args.wait, every_s=max(1, args.every), verbose=args.verbose)
        else:
            info = get_batch_status(client, batch_id)

        st = info.get("status")
        if st != "completed":
            print(f"[warn] batch status = {st}. Use --poll or fetch later.", file=sys.stderr)
            # still dump meta to track
            meta = {
                "draft": str(args.draft),
                "mode": args.mode,
                "model": args.model,
                "max_output_tokens": args.max_output_tokens,
                "whole": bool(args.whole),
                "num_sections": len(sections),
                "outdir": str(run_dir.resolve()),
                "batch_id": batch_id,
                "batch_status": st,
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return

        out_id = info.get("output_file_id")
        if not out_id:
            print("[error] completed but no output_file_id", file=sys.stderr)
            return

        raw = _download_file_bytes(client, out_id)
        (run_dir / "batch_output.raw.ndjson").write_bytes(raw)

        # Try to load cid->heading map if present
        cid_map_path = run_dir / "batch_map.json"
        cid_map = None
        if cid_map_path.exists():
            try:
                cid_map = json.loads(cid_map_path.read_text(encoding="utf-8"))
            except Exception:
                cid_map = None

        triples = parse_batch_output_ndjson(raw, cid_map or {})
        # Write outputs
        title = ("Copyedit Report" if args.mode.startswith("copyedit")
                 else "Concise Report" if args.mode.startswith("concise")
                 else args.mode.replace("_", " ").title())
        write_markdown([(h, t) for (h, t, *_rest) in triples], md_path, title)
        write_jsonl([(h, t) for (h, t, *_rest) in triples], jsonl_path)

        meta = {
            "draft": str(args.draft),
            "mode": args.mode,
            "model": args.model,
            "max_output_tokens": args.max_output_tokens,
            "whole": bool(args.whole),
            "num_sections": len(sections),
            "outdir": str(run_dir.resolve()),
            "batch_id": batch_id,
            "batch_status": "completed",
            "outputs": {"markdown": md_path.name, "jsonl": jsonl_path.name},
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if args.verbose:
            print(f"[ok] fetched batch into {run_dir}", file=sys.stderr)
        return

    # --- BATCH PATH ---
    if args.batch or args.fetch_id:
        # Build run directory (same scheme as sync)
        stem = args.basename or f"{Path(args.draft).stem}.{args.mode}.{args.model}"
        base_out = Path(args.outdir)
        if args.flat:
            run_dir = base_out
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            tag = f".{args.run_tag}" if args.run_tag else ""
            run_dir = base_out / f"{stem}{tag}.{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        client = OpenAI()

        if args.fetch_id:
            batch_id = args.fetch_id
            if args.verbose:
                print(f"[batch] fetching existing batch: {batch_id}", file=sys.stderr)
        else:
            # Submit ONCE
            ret = run_batch(
                sections=sections,
                mode=args.mode,
                model=args.model,
                out_dir=run_dir,
                max_output_tokens=args.max_output_tokens,
                verbose=args.verbose,
            )
            # Accept 4-tuple or plain string
            if isinstance(ret, tuple):
                # (ndjson_path, batch_map, endpoint, batch_id)
                if len(ret) != 4:
                    raise RuntimeError(f"Unexpected run_batch() return arity: {len(ret)}")
                ndjson_path, batch_map, endpoint, batch_id = ret
            elif isinstance(ret, str):
                batch_id = ret
            else:
                raise RuntimeError(f"Unexpected run_batch() return type: {type(ret).__name__}")
            # (ntk_batch.run_batch already printed “[ok] submitted batch: …”)

        # Optional polling
        if args.poll:
            if not args.wait:
                args.wait = 300
            info = poll_batch(client, batch_id, timeout_s=args.wait, every_s=max(1, args.every), verbose=args.verbose)
        else:
            info = get_batch_status(client, batch_id)

        status = info.get("status")
        if status != "completed":
            print(f"[warn] batch status = {status}. Fetch later with --fetch-id {batch_id}.", file=sys.stderr)
            # Also write a tiny meta file so you have the ID recorded
            (run_dir / "submitted.json").write_text(
                json.dumps({
                    "draft": str(args.draft),
                    "mode": args.mode,
                    "model": args.model,
                    "max_output_tokens": args.max_output_tokens,
                    "whole": bool(args.whole),
                    "num_sections": len(sections),
                    "batch_id": batch_id,
                    "status": status,
                }, indent=2),
                encoding="utf-8"
            )
            sys.exit(0)

        # Completed → download + render
        out_id = info.get("output_file_id")
        if not out_id:
            print("[error] completed but no output_file_id", file=sys.stderr)
            sys.exit(1)

        raw = _download_file_bytes(client, out_id)
        (run_dir / "batch_output.raw.ndjson").write_bytes(raw)

        # Read the id→heading map if present
        batch_map_path = run_dir / "batch_map.json"
        batch_map = {}
        if batch_map_path.exists():
            try:
                batch_map = json.loads(batch_map_path.read_text(encoding="utf-8"))
            except Exception:
                batch_map = {}

        triples = parse_batch_output_ndjson(raw, batch_map)

        # Save outputs
        title = ("Copyedit Report" if args.mode.startswith("copyedit")
                 else "Concise Report" if args.mode.startswith("concise")
                 else args.mode.replace("_", " ").title())
        md_path = run_dir / f"{stem}.md"
        jsonl_path = run_dir / f"{stem}.jsonl"
        meta_path = run_dir / f"{stem}.meta.json"

        write_markdown([(h, t) for (h, t, _) in triples], md_path, title)
        write_jsonl([(h, t) for (h, t, _) in triples], jsonl_path)

        meta = {
            "draft": str(args.draft),
            "mode": args.mode,
            "model": args.model,
            "max_output_tokens": args.max_output_tokens,
            "whole": bool(args.whole),
            "num_sections": len(sections),
            "batch_id": batch_id,
            "outdir": str(run_dir.resolve()),
            "outputs": {
                "markdown": md_path.name,
                "jsonl": jsonl_path.name,
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if args.verbose:
            print("[ok] wrote:", file=sys.stderr)
            print(f"  • {md_path}", file=sys.stderr)
            print(f"  • {jsonl_path}", file=sys.stderr)
            print(f"  • {meta_path}", file=sys.stderr)
        sys.exit(0)

if __name__ == "__main__":
    main()