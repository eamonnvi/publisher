#!/usr/bin/env python3
# novel_toolkit_cli.py
import argparse, json, sys, time
from pathlib import Path
from datetime import datetime

from ntk_core import iter_sections, run_sync, write_markdown, write_jsonl
from ntk_prompts import PROMPTS
from ntk_batch import run_batch, fetch_batch, parse_batch_output_ndjson

def parse_args() -> argparse.Namespace:
    modes = sorted(PROMPTS.keys())

    p = argparse.ArgumentParser(description="Arbuthnot Books — Novel Toolkit (sync + minimal batch)")
    p.add_argument("draft", help="Input .md or .txt file")
    p.add_argument("--mode", required=True, choices=modes, help="Processing mode (from ntk_prompts.PROMPTS)")

    # sectioning
    p.add_argument("--heading-regex", default=r"(?m)^\s*#{3,6}\s+(.+?)\s*$",
                   help="Regex that captures ATX headings in group(1). Default matches ### .. ######")
    p.add_argument("--list-sections", action="store_true", help="List detected sections and exit")
    p.add_argument("--sections-first", type=int, metavar="N",
                   help="SYNC ONLY: process only the first N sections")
    p.add_argument("--whole", action="store_true",
                   help="Treat the entire file as a single section (bypass heading splitting)")

    # model/runtime
    p.add_argument("--model", default="gpt-4.1",
                   choices=["gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4.1-mini"])
    p.add_argument("--max-output-tokens", type=int, default=2000)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--retries", type=int, default=2)

    # outputs
    p.add_argument("--outdir", default="outputs",
                   help="Base directory to save outputs (default: ./outputs)")
    p.add_argument("--basename", default=None,
                   help="Base filename (default: draft stem + .mode.model)")
    p.add_argument("--run-tag", default=None,
                   help="Optional short tag included in the run subfolder name (e.g. 'pass1')")
    p.add_argument("--flat", action="store_true",
                   help="Do NOT create a timestamped subfolder; write directly into --outdir")

    # batch
    p.add_argument("--batch", action="store_true", help="Submit to OpenAI Batch API instead of running sync")
    p.add_argument("--fetch-id", default=None, help="Fetch a previously submitted batch id and write outputs")
    p.add_argument("--poll", action="store_true", help="When fetching, poll until completion")
    p.add_argument("--every", type=int, default=5, help="Polling interval seconds")
    p.add_argument("--wait", type=int, default=300, help="Max seconds to poll when --poll is set")

    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def _build_run_dir(outdir: Path, draft_path: Path, mode: str, model: str, flat: bool, run_tag: str | None) -> Path:
    stem = f"{(draft_path.stem)}.{mode}.{model}"
    base = Path(outdir)
    if flat:
        rd = base
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f".{run_tag}" if run_tag else ""
        rd = base / f"{stem}{tag}.{ts}"
    rd.mkdir(parents=True, exist_ok=True)
    return rd

def main():
    args = parse_args()
    draft_path = Path(args.draft)
    text = draft_path.read_text(encoding="utf-8")

    # list-only path
    if args.list_sections:
        sections_all = iter_sections(text, args.heading_regex)
        for i, (h, b) in enumerate(sections_all, 1):
            print(f"{i:03d} | {h} | {len((b or '').strip())} chars")
        return

    # build sections
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

    run_dir = _build_run_dir(Path(args.outdir), draft_path, args.mode, args.model, args.flat, args.run_tag)
    stem = (args.basename or f"{draft_path.stem}.{args.mode}.{args.model}")
    md_path = run_dir / f"{stem}.md"
    jl_path = run_dir / f"{stem}.jsonl"
    meta_path = run_dir / f"{stem}.meta.json"

    # ---- batch fetch path (id provided) ----
    if args.fetch_id:
        # read sidecar map if present (nice headings in the output)
        batch_map = {}
        sidecar = run_dir / "batch_map.json"
        if sidecar.exists():
            try:
                batch_map = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:
                batch_map = {}

        info = fetch_batch(args.fetch_id, run_dir, verbose=args.verbose, poll=args.poll, every_s=args.every, timeout_s=args.wait)
        st = info.get("status")
        if st != "completed":
            print(f"[warn] batch status = {st}. Fetch later.", file=sys.stderr)
            return

        raw = info.get("raw", b"")
        triples = parse_batch_output_ndjson(raw, batch_map)
        # write
        title = ("Copyedit Report" if args.mode.startswith("copyedit")
                 else "Concise Report" if args.mode.startswith("concise")
                 else args.mode.replace("_"," ").title())
        write_markdown([(h,t) for (h,t,_) in triples], md_path, title)
        write_jsonl([(h,t) for (h,t,_) in triples], jl_path)
        meta = {
            "draft": str(args.draft),
            "mode": args.mode,
            "model": args.model,
            "max_output_tokens": args.max_output_tokens,
            "whole": bool(args.whole),
            "num_sections": len(triples),
            "outdir": str(run_dir.resolve()),
            "outputs": {"markdown": md_path.name, "jsonl": jl_path.name},
            "batch_id": args.fetch_id,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if args.verbose:
            print(f"[ok] fetched batch into {run_dir}", file=sys.stderr)
        return

    # ---- batch submit path ----
    if args.batch:
        batch_id = run_batch(
            sections=sections,
            mode=args.mode,
            model=args.model,
            out_dir=run_dir,
            max_out=args.max_output_tokens,
            verbose=args.verbose,
        )
        print(f"[ok] submitted batch: {batch_id}", file=sys.stderr)
        # also write a tiny meta for convenience
        (run_dir / "submitted.json").write_text(
            json.dumps({
                "draft": str(args.draft),
                "mode": args.mode,
                "model": args.model,
                "max_output_tokens": args.max_output_tokens,
                "whole": bool(args.whole),
                "num_sections": len(sections),
                "batch_id": batch_id,
            }, indent=2),
            encoding="utf-8"
        )
        return

    # ---- sync path ----
    run_sync(
        sections=sections,
        mode=args.mode,
        model=args.model,
        timeout=args.timeout,
        max_out=args.max_output_tokens,   # ntk_core.run_sync expects max_out
        retries=args.retries,
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()