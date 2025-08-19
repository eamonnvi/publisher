# novel_toolkit_cli.py
import argparse, json, sys
from pathlib import Path
import ntk_core as core
import ntk_batch as bt

def parse_args():
    p = argparse.ArgumentParser("NTK CLI")
    p.add_argument("draft", nargs="?")
    p.add_argument("--mode", choices=["concise","discursive","critique","entities","continuity","overview","inline-edit","copyedit","improvement_suggestions","detailed-summary","sensitivity-analysis","plot-points"], default="concise")
    p.add_argument("--format", choices=["md","json","both"], default="md")
    p.add_argument("--model", default="gpt-4.1-mini", choices=["gpt-5","gpt-5-mini","gpt-4.1","gpt-4.1-mini"])
    p.add_argument("--engine", choices=["auto","responses","chat"], default="auto")
    p.add_argument("--min-heading-level", type=int, default=3)
    p.add_argument("--heading-regex")
    p.add_argument("--ignore-headings")
    p.add_argument("--list-sections", action="store_true")
    p.add_argument("--sections-first", type=int)
    p.add_argument("--sections-range", nargs=2, type=int)
    p.add_argument("--max-tokens-out", type=int, default=1024)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--verbose", action="store_true")
    # batch
    p.add_argument("--batch-dry-run", action="store_true")
    p.add_argument("--batch-submit", action="store_true")
    p.add_argument("--batch-status")
    p.add_argument("--batch-fetch")
    p.add_argument("--batch-wait", type=int)
    p.add_argument("--batch-poll-every", type=int, default=5)
    # router overrides
    p.add_argument("--force-sync", action="store_true")
    p.add_argument("--force-batch", action="store_true")
    return p.parse_args()

def should_batch(a) -> bool:
    if a.force_sync: return False
    if a.force_batch: return True
    if a.batch_dry_run or a.batch_submit or a.batch_status or a.batch_fetch: return True
    return False

def main():
    args = parse_args()
    out_dir = args.out_dir or Path(f"{args.mode}_{core.safe_slug(args.model)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    client = bt.get_client()

    # Batch status/fetch early exits
    if args.batch_status:
        print(json.dumps(bt.batch_status(client, args.batch_status), indent=2))
        return
    if args.batch_fetch:
        info = bt.poll_batch(client, args.batch_fetch, args.batch_wait or 0, args.batch_poll_every, args.verbose) if args.batch_wait else bt.batch_status(client, args.batch_fetch)
        if info.get("status") != "completed":
            print(f"[error] batch status={info.get('status')}", file=sys.stderr); return
        out_id = info.get("output_file_id")
        if not out_id: print("[error] no output_file_id", file=sys.stderr); return
        ndjson = bt.download_file(client, out_id)
        (out_dir / "batch_output.raw.ndjson").write_bytes(ndjson)
        id_to_heading = core._json_load_or_empty(out_dir / "batch_map.json") if hasattr(core, "_json_load_or_empty") else {}
        triples = core.parse_batch_output_ndjson(ndjson, id_to_heading)
        core.write_outputs_md_jsonl(triples, args, out_dir) if hasattr(core, "write_outputs_md_jsonl") else None
        print(f"[ok] fetched into {out_dir}")
        return

    # Sync / Submit / Dry-run need draft
    if not args.draft:
        print("[error] draft path required for sync or submit", file=sys.stderr); return

    text = core.read_text(Path(args.draft))
    rx = core.compile_heading_regex(args.heading_regex, args.min_heading_level)
    sections = core.iter_sections(text, rx)
    if args.ignore_headings:
        import re
        rx_ign = re.compile(args.ignore_headings, re.IGNORECASE)
        sections = [(h,b) for (h,b) in sections if not rx_ign.search(h)]

    # list-only
    if args.list_sections:
        for i,(h,b) in enumerate(sections,1):
            print(f"[slice] {i:03d} | {h} | {len((b or '').strip())} chars")
        return

    if should_batch(args) and (args.batch_dry_run or args.batch_submit):
        recs, id_map, endpoint = core.make_batch_records(sections, args)
        nd = out_dir / "batch_input.jsonl"
        with nd.open("w", encoding="utf-8") as f:
            for r in recs: f.write(json.dumps(r, ensure_ascii=False)+"\n")
        (out_dir / "batch_map.json").write_text(json.dumps(id_map, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] wrote NDJSON: {nd}")
        if args.batch_submit:
            bid = bt.submit_batch(client, nd, endpoint)
            print(f"[ok] submitted: {bid}")
        return

    # Sync path: call your existing LLM loop (omitted here for brevity)
    print("[todo] sync path loop (reuse your call_llm + writers)")

if __name__ == "__main__":
    main()
