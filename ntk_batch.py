# ntk_batch.py
import json, os, sys, time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from openai import OpenAI
from ntk_prompts import build_prompt  # ← use the real prompt builder

def _is_gpt5(model: str) -> bool:
    return str(model or "").lower().startswith("gpt-5")

def run_batch(
    sections: List[Tuple[str, str]],
    mode: str,
    model: str = "gpt-5",
    out_dir: Path = Path("batch_out"),
    max_out: int = 800,
    verbose: bool = False,
) -> str:
    """
    Minimal batch submit:
      - Writes batch_input.jsonl (+ batch_map.json sidecar)
      - Submits via Files + Batches API
      - Returns batch id
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    client = OpenAI(api_key=api_key)

    out_dir.mkdir(parents=True, exist_ok=True)
    ndjson_path = out_dir / "batch_input.jsonl"

    # Decide endpoint + body factory
    if _is_gpt5(model):
        endpoint = "/v1/responses"
        def make_body(prompt: str) -> Dict:
            # Responses API uses 'max_output_tokens'; do NOT set temperature for gpt-5.
            return {
                "model": model,
                "input": prompt,
                "max_output_tokens": max_out,
            }
    else:
        endpoint = "/v1/chat/completions"
        def make_body(prompt: str) -> Dict:
            # Chat (non-gpt-5) uses 'messages' + 'max_tokens'; temperature allowed.
            return {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_out,
                "temperature": 0,
            }
        # NOTE: If you ever route gpt-5 to chat, use 'max_completion_tokens' not 'max_tokens'.

    # Write NDJSON + sidecar map
    count = 0
    cid_to_heading: Dict[str, str] = {}
    with ndjson_path.open("w", encoding="utf-8") as fh:
        for idx, (heading, body) in enumerate(sections, 1):
            if not (body or "").strip():
                continue
            prompt = build_prompt(mode, heading, body)
            cid = f"sec_{idx:04d}"
            obj = {
                "custom_id": cid,
                "method": "POST",
                "url": endpoint,
                "body": make_body(prompt),
            }
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            cid_to_heading[cid] = heading
            count += 1

    (out_dir / "batch_map.json").write_text(
        json.dumps(cid_to_heading, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    if verbose:
        print(f"[batch] wrote {count} lines -> {ndjson_path}", file=sys.stderr)
        print(f"[batch] wrote batch_map.json ({len(cid_to_heading)} entries)", file=sys.stderr)

    # Submit
    file_obj = client.files.create(file=ndjson_path.open("rb"), purpose="batch")
    b = client.batches.create(input_file_id=file_obj.id,
                              endpoint=endpoint,
                              completion_window="24h")
    if verbose:
        print(f"[ok] submitted batch: {b.id}", file=sys.stderr)
    return b.id

def fetch_batch(
    batch_id: str,
    out_dir: Path,
    verbose: bool = True,
    poll: bool = False,
    every_s: int = 5,
    timeout_s: int = 300,
) -> Dict:
    """
    Fetch a completed (or poll until completed) batch and return a dict with:
      {
        "status": "...",
        "output_file_id": "...",
        "raw": b"...",             # raw NDJSON bytes (if completed)
      }
    """
    client = OpenAI()

    def _get_status(bid: str) -> Dict:
        r = client.batches.retrieve(bid)
        d = r.model_dump()
        return {
            "status": d.get("status"),
            "output_file_id": d.get("output_file_id"),
            "response_count": d.get("request_counts", {}).get("completed", 0),
            "raw": d,
        }

    start = time.time()
    info = _get_status(batch_id)

    if poll and info["status"] not in ("completed", "failed", "expired", "canceled"):
        if verbose:
            print(f"[batch] polling {batch_id} until completion…", file=sys.stderr)
        while time.time() - start < timeout_s:
            time.sleep(max(1, int(every_s)))
            info = _get_status(batch_id)
            st = info["status"]
            if verbose:
                print(f"[batch] status={st}", file=sys.stderr)
            if st in ("completed", "failed", "expired", "canceled"):
                break

    if info["status"] != "completed":
        return info

    # Download output NDJSON
    out_id = info["output_file_id"]
    if not out_id:
        return info

    file_obj = client.files.content(out_id)
    raw = file_obj.read() if hasattr(file_obj, "read") else file_obj
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "batch_output.raw.ndjson").write_bytes(raw)
    info["raw"] = raw
    return info

def parse_batch_output_ndjson(raw_bytes: bytes, batch_map: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """
    Parse Batch NDJSON (raw bytes) into a list of (heading, text, custom_id).
    """
    out: List[Tuple[str,str,str]] = []
    for line in (raw_bytes or b"").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line.decode("utf-8"))
        except Exception:
            continue
        cid = (d.get("custom_id") or "").strip()
        body = d.get("response", {}).get("body", {})
        text = ""
        # Try responses shape first
        text = body.get("output_text") or body.get("text") or ""
        if not text:
            # Try chat
            choices = body.get("choices") or []
            if choices and "message" in choices[0]:
                text = (choices[0]["message"].get("content") or "").strip()
        heading = batch_map.get(cid, cid or "Unknown")
        out.append((heading, text or "", cid))
    return out