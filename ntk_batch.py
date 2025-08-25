# ntk_batch.py
import json, os, sys, time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from openai import OpenAI

# Keep batch construction independent of ntk_core;
# just render prompt text using the prompt module.
from ntk_prompts import render_prompt_text


def _is_gpt5(model: str) -> bool:
    return str(model or "").lower().startswith("gpt-5")


def run_batch(
    sections: List[Tuple[str, str]],
    mode: str,
    model: str = "gpt-4.1-mini",
    out_dir: Path = Path("outputs"),
    max_output_tokens: int = 800,
    verbose: bool = False,
) -> str:
    """
    Minimal batch submit:
      - Writes batch_input.jsonl and batch_map.json
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

        def make_body(prompt: str) -> Dict[str, Any]:
            # Responses API uses 'max_output_tokens' and ignores temperature
            return {
                "model": model,
                "input": prompt,
                "max_output_tokens": max_output_tokens,
            }
    else:
        endpoint = "/v1/chat/completions"

        def make_body(prompt: str) -> Dict[str, Any]:
            # Chat (non-gpt-5) uses 'messages' + 'max_tokens'. Allow temperature=0.
            return {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_output_tokens,
                "temperature": 0,
            }

    # Write NDJSON + id→heading map
    count = 0
    cid_to_heading: Dict[str, str] = {}
    with ndjson_path.open("w", encoding="utf-8") as fh:
        for idx, (heading, body) in enumerate(sections, 1):
            if not (body or "").strip():
                continue
            prompt = render_prompt_text(mode, heading, body)
            cid = f"sec_{idx:04d}"
            obj = {
                "custom_id": cid,
                "method": "POST",
                "url": endpoint,
                "body": make_body(prompt),
            }
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
            cid_to_heading[cid] = heading

    (out_dir / "batch_map.json").write_text(
        json.dumps(cid_to_heading, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if verbose:
        print(f"[batch] wrote {count} lines -> {ndjson_path}", file=sys.stderr)
        print(f"[batch] wrote batch_map.json ({len(cid_to_heading)} entries)", file=sys.stderr)

    # Submit
    file_obj = client.files.create(file=ndjson_path.open("rb"), purpose="batch")
    b = client.batches.create(
        input_file_id=file_obj.id,
        endpoint=endpoint,
        completion_window="24h",
    )
    if verbose:
        print(f"[ok] submitted batch: {b.id}", file=sys.stderr)
    return b.id


def get_batch_status(client: OpenAI, batch_id: str) -> Dict[str, Any]:
    """Return a dict snapshot of batch status (id, status, output_file_id, etc.)."""
    b = client.batches.retrieve(batch_id)
    d = b.model_dump() if hasattr(b, "model_dump") else dict(b)
    # Lift output file id if present
    out_id = None
    try:
        out = d.get("output_file_id") or d.get("results_file_id")
        out_id = out
    except Exception:
        pass
    return {
        "id": d.get("id"),
        "status": d.get("status"),
        "output_file_id": out_id,
        "raw": d,
    }


def poll_batch(
    client: OpenAI,
    batch_id: str,
    timeout_s: int = 300,
    every_s: int = 5,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Poll until completed/failed or timeout; return final info dict."""
    t0 = time.time()
    if verbose:
        print(f"[batch] polling {batch_id} until completion…", file=sys.stderr)
    while True:
        info = get_batch_status(client, batch_id)
        st = info.get("status")
        if verbose:
            print(f"[batch] status={st}", file=sys.stderr)
        if st in ("completed", "failed", "expired", "cancelled"):
            return info
        if time.time() - t0 > timeout_s:
            return info  # return whatever current state is
        time.sleep(max(1, int(every_s)))


def _download_file_bytes(client: OpenAI, file_id: str) -> bytes:
    """Return raw bytes of a file id from OpenAI Files API."""
    content = client.files.content(file_id)
    return content.read()


def _extract_text_from_responses_obj(obj: Dict[str, Any]) -> str:
    """
    Extract text from a typical /v1/responses item content.
    Try 'output_text', then 'output' blocks, then fallback to empty.
    """
    # Direct field
    t = obj.get("output_text") or obj.get("text")
    if isinstance(t, str) and t.strip():
        return t.strip()

    # Newer Responses shape: {"output": [... blocks ...]}
    out = obj.get("output")
    if isinstance(out, list):
        for block in out:
            if isinstance(block, dict):
                # text in block?
                if isinstance(block.get("text"), str) and block["text"].strip():
                    return block["text"].strip()
                # message content array
                content = block.get("content")
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and isinstance(c.get("text"), str):
                            txt = c["text"].strip()
                            if txt:
                                return txt
    return ""


def _extract_text_from_chat_obj(obj: Dict[str, Any]) -> str:
    """
    Extract assistant text from a typical /v1/chat/completions response.
    """
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict):
            t = msg.get("content")
            if isinstance(t, str) and t.strip():
                return t.strip()
    return ""


def parse_batch_output_ndjson(raw_bytes: bytes, cid_to_heading: Optional[Dict[str, str]] = None):
    """
    Parse NDJSON output file into a list of tuples (heading, text, raw_item).
    - cid_to_heading: optional mapping from custom_id -> heading text (for pretty output).
    - Robust to both /v1/responses and /v1/chat/completions formats.
    """
    triples = []
    cid_to_heading = cid_to_heading or {}
    for line in raw_bytes.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line.decode("utf-8", errors="replace"))
        except Exception:
            try:
                item = json.loads(line)
            except Exception:
                continue

        # Extract text
        text = ""
        # Some exports wrap inside "response" key
        response = item.get("response") or item.get("output") or {}
        if isinstance(response, dict):
            text = _extract_text_from_responses_obj(response)
            if not text:
                text = _extract_text_from_chat_obj(response)
        else:
            # Sometimes the root is already a chat-like response
            text = _extract_text_from_chat_obj(item)
            if not text:
                text = _extract_text_from_responses_obj(item)

        # Heading
        cid = item.get("custom_id") or "sec_0000"
        heading = cid_to_heading.get(cid, cid)

        # Ensure text is a string
        if not isinstance(text, str):
            try:
                text = json.dumps(text, ensure_ascii=False)
            except Exception:
                text = ""

        triples.append((heading, text, item))
    return triples