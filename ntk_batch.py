# ntk_batch.py
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from openai import OpenAI
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
    Prepare NDJSON and submit a Batch.
    - Writes batch_input.jsonl and batch_map.json into out_dir
    - Submits via Files + Batches API
    - Prints one line: [ok] submitted batch: <batch_id>
    - Returns the batch_id as str
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    client = OpenAI(api_key=api_key)

    out_dir.mkdir(parents=True, exist_ok=True)
    ndjson_path = out_dir / "batch_input.jsonl"
    run_tag = time.strftime("%Y%m%d_%H%M%S")  # uniqueness for this submit


    # Endpoint + request body factory
    if _is_gpt5(model):
        endpoint = "/v1/responses"

        def make_body(prompt: str) -> Dict[str, Any]:
            # Responses API (gpt-5*): text-centric, low/no reasoning, UK copyedit
            return {
                "model": model,
                "input": [
                    {"role": "system", "content": "You are a careful copy editor. Output only plain text in the requested sections."},
                    {"role": "user", "content": prompt},
                ],
                # Allow a decent budget; 5-mini sometimes needs room
                "max_output_tokens": min(int(max_output_tokens or 0) or 1800, 3000),
                # Steer away from hidden chain-of-thought
                "reasoning": {"effort": "low"},   # try "none" if accepted in your SDK
                # Tell the Responses API we want *text* output
                "text": {
                    "format": {"type": "text"},
                    "verbosity": "low",
                },
                # Do NOT pass temperature for gpt-5*
            }




    else:
        endpoint = "/v1/chat/completions"

        def make_body(prompt: str) -> Dict[str, Any]:
            # Chat (non-gpt-5) uses max_tokens, and we set temperature=0 for determinism.
            return {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_output_tokens,
                "temperature": 0,
            }

    # Write NDJSON + sidecar map
    count = 0
    id_to_heading: Dict[str, str] = {}
    with ndjson_path.open("w", encoding="utf-8") as fh:
        for idx, (heading, body) in enumerate(sections, 1):
            if not (body or "").strip():
                continue
            prompt = render_prompt_text(mode, heading, body)
            # Add a per-run suffix to make ids idempotent across retries/batches.
            cid = f"sec_{idx:04d}__{run_tag}"            

            obj = {
                "custom_id": cid,
                "method": "POST",
                "url": endpoint,
                "body": make_body(prompt),
            }
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
            id_to_heading[cid] = heading

    # Save id->heading map (used at fetch time to restore original headings)
    (out_dir / "batch_map.json").write_text(
        json.dumps(id_to_heading, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Optional: persist sections so a rebatch can recreate prompts for selected cids
    try:
        (out_dir / "sections.json").write_text(
            json.dumps(sections, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass

    if verbose:
        print(f"[batch] wrote {count} lines -> {ndjson_path}", file=sys.stderr)
        print(f"[batch] wrote batch_map.json ({len(id_to_heading)} entries)", file=sys.stderr)

    # Guard: do not upload empty NDJSONs; warn on very large ones.
    if count == 0:
        print(f"[warn] {ndjson_path} has 0 lines; skipping batch upload.", file=sys.stderr)
        # Raise a controlled error; the shell wrappers tolerate no-batch-id rows.
        raise RuntimeError("empty-batch")
    if count > 200:
        print(f"[warn] NDJSON contains {count} lines; consider smaller ranges for better resilience.", file=sys.stderr)

    # Submit the batch
    file_obj = client.files.create(file=ndjson_path.open("rb"), purpose="batch")
    b = client.batches.create(
        input_file_id=file_obj.id,
        endpoint=endpoint,
        completion_window="24h",
    )
    print(f"[ok] submitted batch: {b.id}", file=sys.stderr)
    return b.id


def get_batch_status(client: OpenAI, batch_id: str) -> Dict[str, Any]:
    """Return the batch status object from the API."""
    try:
        b = client.batches.retrieve(batch_id)
        # Normalize to dict for callers
        try:
            return b.model_dump()
        except Exception:
            return dict(b or {})
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve batch {batch_id}: {e}")


def poll_batch(
    client: OpenAI,
    batch_id: str,
    timeout_s: int = 0,
    every_s: int = 5,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Poll a batch until it completes or until timeout_s (0 or None means no timeout).
    Returns the final status dict (even if not completed).
    """
    start = time.time()
    while True:
        info = get_batch_status(client, batch_id)
        st = info.get("status")
        if verbose:
            print(f"[batch] status={st}", file=sys.stderr)

        if st in {"completed", "failed", "expired", "cancelling", "cancelled"}:
            return info

        if timeout_s and (time.time() - start) > timeout_s:
            return info

        time.sleep(max(1, every_s))


def _download_file_bytes(client: OpenAI, file_id: str) -> bytes:
    """Download a file’s raw bytes from the Files API."""
    try:
        resp = client.files.content(file_id)
        return resp.read()
    except Exception as e:
        raise RuntimeError(f"Failed to download file {file_id}: {e}")


# -------------------------
# Batch output NDJSON parser
# -------------------------

def _extract_text_from_responses_body(body: Dict[str, Any]) -> str:
    """
    Extract assistant text from a /v1/responses body.
    Handles multiple shapes seen in the field.

    Known shapes:
      A) {"output_text": "..."}
      B) {"text": "..."}                          # some SDKs
      C) {"output": [{"type":"message","content":[{"type":"text","text":"..."}]}]}
      D) {"output": [{"type":"message","content":[{"type":"output_text","text":"..."}]}]}
      E) {"output": [{"type":"output_text","text":"..."}]}
      F) {"output": [{"text":"..."}]}             # minimal
      G) {"message": {"content": "..."}}
    """
    # A / B — direct fields
    for k in ("output_text", "text"):
        t = body.get(k)
        if isinstance(t, str) and t.strip():
            return t.strip()

    out = body.get("output")

    # E / F — top-level blocks with text
    if isinstance(out, list):
        for block in out:
            if not isinstance(block, dict):
                continue
            # E: explicit output_text block
            if block.get("type") == "output_text":
                txt = block.get("text")
                if isinstance(txt, str) and txt.strip():
                    return txt.strip()
            # F: minimal block with a direct "text" field
            txt = block.get("text")
            if isinstance(txt, str) and txt.strip():
                return txt.strip()

        # C / D — message blocks with a list of content items
        for block in out:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "message":
                content = block.get("content")
                if isinstance(content, list):
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        # support both {"type":"text","text":"..."} and {"type":"output_text","text":"..."}
                        txt = item.get("text")
                        if isinstance(txt, str) and txt.strip():
                            return txt.strip()
                # very defensive: some providers put a string directly under "content"
                if isinstance(content, str) and content.strip():
                    return content.strip()

        # Last-ditch: nested "content" lists under non-message blocks
        for block in out:
            if not isinstance(block, dict):
                continue
            content = block.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        txt = item.get("text")
                        if isinstance(txt, str) and txt.strip():
                            return txt.strip()

    # G — some SDKs mimic chat shape inside responses
    msg = body.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text")
                    if isinstance(txt, str) and txt.strip():
                        return txt.strip()

    return ""



def _extract_text_from_chat_body(body: Dict[str, Any]) -> str:
    """
    Extract assistant text from /v1/chat/completions body.
    """
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    # In some SDKs, 'content' may be directly under body['message']
    message = body.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def parse_batch_output_ndjson(raw: bytes, id_to_heading: Optional[Dict[str, str]] = None) -> List[Tuple[str, str, str]]:
    """
    Parse the NDJSON bytes downloaded from the batch output file.
    Returns a list of (heading, text, custom_id).

    - Supports outputs from both /v1/chat/completions and /v1/responses.
    - Uses id_to_heading (from batch_map.json) to restore human headings; falls
      back to the custom_id if no mapping is available.
    """
    id_to_heading = id_to_heading or {}
    triples: List[Tuple[str, str, str]] = []

    # Each line should be a JSON object like:
    # { "custom_id": "...", "response": { "status_code": 200, "body": { ... } } }
    # If "error" is present, include an error marker in the text.
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line.decode("utf-8") if isinstance(line, (bytes, bytearray)) else line)
        except Exception:
            continue

        cid = rec.get("custom_id") or "sec_????"
        heading = id_to_heading.get(cid, cid)

        # Prefer 'response' success body; handle error cases explicitly
        if "response" in rec:
            resp = rec["response"] or {}
            status = resp.get("status_code", 0)
            body = resp.get("body") or {}
            text = ""

            # Decide extraction strategy by shape
            if isinstance(body, dict):
                if "output" in body or "output_text" in body or "text" in body:
                    text = _extract_text_from_responses_body(body)
                elif "choices" in body:
                    text = _extract_text_from_chat_body(body)

            if not text:
                # Provide some diagnostic if empty
                text = f"[empty or unparsable output; status={status}]"

            triples.append((heading, text, cid))
            continue

        if "error" in rec:
            err = rec["error"]
            triples.append((heading, f"[error] {json.dumps(err, ensure_ascii=False)}", cid))
            continue

        # Unknown shape
        triples.append((heading, "[unrecognized record in NDJSON output]", cid))

    return triples