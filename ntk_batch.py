# ntk_batch.py
import json, os, sys
from pathlib import Path
from typing import List, Tuple
from openai import OpenAI
from ntk_prompts import build_prompt

def _is_gpt5(model: str) -> bool:
    return str(model or "").lower().startswith("gpt-5")

def run_batch(sections: List[Tuple[str,str]],
              mode: str,
              model: str = "gpt-5",
              out_dir: Path = Path("batch_out"),
              max_output_tokens: int = 800,
              verbose: bool = False) -> str:
    """
    Minimal batch submit:
      - Writes batch_input.jsonl (no 'metadata' fields)
      - Submits via Files + Batches API
      - Returns batch id
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    client = OpenAI(api_key=api_key)

    out_dir.mkdir(parents=True, exist_ok=True)
    ndjson_path = out_dir / "batch_input.jsonl"

    if _is_gpt5(model):
        endpoint = "/v1/responses"
        def make_body(prompt: str) -> dict:
            return {"model": model, "input": prompt, "max_output_tokens": max_output_tokens}
    else:
        endpoint = "/v1/chat/completions"
        def make_body(prompt: str) -> dict:
            # IMPORTANT: for non-gpt-5 chat we must use 'messages' + 'max_tokens'
            return {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_output_tokens,
            }

    # write NDJSON
    count = 0
    with ndjson_path.open("w", encoding="utf-8") as fh:
        for idx, (heading, body) in enumerate(sections, 1):
            if not (body or "").strip():
                continue
            prompt = build_prompt(mode, heading, body)
            obj = {
                "custom_id": f"sec_{idx:04d}",
                "method": "POST",
                "url": endpoint,
                "body": make_body(prompt),
            }
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1

    if verbose:
        print(f"[batch] wrote {count} lines -> {ndjson_path}", file=sys.stderr)

    # submit
    file_obj = client.files.create(file=ndjson_path.open("rb"), purpose="batch")
    b = client.batches.create(input_file_id=file_obj.id,
                              endpoint=endpoint,
                              completion_window="24h")
    if verbose:
        print(f"[ok] submitted batch: {b.id}", file=sys.stderr)
    return b.id