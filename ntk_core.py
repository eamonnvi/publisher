# ntk_core.py
import os, re, sys, time, json
from typing import List, Tuple
from pathlib import Path
from openai import OpenAI
from ntk_prompts import build_prompt

# --- basic md splitter (ATX headings ### and above by default via CLI) ---
def iter_sections(text: str, heading_rx: str = r"(?m)^\s*#{1,6}\s+(.+?)\s*$") -> List[Tuple[str, str]]:
    rx = re.compile(heading_rx)
    sections: List[Tuple[str, str]] = []
    last_pos = 0
    last_head = None
    for m in rx.finditer(text):
        if last_head is not None:
            body = text[last_pos:m.start()].strip()
            sections.append((last_head, body))
        last_head = m.group(1).strip()
        last_pos = m.end()
    if last_head is not None:
        sections.append((last_head, text[last_pos:].strip()))
    elif text.strip():
        sections.append(("Full Manuscript", text.strip()))
    return sections

def normalize_heading(h: str) -> str:
    return re.sub(r"\s#+\s*$", "", (h or "").strip())

# --- OpenAI calling ---
def call_openai(
    *,
    model: str,
    prompt: str,
    max_out: int = 1024,
    timeout: int = 60,
    retries: int = 2,
    verbose: bool = False,
    no_fallback: bool = False,
    prefer_non_reasoning: bool = False,
) -> str:
    """
    Prefer /v1/responses for gpt-5*, else chat. Optionally favour non-reasoning model first.
    """
    client = OpenAI()
    is_gpt5 = (model or "").lower().startswith("gpt-5")
    SYS_TXT = ("Write the answer as plain visible text for the user. "
               "Do not omit the answer or hide content. If you need to reason, keep it brief "
               "and output the final answer as text.")

    def _extract_responses_text(r) -> str:
        try:
            t = getattr(r, "output_text", None)
            if isinstance(t, str) and t.strip():
                return t.strip()
        except Exception:
            pass
        try:
            d = r if isinstance(r, dict) else r.model_dump()
            t = d.get("text") or d.get("output_text")
            if isinstance(t, str) and t.strip():
                return t.strip()
        except Exception:
            pass
        try:
            out = getattr(r, "output", None)
            if isinstance(out, list):
                for block in out:
                    if isinstance(block, dict):
                        if isinstance(block.get("text"), str) and block["text"].strip():
                            return block["text"].strip()
                        if isinstance(block.get("content"), str) and block["content"].strip():
                            return block["content"].strip()
                        if block.get("type") == "message":
                            content = block.get("content", [])
                            if isinstance(content, list):
                                for c in content:
                                    if isinstance(c, dict) and isinstance(c.get("text"), str):
                                        t2 = c["text"].strip()
                                        if t2:
                                            return t2
        except Exception:
            pass
        return ""

    def _log_empty(label: str, resp_obj):
        if not verbose: return
        try:
            raw = resp_obj.model_dump_json() if hasattr(resp_obj, "model_dump_json") else json.dumps(
                resp_obj if isinstance(resp_obj, dict) else {}, ensure_ascii=False)
        except Exception:
            raw = "<unserialisable>"
        sys.stderr.write(f"[debug] {label} returned empty text; raw follows:\n{raw}\n")

    def call_responses(curr_model: str) -> str:
        if verbose: sys.stderr.write("[info] using /v1/responses\n")
        try:
            kwargs = dict(model=curr_model, input=prompt, max_output_tokens=max_out, instructions=SYS_TXT, timeout=timeout)
            r = client.responses.create(**kwargs)
            txt = _extract_responses_text(r)
            if not txt: _log_empty("responses", r)
            return txt
        except Exception as e:
            if verbose: sys.stderr.write(f"[warn] responses error: {e}\n")
            return ""

    def call_chat(curr_model: str) -> str:
        if verbose: sys.stderr.write("[info] using /v1/chat/completions\n")
        try:
            tok_field = "max_completion_tokens" if curr_model.lower().startswith("gpt-5") else "max_tokens"
            kwargs = {
                "model": curr_model,
                "messages": [{"role": "system", "content": SYS_TXT}, {"role": "user", "content": prompt}],
                tok_field: max_out,
                "timeout": timeout,
            }
            r = client.chat.completions.create(**kwargs)
            msg = r.choices[0].message if r and r.choices else None
            return (getattr(msg, "content", None) or "").strip() if msg else ""
        except Exception as e:
            if verbose: sys.stderr.write(f"[warn] chat error: {e}\n")
            return ""

    # order
    attempts = []
    if prefer_non_reasoning and not is_gpt5:
        attempts = [lambda: call_chat(model), lambda: call_responses(model)]
    else:
        attempts = [lambda: call_responses(model), lambda: call_chat(model)] if is_gpt5 else [lambda: call_chat(model), lambda: call_responses(model)]

    for i, fn in enumerate(attempts, 1):
        txt = fn()
        if txt:
            return txt
        if verbose:
            sys.stderr.write(f"[warn] attempt {i}/{len(attempts)}: empty text; retrying…\n")
        time.sleep(0.3)

    if not no_fallback and not is_gpt5:
        if verbose: sys.stderr.write("[info] falling back to gpt-4.1-mini\n")
        txt = call_chat("gpt-4.1-mini")
        if txt: return txt

    raise RuntimeError("Empty text from endpoint")

# --- run helpers ---
def run_sync(
    *, sections: List[Tuple[str, str]], mode: str, model: str = "gpt-4.1",
    timeout: int = 60, max_out: int = 1024, retries: int = 2, verbose: bool = False,
    prefer_non_reasoning: bool = False, no_fallback: bool = False
) -> None:
    for i, (heading, body) in enumerate(sections, 1):
        if not (body or "").strip():
            if verbose: print(f"[warn] empty body — skipping: {heading!r}", file=sys.stderr)
            continue
        heading = normalize_heading(heading)
        prompt = build_prompt(mode, heading, body)
        if verbose: print(f"[sync] section {i} • {heading!r} • prompt_len={len(prompt)}", file=sys.stderr)
        text = call_openai(model=model, prompt=prompt, timeout=timeout, max_out=max_out,
                           retries=retries, verbose=verbose,
                           prefer_non_reasoning=prefer_non_reasoning, no_fallback=no_fallback)
        print(f"\n**{heading}**\n\n{text.strip()}\n")

def run_sync_collect(
    *, sections: List[Tuple[str,str]], mode: str, model: str = "gpt-4.1",
    timeout: int = 60, max_out: int = 1024, retries: int = 2, verbose: bool = False,
    prefer_non_reasoning: bool = False, no_fallback: bool = False
):
    out = []
    for i, (heading, body) in enumerate(sections, 1):
        if not (body or "").strip():
            if verbose: print(f"[warn] empty body — skipping: {heading!r}", file=sys.stderr)
            continue
        heading = normalize_heading(heading)
        prompt = build_prompt(mode, heading, body)
        if verbose: print(f"[sync] section {i} • {heading!r} • prompt_len={len(prompt)}", file=sys.stderr)
        text = call_openai(model=model, prompt=prompt, timeout=timeout, max_out=max_out,
                           retries=retries, verbose=verbose,
                           prefer_non_reasoning=prefer_non_reasoning, no_fallback=no_fallback)
        out.append((heading, text))
    return out

# --- writers ---
def write_markdown(sections_text: List[Tuple[str, str]], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n\n")
        for heading, text in sections_text:
            fh.write(f"**{heading}**\n\n{text.strip()}\n\n")

def write_jsonl(sections_text: List[Tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for heading, text in sections_text:
            fh.write(json.dumps({"heading": heading, "text": text}, ensure_ascii=False) + "\n")