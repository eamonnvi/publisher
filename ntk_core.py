# ntk_core.py
import os, re, sys, time, json
from typing import List, Tuple
from pathlib import Path
from openai import OpenAI
from ntk_prompts import build_prompt


# --- basic md splitter (ATX headings ### and above by default via CLI) ---

def iter_sections(text: str, heading_rx: str = r"(?m)^\s*#{1,6}\s+(.+?)\s*$") -> List[Tuple[str, str]]:
    """
    Return [(heading_text, body_text)] for ATX-style markdown headings.
    `heading_rx` must capture the heading text in group(1).
    If no headings are found, treat the whole file as a single section.
    """
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

    # tail
    if last_head is not None:
        sections.append((last_head, text[last_pos:].strip()))
    elif text.strip():
        sections.append(("Full Manuscript", text.strip()))
    return sections


def normalize_heading(h: str) -> str:
    # trims trailing hashes like "Title ###"
    return re.sub(r"\s#+\s*$", "", (h or "").strip())


# --- response extraction + OpenAI calls --------------------------------------

def call_openai(
    *,
    model: str,
    prompt: str,
    max_out: int = 1024,
    timeout: int = 60,
    retries: int = 2,
    verbose: bool = False,
    fallback_model: str = "gpt-4.1-mini",
    prefer_non_reasoning: bool = False,
) -> str:
    """
    Prefer /v1/responses for gpt-5*. For non-gpt-5 use chat by default.
    If prefer_non_reasoning=True, try fallback_model chat first (good for copyedit modes).

    Retries across endpoints/models are handled here (not per-request retries).
    """
    client = OpenAI()
    is_gpt5 = (model or "").lower().startswith("gpt-5")

    SYS_TXT = (
        "Write the answer as plain visible text for the user. "
        "Do not omit the answer. Do not hide content in private thinking. "
        "If you need to reason, keep it brief and output the final answer as text."
    )

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
            if not out and not isinstance(r, dict):
                try:
                    out = r.model_dump().get("output")
                except Exception:
                    out = None
            if isinstance(out, list):
                for block in out:
                    if isinstance(block, dict):
                        if block.get("type") == "message":
                            content = block.get("content", [])
                            if isinstance(content, list):
                                for c in content:
                                    if isinstance(c, dict) and isinstance(c.get("text"), str):
                                        txt = c["text"].strip()
                                        if txt:
                                            return txt
                        if isinstance(block.get("text"), str):
                            txt = block["text"].strip()
                            if txt:
                                return txt
                        if isinstance(block.get("content"), str):
                            txt = block["content"].strip()
                            if txt:
                                return txt
        except Exception:
            pass
        return ""

    def _log_empty(label: str, resp_obj):
        if not verbose:
            return
        try:
            if hasattr(resp_obj, "model_dump_json"):
                raw = resp_obj.model_dump_json()
            elif hasattr(resp_obj, "model_dump"):
                raw = json.dumps(resp_obj.model_dump(), ensure_ascii=False)
            else:
                raw = json.dumps(resp_obj if isinstance(resp_obj, dict) else {}, ensure_ascii=False)
            sys.stderr.write(f"[debug] {label} returned empty text; raw payload follows:\n{raw}\n")
        except Exception as e:
            sys.stderr.write(f"[debug] {label} empty; could not dump payload: {e}\n")

    def call_responses(curr_model: str) -> str:
        if verbose:
            sys.stderr.write("[info] using /v1/responses\n")
        try:
            kwargs = dict(
                model=curr_model,
                input=prompt,
                max_output_tokens=max_out,
                instructions=SYS_TXT,
                timeout=timeout,
            )
            # do NOT set temperature for gpt-5; for other models we avoid responses anyway
            r = client.responses.create(**kwargs)
            txt = _extract_responses_text(r)
            if not txt:
                _log_empty("responses/plain", r)
            return txt
        except Exception as e:
            if verbose:
                sys.stderr.write(f"[warn] responses error: {e}\n")
            return ""

    def call_chat(curr_model: str) -> str:
        if verbose:
            sys.stderr.write("[info] using /v1/chat/completions\n")
        try:
            tok_field = "max_completion_tokens" if curr_model.lower().startswith("gpt-5") else "max_tokens"
            kwargs = {
                "model": curr_model,
                "messages": [
                    {"role": "system", "content": SYS_TXT},
                    {"role": "user", "content": prompt},
                ],
                tok_field: max_out,
                "timeout": timeout,
            }
            # non-gpt-5 chat allows temperature=0 (deterministic-ish)
            if not curr_model.lower().startswith("gpt-5"):
                kwargs["temperature"] = 0

            r = client.chat.completions.create(**kwargs)
            msg = r.choices[0].message if r and r.choices else None
            txt = (getattr(msg, "content", None) or "").strip() if msg else ""
            if not txt and verbose:
                try:
                    raw = r.model_dump_json()
                except Exception:
                    raw = ""
                sys.stderr.write(f"[debug] chat returned empty text; raw:\n{raw}\n")
            return txt
        except Exception as e:
            if verbose:
                sys.stderr.write(f"[warn] chat error: {e}\n")
            return ""

    def _attempt(curr_model: str, prefer_resp: bool) -> str:
        order = [call_responses, call_chat] if prefer_resp else [call_chat, call_responses]
        for i, fn in enumerate(order, 1):
            txt = fn(curr_model)
            if txt:
                return txt
            if verbose:
                sys.stderr.write(f"[warn] attempt {i}/{len(order)} with {curr_model}: empty text; retrying…\n")
            time.sleep(0.3)
        return ""

    # Routing
    if prefer_non_reasoning:
        if verbose:
            sys.stderr.write(f"[info] prefer_non_reasoning=True → trying {fallback_model} chat first\n")
        out = _attempt(fallback_model, prefer_resp=False)
        if out:
            return out
        # Then try the requested model
        out = _attempt(model, prefer_resp=is_gpt5)
        if out:
            return out
    else:
        out = _attempt(model, prefer_resp=is_gpt5)
        if out:
            return out
        if fallback_model and fallback_model.lower() != (model or "").lower():
            if verbose:
                sys.stderr.write(f"[info] falling back to {fallback_model}\n")
            out = _attempt(fallback_model, prefer_resp=False)
            if out:
                return out

    raise RuntimeError("Empty text from endpoint")


# --- run helpers --------------------------------------------------------------

def run_sync(
    sections: List[Tuple[str, str]],
    mode: str,
    model: str = "gpt-5",
    timeout: int = 60,
    max_out: int = 1024,
    retries: int = 2,   # reserved hook
    verbose: bool = False,
) -> None:
    """
    Build prompts and call the OpenAI API, printing results to stdout.
    """
    for i, (heading, body) in enumerate(sections, 1):
        if not (body or "").strip():
            if verbose:
                print(f"[warn] empty body — skipping: {heading!r}", file=sys.stderr)
            continue

        heading = normalize_heading(heading)
        prompt = build_prompt(mode, heading, body)

        if verbose:
            print(f"[sync] section {i} • {heading!r} • prompt_len={len(prompt)}", file=sys.stderr)

        prefer_nr = bool(mode.startswith("copyedit"))
        text = call_openai(
            model=model,
            prompt=prompt,
            timeout=timeout,
            max_out=max_out,
            retries=retries,
            verbose=verbose,
            fallback_model="gpt-4.1-mini",
            prefer_non_reasoning=prefer_nr,
        )

        print(f"\n**{heading}**\n\n{text.strip()}\n")


def run_sync_collect(
    sections: List[Tuple[str, str]],
    mode: str,
    model: str = "gpt-5",
    timeout: int = 60,
    max_out: int = 1024,
    retries: int = 2,
    verbose: bool = False,
) -> List[Tuple[str, str]]:
    """
    Same as run_sync, but returns a list of (heading, text) instead of printing.
    """
    out: List[Tuple[str, str]] = []
    for i, (heading, body) in enumerate(sections, 1):
        if not (body or "").strip():
            if verbose:
                print(f"[warn] empty body — skipping: {heading!r}", file=sys.stderr)
            continue

        heading = normalize_heading(heading)
        prompt = build_prompt(mode, heading, body)

        if verbose:
            print(f"[sync] section {i} • {heading!r} • prompt_len={len(prompt)}", file=sys.stderr)

        prefer_nr = bool(mode.startswith("copyedit"))
        text = call_openai(
            model=model,
            prompt=prompt,
            timeout=timeout,
            max_out=max_out,
            retries=retries,
            verbose=verbose,
            fallback_model="gpt-4.1-mini",
            prefer_non_reasoning=prefer_nr,
        )

        out.append((heading, text))
    return out


# --- safe writers -------------------------------------------------------------

def _coerce_text(val) -> str:
    """
    Convert various response payloads into a printable string.
    Handles:
      - plain str
      - dicts with 'text' / 'output_text' / 'content'
      - any other type via str()
    """
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        for k in ("text", "output_text", "content"):
            v = val.get(k)
            if isinstance(v, str):
                return v
        # If dict but no known field, emit the JSON; helpful for debugging
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def write_markdown(sections_text: List[Tuple[str, str]], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n\n")
        for heading, text in sections_text:
            fh.write(f"**{heading}**\n\n{_coerce_text(text).strip()}\n\n")


def write_jsonl(sections_text: List[Tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for heading, text in sections_text:
            fh.write(json.dumps({"heading": heading, "text": _coerce_text(text)}, ensure_ascii=False) + "\n")