# ntk_core.py

import re
import sys
import time            # used for retry sleeps/backoff in call_openai
from typing import List, Tuple

from openai import OpenAI
from ntk_prompts import PROMPTS

def render_prompt_text(mode: str, heading: str, body: str) -> str:
    """
    Render PROMPTS[mode] into a single text block suitable for /v1/responses
    (and fine to send as a single chat message if you fall back).
    """
    spec = PROMPTS.get(mode)
    if spec is None:
        raise KeyError(f"Unknown mode: {mode!r}")

    # Back-compat: allow string templates if any remain
    if isinstance(spec, str):
        return spec.format(body=body, heading=heading)

    if not isinstance(spec, dict):
        raise TypeError(f"PROMPTS[{mode!r}] must be str or dict with system/user")

    sys_txt = (spec.get("system") or "").format(body=body, heading=heading).strip()
    usr_txt = (spec.get("user")   or "").format(body=body, heading=heading).strip()

    parts = []
    if sys_txt:
        parts.append(f"[SYSTEM]\n{sys_txt}")
    if usr_txt:
        parts.append(usr_txt)
    return "\n\n".join(parts)

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


def call_openai(
    model: str,
    prompt: str,
    max_out: int = 1024,
    timeout: int = 60,
    retries: int = 2,
    verbose: bool = False,
) -> str:
    """
    Prefer /v1/responses for gpt-5*, with robust output extraction.
    Retry order: responses → responses → chat (or chat → responses for non-gpt-5).
    Translates token limit to the correct parameter per endpoint.
    """
    from openai import OpenAI
    client = OpenAI()

    # --- helpers -------------------------------------------------------------

    def _extract_from_responses(r) -> str:
        """
        Try several shapes the Responses API may return.
        """
        # 1) SDK convenience
        try:
            txt = getattr(r, "output_text", None)
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
        except Exception:
            pass

        # 2) Structured blocks
        try:
            out = getattr(r, "output", None)
            if isinstance(out, list) and out:
                blk = out[0]
                if isinstance(blk, dict):
                    content = blk.get("content")
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and isinstance(c.get("text"), str):
                                t = c["text"].strip()
                                if t:
                                    return t
        except Exception:
            pass

        # 3) Dict fallback
        try:
            d = r if isinstance(r, dict) else r.model_dump()
            for k in ("output_text", "text"):
                t = d.get(k)
                if isinstance(t, str) and t.strip():
                    return t.strip()
        except Exception:
            pass

        return ""

    # Prefer /v1/responses for gpt-5 family
    prefer_responses = str(model or "").lower().startswith("gpt-5")

    def call_responses_once() -> str:
        if verbose:
            sys.stderr.write("[info] using /v1/responses\n")
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=max_out,
            )
            txt = _extract_from_responses(resp)
            return txt
        except Exception as e:
            if verbose:
                sys.stderr.write(f"[warn] responses call error: {e}\n")
            return ""

    def call_chat_once() -> str:
        if verbose:
            sys.stderr.write("[info] using /v1/chat/completions\n")
        try:
            # gpt-5 chat expects max_completion_tokens; others use max_tokens
            tok_field = "max_completion_tokens" if str(model).lower().startswith("gpt-5") else "max_tokens"
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                tok_field: max_out,
            }
            resp = client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message if (resp and resp.choices) else None
            return (getattr(msg, "content", None) or "").strip() if msg else ""
        except Exception as e:
            if verbose:
                sys.stderr.write(f"[warn] chat call error: {e}\n")
            return ""

    # Attempt order & simple retry loop
    attempts = (
        [call_responses_once, call_responses_once, call_chat_once]
        if prefer_responses
        else [call_chat_once, call_responses_once]
    )

    tries = 0
    for fn in attempts:
        tries += 1
        txt = fn()
        if txt:
            return txt
        if verbose:
            sys.stderr.write(f"[warn] attempt {tries}/{len(attempts)}: empty text; retrying…\n")
        time.sleep(0.4)

    raise RuntimeError("Empty text from endpoint")


def run_sync(
    sections: List[Tuple[str, str]],
    mode: str,
    model: str = "gpt-5",
    timeout: int = 60,
    max_out: int = 1024,
    retries: int = 2,   # reserved for future per-call retry; we do multi-endpoint tries already
    verbose: bool = False,
) -> None:
    """
    Build prompts and call the real OpenAI API, printing results to stdout.
    CLI handles file reading & section slicing; this just runs the loop.
    """
    for i, (heading, body) in enumerate(sections, 1):
        if not (body or "").strip():
            if verbose:
                print(f"[warn] empty body — skipping: {heading!r}", file=sys.stderr)
            continue

        heading = normalize_heading(heading)
        prompt = render_prompt_text(mode, heading, body)

        if verbose:
            print(f"[sync] section {i} • {heading!r} • prompt_len={len(prompt)}", file=sys.stderr)

        text = call_openai(
            model=model,
            prompt=prompt,
            max_out=max_out,
            timeout=timeout,
            retries=retries,
            verbose=verbose,
        )

        print(f"\n**{heading}**\n\n{text.strip()}\n")