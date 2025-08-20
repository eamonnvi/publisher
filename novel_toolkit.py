#!/usr/bin/env python3
"""
novel_toolkit.py – Arbuthnot Books helper
v5.4.6 (2025-08-13)

Adds:
- --profile-on-fetch: estimate tokens & costs on batch fetch (re-encode outputs and, if draft provided, prompts)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

from tqdm import tqdm  # type: ignore
import tiktoken  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except Exception:
    sys.stderr.write("Install deps: pip install openai tiktoken tqdm\n")
    raise

################################################################################
# Prompt templates                                                              #
################################################################################

PROMPT_TEMPLATES: Dict[str, str] = {
    "concise": (
        "You are an experienced British literary editor.\n"
        "Write a clear, accurate SUMMARY and a concise TITLE for the {label} titled '{heading}'.\n"
        "Output format (exactly):\n"
        "SUMMARY:\n"
        "(2–4 short paragraphs in British English)\n"
        "TITLE: <a short, vivid title>\n\n"
        "----- BEGIN {LABEL} -----\n{body}\n----- END {LABEL} -----\n"
    ),
    "discursive": (
        "Provide an in-depth, analytical synopsis of the section titled '{heading}'. "
        "Discuss character motivations, conflicts, themes, subtext, narrative strategy, pacing, suspense and misdirection. "
        "Highlight details with possible later impact. Do not avoid spoilers.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "critique_md": (
        "You are an experienced developmental editor. Evaluate the section titled '{heading}' as follows:\n\n"
        "1. **Strengths:** voice, pacing, originality, structure, style.\n"
        "2. **Weaknesses:** inconsistencies, clichés, slack, unresolved setups.\n"
        "3. **Genre fit and innovation.**\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "critique_json": (
        "You are an experienced developmental editor. Return STRICT JSON with keys 'strengths','weaknesses','improvements' – arrays of strings.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "improvement_suggestions": (
        "For the section titled '{heading}', list specific, actionable suggestions for revision: clarity, motivation, suspense, pacing, ambiguities. "
        "Include examples or sample rewrites if relevant.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "entities_json": (
        "Extract structured facts from the section titled '{heading}'. "
        "Return only a single valid JSON object with keys: "
        "'section_heading','characters','pov_character','themes','locations','timestamp'.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "continuity_md": (
        "You are a continuity editor. Compare the **current section** to the following **prior facts**. "
        "Flag contradictions (dates, ages, locations, actions) and note ambiguous or potentially inconsistent elements.\n\n"
        "### Prior facts\n```json\n{prior_entities}\n```\n\n### Current section\n{body}\n"
    ),
    "continuity_json": (
        "Compare current section to prior facts. Return ONLY a JSON array of inconsistency strings (empty if none).\n\n"
        "Prior facts: {prior_entities}\n\nCurrent section:\n{body}\n"
    ),
    "copyedit_md": (
        "## Identity\nYou are a line-by-line copy editor in a British literary publishing house.\n\n"
        "## Task\nIdentify typos, grammar issues, redundancies, repetitions.\n\n"
        "## Prohibitions\nDo not break sentences, reduce adverbs, spot clichés, or flag contractions.\n\n"
        "## Output\nBulleted Markdown: line number, snippet, issue, suggestion.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "copyedit_json": (
        "(Same identity/task) Return JSON array of {line_number:int, original:str, issue:str, suggestion:str}.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "overview": (
        "Write a 2,000–2,500-word editorial overview of the manuscript below. "
        "Cover plot, character arcs, themes, style, structure, pacing, genre conventions and subversions, and any continuity issues. "
        "Include major strengths and weaknesses, and possible revisions.\n\n"
        "----- BEGIN MANUSCRIPT -----\n{body}\n----- END MANUSCRIPT -----"
    ),
    "detailed-summary": (
        "Write a 5,000–10,000-word chapter-by-chapter summary of the manuscript below.\n\n"
        "----- BEGIN MANUSCRIPT -----\n{body}\n----- END MANUSCRIPT -----"
    ),
    "sensitivity-analysis": (
        "Read the sexually explicit sections of the manuscript below. Assess whether they are justified by the period and situations. "
        "Suggest rewordings for lines which are borderline unacceptable.\n\n"
        "----- BEGIN MANUSCRIPT -----\n{body}\n----- END MANUSCRIPT -----"
    ),
    "plot-points": (
        "Answer the specific plot questions below based on the manuscript section provided.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
}

################################################################################
# Helpers                                                                       #
################################################################################



def prepend_cli_metadata_to_output(output_path: Path, args: argparse.Namespace, mode: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_abs_path = output_path.resolve()
    arglist = []
    for key, value in vars(args).items():
        if key in ("draft", "out_dir"):
            continue
        if value in (None, False):
            continue
        if isinstance(value, Path):
            value = str(value)
        k = f"--{key.replace('_','-')}"
        if value is True:
            arglist.append(k)
        else:
            arglist.append(f"{k} {value}")
    cli = f"python3 novel_toolkit.py {args.draft} " + " ".join(arglist) if getattr(args, "draft", None) else "novel_toolkit.py (batch fetch)"
    header = (
        f"# {mode.capitalize()} Report\n\n"
        f"**Generated:** {ts}\n"
        f"**Output file:** `{output_abs_path}`\n"
        f"**CLI command:**\n\n```bash\n{cli}\n```\n\n---\n\n"
    )
    original = output_path.read_text(encoding="utf-8")
    output_path.write_text(header + original, encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def _norm_heading(h: str | None) -> str:
    """Normalise headings for stable comparisons."""
    if not h:
        return ""
    return re.sub(r"\s+", " ", h).strip()

def compile_heading_regex(custom: Optional[str], min_level: int = 1) -> re.Pattern[str]:
    if custom:
        return re.compile(custom, re.MULTILINE)
    return re.compile(rf"^\s*#{{{min_level},6}}\s*([^#\n]+?)\s*#*\s*$", re.MULTILINE)


def iter_sections(text: str, heading_rx: re.Pattern[str]) -> Iterable[Tuple[str, str]]:
    matches = list(heading_rx.finditer(text))
    if not matches:
        yield "Full Manuscript", text.strip()
        return
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = m.group(1).strip()
        yield heading, text[start:end].strip()

def _select_sections_for_batch(sections, args):
    """
    Returns (subset, offset, selection_desc). Offset is the index offset (0-based)
    of the first selected item in the original sections list.
    """
    n = len(sections)
    if args.batch_range:
        start, end = args.batch_range
        # clamp to [1, n], convert to 0-based
        start = max(1, start)
        end = min(n, end)
        if start > end:
            return [], 0, f"empty (normalized start {start} > end {end})"
        offset = start - 1
        subset = sections[offset:end]
        return subset, offset, f"{start}..{end} of {n}"
    elif args.batch_first:
        k = max(0, min(args.batch_first, n))
        subset = sections[:k]
        return subset, 0, f"first {k} of {n}"
    else:
        return sections, 0, f"all {n}"

def num_tokens(model: str, *chunks: str) -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")
    return sum(len(enc.encode(c)) for c in chunks)


def build_prompt(mode: str, heading: str, body: str, fmt: str, prior: Optional[str] = None) -> str:
    key_map = {
        "critique": "critique_json" if fmt in {"json","both"} else "critique_md",
        "entities": "entities_json",
        "continuity": "continuity_json" if fmt in {"json","both"} else "continuity_md",
        "copyedit": "copyedit_json" if fmt in {"json","both"} else "copyedit_md",
    }
    template_key = key_map.get(mode, mode)
    if template_key not in PROMPT_TEMPLATES:
        raise KeyError(f"No prompt template for mode '{mode}'.")
    template = PROMPT_TEMPLATES[template_key]
    norm = heading.strip().lower()
    label = "manuscript" if norm in {"full manuscript","manuscript"} else "section"
    return template.format(
        label=label, LABEL=label.upper(), heading=heading, body=body, prior_entities=prior or "[]"
    )

_TITLE_RX = re.compile(r"(?im)^(?:\*Suggested\s+title:\*|TITLE:)\s*(.+?)\s*$")

def normalize_concise(text: str) -> tuple[str, Optional[str]]:
    m = _TITLE_RX.search(text)
    title = m.group(1).strip() if m else None
    if "SUMMARY:" in text:
        summary = text.split("SUMMARY:", 1)[1].strip()
        if m:
            summary = _TITLE_RX.sub("", summary).strip()
    else:
        summary = text.strip()
    return summary, title

def _looks_truncated(s: str) -> bool:
    s = s.rstrip()
    return bool(s) and s[-1] not in ".!?”’\"'）】〉>]"

def postprocess_markdown_sections(md_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    header_rx = re.compile(r"^\*\*([^\*\n]+)\*\*\s*$", re.MULTILINE)
    matches = list(header_rx.finditer(text))
    if not matches:
        return
    chunks: List[str] = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if (idx + 1) < len(matches) else len(text)
        block = text[start:end]
        lines = block.splitlines()
        if not lines:
            chunks.append(block)
            continue
        header = lines[0]
        body = "\n".join(lines[1:]).strip("\n")
        tm = _TITLE_RX.search(body)
        if tm:
            title_text = tm.group(1).strip()
            body = _TITLE_RX.sub("", body, count=1).strip()
            body = (body.rstrip() + "\n\n*Suggested title:* " + title_text + "\n").lstrip("\n")
        else:
            body = body.rstrip() + "\n"
        cleaned = header + "\n\n" + body.strip("\n") + "\n"
        chunks.append(cleaned)
    cleaned_text = ("\n\n").join(s.rstrip() for s in chunks).rstrip() + "\n"
    md_path.write_text(cleaned_text, encoding="utf-8")

################################################################################
# Pricing                                                                       #
################################################################################

PRICE = {
    "gpt-5":        {"in": 0.010, "out": 0.030},
    "gpt-5-mini":   {"in": 0.003, "out": 0.006},
    "gpt-4.1":      {"in": 0.005, "out": 0.015},
    "gpt-4.1-mini": {"in": 0.001, "out": 0.003},
}
CURRENCY = "USD"

def estimate_cost(model: str, tokens_in: int, tokens_out: int, fx_rate: Optional[float]) -> tuple[str,str,str,str]:
    r = PRICE.get(model)
    if not r:
        return "","","",""
    fin  = (tokens_in  / 1000.0) * r["in"]
    fout = (tokens_out / 1000.0) * r["out"]
    total = fin + fout
    if fx_rate and fx_rate > 0:
        fin, fout, total = fin * fx_rate, fout * fx_rate, total * fx_rate
        cur = "GBP"
    else:
        cur = CURRENCY
    return (f"{fin:.6f}", f"{fout:.6f}", f"{total:.6f}", cur)

def estimate_cost_numeric(model: str, tokens_in: float, tokens_out: float, fx_rate: Optional[float]):
    r = PRICE.get(model)
    if not r:
        cur = "GBP" if (fx_rate and fx_rate > 0) else CURRENCY
        return 0.0, 0.0, 0.0, cur
    cin  = (tokens_in  / 1000.0) * r["in"]
    cout = (tokens_out / 1000.0) * r["out"]
    total = cin + cout
    cur = "GBP" if (fx_rate and fx_rate > 0) else CURRENCY
    if fx_rate and fx_rate > 0:
        cin *= fx_rate; cout *= fx_rate; total *= fx_rate
    return cin, cout, total, cur

################################################################################
# OpenAI calls                                                                  #
################################################################################

def call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    max_out: int,
    want_json: bool,
    engine: str,
    fallback_model: Optional[str],
    verbose: bool,
    dump_raw_dir: Optional[str],
    heading: str,
    request_timeout: int,
    max_retries: int,
) -> Optional[str]:

    def chat_call(m: str) -> Optional[str]:
        if verbose:
            sys.stderr.write(f"[info] Using Chat Completions with model '{m}' (json={want_json})\n")

        kwargs = dict(
            model=m,
            messages=[{"role": "user", "content": prompt}],
            timeout=request_timeout,
        )

        # Param compatibility (gpt-5* chat needs max_completion_tokens)
        if m.startswith("gpt-5"):
            kwargs["max_completion_tokens"] = max_out
            # some gpt-5* chat variants only allow default temperature; omit unless you need it
        else:
            kwargs["max_tokens"] = max_out
            # kwargs["temperature"] = 0.3  # uncomment if you really want to set it for 4.1*

        resp = client.chat.completions.create(**kwargs)

        # Try to extract text robustly
        text = ""
        try:
            # modern SDKs
            text = (resp.choices[0].message.content or "").strip()
        except Exception:
            # fallback shapes
            try:
                content = getattr(resp.choices[0], "message", None)
                if content:
                    text = (getattr(content, "content", "") or "").strip()
            except Exception:
                text = ""

        if dump_raw_dir:
            Path(dump_raw_dir, f"{safe_slug(heading)}__chat.json").write_text(resp.model_dump_json(), encoding="utf-8")

        return text or None

    def responses_call(m: str) -> Optional[str]:
        if verbose:
            sys.stderr.write(f"[info] Using Responses API with model '{m}' (json={want_json})\n")

        # Responses prefers 'input' + generation_config
        kwargs = dict(
            model=m,
            input=[{"role": "user", "content": prompt}],
            timeout=request_timeout,
        )
        try:
            resp = client.responses.create(**kwargs, generation_config={"max_output_tokens": max_out})
        except TypeError:
            # older SDK signature
            resp = client.responses.create(**kwargs, max_output_tokens=max_out)

        text = _extract_text_from_responses(resp)

        if dump_raw_dir:
            Path(dump_raw_dir, f"{safe_slug(heading)}__responses.json").write_text(resp.model_dump_json(), encoding="utf-8")

        return text or None

    prefer_responses = (engine == "responses") or (engine == "auto" and model.startswith("gpt-5"))
    primary = responses_call if prefer_responses else chat_call
    secondary = chat_call if prefer_responses else responses_call

    # Build the sequence of (model, fn) tries
    tries_plan: list[tuple[str, Any]] = [(model, primary), (model, secondary)]
    if fallback_model and fallback_model != model:
        tries_plan += [(fallback_model, primary), (fallback_model, secondary)]

    last_exc: Optional[Exception] = None

    for m, fn in tries_plan:
        attempt = 0
        while attempt <= max_retries:
            try:
                out = fn(m)

                # Case A: endpoint actually returned nothing
                if out is None or not isinstance(out, str) or not out.strip():
                    if verbose:
                        sys.stderr.write(f"[warn] Empty text from endpoint for {m}; attempt {attempt+1}/{max_retries+1}\n")
                    if attempt < max_retries:
                        _sleep_backoff(attempt)
                        attempt += 1
                        continue
                    else:
                        break  # move to next (model,fn) in tries_plan

                # Case B: non-empty text—return it even if JSON was requested.
                # Upstream (entities/continuity) will attempt to parse/repair.
                return out.strip()

            except Exception as e:
                last_exc = e
                if verbose:
                    sys.stderr.write(f"[warn] endpoint error for {m}: {e} (attempt {attempt+1}/{max_retries+1})\n")
                if attempt < max_retries:
                    _sleep_backoff(attempt)
                    attempt += 1
                    continue
                else:
                    break  # try next (model,fn)

    if verbose and last_exc:
        sys.stderr.write(f"[warn] All attempts failed: {last_exc}\n")
    return None


def safe_slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)[:120]

def _pretty_rel(p: Path, base: Path) -> str:
    try:
        return str(p.resolve().relative_to(base.resolve()))
    except Exception:
        return str(p.resolve())


def _sleep_backoff(attempt: int, base: float = 0.6, cap: float = 12.0) -> None:
    """
    Exponential backoff with jitter.
    attempt: 0,1,2,...
    base: initial delay seconds
    cap:  max delay
    """
    #  backoff = min(cap, base * 2^attempt) + jitter(0..250ms)
    delay = min(cap, base * (2 ** attempt)) + random.random() * 0.25
    time.sleep(delay)

def _extract_text_from_responses(resp) -> str:
    """
    Try several shapes for the Responses API. Returns '' if nothing useful.
    """
    # 1) Modern convenience property
    try:
        t = getattr(resp, "output_text", None)
        if isinstance(t, str) and t.strip():
            return t.strip()
    except Exception:
        pass

    # 2) Generic content list (SDKs can vary)
    try:
        content = getattr(resp, "output", None)
        if content and isinstance(content, list):
            # typical shape: [{"content": [{"type":"output_text","text":"..."}], ...}]
            block = content[0]
            inner = block.get("content") if isinstance(block, dict) else None
            if inner and isinstance(inner, list) and len(inner):
                maybe = inner[0]
                # dict or attr; be defensive
                if isinstance(maybe, dict):
                    t = maybe.get("text", "")
                else:
                    t = getattr(maybe, "text", "") or ""
                if isinstance(t, str) and t.strip():
                    return t.strip()
    except Exception:
        pass

    # 3) Fallback to message-style (rare for responses)
    try:
        ch0 = resp.choices[0]
        msg = getattr(ch0, "message", None)
        if msg:
            t = getattr(msg, "content", "") or ""
            if isinstance(t, str) and t.strip():
                return t.strip()
    except Exception:
        pass

    return ""

################################################################################
# Batch helpers                                                                 #
################################################################################

def _batch_endpoint_and_body(model: str, prompt: str, max_out: int):
    """
    Returns (endpoint_name, url, body_dict) based on the model family.
    - gpt-5*  -> Responses API (/v1/responses) with input + max_output_tokens
    - gpt-4.1*-> Chat Completions (/v1/chat/completions) with messages + max_tokens
    """
    m = (model or "").strip().lower()
    if m.startswith("gpt-5"):
        # Responses API shape
        url = "/v1/responses"
        body = {
            "model": model,
            "input": [{"role": "user", "content": prompt}],
            "max_output_tokens": max_out,
            # DO NOT set temperature; gpt-5* often only accepts default
        }
        return ("responses", url, body)
    else:
        # Chat Completions shape
        url = "/v1/chat/completions"
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_out,
        }
        return ("chat", url, body)

BATCH_TERMINAL_STATES = {"completed", "failed", "cancelled", "expired"}

def get_batch_status(client: OpenAI, batch_id: str) -> dict:
    try:
        b = client.batches.retrieve(batch_id)
    except Exception as e:
        return {"id": batch_id, "status": "unknown", "error": str(e)}
    return dict(
        id=b.id, status=b.status,
        input_file_id=getattr(b, "input_file_id", None),
        output_file_id=getattr(b, "output_file_id", None),
        error_file_id=getattr(b, "error_file_id", None),
        created_at=getattr(b, "created_at", None),
        cancelled_at=getattr(b, "cancelled_at", None),
        completed_at=getattr(b, "completed_at", None),
        in_progress_at=getattr(b, "in_progress_at", None),
    )

def poll_batch_until_done(client: OpenAI, batch_id: str, max_wait: Optional[int], poll_every: int) -> dict:
    start = time.time()
    while True:
        info = get_batch_status(client, batch_id)
        sys.stderr.write(f"[info] batch {batch_id} status: {info.get('status','unknown')}\n")
        if info.get("status") in BATCH_TERMINAL_STATES:
            return info
        if max_wait is not None and (time.time() - start) >= max_wait:
            sys.stderr.write("[warn] Batch still not terminal at max-wait limit.\n")
            return info
        time.sleep(max(1, poll_every))

def download_openai_file(client: OpenAI, file_id: str) -> bytes:
    f = client.files.content(file_id)
    try:
        return f.read()
    except AttributeError:
        return f

def process_batch_results_ndjson(ndjson_bytes: bytes) -> List[Tuple[str, str, Optional[str]]]:
    """Return list of (heading, text, model_if_any)."""
    out: List[Tuple[str, str, Optional[str]]] = []
    for line in ndjson_bytes.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        heading = (obj.get("metadata") or {}).get("heading") or obj.get("custom_id") or "Unknown"
        model = None
        text = ""
        # Try chat-completions style
        try:
            body = obj["response"]["body"]
            model = body.get("model") or model
            text = body["choices"][0]["message"]["content"]
        except Exception:
            # Try responses style
            try:
                body = obj["response"]["body"]
                model = body.get("model") or model
                part = body["output"][0]
                text = part.get("content") or part.get("text") or ""
            except Exception:
                text = ""
        out.append((heading, (text or "").strip(), model))
    return out

################################################################################
# Main                                                                          #
################################################################################

md_fh = None
json_fh = None
md_path = None
jsonl_path = None

def main() -> None:
    p = argparse.ArgumentParser(description="Arbuthnot Books novel toolkit")
    p.add_argument("draft", nargs="?", help="Input .md or .txt file (not required for --batch-status/--batch-fetch)")
    modes = [
        "concise","discursive","critique","entities","continuity","overview",
        "inline-edit","copyedit","improvement_suggestions","detailed-summary",
        "sensitivity-analysis","plot-points",
    ]
    p.add_argument("--mode", choices=modes, default="concise")
    p.add_argument("--whole", action="store_true", help="Treat entire file as one section")
    p.add_argument("--format", choices=["md","json","both"], default="md")
    p.add_argument("--model", default="gpt-4.1-mini", choices=["gpt-5","gpt-5-mini","gpt-4.1","gpt-4.1-mini"])
    p.add_argument("--engine", choices=["auto","responses","chat"], default="chat")
    p.add_argument("--fallback-model", choices=["gpt-5","gpt-5-mini","gpt-4.1","gpt-4.1-mini"], help="Retry with this model if primary fails")
    # batch controls
    p.add_argument("--batch-submit", action="store_true", help="Prepare and submit a discounted Batch job")
    p.add_argument("--batch-fetch", help="Fetch results of an existing Batch job id")
    p.add_argument("--batch-dry-run", action="store_true", help="Prepare Batch input only (no submit)")
    p.add_argument("--batch-status", help="Print Batch status and exit")
    p.add_argument("--batch-wait", type=int, help="With --batch-fetch: wait up to N seconds for completion")
    p.add_argument("--batch-poll-every", type=int, default=10, help="Polling interval seconds for --batch-wait")
    p.add_argument("--batch-first", type=int, help="Batch debug: include only the first N sections when building/submitting a batch")
    p.add_argument("--batch-range", nargs=2, type=int, metavar=("START", "END"), help="Batch debug: include only sections START..END (1-based, inclusive) when building/submitting a batch")
    # token/limits
    p.add_argument("--max-tokens-out", type=int, default=1024)
    p.add_argument("--max-section-tokens", type=int, default=120_000)
    p.add_argument("--heading-regex")
    p.add_argument("--min-heading-level", type=int, default=1, help="Minimum ATX heading level (1..6) considered a section")
    p.add_argument("--ignore-headings", help="Regex; skip any section whose heading matches (case-insensitive)")
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--list-sections", action="store_true", help="Print detected sections with body lengths and exit")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dump-raw", action="store_true")
    # profiling / pricing
    p.add_argument("--profile", action="store_true")
    p.add_argument("--price", action="store_true", help="Add token cost estimates to profile rows")
    p.add_argument("--fx-rate", type=float, help="Optional USD→GBP multiplier for cost columns")
    p.add_argument("--assume-cached-percent", type=float, default=0.0, help="Reserved for caching models (0–100)")
    p.add_argument("--estimate-only", action="store_true", help="Do not call the API; write profile.csv with section-by-section token and cost estimates, then exit")
    p.add_argument("--estimate-out", type=int, default=None, help="Estimated tokens OUT per section (default: use --max-tokens-out if set, else 600)")
    # reliability
    p.add_argument("--request-timeout", type=int, default=60)
    p.add_argument("--max-retries", type=int, default=2)
    # continuity
    p.add_argument("--entities-file", type=Path, help="Path to entities.jsonl for continuity mode")
    p.add_argument("--resume-entities", type=Path, help="Resume/patch entities: path to existing entities.jsonl; skip headings already present and append only new ones.")
    # concise shaping
    p.add_argument("--normalize-output", action="store_true")
    p.add_argument("--continue-on-truncate", action="store_true")
    p.add_argument("--continue-max", type=int, default=2)
    # NEW: profile on fetch
    p.add_argument("--profile-on-fetch", action="store_true",
                   help="When fetching a completed batch, estimate tokens & costs from outputs (and prompts if draft is supplied).")
    p.add_argument("--cost-from-profile", help="Estimate total cost from a profile.csv (add costs if missing).")

    args = p.parse_args()

    # --- cost-from-profile subcommand -------------------------------------------
    if getattr(args, "cost_from_profile", None) is not None:
        import pandas as pd

        csv_path = args.cost_from_profile  # may be str or Path
        if not os.path.exists(csv_path):
            sys.exit(f"[error] Not found: {csv_path}")

        # require token columns so we can compute costs
        df = pd.read_csv(csv_path)
        if "model" not in df.columns:
            sys.exit("[error] profile missing 'model' column")
        if "tokens_in_est" not in df.columns or "tokens_out_est" not in df.columns:
            sys.exit("[error] profile missing 'tokens_in_est' / 'tokens_out_est' (re-run with --profile / --profile-on-fetch)")

        # ensure columns exist
        for c in ("cost_input","cost_output","cost_total"):
            if c not in df.columns:
                df[c] = 0.0
        if "currency" not in df.columns:
            df["currency"] = ""

        fx = getattr(args, "fx_rate", None)

        # numeric re-use of your PRICE table
        def _estimate_cost_numeric(model: str, tin: float, tout: float, fx_rate: float | None):
            r = PRICE.get(model)
            if not r:
                return 0.0, 0.0, 0.0, CURRENCY
            cin  = (tin  / 1000.0) * r["in"]
            cout = (tout / 1000.0) * r["out"]
            total = cin + cout
            cur = "GBP" if (fx_rate and fx_rate > 0) else CURRENCY
            if fx_rate and fx_rate > 0:
                cin *= fx_rate; cout *= fx_rate; total *= fx_rate
            return cin, cout, total, cur

        # compute / overwrite cost columns from tokens
        rows = []
        for _, row in df.iterrows():
            model = str(row.get("model","")).strip()
            tin = float(row.get("tokens_in_est", 0) or 0.0)
            tout = float(row.get("tokens_out_est", 0) or 0.0)
            cin, cout, ctot, cur = _estimate_cost_numeric(model, tin, tout, fx)
            rows.append((cin, cout, ctot, cur))

        if rows:
            df["cost_input"]  = [r[0] for r in rows]
            df["cost_output"] = [r[1] for r in rows]
            df["cost_total"]  = [r[2] for r in rows]
            # keep any existing non-empty currency, else set computed
            df["currency"] = [
                (str(df.at[i, "currency"]).strip() or rows[i][3]) for i in range(len(df))
            ]

        total_in  = float(df["cost_input"].sum())
        total_out = float(df["cost_output"].sum())
        total_all = float(df["cost_total"].sum())
        currencies = df["currency"].dropna().unique().tolist() or [("GBP" if (fx and fx > 0) else CURRENCY)]
        cur = currencies[0]

        print(f"[ok] Cost summary ({cur}):")
        print(f"  Input:  {total_in:.4f} {cur}")
        print(f"  Output: {total_out:.4f} {cur}")
        print(f"  Total:  {total_all:.4f} {cur}")

        base, ext = os.path.splitext(str(csv_path))
        out_file = f"{base}_with_costs{ext}"
        df.to_csv(out_file, index=False)
        print(f"[ok] Updated file written: {out_file}")
        sys.exit(0)

    # Non-file batch ops can run without draft
    if (args.batch_status or args.batch_fetch) and not args.draft:
        pass
    elif not args.draft:
        # draft required for normal run / submit / dry-run
        if not (args.batch_status or args.batch_fetch):
            p.error("draft is required unless using --batch-status/--batch-fetch")

    if args.mode == "concise" and not args.normalize_output:
        args.normalize_output = True

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("[error] OPENAI_API_KEY not set")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = ("{}_whole".format(args.mode)) if args.whole else args.mode
    out_dir = args.out_dir or Path(f"{base}_{ts}")

    # --- batch status ---
    if args.batch_status:
        info = get_batch_status(client, args.batch_status)
        sys.stderr.write(json.dumps(info, indent=2) + "\n")
        return

    # --- batch fetch (with auto-wait / profile-on-fetch) ---
    if args.batch_fetch:
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.batch_wait:
            info = poll_batch_until_done(client, args.batch_fetch, args.batch_wait, args.batch_poll_every)
        else:
            info = get_batch_status(client, args.batch_fetch)
            sys.stderr.write(f"[info] batch {args.batch_fetch} status: {info.get('status','unknown')}\n")

        status = info.get("status")
        if status == "completed":
            out_id = info.get("output_file_id")
            if not out_id:
                sys.stderr.write("[error] Completed but no output_file_id.\n")
                return
            ndjson_bytes = download_openai_file(client, out_id)
            triples = process_batch_results_ndjson(ndjson_bytes)  # [(heading, text, model?), ...]

            # Write outputs (md/jsonl) like normal
            md_path = out_dir / f"{args.mode}.md" if args.format in {"md","both"} and args.mode != "entities" else None
            jsonl_path = out_dir / f"{args.mode}.jsonl" if args.format in {"json","both"} else None

            if md_path:
                with md_path.open("w", encoding="utf-8") as md_fh:
                    for heading, text, _model in triples:
                        if args.mode == "concise" and args.normalize_output and text:
                            summary, title = normalize_concise(text)
                            norm = summary.strip()
                            if title:
                                norm = f"{norm}\n\n*Suggested title:* {title}"
                            text = norm
                        md_fh.write(f"**{heading}**\n\n{(text or '').strip()}\n\n")
                postprocess_markdown_sections(md_path)
                prepend_cli_metadata_to_output(md_path, args, args.mode)

            if jsonl_path:
                with jsonl_path.open("w", encoding="utf-8") as jf:
                    for heading, text, _model in triples:
                        jf.write(json.dumps({"section_heading": heading, "raw": text}, ensure_ascii=False) + "\n")

            # Optional: profile-on-fetch
            if args.profile_on_fetch:
                # If draft is provided, reconstruct prompts to estimate input token counts.
                tokens_in_map: Dict[str, int] = {}
                if args.draft:
                    draft_text = read_text(Path(args.draft))
                    # slice by headings so we can rebuild the same prompts
                    if args.whole:
                        sections = [("Full Manuscript", draft_text.strip())]
                    else:
                        heading_rx = compile_heading_regex(args.heading_regex, args.min_heading_level)
                        sections = list(iter_sections(draft_text, heading_rx))
                        if args.ignore_headings:
                            rx_ign = re.compile(args.ignore_headings, re.IGNORECASE)
                            sections = [(h,b) for (h,b) in sections if not rx_ign.search(h)]
                    # build prompt per heading (match by heading text)
                    for h, b in sections:
                        ptxt = build_prompt(args.mode, h, b, args.format)
                        tokens_in_map[h] = num_tokens(args.model, ptxt)

                # Write profile.csv
                import csv
                prof_path = out_dir / "profile.csv"
                with open(prof_path, "w", newline="", encoding="utf-8") as pf:
                    pw = csv.writer(pf)
                    pw.writerow(["idx","heading","endpoint","model","tokens_in_est","tokens_out_est","latency_s","cost_input","cost_output","cost_total","currency"])
                    for i, (heading, text, model_line) in enumerate(triples, 1):
                        tokens_out_est = num_tokens(model_line or args.model, text or "")
                        tokens_in_est = tokens_in_map.get(heading, 0) if tokens_in_map else 0
                        ci, co, ct, cur = estimate_cost(model_line or args.model, tokens_in_est, tokens_out_est, args.fx_rate) if args.price else ("","","","")
                        pw.writerow([i, heading, "batch", model_line or args.model, tokens_in_est, tokens_out_est, "", ci, co, ct, cur])

            print(f"[ok] Fetched batch {args.batch_fetch} into {out_dir.resolve()}")
            return

        elif status in {"failed","cancelled","expired"}:
            sys.stderr.write(f"[error] Batch is terminal with status '{status}'.\n")
            err_id = info.get("error_file_id")
            if err_id:
                try:
                    err_bytes = download_openai_file(client, err_id)
                    err_path = out_dir / "batch_errors.ndjson"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    err_path.write_bytes(err_bytes)
                    sys.stderr.write(f"[info] Wrote error details: {err_path}\n")
                except Exception as e:
                    sys.stderr.write(f"[warn] Could not download error file: {e}\n")
            return
        else:
            sys.stderr.write("[error] Batch not completed yet. Use --batch-wait or try later.\n")
            return

    # ---- submit / dry-run need draft ----
    if args.batch_submit or args.batch_dry_run:
        if not args.draft:
            sys.exit("[error] --batch-submit/--batch-dry-run require a draft file path")
    draft_path = Path(args.draft) if args.draft else None
    text = read_text(draft_path) if draft_path else ""

    # Slice draft
    if args.whole:
        sections = [("Full Manuscript", text.strip())]
    else:
        heading_rx = compile_heading_regex(args.heading_regex, args.min_heading_level)
        sections = list(iter_sections(text, heading_rx))
        if args.ignore_headings:
            rx_ign = re.compile(args.ignore_headings, re.IGNORECASE)
            before = len(sections)
            sections = [(h,b) for (h,b) in sections if not rx_ign.search(h)]
            if args.verbose:
                sys.stderr.write(f"[info] Ignored {before - len(sections)} sections via --ignore-headings\n")
        if args.list_sections:
            for idx, (h,b) in enumerate(sections, 1):
                sys.stderr.write(f"[slice] {idx:03d} | {h} | {len(b.strip())} chars\n")
            return

    # === counters for the run ===
    processed = 0
    # attempted = 0   # (optional) if you later want to report attempts separately

    # ---- resume-entities preload (only for entities/json paths) ----
    done_headings: set[str] = set()
    if args.mode == "entities" and args.resume_entities:
        try:
            with args.resume_entities.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        h = obj.get("section_heading") or obj.get("heading")
                        if h:
                            done_headings.add(_norm_heading(h))
                    except Exception:
                        # skip malformed lines
                        pass
            if args.verbose:
                sys.stderr.write(
                    f"[resume] Preloaded {len(done_headings)} existing entity records from {args.resume_entities}\n"
                )
        except FileNotFoundError:
            sys.stderr.write(
                f"[warn] --resume-entities file not found: {args.resume_entities}\n"
            )

    # --- estimate-only early exit ------------------------------------------------
    if args.estimate_only:
        import csv

        out_dir.mkdir(parents=True, exist_ok=True)
        prof_path = out_dir / "profile.csv"

        # choose an OUT estimate
        est_out = args.estimate_out if args.estimate_out is not None else (
            args.max_tokens_out if args.max_tokens_out else 600
        )
        fx = getattr(args, "fx_rate", None)

        # Reuse the subset selector so estimate-only honors --batch-first / --batch-range
        try:
            sel_sections, index_offset, selection_desc = _select_sections_for_batch(sections, args)
        except NameError:
            sel_sections, index_offset, selection_desc = sections, 0, f"all {len(sections)}"

        if args.verbose:
            sys.stderr.write(f"[estimate] selection: {selection_desc}\n")

        with prof_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow([
                "idx","heading","endpoint","model",
                "tokens_in_est","tokens_out_est","latency_s",
                "cost_input","cost_output","cost_total","currency"
            ])
            processed = 0
            t0 = time.time()

            for j, (heading, body) in enumerate(sel_sections, 1):
                global_idx = index_offset + j  # 1-based position in original sequence
                if not body.strip():
                    continue
                prompt = build_prompt(args.mode, heading, body, args.format)
                t_in  = num_tokens(args.model, prompt)
                t_out = int(est_out)

                fin, fout, ftot, cur = estimate_cost_numeric(args.model, t_in, t_out, fx)
                w.writerow([
                    global_idx, heading, "estimate", args.model,
                    t_in, t_out, 0.0,
                    f"{fin:.6f}", f"{fout:.6f}", f"{ftot:.6f}", cur
                ])
                processed += 1

        if processed == 0:
            sys.exit("[error] No sections to estimate. Check headings/filters or selection flags.")
        print(f"[ok] Wrote estimate-only profile: {prof_path}")

        # Friendly summary (uses processed count, not total sections)
        try:
            import pandas as _pd
            df = _pd.read_csv(prof_path)
            total = float(df["cost_total"].astype(float).sum())
            cur = (df["currency"].dropna().unique().tolist() or [CURRENCY])[0]
            print(f"[ok] Estimated total: {total:.4f} {cur} across {processed} sections")
            if processed > 0:
                avg_cost = total / processed
                if cur in ("GBP", "EUR"):
                    print(f"[ok] Average per section: {avg_cost:.2f} {cur}")
                else:
                    print(f"[ok] Average per section: {avg_cost:.4f} {cur}")
        except Exception:
            pass
        sys.exit(0)

    # --- batch submit/dry-run ---
    if args.batch_submit or args.batch_dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        ndjson_path = out_dir / "batch_input.jsonl"

        # Decide batch endpoint family from model once (homogeneous batch)
        model_lc = (args.model or "").strip().lower()
        if model_lc.startswith("gpt-5"):
            batch_endpoint = "/v1/responses"
            def make_body(prompt: str) -> dict:
                return {
                    "model": args.model,
                    # Plain string input for maximum compatibility in batches
                    "input": prompt,
                    "max_output_tokens": args.max_tokens_out,
                }
        else:
            batch_endpoint = "/v1/chat/completions"
            def make_body(prompt: str) -> dict:
                return {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": args.max_tokens_out,
                }

        # Select subset for batch (supports --batch-first / --batch-range)
        batch_sections, index_offset, selection_desc = _select_sections_for_batch(sections, args)
        if args.verbose:
            sys.stderr.write(f"[batch] selection: {selection_desc}\n")
        if not batch_sections:
            sys.exit("[error] No sections selected for batch (check --batch-first/--batch-range).")

        count = 0
        with ndjson_path.open("w", encoding="utf-8") as f:
            for j, (heading, body_text) in enumerate(batch_sections, 1):
                global_idx = index_offset + j  # 1-based

                if not body_text.strip():
                    if args.verbose:
                        sys.stderr.write(f"[warn] Skipping empty body at idx {global_idx}: '{heading}'\n")
                    continue

                prompt = build_prompt(args.mode, heading, body_text, args.format)

                req = {
                    "custom_id": f"sec_{global_idx:04d}",
                    "method": "POST",
                    "url": batch_endpoint,      # match the batch create endpoint
                    "body": make_body(prompt),  # correct schema for the endpoint
                }
                # Intentionally NO "metadata" here (avoid schema surprises)
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
                count += 1

        if args.verbose:
            sys.stderr.write(f"[info] Batched {count} items to {ndjson_path}\n")

        if args.batch_dry_run:
            print(f"[ok] Wrote batch input (dry-run): {ndjson_path}")
            return

        # Upload & submit; endpoint must match all per-line 'url's
        file_obj = client.files.create(file=ndjson_path.open("rb"), purpose="batch")
        b = client.batches.create(
            input_file_id=file_obj.id,
            endpoint=batch_endpoint,     # << match the NDJSON 'url' above
            completion_window="24h",
        )
        print(f"[ok] Submitted batch: {b.id}\n    Save this id and later run with: --batch-fetch {b.id}")
        return

        # Upload & submit
        file_obj = client.files.create(file=ndjson_path.open("rb"), purpose="batch")
        # NB: endpoint here is only advisory for the batch; each line has its own 'url'
        b = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",  # kept for compatibility; per-line 'url' wins
            completion_window="24h",
        )
        print(f"[ok] Submitted batch: {b.id}\n    Save this id and later run with: --batch-fetch {b.id}")
        return

    # --- synchronous path ---
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- profiling writer (initialise to None so guards are safe) ---
    prof_writer = None

    # --- initialise file handle vars ---
    md_fh = None
    json_fh = None
    md_path = None
    jsonl_path = None
    prof_writer = None

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Markdown path / handle
    if args.format in {"md", "both"} and args.mode != "entities":
        md_path = out_dir / f"{args.mode}.md"
        md_fh = open(md_path, "w", encoding="utf-8")

    # JSONL path / handle
    if args.format in {"json", "both"}:
        if args.mode == "entities" and args.resume_entities:
            jsonl_path = Path(args.resume_entities)
            json_fh = open(jsonl_path, "a", encoding="utf-8")  # append to existing
            if args.verbose:
                sys.stderr.write(f"[resume] Appending new entities to: {jsonl_path}\n")
        else:
            jsonl_path = out_dir / f"{args.mode}.jsonl"
            json_fh = open(jsonl_path, "w", encoding="utf-8")

    # Profile writer if enabled
    if args.profile:
        import csv
        prof_path = out_dir / "profile.csv"
        prof_writer = csv.writer(open(prof_path, "w", newline="", encoding="utf-8"))
        prof_writer.writerow(["idx", "heading", "endpoint", "model",
                              "prompt_toks", "completion_toks", "total_toks",
                              "elapsed", "cost", "cached", "status"])
    CTX_LIMITS = {"gpt-5-mini": 1_000_000, "gpt-5": 1_000_000, "gpt-4.1": 128_000, "gpt-4.1-mini": 1_000_000}
    CTX_LIMIT = CTX_LIMITS.get(args.model, 128_000)

    processed = 0
    try:
        # Accumulate sections that produced no output (for later patch runs)
        missing_sections: list[tuple[int, str]] = []

        for idx, (heading, body) in enumerate(tqdm(sections, desc="Processing", unit="section"), 1):
            if (not body.strip()) and (not args.whole):
                if args.verbose:
                    sys.stderr.write(f"[warn] '{heading}' has no body text; skipped.\n")
                if args.profile and prof_writer:
                    prof_writer.writerow([idx, heading, "chat" if args.engine=="chat" else "responses", args.model, 0, 0, 0.0, "", "", "", "skipped"])
                continue

            prompt = build_prompt(args.mode, heading, body, args.format, prior=None)
            total_in = num_tokens(args.model, prompt)
            if args.verbose:
                sys.stderr.write(f"[debug] {heading}: {total_in} tokens in\n")
            if total_in + args.max_tokens_out + 10 > CTX_LIMIT:
                sys.stderr.write(f"[warn] '{heading}' exceeds {args.model} context window; skipped.\n")
                if args.profile and prof_writer:
                    prof_writer.writerow([idx, heading, "chat" if args.engine=="chat" else "responses", args.model, total_in, 0, 0.0, "", "", "", "skipped_ctx"])
                continue

            t0 = time.time()
            response = call_llm(
                client=client,
                model=args.model,
                prompt=prompt,
                max_out=args.max_tokens_out,
                want_json=(args.format in {"json","both"} and args.mode in {"entities","continuity"}),
                engine=args.engine,
                fallback_model=args.fallback_model,
                verbose=args.verbose,
                dump_raw_dir=str(out_dir) if args.dump_raw else None,
                heading=heading,
                request_timeout=args.request_timeout,
                max_retries=args.max_retries,
            )
            dt = time.time() - t0

            # If the model failed to return usable text, record for patching and skip
            if response is None or (isinstance(response, str) and not response.strip()):
                sys.stderr.write(f"[warn] Empty/no response for '{heading}'. Marking missing.\n")
                missing_sections.append((idx, heading))
                if args.profile and prof_writer:
                    # You can refine 'endpoint' string if you track which branch was used
                    prof_writer.writerow([idx, heading, "no-output", args.model, 0, 0, 0.0, "", "", "", "skipped"])
                continue

            if not response or not response.strip():
                sys.stderr.write(f"[warn] Empty response for '{heading}'. Skipping write.\n")
                if args.profile and prof_writer:
                    ci, co, ct, cur = estimate_cost(args.model, total_in, 0, args.fx_rate) if args.price else ("","","","")
                    prof_writer.writerow([idx, heading, "chat" if args.engine=="chat" else "responses", args.model, total_in, 0, round(dt,3), ci, co, ct, cur])
                continue

            if args.mode == "concise" and args.format in {"md","both"}:
                if args.continue_on_truncate:
                    tries = 0
                    while _looks_truncated(response) and tries < args.continue_max:
                        tries += 1
                        cont_prompt = (
                            "Continue the previous SUMMARY from exactly where it stopped. "
                            "Do not repeat earlier text. Finish cleanly.\n\n"
                            f"PREVIOUS END:\n{response[-600:]}"
                        )
                        cont = call_llm(
                            client=client,
                            model=args.model,
                            prompt=cont_prompt,
                            max_out=min(768, args.max_tokens_out),
                            want_json=False,
                            engine=args.engine,
                            fallback_model=args.fallback_model,
                            verbose=args.verbose,
                            dump_raw_dir=str(out_dir) if args.dump_raw else None,
                            heading=heading + " (cont)",
                            request_timeout=args.request_timeout,
                            max_retries=args.max_retries,
                        ) or ""
                        response = (response + "\n" + cont.strip()).strip()
                if args.normalize_output:
                    summary, title = normalize_concise(response)
                    normalized = summary.strip()
                    if title:
                        normalized = f"{normalized}\n\n*Suggested title:* {title}"
                    response = normalized

            if md_fh:
                md_fh.write(f"**{heading}**\n\n{response.strip()}\n\n")

            if json_fh and args.format in {"json","both"}:
                if args.mode in {"entities","continuity"}:
                    try:
                        data = json.loads(response)
                    except Exception:
                        data = {"section_heading": heading, "raw": response}
                    json_fh.write(json.dumps(data, ensure_ascii=False) + "\n")
                else:
                    json_fh.write(json.dumps({"section_heading": heading, "raw": response}, ensure_ascii=False) + "\n")

            if args.profile and prof_writer:
                tokens_out_est = num_tokens(args.model, response)
                ci, co, ct, cur = estimate_cost(args.model, total_in, tokens_out_est, args.fx_rate) if args.price else ("","","","")
                endpoint = "responses" if (args.engine=="responses" or (args.engine=="auto" and args.model.startswith("gpt-5"))) else "chat"
                prof_writer.writerow([idx, heading, endpoint, args.model, total_in, tokens_out_est, round(dt,3), ci, co, ct, cur])

            processed += 1

    finally:
        try:
            md_fh and md_fh.close()
        except Exception:
            pass
        try:
            json_fh and json_fh.close()
        except Exception:
            pass

    if processed == 0:
        sys.stderr.write("[warn] No sections produced (all empty or skipped).\n")
        return

    if args.format in {"md","both"} and args.mode != "entities":
        postprocess_markdown_sections(md_path)
        prepend_cli_metadata_to_output(md_path, args, args.mode)

        # After processing loop: persist and summarize any missing sections
        if missing_sections:
            # Write a simple list
            miss_path = out_dir / "missing_sections.txt"
            with miss_path.open("w", encoding="utf-8") as mf:
                for i, h in missing_sections:
                    mf.write(f"{i:03d} | {h}\n")

            # Also print condensed ranges, e.g., 12..14, 27..27
            ranges = []
            for i, _ in sorted(missing_sections, key=lambda t: t[0]):
                if not ranges or i != ranges[-1][1] + 1:
                    ranges.append([i, i])
                else:
                    ranges[-1][1] = i

            sys.stderr.write(f"[info] Missing sections written to: {miss_path}\n")
            if ranges:
                sys.stderr.write("[hint] You can patch just these with --batch-range:\n")
                for a, b in ranges:
                    if a == b:
                        sys.stderr.write(f"       --batch-range {a} {b}\n")
                    else:
                        sys.stderr.write(f"       --batch-range {a} {b}\n")

print(f"[ok] Wrote {processed}/{len(sections)} sections to {out_dir.resolve()}")

if md_fh and args.format in {"md", "both"} and md_path:
    print(f"  • Markdown: {_pretty_rel(md_path, out_dir.parent)}")

if json_fh and args.format in {"json", "both"} and jsonl_path:
    print(f"  • JSONL:   {_pretty_rel(jsonl_path, out_dir.parent)}")

if __name__ == "__main__":
    main()
