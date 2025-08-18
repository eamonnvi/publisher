#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
novel_toolkit_allin.py — Arbuthnot Books helper (clean, integrated build)
=========================================================================
Features
--------
• Modes: concise, discursive, critique, entities, continuity, overview,
  inline-edit, copyedit, improvement_suggestions, detailed-summary,
  sensitivity-analysis, plot-points
• Engines: Chat Completions + Responses (auto-switch + future-proofed params)
• Batch: submit / dry-run / status / fetch, plus profile-on-fetch & pricing
• Slicer: --batch-first / --batch-range available in sync & batch paths
• Resume: --resume-entities to append skipped sections to existing JSONL
• Profiling: tokens/latency per section; pricing with optional GBP FX rate
• Estimate-only mode for quick cost estimation without calling the API
• Heading filters: --min-heading-level, --ignore-headings (regex), --whole
• Markdown post-process cleaner + CLI metadata header
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

# -------------------------- optional dependencies ----------------------------
try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None  # fall back to rough estimator

try:
    from openai import OpenAI
except Exception as e:
    sys.stderr.write("Install deps: pip install openai tiktoken tqdm\n")
    raise

# tqdm only for progress bar; we keep optional
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(x, **kwargs):  # type: ignore
        return x

# ----------------------------- prompt templates ------------------------------
PROMPT_TEMPLATES: Dict[str, str] = {
    # Summaries
    "concise": (
        "Summarise the following {label} of a novel entitled '{heading}'. "
        "Include major events, character actions, and information that may be important for the overall plot. "
        "Suggest a more suitable title for the chapter. "
        "Write in clear British English prose.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "discursive": (
        "Provide an in-depth, analytical synopsis of the section titled '{heading}'. "
        "Discuss character motivations, conflicts, thematic significance, subtext, and narrative strategies. "
        "Highlight details with later narrative impact. Do not omit spoilers.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    # Developmental critique
    "critique_md": (
        "You are an experienced developmental editor. Evaluate the section titled '{heading}' as follows:\n\n"
        "1. **Strengths**\n2. **Weaknesses**\n3. **Genre fit and innovation**\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "critique_json": (
        "You are an experienced developmental editor. Return STRICT JSON with keys "
        "'strengths', 'weaknesses', 'improvements' – each an array of strings.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    # Improvement suggestions
    "improvement_suggestions": (
        "For the section titled '{heading}', list specific, actionable revision suggestions. "
        "Focus on clarity, character motivation, suspense, pacing, and ambiguity resolution. "
        "Include sample rewrites if apt.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    # Entity extraction
    "entities_json": (
        "Extract structured facts from the section titled '{heading}'. "
        "Return only a single valid JSON object with keys: "
        "'section_heading', 'characters', 'pov_character', 'themes', 'locations', 'timestamp'. "
        "Do not include explanations, markdown, or any non-JSON output.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    # Continuity
    "continuity_md": (
        "You are a continuity editor. Compare the **current section** to these **prior facts**. "
        "Flag contradictions (dates, ages, locations, actions) and note ambiguous or potentially inconsistent elements.\n\n"
        "### Prior facts\n```json\n{prior_entities}\n```\n\n### Current section\n{body}\n"
    ),
    "continuity_json": (
        "Compare current section to prior facts. Return ONLY a JSON array of inconsistency strings (empty if none).\n\n"
        "Prior facts: {prior_entities}\n\nCurrent section:\n{body}\n"
    ),
    # Copyedit
    "copyedit_md": (
        "## Identity\nYou are a line-by-line copy editor in a British literary publishing house.\n\n"
        "## Task\nIdentify typos, grammar issues, redundancies, repetitions.\n\n"
        "## Prohibitions\nDo not break sentences, reduce the number of adverbs, spot clichés, or flag contractions.\n\n"
        "## Output\nBulleted Markdown: line number, snippet, issue, suggestion.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "copyedit_json": (
        "(Same identity/task) Return JSON array of objects "
        "{line_number:int; original:str; issue:str; suggestion:str}.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    # Overview / detailed
    "overview": (
        "Write a 2,000–2,500-word editorial overview of the manuscript below. "
        "Cover plot, character arcs, themes, style, structure, pacing, genre conventions, and continuity issues.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "detailed-summary": (
        "Write a 5,000–10,000-word chapter-by-chapter summary of the manuscript below.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "sensitivity-analysis": (
        "Read the sexually explicit sections of the manuscript below. "
        "Assess whether they are justified by period and context. "
        "Suggest rewordings for borderline passages.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "plot-points": (
        "Analyse whether the current draft telegraphs that Mrs Nairn and the glamorous actress are Sheena in disguise; "
        "and that Marlboro cigarettes mark Katrin. Evaluate the Paddington Green scene for whether Ulrike is truly Ulrike "
        "(declining cigarettes). Consider whether readers can infer Grace's developing attraction to Ulrike at Newnham. "
        "Identify plot points that fail or could be sharpened.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
}

# ------------------------------ pricing table --------------------------------
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

def estimate_cost_numeric(model: str, tokens_in: float, tokens_out: float, fx_rate: Optional[float]) -> tuple[float,float,float,str]:
    r = PRICE.get(model)
    if not r:
        return (0.0, 0.0, 0.0, CURRENCY)
    fin  = (tokens_in  / 1000.0) * r["in"]
    fout = (tokens_out / 1000.0) * r["out"]
    total = fin + fout
    cur = "GBP" if (fx_rate and fx_rate > 0) else CURRENCY
    if fx_rate and fx_rate > 0:
        fin *= fx_rate; fout *= fx_rate; total *= fx_rate
    return (fin, fout, total, cur)

# ------------------------------- helpers -------------------------------------
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def compile_heading_regex(custom: Optional[str], min_level: Optional[int]) -> re.Pattern[str]:
    if custom:
        return re.compile(custom, re.MULTILINE)
    # default: ATX headings or **Heading**
    levels = min_level or 1
    # at least `levels` hashes, up to 6
    default = rf"^\s*(?:#{{{levels},6}}\s*([^#\n]+?)\s*#*\s*$|\*\*([^*]+?)\*\*\s*$)"
    return re.compile(default, re.MULTILINE)

def iter_sections(text: str, heading_rx: re.Pattern[str], min_level: Optional[int]=None) -> Iterable[Tuple[str, str]]:
    matches = list(heading_rx.finditer(text))
    if not matches:
        yield "Full Manuscript", text.strip(); return
    for i, m in enumerate(matches):
        start = m.end(); end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = next((g for g in m.groups() if g), "").strip()
        yield heading, text[start:end].strip()

def _encoding_for_model(model: str):
    if tiktoken:
        try:
            return tiktoken.encoding_for_model(model)
        except Exception:
            try:
                return tiktoken.get_encoding("o200k_base")
            except Exception:
                return None
    return None

def num_tokens(model: str, *chunks: str) -> int:
    enc = _encoding_for_model(model)
    if enc:
        return sum(len(enc.encode(c)) for c in chunks if c)
    # rough fallback: 1 token ≈ 4 chars (very approximate)
    return sum(max(1, len(c)//4) for c in chunks if c)

def safe_slug(s: str) -> str:
    s2 = re.sub(r"[^a-zA-Z0-9._-]+", "_", s.strip())
    return s2[:80] if len(s2) > 80 else s2

def prepend_cli_metadata_to_output(output_path: Path, args: argparse.Namespace, mode: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_abs_path = output_path.resolve()
    arglist = []
    for key, value in vars(args).items():
        if key not in ("draft", "out_dir") and value not in (None, False):
            if isinstance(value, Path):
                value = str(value)
            keyflag = f"--{key.replace('_','-')}"
            arglist.append(keyflag if value is True else f"{keyflag} {value}")
    cli_command = f"python3 {Path(sys.argv[0]).name} {args.draft or ''} " + " ".join(arglist)
    header = (
        f"# {mode.capitalize()} Report\n\n"
        f"**Generated:** {timestamp}\n"
        f"**Output file:** `{output_abs_path}`\n"
        f"**CLI command:**\n\n"
        f"```bash\n{cli_command}\n```\n\n---\n\n"
    )
    original_text = output_path.read_text(encoding="utf-8")
    output_path.write_text(header + original_text, encoding="utf-8")

def postprocess_markdown_sections(md_path: Path):
    """Ensure a blank line between sections and Suggested title placed last."""
    text = md_path.read_text(encoding="utf-8")
    # Add a blank line before any bold heading **...**
    text = re.sub(r"\n(\*\*[^*\n]+?\*\*)", r"\n\n\1", text)
    # Coerce "*Suggested title:* ..." to be last item in a block is handled upstream,
    # but we ensure at least one trailing newline:
    if not text.endswith("\n"):
        text += "\n"
    md_path.write_text(text, encoding="utf-8")

def normalize_concise(response: str) -> Tuple[str, Optional[str]]:
    """
    Accepts any style and tries to return (SUMMARY text, title or None).
    Handles cases where model emitted SUMMARY:/TITLE: or plain text.
    """
    r = response.strip()
    title = None
    # Try to parse a TITLE line
    m = re.search(r"(?im)^\s*(?:title|suggested\s*title)\s*:\s*(.+?)\s*$", r)
    if m:
        title = m.group(1).strip()
    # Try to remove SUMMARY: prefix if present
    r = re.sub(r"(?im)^\s*summary\s*:\s*", "", r).strip()
    # If the model emitted a Markdown section, keep it; otherwise just the text
    return (r.strip(), title)

def _pretty_rel(p: Path, parent: Path) -> str:
    try:
        return str(p.relative_to(parent))
    except Exception:
        return str(p)

# ----------------------- prompt builder & mode mapping ------------------------
def build_prompt(mode: str, heading: str, body: str, fmt: str, prior: Optional[str]=None) -> str:
    key_map = {
        "critique": "critique_json" if fmt in {"json","both"} else "critique_md",
        "entities": "entities_json",
        "continuity": "continuity_json" if fmt in {"json","both"} else "continuity_md",
        "copyedit": "copyedit_json" if fmt in {"json","both"} else "copyedit_md",
        # direct keys: concise, discursive, overview, inline-edit, etc.
    }
    template_key = key_map.get(mode, mode)
    if template_key not in PROMPT_TEMPLATES:
        raise KeyError(f"No prompt template for mode '{mode}'.")
    template = PROMPT_TEMPLATES[template_key]
    norm = (heading or "").strip().lower()
    label = "manuscript" if norm in {"full manuscript", "manuscript"} else "section"
    return template.format(
        label=label, LABEL=label.upper(), heading=heading, body=body,
        prior_entities=prior or "[]"
    )

# ------------------- OpenAI call (chat vs responses) -------------------------
def _chat_param_name_for_max_tokens(model: str) -> str:
    """Some models require max_completion_tokens instead of max_tokens."""
    m = (model or "").strip().lower()
    return "max_completion_tokens" if m.startswith("gpt-5") else "max_tokens"

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
        token_key = _chat_param_name_for_max_tokens(m)
        kwargs = dict(
            model=m,
            messages=[{"role": "user", "content": prompt}],
            timeout=request_timeout,
        )
        kwargs[token_key] = max_out
        # NB: temperature defaults are fine; some gpt-5-mini variants only support default=1
        resp = client.chat.completions.create(**kwargs)
        if dump_raw_dir:
            Path(dump_raw_dir, f"{safe_slug(heading)}__chat.json").write_text(resp.model_dump_json(), encoding="utf-8")
        try:
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return None

    def responses_call(m: str) -> Optional[str]:
        if verbose:
            sys.stderr.write(f"[info] Using Responses API with model '{m}' (json={want_json})\n")
        kwargs = dict(
            model=m,
            input=[{"role": "user", "content": prompt}],
            timeout=request_timeout,
        )
        # new SDK prefers generation_config
        try:
            resp = client.responses.create(**kwargs, generation_config={"max_output_tokens": max_out})
        except TypeError:
            resp = client.responses.create(**kwargs, max_output_tokens=max_out)

        if dump_raw_dir:
            Path(dump_raw_dir, f"{safe_slug(heading)}__responses.json").write_text(resp.model_dump_json(), encoding="utf-8")

        text = None
        # Try modern field
        try:
            text = (resp.output_text or "").strip()  # type: ignore
        except Exception:
            pass
        # Fallback shape
        if not text:
            try:
                content = resp.output[0].content  # type: ignore
                if content and len(content) and "text" in content[0]:
                    text = (content[0]["text"] or "").strip()
            except Exception:
                text = None
        return text or None

    prefer_responses = (engine == "responses") or (engine == "auto" and model.strip().lower().startswith("gpt-5"))
    primary = responses_call if prefer_responses else chat_call
    secondary = chat_call if prefer_responses else responses_call

    models_to_try = [(model, primary), (model, secondary)]
    if fallback_model and fallback_model != model:
        models_to_try += [(fallback_model, primary), (fallback_model, secondary)]

    err_last = None
    for m, fn in models_to_try:
        tries = 0
        while tries <= max_retries:
            try:
                out = fn(m)
                if out:
                    return out
                else:
                    if verbose:
                        sys.stderr.write(f"[warn] Empty text from endpoint for {m}; retrying...\n")
            except Exception as e:
                err_last = e
                if verbose:
                    sys.stderr.write(f"[warn] endpoint error for {m}: {e}\n")
            tries += 1
            time.sleep(min(2.0, 0.5 + tries * 0.25))
    if verbose and err_last:
        sys.stderr.write(f"[warn] All attempts failed: {err_last}\n")
    return None

# ---------------------------- batch helpers ----------------------------------
def _batch_is_gpt5(model: str) -> bool:
    return (model or "").strip().lower().startswith("gpt-5")

def _batch_chat_token_key(model: str) -> str:
    return _chat_param_name_for_max_tokens(model)

def _batch_make_body(model: str, prompt: str, max_out: int) -> Tuple[str, dict]:
    if _batch_is_gpt5(model):
        endpoint = "/v1/responses"
        body = {"model": model, "input": prompt, "max_output_tokens": max_out}
    else:
        endpoint = "/v1/chat/completions"
        token_key = _batch_chat_token_key(model)
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], token_key: max_out}
    return endpoint, body

def _batch_write_ndjson(ndjson_path: Path, sections: List[Tuple[int, str, str]],
                        model: str, max_out: int, verbose: bool) -> int:
    count = 0
    with ndjson_path.open("w", encoding="utf-8") as f:
        for idx, heading, body in sections:
            if not (body or "").strip():
                if verbose:
                    sys.stderr.write(f"[warn] Skipping empty body idx={idx}: {heading}\n")
                continue
            prompt = build_prompt(args.mode, heading, body, args.format)  # uses global args within main()
            url, body_dict = _batch_make_body(model, prompt, max_out)
            row = {
                "custom_id": f"sec_{idx:04d}",
                "method": "POST",
                "url": url,
                "body": body_dict,
                "metadata": {"idx": idx, "heading": heading},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count

def get_batch_status(client: OpenAI, batch_id: str) -> dict:
    b = client.batches.retrieve(batch_id)
    return {
        "id": b.id,
        "status": b.status,
        "input_file_id": getattr(b, "input_file_id", None),
        "output_file_id": getattr(b, "output_file_id", None),
        "error_file_id": getattr(b, "error_file_id", None),
        "created_at": getattr(b, "created_at", None),
        "in_progress_at": getattr(b, "in_progress_at", None),
        "completed_at": getattr(b, "completed_at", None),
        "cancelled_at": getattr(b, "cancelled_at", None),
        "expires_at": getattr(b, "expires_at", None),
    }

def _poll_batch(
    client,
    batch_id: str,
    timeout_s: int | None,
    every_s: int,
    verbose: bool = True,
) -> dict:
    """
    Poll a batch until it reaches a terminal state or timeout.
    - timeout_s: None or <=0 means 'wait indefinitely'
    - every_s: poll interval in seconds (min 1s)
    """
    import time, random
    start = time.time()
    last_print = 0.0
    spinner = "|/-\\"
    si = 0

    # treat None or <=0 as infinite wait
    infinite = (timeout_s is None) or (timeout_s <= 0)
    interval = max(1, int(every_s))

    while True:
        try:
            info = get_batch_status(client, batch_id)
            status = info.get("status", "unknown")
        except Exception as e:
            # transient fetch problem; log once per interval and keep going
            status = "unknown"
            info = {"id": batch_id, "status": status, "error": str(e)}

        now = time.time()
        if verbose and (now - last_print) >= max(0.5, min(2, interval)):
            elapsed = int(now - start)
            rem = None if infinite else max(0, int(timeout_s - elapsed))  # type: ignore[arg-type]
            tick = spinner[si % len(spinner)]; si += 1
            msg = f"[poll {tick}] id={batch_id} status={status} elapsed={elapsed}s"
            if rem is not None:
                msg += f" remaining={rem}s"
            sys.stderr.write(msg + "\n")
            last_print = now

        if status in {"completed", "failed", "cancelled", "expired"}:
            return info

        if (not infinite) and (time.time() - start) >= timeout_s:  # type: ignore[operator]
            # Return the latest snapshot (likely still validating/in_progress)
            return info

        # Small jitter so multiple clients don’t sync-stampede the API
        time.sleep(interval + random.uniform(0, 0.25))

        if status in {"completed", "failed", "cancelled", "expired"}:
            if verbose:
                elapsed = int(time.time() - start)
                sys.stderr.write(
                    f"[done] id={batch_id} status={status} total_elapsed={elapsed}s\n"
                )
            return info

def download_openai_file(client: OpenAI, file_id: str) -> bytes:
    f = client.files.content(file_id)
    return f.read()  # type: ignore

def process_batch_results_ndjson(ndjson_bytes: bytes) -> List[Tuple[int, str, str, Optional[str]]]:
    out: List[Tuple[int, str, str, Optional[str]]] = []
    for line in ndjson_bytes.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        meta = rec.get("response", {}).get("metadata") or rec.get("metadata") or {}
        meta_req = rec.get("request", {}).get("metadata") or {}
        idx = meta.get("idx") or meta_req.get("idx")
        heading = meta.get("heading") or meta_req.get("heading") or ""
        text = None
        model_seen = rec.get("response", {}).get("body", {}).get("model")
        body = rec.get("response", {}).get("body") or {}

        if "output_text" in body and body["output_text"]:
            text = str(body["output_text"]).strip()
        elif "output" in body and isinstance(body["output"], list):
            try:
                content = body["output"][0]["content"]
                if content and isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            text = (part["text"] or "").strip()
                            break
            except Exception:
                pass

        if not text:
            try:
                choices = body.get("choices") or []
                if choices:
                    text = (choices[0]["message"]["content"] or "").strip()
            except Exception:
                pass

        try:
            idx = int(idx)
        except Exception:
            idx = -1

        out.append((idx, heading, text or "", model_seen))
    return out

# ------------------------------- slicer --------------------------------------
def select_sections_for_batch(sections: List[Tuple[str, str]], args: argparse.Namespace) -> List[Tuple[int, str, str]]:
    """Return [(idx, heading, body)] after applying --batch-first / --batch-range."""
    indexed = [(i, h, b) for i, (h, b) in enumerate(sections, 1)]
    if getattr(args, "batch_first", None):
        n = int(args.batch_first)
        return indexed[:max(0, n)]
    if getattr(args, "batch_range", None) and len(args.batch_range) == 2:
        a, b = int(args.batch_range[0]), int(args.batch_range[1])
        a = max(1, a)
        b = max(a, b)
        return [x for x in indexed if a <= x[0] <= b]
    return indexed

def _select_sections_for_batch(sections, args):
    """
    Adapter around select_sections_for_batch() to provide:
      (batch_sections, index_offset, selection_desc)

    - batch_sections: List[(heading, body)]
    - index_offset:   0-based offset of first selected item in original list
    - selection_desc: human-readable description ("first K of N", "A..B of N", "all N")
    """
    n = len(sections)
    indexed = select_sections_for_batch(sections, args)  # [(idx, heading, body)]

    if not indexed:
        return [], 0, "none selected"

    # Convert back to (heading, body)
    sel = [(h, b) for (i, h, b) in indexed]

    # Compute offset from original sequence (0-based)
    index_offset = max(0, indexed[0][0] - 1)

    # Build description
    if getattr(args, "batch_first", None):
        k = len(sel)
        return sel, 0, f"first {k} of {n}"

    if getattr(args, "batch_range", None) and len(args.batch_range) == 2:
        a, b = int(args.batch_range[0]), int(args.batch_range[1])
        a = max(1, a); b = max(a, b)
        return sel, (a - 1), f"{a}..{b} of {n}"

    return sel, 0, f"all {n}"

# ---------- Batch sidecar + results parsing (metadata-free) ----------
def _load_batch_map(map_path: Path) -> dict[str, str]:
    """Load custom_id -> heading mapping saved at submit time."""
    try:
        return json.loads(map_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _parse_batch_output_ndjson(ndjson_bytes: bytes, id_to_heading: dict[str, str]) -> list[tuple[str, str, str]]:
    """
    Return [(heading, text, model)] from the batch output NDJSON.
    Handles both /v1/responses and /v1/chat/completions result shapes,
    including the Batch wrapper where the real payload lives in response.body.
    """
    import json, re

    def _first_text_from_content(block):
        try:
            if isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    return block["text"]
                if isinstance(block.get("content"), list):
                    for c in block["content"]:
                        if isinstance(c, dict) and isinstance(c.get("text"), str):
                            return c["text"]
        except Exception:
            pass
        return ""

    triples: list[tuple[str, str, str]] = []

    for raw in ndjson_bytes.splitlines():
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue

        custom_id = obj.get("custom_id") or obj.get("id")
        heading = id_to_heading.get(custom_id, custom_id or "UNKNOWN")

        text = ""
        model = ""

        # ---- Batch wrapper: response.body is the actual API response ----
        rsp = obj.get("response")
        if isinstance(rsp, dict) and isinstance(rsp.get("body"), dict):
            payload = rsp["body"]
        else:
            payload = rsp if isinstance(rsp, dict) else {}

        # Try Responses API shapes first (payload from /v1/responses)
        if isinstance(payload, dict):
            model = payload.get("model", "") or model

            # 1) Convenience field
            if isinstance(payload.get("output_text"), str):
                text = (payload["output_text"] or "").strip()

            # 2) Canonical content blocks: payload["output"][0]["content"][0]["text"]
            if not text and isinstance(payload.get("output"), list) and payload["output"]:
                blk = payload["output"][0]
                if isinstance(blk, dict):
                    if isinstance(blk.get("text"), str):
                        text = blk["text"].strip()
                    if not text and isinstance(blk.get("content"), list):
                        for c in blk["content"]:
                            t = _first_text_from_content(c)
                            if t:
                                text = t.strip()
                                break

            # 3) Some SDKs stash assistant content in payload["message"]["content"]
            if not text and isinstance(payload.get("message"), dict):
                mc = payload["message"].get("content")
                if isinstance(mc, str):
                    text = mc.strip()
                elif isinstance(mc, list) and mc:
                    t = _first_text_from_content(mc[0])
                    if t:
                        text = t.strip()

            # Chat Completions shape inside payload
            if not text and isinstance(payload.get("choices"), list):
                try:
                    msg = (payload["choices"][0].get("message") or {})
                    ct = msg.get("content")
                    if isinstance(ct, str):
                        text = ct.strip()
                    model = model or (payload.get("model") or "")
                except Exception:
                    pass

            # Last-ditch recursive search for any 'text' field
            if not text:
                def _search_text(d):
                    if isinstance(d, dict):
                        for k, v in d.items():
                            if k == "text" and isinstance(v, str) and v.strip():
                                return v.strip()
                            found = _search_text(v)
                            if found:
                                return found
                    elif isinstance(d, list):
                        for it in d:
                            found = _search_text(it)
                            if found:
                                return found
                    return ""
                t = _search_text(payload)
                if t:
                    text = t

        triples.append((heading, text, model or ""))

    return triples

def _pretty_heading_from_cid(h: str) -> str:
    # turn "sec_0001_MI6_-_Borough" -> "MI6 - Borough"
    return re.sub(r"^sec_\d{4}_", "", h).replace("_", " ").strip()

# ------------------------------- CLI -----------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arbuthnot Books novel toolkit (all-in build)")
    p.add_argument("draft", nargs="?", help="Input .md or .txt file (not required for --batch-status/--batch-fetch)")

    modes = [
        "concise", "discursive", "critique", "entities", "continuity",
        "overview", "inline-edit", "copyedit", "improvement_suggestions",
        "detailed-summary", "sensitivity-analysis", "plot-points",
    ]
    p.add_argument("--mode", choices=modes, default="concise")
    p.add_argument("--whole", action="store_true", help="Treat entire file as one section")
    p.add_argument("--format", choices=["md", "json", "both"], default="md")

    p.add_argument("--model", default="gpt-4.1-mini",
                   choices=["gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4.1-mini"])
    p.add_argument("--engine", choices=["auto", "responses", "chat"], default="auto")
    p.add_argument("--fallback-model", choices=["gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4.1-mini"], default=None)

    # Slicing / filtering
    p.add_argument("--heading-regex")
    p.add_argument("--min-heading-level", type=int, default=3)
    p.add_argument("--ignore-headings", help="Regex to ignore headings (e.g. '^(week|part)\\b')")
    p.add_argument("--list-sections", action="store_true", help="List detected sections and exit")
    # Non-batch slicing
    p.add_argument("--first", type=int, help="Process only the first N sections (non-batch)")
    p.add_argument("--last", type=int, help="Process only the last N sections (non-batch)")
    p.add_argument("--range", nargs=2, type=int, metavar=("START", "END"),
                   help="Process 1-based inclusive range START..END (non-batch)")
    p.add_argument("--batch-first", type=int, help="Take only the first N sections")
    p.add_argument("--batch-range", nargs=2, metavar=("START", "END"), help="Take inclusive range of sections")

    # Output / runtime
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dump-raw", action="store_true")
    p.add_argument("--request-timeout", type=int, default=60)
    p.add_argument("--max-retries", type=int, default=2)

    # Tokens / limits
    p.add_argument("--max-tokens-out", type=int, default=1024)
    p.add_argument("--max-section-tokens", type=int, default=120_000)

    # Profiling & pricing
    p.add_argument("--profile", action="store_true", help="Write profile.csv in out dir")
    p.add_argument("--price", action="store_true", help="Add cost columns to profile")
    p.add_argument("--fx-rate", type=float, help="Multiply USD costs by FX to display in GBP")

    # Resume entities JSONL
    p.add_argument("--resume-entities", type=str, help="Append new entities to an existing entities.jsonl")

    # Concise niceties
    p.add_argument("--normalize-output", action="store_true", help="Normalise concise outputs")
    p.add_argument("--continue-on-truncate", action="store_true")
    p.add_argument("--continue-max", type=int, default=1)

    # Estimate-only
    p.add_argument("--estimate-only", action="store_true", help="Do not call API, just write estimate profile.csv")
    p.add_argument("--estimate-out", type=int, help="Override estimated tokens_out per section in estimate-only")

    # Batch API
    p.add_argument("--batch-submit", action="store_true", help="Create NDJSON and submit Batch")
    p.add_argument("--batch-dry-run", action="store_true", help="Write NDJSON only")
    p.add_argument("--batch-status", metavar="BATCH_ID", help="Show status")
    p.add_argument("--batch-fetch", metavar="BATCH_ID", help="Fetch completed batch outputs")
    p.add_argument("--batch-wait", type=int, metavar="SECONDS",
                   help="If set with --batch-fetch, poll until completion or timeout.")
    p.add_argument("--batch-poll-every", type=int, default=5, metavar="SECONDS",
                   help="Polling interval (seconds) for --batch-fetch when --batch-wait is set. Default: 5.")    
    p.add_argument("--profile-on-fetch", action="store_true", help="Rebuild tokens/costs on fetch")

    return p.parse_args()

# -------------------------------- main ---------------------------------------
def main() -> None:
    global args  # for _batch_write_ndjson prompt builder
    args = parse_args()

    # cost-from-profile subcommand (optional: pass a CSV to compute totals)
    if getattr(args, "cost_from_profile", None):
        import pandas as pd
        fx = args.fx_rate if hasattr(args, "fx_rate") else None
        df = pd.read_csv(args.cost_from_profile)
        if "cost_input" not in df.columns or "cost_output" not in df.columns:
            df["cost_input"] = 0.0; df["cost_output"] = 0.0; df["cost_total"] = 0.0
        for idx, row in df.iterrows():
            if row.get("tokens_in_est", 0) > 0 or row.get("tokens_out_est", 0) > 0:
                ci, co, ct = estimate_cost_numeric(
                    row.get("model", ""), float(row.get("tokens_in_est", 0)),
                    float(row.get("tokens_out_est", 0)), fx
                )[:3]
                df.at[idx, "cost_input"] = ci
                df.at[idx, "cost_output"] = co
                df.at[idx, "cost_total"] = ct
        total = df["cost_total"].sum()
        cur = "GBP" if fx else "USD"
        print(f"[ok] Total: {total:.4f} {cur}")
        base, ext = os.path.splitext(args.cost_from_profile)
        out = f"{base}_with_costs{ext}"
        df.to_csv(out, index=False)
        print(f"[ok] Updated: {out}")
        return

    # For batch status/fetch we don't require a draft
    if (args.batch_status or args.batch_fetch) and not args.draft:
        pass
    elif not args.draft:
        # Draft required for sync / submit / dry-run
        if not (args.batch_status or args.batch_fetch):
            sys.exit("draft is required unless using --batch-status/--batch-fetch")

    if args.mode == "concise" and not args.normalize_output:
        args.normalize_output = True

    # API & client
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("[error] OPENAI_API_KEY not set")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = ("{}_whole".format(args.mode)) if args.whole else args.mode
    out_dir = args.out_dir or Path(f"{base}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Batch status early exit ---
    if args.batch_status:
        info = get_batch_status(client, args.batch_status)
        sys.stderr.write(json.dumps(info, indent=2) + "\n")
        return

    # --- batch fetch (with optional wait & profile-on-fetch) ---
    if args.batch_fetch:
        out_dir.mkdir(parents=True, exist_ok=True)

        # Either poll until done or just read once
        if args.batch_wait:
            info = _poll_batch(
                client,
                args.batch_fetch,
                timeout_s=args.batch_wait,
                every_s=max(1, getattr(args, "batch_poll_every", 5)),
                verbose=args.verbose,
            )
        else:
            info = get_batch_status(client, args.batch_fetch)
            if args.verbose:
                sys.stderr.write(f"[info] batch {args.batch_fetch} status: {info.get('status','unknown')}\n")

        status = info.get("status")

        # Surface server-side failures and try to pull the error file if present
        if status == "failed":
            err_id = info.get("error_file_id")
            if err_id:
                sys.stderr.write("[info] Batch failed, retrieving error file...\n")
                try:
                    err_bytes = download_openai_file(client, err_id)
                    err_path = out_dir / "batch_errors.ndjson"
                    err_path.write_bytes(err_bytes)
                    sys.stderr.write(f"[info] Wrote error details: {err_path}\n")
                    # show a short preview to the terminal
                    try:
                        preview = err_bytes.decode("utf-8", "replace").splitlines()[:50]
                        if preview:
                            sys.stderr.write("\n".join(preview) + ("\n...\n" if len(preview) == 50 else "\n"))
                    except Exception:
                        pass
                except Exception as e:
                    sys.stderr.write(f"[warn] Could not download error file: {e}\n")
            else:
                sys.stderr.write("[error] Batch failed and no error_file_id is present.\n")
            return

        if status != "completed":
            sys.stderr.write(f"[error] Batch not completed (status={status}). Use --batch-wait or try later.\n")
            return

        out_id = info.get("output_file_id")
        if not out_id:
            sys.stderr.write("[error] Completed but no output_file_id.\n")
            return

        # Save raw NDJSON for debugging
        ndjson_bytes = download_openai_file(client, out_id)
        (out_dir / "batch_output.raw.ndjson").write_bytes(ndjson_bytes)  # handy for debugging

        # Download batch output NDJSON
        ndjson_bytes = download_openai_file(client, out_id)

        # Load sidecar mapping custom_id -> heading (if you created one on submit)
        batch_map_path = out_dir / "batch_map.json"
        id_to_heading = _load_batch_map(batch_map_path)  # returns {} if file absent

        # Parse output (expects no top-level 'metadata' in NDJSON items)
        triples = _parse_batch_output_ndjson(ndjson_bytes, id_to_heading)  # [(heading, text, model)]

        # Write outputs (md/jsonl) like normal
        md_path = out_dir / f"{args.mode}.md" if args.format in {"md","both"} and args.mode != "entities" else None
        jsonl_path = out_dir / f"{args.mode}.jsonl" if args.format in {"json","both"} else None

        if md_path:
            with md_path.open("w", encoding="utf-8") as md_fh:
                for heading, text, _model in triples:
                    # prettify only for display
                    h_disp = _pretty_heading_from_cid(heading) if heading and heading.startswith("sec_") else heading

                    if args.mode == "concise" and args.normalize_output and text:
                        summary, title = normalize_concise(text)
                        norm = summary.strip()
                        if title:
                            norm = f"{norm}\n\n*Suggested title:* {title}"
                        text = norm

                    md_fh.write(f"**{h_disp}**\n\n{(text or '').strip()}\n\n")
                    
            postprocess_markdown_sections(md_path)
            prepend_cli_metadata_to_output(md_path, args, args.mode)

        if jsonl_path:
            with jsonl_path.open("w", encoding="utf-8") as jf:
                for heading, text, _model in triples:
                    h_disp = _pretty_heading_from_cid(heading) if heading and heading.startswith("sec_") else heading
                    jf.write(json.dumps({"section_heading": h_disp, "raw": text}, ensure_ascii=False) + "\n")

        # Optional: profile-on-fetch
        if args.profile_on_fetch:
            import csv
            prof_path = out_dir / "profile.csv"

            # Rebuild input-token estimates if draft available
            tokens_in_map = {}
            if args.draft:
                draft_text = read_text(Path(args.draft))
                if args.whole:
                    src_sections = [("Full Manuscript", draft_text.strip())]
                else:
                    heading_rx = compile_heading_regex(args.heading_regex, args.min_heading_level)
                    src_sections = list(iter_sections(draft_text, heading_rx))
                    if args.ignore_headings:
                        rx_ign = re.compile(args.ignore_headings, re.IGNORECASE)
                        src_sections = [(h, b) for (h, b) in src_sections if not rx_ign.search(h)]
                for h, b in src_sections:
                    ptxt = build_prompt(args.mode, h, b, args.format)
                    tokens_in_map[h] = num_tokens(args.model, ptxt)

            with prof_path.open("w", newline="", encoding="utf-8") as pf:
                w = csv.writer(pf)
                w.writerow(["idx","heading","endpoint","model","tokens_in_est","tokens_out_est","latency_s","cost_input","cost_output","cost_total","currency"])
                for i, (heading, text, model_line) in enumerate(triples, 1):
                    tout = num_tokens(model_line or args.model, text or "")
                    tin  = tokens_in_map.get(heading, 0) if tokens_in_map else 0
                    if args.price:
                        ci, co, ct, cur = estimate_cost(model_line or args.model, tin, tout, args.fx_rate)
                    else:
                        ci, co, ct, cur = ("", "", "", "")
                    w.writerow([i, heading, "batch", model_line or args.model, tin, tout, "", ci, co, ct, cur])

        print(f"[ok] Fetched batch {args.batch_fetch} into {out_dir.resolve()}")
        if md_path and args.format in {"md","both"} and args.mode != "entities":
            print(f"  • Markdown: {_pretty_rel(md_path, out_dir.parent)}")
        if jsonl_path and args.format in {"json","both"}:
            print(f"  • JSONL:   {_pretty_rel(jsonl_path, out_dir.parent)}")
        if args.profile_on_fetch:
            print(f"  • Profile: {_pretty_rel(out_dir / 'profile.csv', out_dir.parent)}")
        return

    # ---- read draft & slice (for sync or submit/dry-run) ----
    draft_path = Path(args.draft) if args.draft else None
    text = read_text(draft_path) if draft_path else ""

    if args.whole:
        sections = [("Full Manuscript", text.strip())]
    else:
        heading_rx = compile_heading_regex(args.heading_regex, args.min_heading_level)
        sections = list(iter_sections(text, heading_rx))
        if args.ignore_headings:
            rx_ign = re.compile(args.ignore_headings, re.IGNORECASE)
            before = len(sections)
            sections = [(h, b) for (h, b) in sections if not rx_ign.search(h)]
            if args.verbose:
                sys.stderr.write(f"[info] Ignored {before - len(sections)} sections via --ignore-headings\n")

    if args.list_sections:
        for idx, (h, b) in enumerate(sections, 1):
            sys.stderr.write(f"[slice] {idx:03d} | {h} | {len((b or '').strip())} chars\n")
        return

    # --- estimate-only early exit ---
    if args.estimate_only:
        import csv
        est_out = args.estimate_out if args.estimate_out is not None else (args.max_tokens_out or 600)
        fx = args.fx_rate if hasattr(args, "fx_rate") else None
        # apply slicer in estimate path too
        sel = select_sections_for_batch(sections, args)
        prof_path = (args.out_dir or Path(f"{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")) / "profile.csv"
        prof_path.parent.mkdir(parents=True, exist_ok=True)
        with prof_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["idx","heading","endpoint","model","tokens_in_est","tokens_out_est","latency_s",
                        "cost_input","cost_output","cost_total","currency"])
            for idx, heading, body in sel:
                if not (body or "").strip():
                    continue
                prompt = build_prompt(args.mode, heading, body, args.format)
                t_in = num_tokens(args.model, prompt)
                t_out = int(est_out)
                cin, cout, ctot, cur = estimate_cost_numeric(args.model, t_in, t_out, fx)
                w.writerow([idx, heading, "estimate", args.model, t_in, t_out, 0.0,
                            f"{cin:.6f}", f"{cout:.6f}", f"{ctot:.6f}", cur])
        # print small summary
        try:
            import pandas as _pd
            df = _pd.read_csv(prof_path)
            total = float(df["cost_total"].astype(float).sum())
            cur = (df["currency"].dropna().unique().tolist() or [CURRENCY])[0]
            print(f"[ok] Wrote estimate-only profile: {prof_path}")
            print(f"[ok] Estimated total: {total:.4f} {cur} across {len(sel)} sections")
        except Exception:
            print(f"[ok] Wrote estimate-only profile: {prof_path}")
        return

    # --- batch submit/dry-run (sanitized: no metadata) ---
    out_dir.mkdir(parents=True, exist_ok=True)
    ndjson_path = out_dir / "batch_input.jsonl"
    batch_map = {}  # custom_id -> heading

    # choose endpoint+body builder based on model family
    model_lc = (args.model or "").strip().lower()
    if model_lc.startswith("gpt-5"):
        batch_endpoint = "/v1/responses"
        def make_body(prompt: str) -> dict:
            return {
                "model": args.model,
                "input": prompt,  # plain string
                "max_output_tokens": args.max_tokens_out,
                "reasoning": {"effort": "low"},             # <-- key change
                "response_format": {"type": "text"},        # <-- ensure visible text
                # optional (often helpful for editing tasks):
                # "temperature": 0.2,
            }
    else:
        batch_endpoint = "/v1/chat/completions"
        def make_body(prompt: str) -> dict:
            # NB: if you MUST use chat for non-gpt-5, keep this
            tok_field = "max_completion_tokens" if "gpt-5" in (args.model or "").lower() else "max_tokens"
            return {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                tok_field: args.max_tokens_out,
            }

    # select which sections to include (respects --batch-first / --batch-range)
    batch_sections, index_offset, selection_desc = _select_sections_for_batch(sections, args)
    if args.verbose:
        sys.stderr.write(f"[batch] selection: {selection_desc}\n")
    if not batch_sections:
        sys.exit("[error] No sections selected for batch.")

    with ndjson_path.open("w", encoding="utf-8") as f:
        count = 0
        for j, (heading, body) in enumerate(batch_sections, 1):
            if not body.strip() and not args.whole:
                if args.verbose:
                    sys.stderr.write(f"[warn] Skipping empty section: '{heading}'\n")
                continue
            global_idx = index_offset + j  # 1-based original position
            prompt = build_prompt(args.mode, heading, body, args.format)
            cid = f"sec_{global_idx:04d}_{safe_slug(heading)[:40]}"
            req = {
                "custom_id": cid,
                "method": "POST",
                "url": batch_endpoint,
                "body": make_body(prompt),
            }  # <- NO metadata
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
            batch_map[cid] = heading
            count += 1

    # save sidecar so we can restore headings on fetch
    (out_dir / "batch_map.json").write_text(json.dumps(batch_map, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.verbose:
        sys.stderr.write(f"[info] Batched {count} items -> {ndjson_path}\n")

    if args.batch_dry_run:
        print(f"[ok] Wrote batch input (dry-run): {ndjson_path}")
        return

    # submit batch
    file_obj = client.files.create(file=ndjson_path.open("rb"), purpose="batch")
    b = client.batches.create(input_file_id=file_obj.id, endpoint=batch_endpoint, completion_window="24h")
    print(f"[ok] Submitted batch: {b.id}\n    Save this id and later run with: --batch-fetch {b.id}")
    return

    # --- synchronous path (LLM calls) ---
    # File handles & profile
    md_fh: Optional[object] = None
    json_fh: Optional[object] = None
    md_path: Optional[Path] = None
    jsonl_path: Optional[Path] = None
    prof_writer = None

    if args.format in {"md","both"} and args.mode != "entities":
        md_path = out_dir / f"{args.mode}.md"
        md_fh = open(md_path, "w", encoding="utf-8")
    if args.format in {"json","both"}:
        if args.mode == "entities" and args.resume_entities:
            jsonl_path = Path(args.resume_entities)
            json_fh = open(jsonl_path, "a", encoding="utf-8")
            if args.verbose:
                sys.stderr.write(f"[resume] Appending new entities to: {jsonl_path}\n")
        else:
            jsonl_path = out_dir / f"{args.mode}.jsonl"
            json_fh = open(jsonl_path, "w", encoding="utf-8")

    if args.profile:
        import csv
        prof_path = out_dir / "profile.csv"
        pfh = open(prof_path, "w", newline="", encoding="utf-8")
        import atexit
        atexit.register(lambda: pfh.close())
        prof_writer = csv.writer(pfh)
        prof_writer.writerow(["idx","heading","endpoint","model",
                              "tokens_in_est","tokens_out_est","latency_s",
                              "cost_input","cost_output","cost_total","currency"])

    # Context window map for guardrails (approx)
    CTX_LIMITS = {"gpt-5": 1_000_000, "gpt-5-mini": 1_000_000, "gpt-4.1": 128_000, "gpt-4.1-mini": 1_000_000}
    CTX_LIMIT = CTX_LIMITS.get(args.model, 128_000)

    # Apply slicer to sync path too
    run_list = select_sections_for_batch(sections, args)

    processed = 0
    missing_sections: List[Tuple[int, str]] = []

    try:
        for idx, heading, body in tqdm(run_list, desc="Processing", unit="section"):
            if (not (body or "").strip()) and (not args.whole):
                if args.verbose:
                    sys.stderr.write(f"[warn] '{heading}' has no body text; skipped.\n")
                if args.profile and prof_writer:
                    prof_writer.writerow([idx, heading,
                                          "responses" if (args.engine=="responses" or (args.engine=="auto" and args.model.startswith("gpt-5"))) else "chat",
                                          args.model, 0, 0, 0.0, "", "", "", "skipped"])
                continue

            prompt = build_prompt(args.mode, heading, body, args.format, prior=None)
            total_in = num_tokens(args.model, prompt)
            if args.verbose:
                sys.stderr.write(f"[debug] {heading}: {total_in} tokens in\n")

            if total_in + args.max_tokens_out + 10 > CTX_LIMIT:
                sys.stderr.write(f"[warn] '{heading}' exceeds {args.model} context window; skipped.\n")
                if args.profile and prof_writer:
                    prof_writer.writerow([idx, heading,
                                          "responses" if (args.engine=="responses" or (args.engine=="auto" and args.model.startswith("gpt-5"))) else "chat",
                                          args.model, total_in, 0, 0.0, "", "", "", "skipped_ctx"])
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

            if not response or not response.strip():
                sys.stderr.write(f"[warn] Empty/no response for '{heading}'. Marking missing.\n")
                missing_sections.append((idx, heading))
                if args.profile and prof_writer:
                    ci, co, ct, cur = estimate_cost(args.model, total_in, 0, args.fx_rate) if args.price else ("","","","")
                    prof_writer.writerow([idx, heading,
                                          "responses" if (args.engine=="responses" or (args.engine=="auto" and args.model.startswith("gpt-5"))) else "chat",
                                          args.model, total_in, 0, round(dt,3), ci, co, ct, cur])
                continue

            text_out = response

            # Auto-continue if concise looks truncated (simple heuristic)
            def _looks_truncated(s: str) -> bool:
                return bool(len(s.strip()) > 200 and not s.strip().endswith((".", "!", "?", "”", "’")))

            if args.mode == "concise" and args.format in {"md","both"} and args.continue_on_truncate:
                tries = 0
                while _looks_truncated(text_out) and tries < args.continue_max:
                    tries += 1
                    cont_prompt = (
                        "Continue the previous SUMMARY from exactly where it stopped. "
                        "Do not repeat earlier text. Finish cleanly.\n\n"
                        f"PREVIOUS END:\n{text_out[-600:]}"
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
                    text_out = (text_out + "\n" + cont.strip()).strip()

            # Normalise concise
            if args.mode == "concise" and args.normalize_output and args.format in {"md","both"}:
                summary, title = normalize_concise(text_out)
                normalized = summary.strip()
                if title:
                    normalized = f"{normalized}\n\n*Suggested title:* {title}"
                text_out = normalized

            # Write outputs
            if md_fh and args.format in {"md","both"} and args.mode != "entities":
                md_fh.write(f"**{h_disp}**\n\n{(text or '').strip()}\n\n")

            if json_fh and args.format in {"json","both"}:
                if args.mode in {"entities", "continuity"}:
                    # Try clean JSON
                    try:
                        obj = json.loads(text_out)
                    except Exception:
                        obj = {"section_heading": heading, "raw": text_out}
                    json_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                else:
                    jf.write(json.dumps({"section_heading": h_disp, "raw": text}, ensure_ascii=False) + "\n")

            # Profiling
            if args.profile and prof_writer:
                tokens_out_est = num_tokens(args.model, text_out)
                ci, co, ct, cur = estimate_cost(args.model, total_in, tokens_out_est, args.fx_rate) if args.price else ("","","","")
                endpoint = "responses" if (args.engine=="responses" or (args.engine=="auto" and args.model.startswith("gpt-5"))) else "chat"
                prof_writer.writerow([idx, heading, endpoint, args.model, total_in, tokens_out_est, round(dt,3), ci, co, ct, cur])

            processed += 1

    finally:
        if md_fh:
            md_fh.close()
        if json_fh:
            json_fh.close()

    if processed == 0:
        sys.exit("[error] No sections processed. Check headings, token limits, or model window.")

    # Post-process & header
    if md_path and args.format in {"md","both"} and args.mode != "entities":
        postprocess_markdown_sections(md_path)
        prepend_cli_metadata_to_output(md_path, args, args.mode)

    print(f"[ok] Wrote {processed}/{len(run_list)} sections to {out_dir.resolve()}")
    if md_path and args.format in {"md","both"} and args.mode != "entities":
        print(f"  • Markdown: {_pretty_rel(md_path, out_dir.parent)}")
    if jsonl_path and args.format in {"json","both"}:
        print(f"  • JSONL:   {_pretty_rel(jsonl_path, out_dir.parent)}")

    if missing_sections:
        miss_str = ", ".join([f"{i}:{h}" for i, h in missing_sections[:8]])
        more = "" if len(missing_sections) <= 8 else f" (+{len(missing_sections)-8} more)"
        sys.stderr.write(f"[info] Missing/empty outputs for: {miss_str}{more}\n")
        if args.mode == "entities":
            sys.stderr.write("      Re-run just those with: --resume-entities <entities.jsonl> --batch-range START END\n")

if __name__ == "__main__":
    main()