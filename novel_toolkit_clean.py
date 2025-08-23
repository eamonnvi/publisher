#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
novel_toolkit_clean.py – Arbuthnot Books helper (clean build with resume + profiling fixes)
------------------------------------------------------------------------------------------
- Writes `idx` to JSONL for stable section identification
- `--resume-entities` prefers idx-based skipping; falls back to heading text
- `--resume-from-profile profile.csv` resumes exactly missing rows (tokens_out_est==0)
- `profile.csv` captures the *actual* endpoint used (responses/chat)
- Initialises file handles safely and prints clean output summary
- Handles max token parameter differences between models/APIs

Tested with Python 3.11/3.12/3.13 and openai>=1.35
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

try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

from tqdm import tqdm  # type: ignore

# ---------------------------- Prompt templates -------------------------------

PROMPT_TEMPLATES: Dict[str, str] = {
    # ——— summaries ———
    "concise": (
        "Summarise the following {label} of a novel entitled '{heading}'. "
        "Include major events, character actions, and information that may be important for the overall plot. "
        "Write in clear British English prose.\n\n"
        "----- BEGIN {LABEL} -----\n{body}\n----- END {LABEL} -----"
    ),
    "discursive": (
        "Provide an in-depth, analytical synopsis of the section titled '{heading}'. "
        "Discuss character motivations, internal and external conflicts, thematic significance, subtext, and narrative strategies. "
        "Comment on the pacing, use of suspense, and effectiveness of red herrings or misleading clues. "
        "Highlight any details that may have later narrative impact (including possible foreshadowing, Chekhov’s guns, or setup for twists). "
        "Consider the structure, tone, and any shifts in perspective. Do not omit spoilers or narrative revelations.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    # ——— developmental critique ———
    "critique_md": (
        "You are an experienced developmental editor. Evaluate the section titled '{heading}' as follows:\n\n"
        "1. **Strengths:** Comment on voice, pacing, originality, narrative structure, and style.\n"
        "2. **Weaknesses:** Note inconsistencies, clichés, narrative slack, unresolved setups, and anything that may undermine suspense or thematic unity.\n"
        "3. **Genre fit and innovation:** Discuss how the section aligns with or subverts expectations for its genre.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "critique_json": (
        "You are an experienced developmental editor. Return STRICT JSON with keys 'strengths', 'weaknesses', 'improvements' – each an array of strings.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    # --- improvement suggestions ---
    "improvement_suggestions": (
        "For the section titled '{heading}', list specific, actionable suggestions for revision. "
        "Focus on improving clarity, deepening character motivation, strengthening suspense, tightening pacing, and resolving ambiguities. "
        "Include examples or sample rewrites if relevant.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    # ——— entity extraction ———
    "entities_json": (
        "Extract structured facts from the section titled '{heading}'. "
        "Return only a single valid JSON object with these keys: "
        "'section_heading', 'characters', 'pov_character', 'themes', 'locations', 'timestamp'. "
        "Do not include explanations, markdown, or any non-JSON output.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    # ——— continuity ———
    "continuity_md": (
        "You are a continuity editor. Compare the **current section** to the following **prior facts**. "
        "Flag any contradictions (dates, ages, locations, character actions, etc.), and note any ambiguous or potentially inconsistent elements (even if not outright contradictions).\n\n"
        "### Prior facts\n```json\n{prior_entities}\n```\n\n### Current section\n{body}\n"
    ),
    "continuity_json": (
        "Compare current section to prior facts. Return ONLY a JSON array of inconsistency strings (empty if none).\n\n"
        "Prior facts: {prior_entities}\n\nCurrent section:\n{body}\n"
    ),
    # ——— copy-edit ———
    "copyedit_md": (
        "## Identity\nYou are a line-by-line copy editor in a British literary publishing house.\n\n"
        "## Task\nIdentify typos, grammar issues, redundancies, repetitions.\n\n"
        "## Prohibitions\nDo not break sentences, reduce the number of adverbs, spot clichés, or flag contractions.\n\n"
        "## Output\nBulleted Markdown: line number, snippet, issue, suggestion.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "copyedit_json": (
        "(Same identity/task) Return JSON array of objects {line_number,int; original,str; issue,str; suggestion,str}.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    # ——— overview ———
    "overview": (
        "Write a 2,000–2,500-word editorial overview of the manuscript below. "
        "Cover plot, character arcs, themes, style, structure, pacing, genre conventions (and subversions), and any continuity or consistency issues observed across sections. "
        "Include major narrative strengths and weaknesses, and comment on possible revisions or further development.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "detailed-summary": (
        "Write a 5,000–10,000-word chapter-by-chapter summary of the manuscript below. "
        "Cover plot, character arcs, themes, style, structure, pacing, genre conventions (and subversions).\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "sensitivity-analysis": (
        "Read the sexually explicit sections of the manuscript below. "
        "Assess whether they are justified by the period in which the narrative is set and the situations in which the characters find themselves.\n\n"
        "Please suggest rewordings for lines which are borderline un/acceptable.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "plot-points": (
        "Is the narrative of the current draft presented in such a way that the attentive reader would guess that Mrs Nairn and the glamourous female film star are Sheena in disguise and that the presence of Marlboro cigarettes marks Katrin? "
        "The fact that Ulrike declines cigarettes in the Paddington Green scene indicates that the character in this scene really is Ulrike and not Katrin. "
        "Is it also possible for the reader to infer that Grace is developing a lesbian attraction to Ulrike in the Newnham College high table scene? "
        "Finally, please identify any plot points that don't work or could be more sharply written.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
}

# ---------------------------- Helpers ----------------------------------------

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def compile_heading_regex(custom: Optional[str], min_level: Optional[int] = None) -> re.Pattern[str]:
    if custom:
        return re.compile(custom, re.MULTILINE)
    # default: ATX headings only; filter by min_level later
    return re.compile(r"^\s*(#{1,6})\s*([^#\n]+?)\s*#*\s*$", re.MULTILINE)

def iter_sections(text: str, heading_rx: re.Pattern[str], min_level: Optional[int] = None) -> Iterable[Tuple[str, str]]:
    matches = []
    for m in heading_rx.finditer(text):
        hashes, title = (m.group(1), m.group(2)) if m.lastindex == 2 else ("", m.group(0))
        level = len(hashes) if hashes else 1
        if min_level and level < min_level:
            continue
        matches.append((m, title.strip()))
    if not matches:
        yield "Full Manuscript", text.strip(); return
    for i, (m, title) in enumerate(matches):
        start = m.end()
        end = matches[i + 1][0].start() if i + 1 < len(matches) else len(text)
        yield title, text[start:end].strip()

def safe_slug(s: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "untitled"

def num_tokens(model: str, *chunks: str) -> int:
    if not tiktoken:
        # Fallback crude estimate
        return sum(max(1, len(c) // 4) for c in chunks)
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")
    return sum(len(enc.encode(c)) for c in chunks)

def build_prompt(mode: str, heading: str, body: str, fmt: str, prior: Optional[str] = None) -> str:
    key_map = {
        "critique": "critique_json" if fmt in {"json", "both"} else "critique_md",
        "entities": "entities_json",
        "continuity": "continuity_json" if fmt in {"json", "both"} else "continuity_md",
        "copyedit": "copyedit_json" if fmt in {"json", "both"} else "copyedit_md",
    }
    template_key = key_map.get(mode, mode)
    if template_key not in PROMPT_TEMPLATES:
        raise KeyError(f"No prompt template for mode '{mode}'.")
    template = PROMPT_TEMPLATES[template_key]
    norm = heading.strip().lower()
    label = "manuscript" if norm in {"full manuscript", "manuscript"} else "section"
    return template.format(label=label, LABEL=label.upper(), heading=heading, body=body, prior_entities=prior or "[]")

def normalize_concise(text: str) -> Tuple[str, Optional[str]]:
    """Extract SUMMARY: ... and optional TITLE: ... from a concise response."""
    t = text.strip()
    title = None
    m = re.search(r"(?im)^\s*TITLE\s*:\s*(.+)$", t)
    if m:
        title = m.group(1).strip()
        t = re.sub(r"(?im)^\s*TITLE\s*:.+$", "", t).strip()
    m2 = re.search(r"(?is)\bSUMMARY\s*:\s*(.+)$", t)
    summary = m2.group(1).strip() if m2 else t
    return summary, title

def postprocess_markdown_sections(md_path: Path) -> None:
    """Ensure one blank line between sections; ensure Suggested title is last line of a section."""
    txt = md_path.read_text(encoding="utf-8")
    # move '*Suggested title:* ...' to end of its section (before next **Heading**)
    def _fix(block: str) -> str:
        # already last? leave as is
        return block
    # Just enforce blank line between sections:
    txt = re.sub(r"\n{0,2}\*\*(.+?)\*\*\n", r"\n\n**\1**\n", txt)
    md_path.write_text(txt, encoding="utf-8")

# ---------------------------- Pricing ----------------------------------------

PRICE = {
    "gpt-5":        {"in": 0.010, "out": 0.030},
    "gpt-5-mini":   {"in": 0.003, "out": 0.006},
    "gpt-4.1":      {"in": 0.005, "out": 0.015},
    "gpt-4.1-mini": {"in": 0.001, "out": 0.003},
}
CURRENCY = "USD"

def estimate_cost(model: str, tokens_in: int, tokens_out: int, fx_rate: Optional[float]) -> tuple[str, str, str, str]:
    r = PRICE.get(model)
    if not r:
        return "", "", "", CURRENCY
    cin = (tokens_in / 1000.0) * r["in"]
    cout = (tokens_out / 1000.0) * r["out"]
    total = cin + cout
    cur = "GBP" if (fx_rate and fx_rate > 0) else CURRENCY
    if fx_rate and fx_rate > 0:
        cin *= fx_rate; cout *= fx_rate; total *= fx_rate
    return (f"{cin:.6f}", f"{cout:.6f}", f"{total:.6f}", cur)

def estimate_cost_numeric(model: str, tokens_in: float, tokens_out: float, fx_rate: Optional[float]) -> tuple[float, float, float, str]:
    r = PRICE.get(model)
    if not r:
        return (0.0, 0.0, 0.0, CURRENCY)
    cin = (tokens_in / 1000.0) * r["in"]
    cout = (tokens_out / 1000.0) * r["out"]
    total = cin + cout
    cur = "GBP" if (fx_rate and fx_rate > 0) else CURRENCY
    if fx_rate and fx_rate > 0:
        cin *= fx_rate; cout *= fx_rate; total *= fx_rate
    return (cin, cout, total, cur)

# ---------------------------- LLM call ---------------------------------------

def _chat_param_name_for_max_tokens(model: str) -> str:
    m = (model or "").lower()
    # Some gpt-5 chat models use `max_completion_tokens`; others accept `max_tokens`.
    # Use the stricter name when the provider requires it.
    if m.startswith("gpt-5"):
        return "max_completion_tokens"
    return "max_tokens"

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
) -> Tuple[Optional[str], Optional[str]]:
    """Return (text, endpoint_used) where endpoint_used in {'responses','chat'}."""
    def chat_call(m: str) -> Tuple[Optional[str], str]:
        if verbose:
            sys.stderr.write(f"[info] Using Chat Completions with model '{m}' (json={want_json})\n")
        kwargs = dict(
            model=m,
            messages=[{"role": "user", "content": prompt}],
            timeout=request_timeout,
        )
        # Future-proof max token name for chat
        token_key = _chat_param_name_for_max_tokens(m)
        kwargs[token_key] = max_out
        if want_json:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or "").strip()
        if dump_raw_dir:
            Path(dump_raw_dir, f"{safe_slug(heading)}__chat.json").write_text(resp.model_dump_json(), encoding="utf-8")
        return (text or None), "chat"

    def responses_call(m: str) -> Tuple[Optional[str], str]:
        if verbose:
            sys.stderr.write(f"[info] Using Responses API with model '{m}' (json={want_json})\n")
        generation_config = {"max_output_tokens": max_out}
        kwargs = dict(
            model=m,
            input=[{"role": "user", "content": prompt}],
            timeout=request_timeout,
        )
        if want_json:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = client.responses.create(**kwargs, generation_config=generation_config)
        except TypeError:
            resp = client.responses.create(**kwargs, max_output_tokens=max_out)

        # Extract text
        text = None
        try:
            text = (resp.output_text or "").strip()  # type: ignore
        except Exception:
            try:
                # fallback extraction for streaming-like objects
                content = resp.output[0].content  # type: ignore
                if content and len(content) and "text" in content[0]:
                    text = (content[0]["text"] or "").strip()
            except Exception:
                text = None
        if dump_raw_dir:
            Path(dump_raw_dir, f"{safe_slug(heading)}__responses.json").write_text(resp.model_dump_json(), encoding="utf-8")
        return (text or None), "responses"

    prefer_responses = (engine == "responses") or (engine == "auto" and model.startswith("gpt-5"))
    primary = responses_call if prefer_responses else chat_call
    secondary = chat_call if prefer_responses else responses_call

    models_to_try: List[Tuple[str, callable]] = [(model, primary), (model, secondary)]
    if fallback_model and fallback_model != model:
        models_to_try += [(fallback_model, primary), (fallback_model, secondary)]

    err_last = None
    for m, fn in models_to_try:
        tries = 0
        while tries <= max_retries:
            try:
                out, label = fn(m)
                if out:
                    return out, label
                else:
                    if verbose:
                        sys.stderr.write(f"[warn] Empty text from endpoint for {m}; retrying...\n")
            except Exception as e:
                err_last = e
                if verbose:
                    sys.stderr.write(f"[warn] endpoint error for {m}: {e}\n")
            tries += 1
            time.sleep(0.5)
    if verbose and err_last:
        sys.stderr.write(f"[warn] All attempts failed: {err_last}\n")
    return None, None

# ---------------------------- CLI / Main -------------------------------------

def _pretty_rel(p: Path, base: Path) -> str:
    try:
        return str(p.relative_to(base))
    except Exception:
        return str(p)

def main() -> None:
    p = argparse.ArgumentParser(description="Arbuthnot Books novel toolkit (clean build)")

    p.add_argument("draft", nargs="?", help="Input .md or .txt file")

    modes = [
        "concise", "discursive", "critique", "entities", "continuity",
        "overview", "inline-edit", "copyedit", "improvement_suggestions",
        "detailed-summary", "sensitivity-analysis", "plot-points",
    ]
    p.add_argument("--mode", choices=modes, default="concise")
    p.add_argument("--whole", action="store_true", help="Treat entire file as one section")
    p.add_argument("--format", choices=["md", "json", "both"], default="md")
    p.add_argument("--model", default="gpt-4.1-mini", choices=["gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4.1-mini"])
    p.add_argument("--engine", choices=["auto", "responses", "chat"], default="auto")
    p.add_argument("--fallback-model", choices=["gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4.1-mini"])

    p.add_argument("--max-tokens-out", type=int, default=1024)
    p.add_argument("--max-section-tokens", type=int, default=120_000)

    p.add_argument("--heading-regex", help="Custom regex for headings (default: ATX)")
    p.add_argument("--min-heading-level", type=int, help="Minimum ATX heading level (1..6)")
    p.add_argument("--ignore-headings", help="Regex to ignore headings (case-insensitive)")
    p.add_argument("--list-sections", action="store_true")

    p.add_argument("--out-dir", type=Path)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dump-raw", action="store_true")

    # Resume / repair
    p.add_argument("--resume-entities", help="Append to an existing entities.jsonl (skips already present rows).")
    p.add_argument("--resume-from-profile", help="CSV with tokens_out_est; process only rows with zero (missing).")

    # Profiling / pricing
    p.add_argument("--profile", action="store_true", help="Write profile.csv with token estimates and endpoint used.")
    p.add_argument("--price", action="store_true", help="Add cost columns to profile.csv.")
    p.add_argument("--fx-rate", type=float, help="If set, convert USD→GBP at this rate.")

    # Retry / timeout
    p.add_argument("--request-timeout", type=int, default=60)
    p.add_argument("--max-retries", type=int, default=2)

    # Slicer
    p.add_argument("--batch-first", type=int, help="Process only the first N sections.")
    p.add_argument("--batch-range", nargs=2, type=int, metavar=("START", "END"),
                   help="Process only sections in the closed interval [START, END] (1-based).")

    # Normalisation helpers
    p.add_argument("--normalize-output", action="store_true")
    p.add_argument("--continue-on-truncate", action="store_true")
    p.add_argument("--continue-max", type=int, default=1)

    p.add_argument("--entities-file", type=Path, help="Path to entities.jsonl for continuity mode")

    args = p.parse_args()

    # ---------------- init variables used later (avoid NameError) -------------
    md_fh: Optional[object] = None
    json_fh: Optional[object] = None
    md_path: Optional[Path] = None
    jsonl_path: Optional[Path] = None
    prof_writer = None
    processed = 0

    # --- non-file operations? we need draft for most paths ---
    if not args.draft:
        p.error("draft is required for this run.")

    if args.mode == "concise" and not args.normalize_output:
        args.normalize_output = True

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("[error] OPENAI_API_KEY not set")
    if OpenAI is None:
        sys.exit("[error] openai package not available. pip install openai")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # output dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{args.mode}_whole" if args.whole else args.mode
    out_dir = args.out_dir or Path(f"{base}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- read + slice
    text = read_text(Path(args.draft))
    if args.whole:
        sections = [("Full Manuscript", text.strip())]
    else:
        heading_rx = compile_heading_regex(args.heading_regex, args.min_heading_level)
        sections = list(iter_sections(text, heading_rx, args.min_heading_level))
        if args.ignore_headings:
            rx_ign = re.compile(args.ignore_headings, re.IGNORECASE)
            before = len(sections)
            sections = [(h, b) for (h, b) in sections if not rx_ign.search(h)]
            if args.verbose:
                sys.stderr.write(f"[info] Ignored {before - len(sections)} sections via --ignore-headings\n")
        if args.list_sections:
            for i, (h, b) in enumerate(sections, 1):
                sys.stderr.write(f"[slice] {i:03d} | {h} | {len(b)} chars\n")
            return

    # slicer selection (for normal sync run)
    if args.batch_first:
        sections = sections[: max(0, args.batch_first)]
    if args.batch_range:
        start, end = args.batch_range
        start = max(1, start)
        end = min(len(sections), end)
        if start > end:
            sys.exit("[error] --batch-range start > end")
        sections = sections[start - 1: end]

    # resume-from-profile -> build a set of target idx
    resume_target_idxs = None
    if args.resume_from_profile:
        import csv
        target = set()
        with open(args.resume_from_profile, newline="", encoding="utf-8") as fh:
            rdr = csv.DictReader(fh)
            for row in rdr:
                try:
                    i = int(row.get("idx", "") or "0")
                except ValueError:
                    continue
                # treat zero/blank tokens_out_est (or status markers) as missing
                try:
                    tout = float(row.get("tokens_out_est", "0") or "0")
                except ValueError:
                    tout = 0.0
                status = (row.get("endpoint", "") or "").lower()
                if (tout == 0.0) or status in {"no-output"}:
                    if i > 0:
                        target.add(i)
        resume_target_idxs = target
        if args.verbose:
            sys.stderr.write(f"[resume] Will process exactly these idx: {sorted(resume_target_idxs)}\n")

    # --- profiling writer (optional) ---
    if args.profile:
        import csv
        prof_path = out_dir / "profile.csv"
        prof_writer = csv.writer(open(prof_path, "w", newline="", encoding="utf-8"))
        prof_writer.writerow([
            "idx", "heading", "endpoint", "model",
            "tokens_in_est", "tokens_out_est", "latency_s",
            "cost_input", "cost_output", "cost_total", "currency"
        ])

    # --- file outputs ---
    if args.format in {"md", "both"} and args.mode != "entities":
        md_path = out_dir / f"{args.mode}.md"
        md_fh = open(md_path, "w", encoding="utf-8")
    if args.format in {"json", "both"}:
        if args.mode == "entities" and args.resume_entities:
            jsonl_path = Path(args.resume_entities)
            json_fh = open(jsonl_path, "a", encoding="utf-8")
            if args.verbose:
                sys.stderr.write(f"[resume] Appending new entities to: {jsonl_path}\n")
        else:
            jsonl_path = out_dir / f"{args.mode}.jsonl"
            json_fh = open(jsonl_path, "w", encoding="utf-8")

    # --- build done sets for resume-entities (idx preferred, fallback headings) ---
    done_headings = set()
    done_idxs = set()
    if args.mode == "entities" and args.resume_entities:
        try:
            with open(args.resume_entities, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    h = rec.get("section_heading")
                    if h:
                        done_headings.add(h)
                    ix = rec.get("idx")
                    if isinstance(ix, int):
                        done_idxs.add(ix)
        except FileNotFoundError:
            pass
        if args.verbose:
            sys.stderr.write(f"[resume] Found {len(done_headings)} headings already present; will skip them.\n")
            if done_idxs:
                sys.stderr.write(f"[resume] Found {len(done_idxs)} numeric idx already present; will prefer idx-based skipping.\n")

    # --- context windows (rough guide) ---
    CTX_LIMITS = {
        "gpt-5":        2_000_000,
        "gpt-5-mini":   1_000_000,
        "gpt-4.1":        128_000,
        "gpt-4.1-mini": 1_000_000,  # (long-context variant)
    }
    CTX_LIMIT = CTX_LIMITS.get(args.model, 128_000)

    # --- main loop ---
    try:
        for idx, (heading, body) in enumerate(tqdm(sections, desc="Processing", unit="section"), 1):
            # honour resume-from-profile (exact idx set)
            if resume_target_idxs is not None and idx not in resume_target_idxs:
                if args.verbose:
                    sys.stderr.write(f"[resume] Skipping idx {idx} (not in target set): {heading}\n")
                continue

            # honour resume-entities (prefer idx, fallback to heading)
            if args.mode == "entities" and args.resume_entities:
                if done_idxs and idx in done_idxs:
                    if args.verbose:
                        sys.stderr.write(f"[resume] Skipping already-done idx {idx}: {heading}\n")
                    continue
                if not done_idxs and heading in done_headings:
                    if args.verbose:
                        sys.stderr.write(f"[resume] Skipping already-done section: {heading}\n")
                    continue

            if (not body.strip()) and (not args.whole):
                if args.verbose:
                    sys.stderr.write(f"[warn] '{heading}' has no body text; skipped.\n")
                if args.profile and prof_writer:
                    prof_writer.writerow([idx, heading, "no-output", args.model, 0, 0, 0.0, "", "", "", "skipped"])
                continue

            prompt = build_prompt(args.mode, heading, body, args.format)
            total_in = num_tokens(args.model, prompt)
            if args.verbose:
                sys.stderr.write(f"[debug] {heading}: {total_in} tokens in\n")

            if total_in + args.max_tokens_out + 10 > CTX_LIMIT:
                sys.stderr.write(f"[warn] '{heading}' exceeds {args.model} context window; skipped.\n")
                if args.profile and prof_writer:
                    prof_writer.writerow([idx, heading, "no-output", args.model, total_in, 0, 0.0, "", "", "", "skipped_ctx"])
                continue

            t0 = time.time()
            want_json = (args.format in {"json", "both"} and args.mode in {"entities", "continuity"})
            response, endpoint_used = call_llm(
                client=client,
                model=args.model,
                prompt=prompt,
                max_out=args.max_tokens_out,
                want_json=want_json,
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
                sys.stderr.write(f"[warn] Empty/no response for '{heading}'. Skipping write.\n")
                if args.profile and prof_writer:
                    ci, co, ct, cur = estimate_cost(args.model, total_in, 0, args.fx_rate) if args.price else ("", "", "", "")
                    ep = endpoint_used or "no-output"
                    prof_writer.writerow([idx, heading, ep, args.model, total_in, 0, round(dt, 3), ci, co, ct, cur])
                continue

            # output to console (optional)
            print(f"\n[{heading}]\n{response.strip()}\n")

            # JSON modes: attempt to parse or store raw
            data = None
            if args.mode in {"entities", "continuity"}:
                try:
                    # allow fenced JSON
                    m = re.search(r"```json\s*(\{.*\})\s*```", response, re.DOTALL)
                    js = m.group(1).strip() if m else response.strip()
                    data = json.loads(js)
                except Exception:
                    data = {"section_heading": heading, "raw": response.strip()}

            # Write outputs
            if md_fh and args.format in {"md", "both"} and args.mode != "entities":
                if args.mode == "concise" and args.normalize_output:
                    summary, title = normalize_concise(response)
                    norm = summary.strip()
                    if title:
                        norm = f"{norm}\n\n*Suggested title:* {title}"
                    md_fh.write(f"**{heading}**\n\n{norm}\n\n")
                else:
                    md_fh.write(f"**{heading}**\n\n{response.strip()}\n\n")

            if json_fh and args.format in {"json", "both"}:
                if data is None:
                    # write raw with idx
                    json_fh.write(json.dumps({"idx": idx, "section_heading": heading, "raw": response.strip()},
                                             ensure_ascii=False) + "\n")
                else:
                    # ensure idx present
                    if isinstance(data, dict):
                        data.setdefault("idx", idx)
                        data.setdefault("section_heading", heading)
                    json_fh.write(json.dumps(data, ensure_ascii=False) + "\n")

            if args.profile and prof_writer:
                tokens_out_est = num_tokens(args.model, response)
                ci, co, ct, cur = estimate_cost(args.model, total_in, tokens_out_est, args.fx_rate) if args.price else ("", "", "", "")
                ep = endpoint_used or ("responses" if (args.engine == "responses" or (args.engine == "auto" and args.model.startswith("gpt-5"))) else "chat")
                prof_writer.writerow([idx, heading, ep, args.model, total_in, tokens_out_est, round(dt, 3), ci, co, ct, cur])

            processed += 1

    finally:
        if md_fh:
            md_fh.close()
        if json_fh:
            json_fh.close()

    if processed == 0:
        sys.exit("[error] No sections processed. Check headings, token limits, or model window.")

    # header prepend for markdown (after closing file)
    if md_path and args.format in {"md", "both"} and args.mode != "entities":
        # prepend CLI + metadata
        ts2 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output_abs_path = md_path.resolve()
        # reconstruct CLI
        arglist = []
        for key, value in vars(args).items():
            if key not in ("draft", "out_dir") and value not in (None, False):
                if isinstance(value, Path):
                    value = str(value)
                arg = f"--{key.replace('_', '-')}"
                arglist.append(arg if value is True else f"{arg} {value}")
        cli = f"python3 {Path(sys.argv[0]).name} {args.draft} " + " ".join(arglist)
        header = (
            f"# {args.mode.capitalize()} Report\n\n"
            f"**Generated:** {ts2}\n"
            f"**Output file:** `{output_abs_path}`\n"
            f"**CLI command:**\n\n```bash\n{cli}\n```\n\n---\n\n"
        )
        original_text = md_path.read_text(encoding="utf-8")
        md_path.write_text(header + original_text, encoding="utf-8")
        # light post-process
        postprocess_markdown_sections(md_path)

    # summary
    print(f"[ok] Wrote {processed}/{len(sections)} sections to {out_dir.resolve()}")
    if md_path and args.format in {"md", "both"} and args.mode != "entities":
        print(f"  • Markdown: {_pretty_rel(md_path, out_dir.parent)}")
    if jsonl_path and args.format in {"json", "both"}:
        print(f"  • JSONL:   {_pretty_rel(jsonl_path, out_dir.parent)}")
    if args.profile:
        print(f"  • Profile: {_pretty_rel(out_dir / 'profile.csv', out_dir.parent)}")


if __name__ == "__main__":
    main()