#!/usr/bin/env python3
"""
novel_toolkit.py – One‑person publishing‑house assistant
=======================================================
Analyse, critique, extract entities, check continuity, copy‑edit, or outline a
novel draft via the OpenAI Chat Completions API.

2025‑07‑03  •  v4.0.1  (bug‑fix: missing code after line 208)
-----------------------------------------------------------
This patch restores the truncated `main()` body so the script runs end‑to‑end.
All functionality introduced in v4.0 (entities + continuity modes) is intact.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import tiktoken  # type: ignore
from tqdm import tqdm  # type: ignore

try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover
    sys.stderr.write("Install deps with: pip install openai tiktoken tqdm\n")
    raise

################################################################################
# Prompt templates                                                              #
################################################################################

PROMPT_TEMPLATES: Dict[str, str] = {
    # ——— summaries ———
    "concise": (
        "Summarise the following section of a novel entitled '{heading}'. "
        "Write in concise British English prose, avoiding spoilers and revealing only essential plot details."\
        "\n\n----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "discursive": (
        "Provide an **expansive synopsis** of the section titled '{heading}'. "
        "Explain character motivations, thematic significance and any subtext for a book‑club discussion (~400–600 words)."\
        "\n\n----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    # ——— developmental critique ———
    "critique_md": (
        "You are an experienced developmental editor. Evaluate '{heading}'.\n\n"
        "1. **Strengths** (voice, pacing, originality).\n2. **Weaknesses** (inconsistencies, clichés, redundancy).\n3. **Concrete improvements** with examples.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "critique_json": (
        "You are an experienced developmental editor. Return STRICT JSON with keys 'strengths', 'weaknesses', 'improvements' (arrays of strings).\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    # ——— entity extraction ———
    "entities_json": (
        "Extract structured facts from '{heading}' and output ONLY this JSON schema:\n"
        "{\n  'section_heading': '{heading}',\n  'characters': [...],\n  'pov_character': <string|null>,\n  'themes': [...],\n  'locations': [...],\n  'timestamp': <string|null>\n}\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    # ——— continuity checker ———
    "continuity_md": (
        "You are a continuity editor. Compare the **current section** to **prior facts** and list contradictions (dates, ages, locations, etc.).\n\n"
        "### Prior facts\n```json\n{prior_entities}\n```\n\n### Current section\n{body}\n"
    ),
    "continuity_json": (
        "Compare current section to prior facts. Return ONLY a JSON array of inconsistency strings (empty array if none).\n\n"
        "Prior facts JSON:\n{prior_entities}\n\nCurrent section:\n{body}\n"
    ),
    # ——— copy‑edit ———
    "copyedit_md": (
        "## Identity\nYou are a line‑by‑line copy editor in a British literary publishing house.\n\n## Task\n1. Review supplied text line by line.\n2. Identify typos, ungrammatical usage, redundancies, repetitions, punctuation issues.\n3. Inside single‑quoted direct speech, flag typos ONLY.\n\n## Prohibitions\n1. Do not break long sentences.\n2. Do not reduce adverbs.\n3. Do not spot clichés.\n4. Do not point out contractions.\n\n## Output\nBulleted Markdown list — line number, excerpt, issue, suggestion.\n\n----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "copyedit_json": (
        "(Same identity/task) Return ONLY a JSON array of objects: {line_number:int, original:str, issue:str, suggestion:str}.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    # ——— overview ———
    "overview": (
        "Write a 2 000–2 500‑word editorial overview of the manuscript below covering plot, character, themes, style, issues.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
}

################################################################################
# Helpers                                                                       #
################################################################################

def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        sys.exit(f"[error] Could not decode {path}: {exc}")


def compile_heading_regex(custom: str | None) -> re.Pattern[str]:
    if custom:
        return re.compile(custom, re.MULTILINE)
    default = r"^\s*(?:#{1,6}\s*([^#\n]+?)\s*#*\s*$|\*\*([^*]+?)\*\*\s*$)"
    return re.compile(default, re.MULTILINE)


def iter_sections(text: str, heading_rx: re.Pattern[str]) -> Iterable[Tuple[str, str]]:
    matches = list(heading_rx.finditer(text))
    if not matches:
        yield "Full Manuscript", text.strip(); return
    for i, m in enumerate(matches):
        start, end = m.end(), matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = next(g for g in m.groups() if g).strip()
        yield heading, text[start:end].strip()


def num_tokens(model: str, *chunks: str) -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")
    return sum(len(enc.encode(c)) for c in chunks)


def build_prompt(mode: str, heading: str, body: str, fmt: str, prior: str | None = None, tpl: Path | None = None) -> str:
    if tpl:
        template = read_text(tpl)
    else:
        key = {
            "critique": "critique_json" if fmt in {"json", "both"} else "critique_md",
            "entities": "entities_json",
            "continuity": "continuity_json" if fmt in {"json", "both"} else "continuity_md",
            "copyedit": "copyedit_json" if fmt in {"json", "both"} else "copyedit_md",
        }.get(mode, mode)
        template = PROMPT_TEMPLATES[key]
    return template.format(heading=heading, body=body, prior_entities=prior or "[]")


def call_openai(model: str, prompt: str, max_out: int) -> str:
    rsp = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=max_out,
    )
    return rsp.choices[0].message.content.strip()

################################################################################
# Main                                                                          #
################################################################################

def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description="Arbuthnot Books novel toolkit")
    parser.add_argument("draft", type=Path, help="Input draft file (.md or .txt)")

    modes = [
        "concise", "discursive", "critique", "entities", "continuity", "overview", "inline-edit", "copyedit",
    ]
    parser.add_argument("--mode", choices=modes, default="concise")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--whole", action="store_true", help="Treat entire file as one section")

    parser.add_argument("--format", choices=["md", "json", "both"], default="md")

    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-tokens-out", type=int, default=1024)
    parser.add_argument("--max-section-tokens", type=int, default=120_000)

    parser.add_argument("--heading-regex")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    print("[debug] novel_toolkit starting", file=sys.stderr, flush=True)

    # Key
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("[error] OPENAI_API_KEY not set")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Output dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = args.mode if not args.whole else f"{args.mode}_whole"
    out_dir = args.out_dir or Path(f"{base}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"{args.mode}.md"
    jsonl_path = out_dir / f"{args.mode}.jsonl"

    # Read + slice
    text = read_text(args.draft)

    # Slice draft into sections
    if args.whole:
        sections = [("Full Manuscript", text.strip())]
    else:
        heading_rx = compile_heading_regex(args.heading_regex)
        sections = list(iter_sections(text, heading_rx))

    # Context‑window map (extend as new models are added)
    CTX_LIMIT = {
        "gpt-4o-mini": 128_000,
        "gpt-4o": 128_000,
        "gpt-4.1-mini": 1_000_000,
    }.get(args.model, 128_000)

    # Prepare optional files
    md_fh = open(md_path, "w", encoding="utf-8") if args.format in {"md", "both"} else None
    json_fh = open(jsonl_path, "w", encoding="utf-8") if args.format in {"json", "both"} else None

    processed = 0
    entity_memory: List[dict] = []  # grows across sections for continuity checks

    try:
        for heading, body in tqdm(sections, desc="Processing", unit="section"):
            # Select prior facts when running continuity mode
            prior_json = json.dumps(entity_memory, ensure_ascii=False) if args.mode == "continuity" else None
            prompt = build_prompt(
                args.mode, heading, body, args.format, prior=prior_json, tpl=args.prompt_file
            )

            total_in = num_tokens(args.model, prompt)
            if total_in + args.max_tokens_out + 10 > CTX_LIMIT:
                sys.stderr.write(
                    f"[warn] '{heading}' exceeds {args.model} context window; skipped.\n"
                )
                continue
            if total_in > args.max_section_tokens:
                sys.stderr.write(
                    f"[warn] '{heading}' prompt size {total_in} > --max-section-tokens {args.max_section_tokens}; skipped.\n"
                )
                continue

            if args.verbose:
                sys.stderr.write(f"[debug] {heading}: {total_in} tokens in")

            try:
                response = call_openai(args.model, prompt, args.max_tokens_out)
            except Exception as exc:
                sys.stderr.write(f"[warn] OpenAI error on '{heading}': {exc}")
                continue

            # Handle JSON parsing for relevant modes
            if args.mode in {"entities", "continuity"}:
                try:
                    data = json.loads(response)
                except json.JSONDecodeError:
                    data = {"section_heading": heading, "raw": response}
            else:
                data = None

            # Write outputs
            if md_fh and args.format in {"md", "both"} and args.mode != "entities":
                md_fh.write(f"**{heading}**{response}")
            if json_fh and args.format in {"json", "both"} and data is not None:
                json_fh.write(json.dumps(data, ensure_ascii=False) + "")

            # Update entity memory for continuity
            if args.mode == "entities" and isinstance(data, dict):
                entity_memory.append(data)
            elif args.mode == "continuity":
                # Still want entities memory to grow: run extraction pass as well
                ent_prompt = build_prompt("entities", heading, body, fmt="json")
                ent_res = call_openai(args.model, ent_prompt, 512)
                try:
                    entity_memory.append(json.loads(ent_res))
                except json.JSONDecodeError:
                    pass

            processed += 1
    finally:
        if md_fh:
            md_fh.close()
        if json_fh:
            json_fh.close()

    if processed == 0:
        sys.exit("[error] No sections processed. Check headings, token limits, or model window.")

    print(f"[ok] Wrote {processed}/{len(sections)} sections to {out_dir.resolve()}")
    if md_fh and args.format in {"md", "both"}:
        print(f"  • Markdown: {md_path.relative_to(out_dir.parent)}")
    if json_fh and args.format in {"json", "both"}:
        print(f"  • JSONL:   {jsonl_path.relative_to(out_dir.parent)}")

