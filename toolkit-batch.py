#!/usr/bin/env python3
"""
novel_toolkit.py – Arbuthnot Books helper
========================================
Clean reference version 2025‑07‑04 • v5.0
----------------------------------------
This file replaces any previous partially edited copies.  It contains:

* Modes: concise, discursive, critique, entities, continuity, overview,
  inline-edit, copyedit.
* Works with Python ≥3.8 (tested with 3.11) and openai ≥1.15.0.
* CTX_LIMIT map for context‑window guards.
* Verbose flag prints token counts and model calls.
* Exits with an explicit error if zero sections processed.

Save this as *novel_toolkit.py* and run:
    python3 -u novel_toolkit.py draft.md --mode entities --format json
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
except ImportError:
    sys.stderr.write("Install deps: pip install openai tiktoken tqdm\n")
    raise

################################################################################
# Prompt templates                                                              #
################################################################################

PROMPT_TEMPLATES: Dict[str, str] = {
    # ——— summaries ———
    "concise": (
    "Summarise the following section of a novel entitled '{heading}'. "
    "Include major events, character actions, and information that may be important for the overall plot, including developments that might be relevant in later chapters. "
    "Flag any details that appear to set up future revelations, red herrings, or narrative misdirection. "
    "Write in clear British English prose.\n\n"
    "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
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
    "For the section titled '{heading}', list specific, actionable suggestions for revision. Focus on improving clarity, deepening character motivation, strengthening suspense, tightening pacing, and resolving ambiguities. "
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
    # ——— copy‑edit ———
    "copyedit_md": (
        "## Identity\nYou are a line‑by‑line copy editor in a British literary publishing house.\n\n"
        "## Task\nIdentify typos, grammar issues, redundancies, repetitions, comma splices.\nInside single‑quoted dialogue flag typos only.\n\n## Prohibitions\nDo not break sentences, reduce adverbs, spot clichés, or flag contractions.\n\n## Output\nBulleted Markdown: line number, snippet, issue, suggestion.\n\n----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
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
}

################################################################################
# Helper functions                                                              #
################################################################################


def prepend_cli_metadata_to_output(output_path: Path, args: argparse.Namespace, mode: str):
    """
    Prepend CLI command, timestamp, and output file path to the beginning of the report.
    Args:
        output_path (Path): Path to the output file.
        args (argparse.Namespace): The parsed command-line arguments.
        mode (str): The current mode (overview, meta-summary, etc.).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_abs_path = output_path.resolve()

    # Reconstruct the CLI command as accurately as possible
    # Filter out arguments with None/False values for clarity
    arglist = []
    for key, value in vars(args).items():
        if key not in ("draft", "out_dir") and value not in (None, False):
            if isinstance(value, Path):
                value = str(value)
            arglist.append(f"--{key.replace('_', '-')}" + (f" {value}" if value is not True else ""))
    cli_command = f"python3 toolkit-batch.py {args.draft} " + " ".join(arglist)

    header = (
        f"# {mode.capitalize()} Report\n\n"
        f"**Generated:** {timestamp}\n"
        f"**Output file:** `{output_abs_path}`\n"
        f"**CLI command:**\n\n"
        f"```bash\n{cli_command}\n```\n\n"
        "---\n\n"
    )

    # Prepend to file
    original_text = output_path.read_text(encoding="utf-8")
    with output_path.open("w", encoding="utf-8") as f:
        f.write(header + original_text)

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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
        start = m.end(); end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = next(g for g in m.groups() if g).strip()
        yield heading, text[start:end].strip()


def num_tokens(model: str, *chunks: str) -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")
    return sum(len(enc.encode(c)) for c in chunks)


def build_prompt(
    mode: str,
    heading: str,
    body: str,
    fmt: str,
    prior: str | None = None
) -> str:
    """
    Selects the appropriate prompt template based on mode and format,
    fills in heading/body (and prior if needed), and returns the final prompt string.
    """
    # Mapping of logical modes to prompt template keys
    key_map = {
        "critique": "critique_json" if fmt in {"json", "both"} else "critique_md",
        "entities": "entities_json",
        "continuity": "continuity_json" if fmt in {"json", "both"} else "continuity_md",
        "copyedit": "copyedit_json" if fmt in {"json", "both"} else "copyedit_md",
        # direct modes: concise, discursive, overview, inline-edit
    }

    # Determine which template key to use
    template_key = key_map.get(mode, mode)  # fallback to mode itself for "concise", etc.

    # Robust error handling with a descriptive message
    if template_key not in PROMPT_TEMPLATES:
        raise KeyError(
            f"\n[error] No prompt template for mode '{mode}' (key '{template_key}')."
            f"\nAvailable templates: {list(PROMPT_TEMPLATES.keys())}"
            f"\nCheck your --mode ('{mode}') and --format ('{fmt}') arguments."
        )

    template = PROMPT_TEMPLATES[template_key]

    # Compose the prompt, supplying prior_entities if required by the template
    # In build_prompt(), dynamically adjust heading label:
    heading_label = "manuscript" if heading.lower() == "full manuscript" else "section"
    prompt = template.format(
        heading=heading,
        body=body,
        prior_entities=prior or "[]"
    )
    return prompt

def call_openai(model: str, prompt: str, max_out: int) -> str:
    rsp = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=max_out,
    )
    return rsp.choices[0].message.content.strip()

def extract_json_from_response(response: str) -> dict:
    import re, json

    # Try ```json ... ``` first
    match = re.search(r"```json\s*(\{.*\})\s*```", response, re.DOTALL)
    if not match:
        # Try generic triple-backtick code block
        match = re.search(r"```\s*(\{.*\})\s*```", response, re.DOTALL)
    json_str = match.group(1).strip() if match else response.strip()
    return json.loads(json_str)

def batch_sections_by_token_limit(sections, max_tokens, model_name, safety_margin=2048):
    """
    Yield (batch_heading, batch_body) tuples, each with total tokens <= max_tokens - safety_margin.
    """
    current_batch = []
    current_tokens = 0
    for heading, body in sections:
        body_tokens = num_tokens(model_name, body)
        # If adding this section would overflow the batch, yield what we have
        if current_tokens + body_tokens > max_tokens - safety_margin and current_batch:
            batch_heading = f"{current_batch[0][0]} to {current_batch[-1][0]}"
            batch_body = "\n\n".join([s[1] for s in current_batch])
            yield (batch_heading, batch_body)
            current_batch = []
            current_tokens = 0
        current_batch.append((heading, body))
        current_tokens += body_tokens
    # Yield the last batch
    if current_batch:
        batch_heading = f"{current_batch[0][0]} to {current_batch[-1][0]}"
        batch_body = "\n\n".join([s[1] for s in current_batch])
        yield (batch_heading, batch_body)

################################################################################
# Main                                                                          #
################################################################################

def main() -> None:  # noqa: C901
    p = argparse.ArgumentParser(description="Arbuthnot Books novel toolkit")
    p.add_argument("draft", type=Path, help="Input .md or .txt file")

    modes = [
        "concise", "discursive", "critique", "entities", "continuity",
        "overview", "inline-edit", "copyedit", "improvement_suggestions",
        "detailed-summary",
    ]
    p.add_argument("--mode", choices=modes, default="concise")
    p.add_argument("--whole", action="store_true", help="Treat entire file as one section")
    p.add_argument("--format", choices=["md", "json", "both"], default="md")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--max-tokens-out", type=int, default=1024)
    p.add_argument("--max-section-tokens", type=int, default=120_000)
    p.add_argument("--heading-regex")
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--entities-file", type=Path, help="Path to entities.jsonl for continuity mode")
    p.add_argument("--batch", action="store_true", help="Enable batching of sections by token limit")
    args = p.parse_args()

    # Enforce mutual exclusion between --whole and --batch
    if args.whole and args.batch:
        sys.exit("[error] Do not use --whole and --batch together. Choose one or neither.")

    #print(f"[debug] args.mode: {args.mode}, args.entities_file: {args.entities_file}")

    # Load entity_memory from file if --entities-file is provided in continuity mode
    entity_memory = []
    if args.mode == "continuity" and args.entities_file:
        with open(args.entities_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entity_memory.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        #print(f"[debug] Loaded {len(entity_memory)} entities from {args.entities_file}")
    # ...rest of your main() function...

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

    # Decide batching strategy
    batching_modes = {"overview", "detailed-summary"}  # extend as needed

    if args.batch:
        batches = list(batch_sections_by_token_limit(
            sections,
            max_tokens=args.max_section_tokens,
            model_name=args.model,
            safety_margin=args.max_tokens_out + 512  # adjust as needed
        ))
        loop_iter = batches
        tqdm_unit = "batch"
    else:
        loop_iter = sections
        tqdm_unit = "section"

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
    #entity_memory: List[dict] = []  # grows across sections for continuity checks

    try:
        for i, (heading, body) in enumerate(tqdm(loop_iter, desc="Processing", unit=tqdm_unit)):
            if args.mode == "continuity" and args.entities_file:
                prior_entities = entity_memory[:i]
                #print(f"[debug] Section {i+1} ({heading}): {len(prior_entities)} prior facts")
                # Optionally, print a sample prior entity:
                if prior_entities:
                    print(f"[debug] Sample prior entity: {prior_entities[-1]}")
                prior_json = json.dumps(prior_entities, ensure_ascii=False)
            else:
                prior_json = json.dumps(entity_memory, ensure_ascii=False) if args.mode == "continuity" else None

            prompt = build_prompt(
                args.mode, heading, body, args.format, prior=prior_json
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
                print(f"\n[{heading}]\n{response}\n")
                #print(f"[entities debug] {response}")   # <--- Add here
            except Exception as exc:
                sys.stderr.write(f"[warn] OpenAI error on '{heading}': {exc}")
                continue

            # Handle JSON parsing for relevant modes
            if args.mode in {"entities", "continuity"}:
                try:
                    data = extract_json_from_response(response)
                except Exception:
                    data = {"section_heading": heading, "raw": response}

            # Write outputs
            if md_fh and args.format in {"md", "both"} and args.mode != "entities":
                md_fh.write(f"**{heading}**\n\n{response}\n")
            if json_fh and args.format in {"json", "both"} and data is not None:
                json_fh.write(json.dumps(data, ensure_ascii=False) + "\n")

            # Update entity memory for continuity
            if args.mode == "entities" and isinstance(data, dict):
                entity_memory.append(data)
            elif args.mode == "continuity":
                # Still want entities memory to grow: run extraction pass as well
                ent_prompt = build_prompt("entities", heading, body, fmt="json")
                ent_res = call_openai(args.model, ent_prompt, 512)
                #print(f"[entities debug] {ent_res}")  # <--- Add here
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

    # <--- Add here:
    if md_fh and args.format in {"md", "both"}:
        prepend_cli_metadata_to_output(md_path, args, args.mode)

    print(f"[ok] Wrote {processed}/{len(sections)} sections to {out_dir.resolve()}")
    if md_fh and args.format in {"md", "both"}:
        print(f"  • Markdown: {md_path.relative_to(out_dir.parent)}")
    if json_fh and args.format in {"json", "both"}:
        print(f"  • JSONL:   {jsonl_path.relative_to(out_dir.parent)}")

if __name__ == "__main__":
    main()

