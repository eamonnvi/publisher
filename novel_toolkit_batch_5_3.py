#!/usr/bin/env python3
"""
novel_toolkit.py – Arbuthnot Books helper
========================================
Clean reference version 2025-08-10 • v5.3

What's new in v5.3
------------------
- --profile: per-section tokens+latency CSV (profile.csv) for sync runs; also for batch fetch (reconstructed).
- --batch-dry-run: build batch_input.jsonl but do not submit (cost audit / inspection).

Other recent features retained
-----------------------------
- GPT-5 family support (Responses or Chat), robust extractors, --dump-raw.
- Parameter quirks handled (no temperature for GPT-5 chat; max_completion_tokens).
- Batch API: --batch-submit / --batch-fetch.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import tiktoken  # type: ignore
from tqdm import tqdm  # type: ignore

try:
    from openai import OpenAI  # modern client
except ImportError:
    sys.stderr.write("Install deps: pip install openai tiktoken tqdm\n")
    raise

################################################################################
# Prompt templates (unchanged from v5.2)                                        #
################################################################################

PROMPT_TEMPLATES: Dict[str, str] = {
    "concise": (
        "Summarise the following section of a novel entitled '{heading}'. "
        "Include major events, character actions, and information that may be important for the overall plot. "
        "Suggest a more suitable title for the chapter. "
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
    "critique_md": (
        "You are an experienced developmental editor. Evaluate the section titled '{heading}' as follows:\n\n"
        "1. **Strengths:** Comment on voice, pacing, originality, narrative structure, and style.\n"
        "2. **Weaknesses:** Note inconsistencies, clichés, narrative slack, unresolved setups, and anything that may undermine suspense or thematic unity.\n"
        "3. **Genre fit and innovation:** Discuss how the section aligns with or subverts expectations for its genre.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "critique_json": (
        "You are an experienced developmental editor. Return STRICT JSON with keys "
        "'strengths', 'weaknesses', 'improvements' – each an array of strings.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "improvement_suggestions": (
        "For the section titled '{heading}', list specific, actionable suggestions for revision. "
        "Focus on improving clarity, deepening character motivation, strengthening suspense, tightening pacing, and resolving ambiguities. "
        "Include examples or sample rewrites if relevant.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "entities_json": (
        "Extract structured facts from the section titled '{heading}'. "
        "Return only a single valid JSON object with these keys: "
        "'section_heading', 'characters', 'pov_character', 'themes', 'locations', 'timestamp'. "
        "Do not include explanations, markdown, or any non-JSON output.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "continuity_md": (
        "You are a continuity editor. Compare the **current section** to the following **prior facts**. "
        "Flag any contradictions (dates, ages, locations, character actions, etc.), and note any ambiguous or potentially inconsistent elements (even if not outright contradictions).\n\n"
        "### Prior facts\n```json\n{prior_entities}\n```\n\n### Current section\n{body}\n"
    ),
    "continuity_json": (
        "Compare current section to prior facts. Return ONLY a JSON array of inconsistency strings (empty if none).\n\n"
        "Prior facts: {prior_entities}\n\nCurrent section:\n{body}\n"
    ),
    "copyedit_md": (
        "## Identity\nYou are a line-by-line copy editor in a British literary publishing house.\n\n"
        "## Task\nIdentify typos, grammar issues, redundancies, repetitions.\n\n"
        "## Prohibitions\nDo not break sentences, reduce the number of adverbs, spot clichés, or flag contractions.\n\n"
        "## Output\nBulleted Markdown: line number, snippet, issue, suggestion.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
    "copyedit_json": (
        "(Same identity/task) Return JSON array of objects "
        "{line_number:int, original:str, issue:str, suggestion:str}.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
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
        "Is the narrative of the current draft presented in such a way that the attentive reader would guess that Mrs Nairn "
        "and the glamourous female film star are Sheena in disguise and that the presence of Marlboro cigarettes marks Katrin? "
        "The fact that Ulrike declines cigarettes in the Paddington Green scene indicates that the character in this scene really is Ulrike and not Katrin. "
        "Is it also possible for the reader to infer that Grace is developing a lesbian attraction to Ulrike in the Newnham College high table scene? "
        "Finally, please identify any plot points that don't work or could be more sharply written.\n\n"
        "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
    ),
}

################################################################################
# Helpers                                                                       #
################################################################################

def prepend_cli_metadata_to_output(output_path: Path, args: argparse.Namespace, mode: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_abs_path = output_path.resolve()
    arglist = []
    for key, value in vars(args).items():
        if key not in ("draft", "out_dir") and value not in (None, False):
            if isinstance(value, Path):
                value = str(value)
            arglist.append(f"--{key.replace('_', '-')}" + (f" {value}" if value is not True else ""))
    cli_command = f"python3 novel_toolkit.py {args.draft} " + " ".join(arglist)
    header = (
        f"# {mode.capitalize()} Report\n\n"
        f"**Generated:** {timestamp}\n"
        f"**Output file:** `{output_abs_path}`\n"
        f"**CLI command:**\n\n"
        f"```bash\n{cli_command}\n```\n\n"
        "---\n\n"
    )
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

################################################################################
# Prompts                                                                       #
################################################################################

def build_prompt(
    mode: str,
    heading: str,
    body: str,
    fmt: str,
    prior: str | None = None
) -> str:
    key_map = {
        "critique": "critique_json" if fmt in {"json", "both"} else "critique_md",
        "entities": "entities_json",
        "continuity": "continuity_json" if fmt in {"json", "both"} else "continuity_md",
        "copyedit": "copyedit_json" if fmt in {"json", "both"} else "copyedit_md",
    }
    template_key = key_map.get(mode, mode)
    if template_key not in PROMPT_TEMPLATES:
        raise KeyError(
            f"\n[error] No prompt template for mode '{mode}' (key '{template_key}')."
            f"\nAvailable templates: {list(PROMPT_TEMPLATES.keys())}"
            f"\nCheck your --mode ('{mode}') and --format ('{fmt}') arguments."
        )
    template = PROMPT_TEMPLATES[template_key]
    return template.format(heading=heading, body=body, prior_entities=prior or "[]")

################################################################################
# Batch helpers                                                                 #
################################################################################

def _mk_batch_line_responses(custom_id: str, model: str, prompt: str, max_out: int, want_json: bool) -> dict:
    body = {"model": model, "input": prompt, "max_output_tokens": max_out}
    if want_json:
        body["response_format"] = {"type": "json_object"}
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/responses", "body": body}

def _mk_batch_line_chat(custom_id: str, model: str, prompt: str, max_out: int, want_json: bool) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if model.startswith("gpt-5"):
        body["max_completion_tokens"] = max_out
    else:
        body["max_tokens"] = max_out
        body["temperature"] = 0.3
    if want_json:
        body["response_format"] = {"type": "json_object"}
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}

################################################################################
# LLM caller (with profiling hook)                                              #
################################################################################

def call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    max_out: int,
    want_json: bool,
    engine: str,
    fallback_model: str,
    verbose: bool = False,
    dump_raw_dir: str | None = None,
    heading: str | None = None,
    profile_hook: Callable[[Dict[str, str]], None] | None = None,
) -> str | None:
    """
    Returns text (or None). Calls profile_hook with {'endpoint','model','stage'}.
    """
    import json as _json
    from pathlib import Path as _Path

    def _dump_raw(tag: str, obj: object):
        if not dump_raw_dir:
            return
        try:
            _Path(dump_raw_dir).mkdir(parents=True, exist_ok=True)
            fname = f"raw_{tag}_{(heading or 'section').replace(' ', '_')}.json"
            fpath = _Path(dump_raw_dir) / fname
            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    _json.dump(obj, f, ensure_ascii=False, default=lambda x: getattr(x, "__dict__", str(x)))
            except Exception:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(str(obj))
            if verbose:
                sys.stderr.write(f"[debug] dumped raw response to {fpath}\n")
        except Exception as _e:
            if verbose:
                sys.stderr.write(f"[debug] failed to dump raw: {_e}\n")

    def use_responses(m: str, eng: str) -> bool:
        if eng == "responses":
            return True
        if eng == "chat":
            return False
        return m.startswith("gpt-5")

    def _extract_responses_text(r) -> str:
        t = getattr(r, "output_text", None)
        if isinstance(t, str) and t.strip():
            return t.strip()
        parsed = getattr(r, "output_parsed", None)
        if parsed is not None:
            try:
                return _json.dumps(parsed, ensure_ascii=False)
            except Exception:
                pass
        chunks = []
        output = getattr(r, "output", None) or []
        for item in output:
            for c in getattr(item, "content", []) or []:
                t2 = getattr(c, "text", None) or getattr(c, "content", None)
                if isinstance(t2, str) and t2.strip():
                    chunks.append(t2.strip())
        return "\n".join(chunks).strip() if chunks else ""

    def _extract_chat_text(r) -> str:
        try:
            msg = r.choices[0].message
            content = getattr(msg, "content", None)
            if isinstance(content, str) and content.strip():
                return content.strip()
        except Exception:
            pass
        t = getattr(r, "output_text", None)
        return t.strip() if isinstance(t, str) else ""

    def responses_call(m: str):
        kwargs = {"model": m, "input": prompt, "max_output_tokens": max_out}
        if not m.startswith("gpt-5"):
            kwargs["generation_config"] = {"temperature": 0.3}
        if want_json:
            kwargs["response_format"] = {"type": "json_object"}
        r = client.responses.create(**kwargs)
        text = _extract_responses_text(r)
        if not text:
            _dump_raw("responses", r)
        return text

    def chat_call(m: str):
        chat_kwargs = {"model": m, "messages": [{"role": "user", "content": prompt}]}
        if m.startswith("gpt-5"):
            chat_kwargs["max_completion_tokens"] = max_out  # no temperature for gpt-5 chat
        else:
            chat_kwargs["max_tokens"] = max_out
            chat_kwargs["temperature"] = 0.3
        if want_json:
            chat_kwargs["response_format"] = {"type": "json_object"}
        r = client.chat.completions.create(**chat_kwargs)
        text = _extract_chat_text(r)
        if not text:
            _dump_raw("chat", r)
        return text

    def call_once(m: str, prefer_responses: bool, stage: str) -> str:
        endpoint = "responses" if prefer_responses else "chat"
        if profile_hook:
            profile_hook({"endpoint": endpoint, "model": m, "stage": stage})
        if prefer_responses:
            if verbose:
                sys.stderr.write(f"[info] Using Responses API with model '{m}' (json={want_json})\n")
            return responses_call(m)
        else:
            if verbose:
                sys.stderr.write(f"[info] Using Chat Completions with model '{m}' (json={want_json})\n")
            return chat_call(m)

    try:
        prefer_responses = use_responses(model, engine)
        text = call_once(model, prefer_responses, stage="primary")
        if text.strip():
            return text
        if model.startswith("gpt-5"):
            if verbose:
                sys.stderr.write("[warn] Empty text from primary endpoint; retrying via alternate endpoint for gpt-5…\n")
            text2 = call_once(model, not prefer_responses, stage="primary-alt")
            if text2.strip():
                return text2
        raise RuntimeError("Empty response text")
    except Exception as e:
        sys.stderr.write(f"[warn] primary model '{model}' failed/empty ({e}); falling back to '{fallback_model}'\n")
        try:
            prefer_responses_fb = use_responses(fallback_model, engine)
            text_fb = call_once(fallback_model, prefer_responses_fb, stage="fallback")
            if text_fb.strip():
                return text_fb
            if fallback_model.startswith("gpt-5"):
                text_fb2 = call_once(fallback_model, not prefer_responses_fb, stage="fallback-alt")
                if text_fb2.strip():
                    return text_fb2
            return None
        except Exception as e2:
            sys.stderr.write(f"[warn] fallback call also failed: {e2}\n")
            return None

def extract_json_from_response(response: str) -> dict:
    try:
        return json.loads(response)
    except Exception:
        pass
    match = re.search(r"```json\s*(\{.*\})\s*```", response, re.DOTALL)
    if not match:
        match = re.search(r"```\s*(\{.*\})\s*```", response, re.DOTALL)
    json_str = match.group(1).strip() if match else response.strip()
    return json.loads(json_str)

################################################################################
# Main                                                                          #
################################################################################

def main() -> None:  # noqa: C901
    p = argparse.ArgumentParser(description="Arbuthnot Books novel toolkit")
    p.add_argument("draft", type=Path, help="Input .md or .txt file")

    modes = [
        "concise", "discursive", "critique", "entities", "continuity",
        "overview", "inline-edit", "copyedit", "improvement_suggestions",
        "detailed-summary", "sensitivity-analysis", "plot-points",
    ]
    p.add_argument("--mode", choices=modes, default="concise")
    p.add_argument("--whole", action="store_true", help="Treat entire file as one section")
    p.add_argument("--format", choices=["md", "json", "both"], default="md")

    MODEL_CHOICES = ["gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4.1-mini"]
    p.add_argument("--model", default="gpt-4.1-mini", choices=MODEL_CHOICES, help="Model to use")
    p.add_argument("--engine", choices=["auto", "responses", "chat"], default="auto",
                   help="auto: Responses for gpt-5*, Chat for others")
    p.add_argument("--fallback-model", default="gpt-4.1-mini", choices=MODEL_CHOICES,
                   help="Fallback model if primary fails")

    # Batch controls
    p.add_argument("--batch-submit", action="store_true",
                   help="Submit a Batch API job instead of running synchronously")
    p.add_argument("--batch-fetch", metavar="BATCH_ID",
                   help="Fetch results for a completed batch id and write outputs")
    p.add_argument("--batch-dry-run", action="store_true",
                   help="Build batch_input.jsonl but do NOT submit")

    p.add_argument("--max-tokens-out", type=int, default=1024)
    p.add_argument("--max-section-tokens", type=int, default=120_000)
    p.add_argument("--heading-regex")
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dump-raw", action="store_true",
                   help="Dump raw API responses to out_dir when output is empty (debug)")
    p.add_argument("--profile", action="store_true",
                   help="Record per-section tokens+latency to profile.csv")
    p.add_argument("--entities-file", type=Path, help="Path to entities.jsonl for continuity mode")
    args = p.parse_args()

    # Load entity_memory for continuity mode (if provided)
    entity_memory: List[dict] = []
    if args.mode == "continuity" and args.entities_file:
        with open(args.entities_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entity_memory.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # API key & client
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("[error] OPENAI_API_KEY not set")
    client = OpenAI()

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

    # Context-window map
    CTX_LIMITS = {
        "gpt-5":         1_000_000,
        "gpt-5-mini":    1_000_000,
        "gpt-4.1":         128_000,
        "gpt-4.1-mini":  1_000_000,
    }
    CTX_LIMIT = CTX_LIMITS.get(args.model)
    if CTX_LIMIT is None:
        sys.exit(f"[error] Model '{args.model}' not supported by this script.")

    # -------------------------------------------------------------------------
    # Batch: SUBMIT (with optional dry-run)
    # -------------------------------------------------------------------------
    if args.batch_submit:
        want_json = (args.format in {"json", "both"}) and args.mode in {"critique", "entities", "continuity", "copyedit"}
        lines = []
        # Choose endpoint consistent with engine/model
        default_endpoint = "/v1/responses" if (args.engine == "responses" or (args.engine == "auto" and args.model.startswith("gpt-5"))) else "/v1/chat/completions"
        for idx, (heading, body) in enumerate(sections, start=1):
            prior_json = json.dumps(entity_memory, ensure_ascii=False) if (args.mode == "continuity" and entity_memory) else ("[]" if args.mode == "continuity" else None)
            prompt = build_prompt(args.mode, heading, body, args.format, prior=prior_json)
            custom_id = f"sec-{idx:04d}"
            if default_endpoint == "/v1/responses":
                lines.append(_mk_batch_line_responses(custom_id, args.model, prompt, args.max_tokens_out, want_json))
            else:
                lines.append(_mk_batch_line_chat(custom_id, args.model, prompt, args.max_tokens_out, want_json))

        batch_in = out_dir / "batch_input.jsonl"
        with open(batch_in, "w", encoding="utf-8") as f:
            for obj in lines:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        if args.batch_dry_run:
            print(f"[ok] Built batch input (dry-run only): {batch_in}")
            print("     Inspect or edit this file; submit later with --batch-submit")
            return

        file_obj = client.files.create(file=open(batch_in, "rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint=default_endpoint,
            completion_window="24h",
        )
        print(f"[ok] Submitted batch: {batch.id}")
        print("    Save this id and later run with: --batch-fetch", batch.id)
        return

    # -------------------------------------------------------------------------
    # Batch: FETCH (plus optional profiling using batch_input.jsonl)
    # -------------------------------------------------------------------------
    if args.batch_fetch:
        batch = client.batches.retrieve(args.batch_fetch)
        status = getattr(batch, "status", None)
        print(f"[info] batch {batch.id} status: {status}")
        if status != "completed":
            sys.exit("[error] Batch not completed yet. Try again later.")

        out_file_id = getattr(batch, "output_file_id", None)
        if not out_file_id:
            sys.exit("[error] No output_file_id on completed batch.")

        content = client.files.content(out_file_id).text
        raw_out = out_dir / "batch_output.jsonl"
        raw_out.write_text(content, encoding="utf-8")

        # Optional: reconstruct prompts to estimate tokens_in if --profile is set
        prompts_by_cid: Dict[str, str] = {}
        batch_in = out_dir / "batch_input.jsonl"
        if args.profile and batch_in.exists():
            with open(batch_in, "r", encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    cid = rec.get("custom_id")
                    body = rec.get("body", {})
                    prompt = body.get("input")
                    if not prompt:
                        # chat form
                        msgs = body.get("messages") or []
                        if msgs and isinstance(msgs, list):
                            prompt = msgs[0].get("content")
                    if cid and isinstance(prompt, str):
                        prompts_by_cid[cid] = prompt

        # Map custom_id -> text
        results: Dict[str, str] = {}
        for line in content.splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            cid = rec.get("custom_id")
            data = rec.get("response", {}) or rec.get("result", {}) or {}
            text = data.get("output_text") or ""
            if not text:
                out = data.get("output") or []
                if out and isinstance(out, list):
                    parts = []
                    for item in out:
                        for c in item.get("content", []) or []:
                            t = c.get("text") or c.get("content")
                            if isinstance(t, str) and t.strip():
                                parts.append(t.strip())
                    if parts:
                        text = "\n".join(parts)
            if not text:
                try:
                    text = data["choices"][0]["message"]["content"]
                except Exception:
                    text = ""
            results[cid] = text or ""

        # Writers
        md_fh = open(md_path, "w", encoding="utf-8") if args.format in {"md", "both"} else None
        json_fh = open(jsonl_path, "w", encoding="utf-8") if args.format in {"json", "both"} else None

        # Optional profiling CSV
        prof_writer = None
        prof_fh = None
        if args.profile:
            prof_fh = open(out_dir / "profile.csv", "w", newline="", encoding="utf-8")
            prof_writer = csv.writer(prof_fh)
            prof_writer.writerow(["idx", "heading", "endpoint", "model", "tokens_in_est", "tokens_out_est", "latency_s"])

        processed = 0
        for idx, (heading, _body) in enumerate(sections, start=1):
            cid = f"sec-{idx:04d}"
            response = results.get(cid, "")

            if md_fh and args.mode != "entities" and args.format in {"md", "both"}:
                md_fh.write(f"**{heading}**\n\n{response}\n")
            if json_fh and args.format in {"json", "both"} and args.mode in {"entities", "continuity"}:
                try:
                    json_fh.write(json.dumps(extract_json_from_response(response), ensure_ascii=False) + "\n")
                except Exception:
                    json_fh.write(json.dumps({"section_heading": heading, "raw": response}, ensure_ascii=False) + "\n")

            if prof_writer:
                prompt_used = prompts_by_cid.get(cid, "")
                tokens_in = num_tokens(args.model, prompt_used) if prompt_used else 0
                tokens_out = num_tokens(args.model, response) if response else 0
                # Batch latency per item is not exposed; write 0.0 as placeholder
                prof_writer.writerow([idx, heading, "batch", args.model, tokens_in, tokens_out, 0.0])

            processed += 1

        if md_fh: md_fh.close()
        if json_fh: json_fh.close()
        if prof_fh: prof_fh.close()

        if md_fh and args.format in {"md", "both"}:
            prepend_cli_metadata_to_output(md_path, args, args.mode)

        print(f"[ok] Wrote batch results to {out_dir.resolve()}")
        return

    # -------------------------------------------------------------------------
    # Synchronous path (original behaviour) with profiling
    # -------------------------------------------------------------------------

    md_fh = open(md_path, "w", encoding="utf-8") if args.format in {"md", "both"} else None
    json_fh = open(jsonl_path, "w", encoding="utf-8") if args.format in {"json", "both"} else None

    # Optional profiling CSV
    prof_writer = None
    prof_fh = None
    if args.profile:
        prof_fh = open(out_dir / "profile.csv", "w", newline="", encoding="utf-8")
        prof_writer = csv.writer(prof_fh)
        prof_writer.writerow(["idx", "heading", "endpoint", "model", "tokens_in_est", "tokens_out_est", "latency_s"])

    processed = 0

    try:
        for i, (heading, body) in enumerate(tqdm(sections, desc="Processing", unit="section")):
            if args.mode == "continuity" and args.entities_file:
                prior_entities = entity_memory[:i]
                prior_json = json.dumps(prior_entities, ensure_ascii=False)
            else:
                prior_json = json.dumps(entity_memory, ensure_ascii=False) if args.mode == "continuity" else None

            prompt = build_prompt(args.mode, heading, body, args.format, prior=prior_json)

            total_in = num_tokens(args.model, prompt)
            if total_in + args.max_tokens_out + 10 > CTX_LIMIT:
                sys.stderr.write(f"[warn] '{heading}' exceeds {args.model} context window; skipped.\n")
                continue
            if total_in > args.max_section_tokens:
                sys.stderr.write(f"[warn] '{heading}' prompt size {total_in} > --max-section-tokens {args.max_section_tokens}; skipped.\n")
                continue

            if args.verbose:
                sys.stderr.write(f"[debug] {heading}: {total_in} tokens in\n")

            # capture endpoint/model chosen via profile_hook
            chosen_endpoint = {"endpoint": "", "model": ""}
            def hook(info: Dict[str, str]):
                # keep the last stage that actually executed (primary or fallback)
                chosen_endpoint["endpoint"] = info.get("endpoint", "")
                chosen_endpoint["model"] = info.get("model", "")

            want_json = (args.format in {"json", "both"}) and args.mode in {"critique", "entities", "continuity", "copyedit"}
            t0 = time.monotonic()
            try:
                response = call_llm(
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
                    profile_hook=hook if args.profile else None,
                )
            except Exception as exc:
                sys.stderr.write(f"[warn] OpenAI error on '{heading}': {exc}\n")
                continue
            dt = time.monotonic() - t0

            # Safeguards
            if response is None:
                sys.stderr.write(f"[warn] Empty response for '{heading}'. Skipping write.\n")
                continue
            if not isinstance(response, str):
                response = str(response)
            if not response.strip():
                sys.stderr.write(f"[warn] Blank response for '{heading}'.\n")

            print(f"\n[{heading}]\n{response}\n")

            # Profiling row
            if prof_writer:
                tokens_out = num_tokens(chosen_endpoint["model"] or args.model, response) if response else 0
                prof_writer.writerow([
                    i + 1,
                    heading,
                    chosen_endpoint["endpoint"] or ("responses" if (args.engine == "responses" or (args.engine == "auto" and args.model.startswith("gpt-5"))) else "chat"),
                    chosen_endpoint["model"] or args.model,
                    total_in,
                    tokens_out,
                    round(dt, 3),
                ])

            # JSON parsing for relevant modes
            data = None
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
                ent_prompt = build_prompt("entities", heading, body, fmt="json")
                ent_t0 = time.monotonic()
                ent_res = call_llm(
                    client, args.model, ent_prompt, 512,
                    want_json=True, engine=args.engine, fallback_model=args.fallback_model,
                    verbose=args.verbose, dump_raw_dir=str(out_dir) if args.dump_raw else None, heading=heading,
                    profile_hook=None,
                )
                # We don't profile the entities pass to keep CSV clean
                try:
                    if ent_res:
                        entity_memory.append(json.loads(ent_res))
                except json.JSONDecodeError:
                    pass

            processed += 1
    finally:
        if md_fh:
            md_fh.close()
        if json_fh:
            json_fh.close()
        if prof_fh:
            prof_fh.close()

    if processed == 0:
        sys.exit("[error] No sections processed. Check headings, token limits, or model window.")

    if md_fh and args.format in {"md", "both"}:
        prepend_cli_metadata_to_output(md_path, args, args.mode)

    print(f"[ok] Wrote {processed}/{len(sections)} sections to {out_dir.resolve()}")
    if md_fh and args.format in {"md", "both"}:
        print(f"  • Markdown: {md_path.relative_to(out_dir.parent)}")
    if json_fh and args.format in {"json", "both"}:
        print(f"  • JSONL:   {jsonl_path.relative_to(out_dir.parent)}")

if __name__ == "__main__":
    main()
