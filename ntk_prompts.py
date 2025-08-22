# ntk_prompts.py
from typing import Dict, Any

# ntk_prompts.py
from typing import Dict, Any

# -----------------------------------------------------------------------------
# Prompt registry
# -----------------------------------------------------------------------------
PROMPTS: Dict[str, Any] = {
    # --- Summarisation modes --------------------------------------------------
    "concise": {
        "system": "You summarise fiction chapters for an editor (UK English).",
        "user": (
            "Summarise the following section in 4–6 crisp bullet points.\n\n"
            "{body}"
        ),
    },
    "concise_para": {
        "system": "You summarise fiction chapters for an editor (UK English).",
        "user": (
            "Summarise the following section in a single paragraph (3–5 sentences).\n\n"
            "{body}"
        ),
    },
    "concise_telegraphic": {
        "system": "You are a professional story analyst.",
        "user": (
            "Summarise the following scene in **5–8 telegraphic bullet points**.\n"
            "Style rules:\n"
            "• Telegraphic (omit subjects if obvious)\n"
            "• No full sentences\n"
            "• Each bullet starts with '- '\n"
            "• No heading, intro, conclusion, or extra text\n\n"
            "{body}"
        ),
    },

    # Prototype “Summaries / Analysis” carried over ---------------------------
    "discursive": {
        "system": "You provide in-depth editorial analysis (UK English).",
        "user": (
            "Provide an analytical synopsis of the section titled '{heading}'. "
            "Discuss character motivations, conflicts, thematic significance, subtext, and narrative strategies. "
            "Highlight details with later narrative impact. Do not omit spoilers.\n\n"
            "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
        ),
    },

    # --- Developmental critique ----------------------------------------------
    "critique_md": {
        "system": "You are an experienced developmental editor (UK English).",
        "user": (
            "Evaluate the section titled '{heading}' in Markdown with:\n\n"
            "1. **Strengths**\n2. **Weaknesses**\n3. **Genre fit and innovation**\n\n"
            "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
        ),
    },
    "critique_json": {
        "system": (
            "You are an experienced developmental editor (UK English). "
            "Return only a single valid JSON object with keys: 'strengths', 'weaknesses', 'improvements' "
            "(each an array of strings). No extra text."
        ),
        "user": (
            "Text title: '{heading}'\n\n"
            "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
        ),
    },

    # --- Improvement suggestions ---------------------------------------------
    "improvement_suggestions": {
        "system": "You propose specific, actionable revision ideas (UK English).",
        "user": (
            "For the section titled '{heading}', list concrete, actionable revision suggestions. "
            "Focus on clarity, character motivation, suspense, pacing, and ambiguity resolution. "
            "Include sample rewrites where helpful.\n\n"
            "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
        ),
    },

    # --- Entity extraction (structured) ---------------------------------------
    "entities_json": {
        "system": (
            "Extract structured facts as JSON only (no markdown or commentary). "
            "Return a single object with keys: "
            "'section_heading', 'characters', 'pov_character', 'themes', 'locations', 'timestamp'."
        ),
        "user": (
            "Section heading: '{heading}'\n\n"
            "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
        ),
    },

    # --- Continuity -----------------------------------------------------------
    # These prompts allow {prior_entities} — if not supplied by the caller,
    # the placeholder is safely formatted as empty.
    "continuity_md": {
        "system": "You are a continuity editor.",
        "user": (
            "Compare the **current section** to these **prior facts**. "
            "Flag contradictions (dates, ages, locations, actions) and note ambiguous or potentially inconsistent elements.\n\n"
            "### Prior facts\n```json\n{prior_entities}\n```\n\n"
            "### Current section\n{body}\n"
        ),
    },
    "continuity_json": {
        "system": (
            "Compare the current section to prior facts and return ONLY a JSON array of inconsistency strings (empty if none). "
            "No extra text."
        ),
        "user": (
            "Prior facts (JSON):\n{prior_entities}\n\n"
            "Current section:\n{body}\n"
        ),
    },

    # --- Copyedit (two shapes) -----------------------------------------------
    # Fully rewritten text (Markdown only; no commentary)
    "copyedit_md": {
        "system": (
            "You are a meticulous UK-English line editor. Improve clarity, flow, grammar, and punctuation "
            "while preserving the author’s voice. Return ONLY the revised text as Markdown—no commentary. "
            "Do not summarise or shorten unless removing redundancy."
        ),
        "user": (
            "# Heading\n{heading}\n\n"
            "# Text\n{body}\n\n"
            "## Edit Rules\n"
            "- Make tangible improvements; do not return the input verbatim.\n"
            "- Preserve meaning, tone, and style; avoid Americanising spelling.\n"
            "- Keep any Markdown the author used (headings, emphasis) unless clearly erroneous.\n"
            "- Do not add bracketed explanations or notes; output must be only the edited text."
        ),
    },

    # Structured suggestions (Markdown with bullets)
    "copyedit_suggestions": {
        "system": "You are a UK-English line editor. Provide actionable copyediting suggestions.",
        "user": (
            "# Heading\n{heading}\n\n"
            "# Text\n{body}\n\n"
            "## Output format\n"
            "Return a Markdown section with:\n"
            "### Summary\n"
            "- 2–4 bullets on overall issues (clarity, rhythm, punctuation, consistency).\n\n"
            "### Line edits\n"
            "Provide 5–12 concrete suggestions. Each as a bullet with:\n"
            "- **Issue:** short description\n"
            "- **Before:** minimal original phrase/sentence (quoted)\n"
            "- **After:** improved wording\n"
            "- **Why:** one short sentence\n\n"
            "Rules:\n"
            "- Keep UK spelling and the author’s voice.\n"
            "- Don’t invent plot facts.\n"
            "- Focus on specificity: tighten wordiness, fix punctuation, untangle syntax, remove repetition.\n"
            "- If the passage truly needs no edits, write: `[No material edits]` and explain why."
        ),
    },

    # JSON envelope with edited text + issues summary
    "copyedit_json": {
        "system": (
            "You are a meticulous UK-English line editor. Improve clarity, flow, grammar, punctuation, and consistency "
            "while preserving the author’s voice and meaning. Return only valid JSON."
        ),
        "user": (
            "{{\n"
            "  \"heading\": \"{heading}\",\n"
            "  \"original\": \"\"\"{body}\"\"\",\n"
            "  \"instructions\": [\n"
            "    \"Rewrite the text fully in the same style and voice.\",\n"
            "    \"Keep all content; do not summarise or add new plot details.\",\n"
            "    \"Fix grammar, punctuation, spelling (UK), and awkward phrasing.\",\n"
            "    \"Maintain any obvious formatting cues (italics, scene breaks).\",\n"
            "    \"Output a single JSON object only—no markdown fences or extra text.\"\n"
            "  ]\n"
            "}}\n\n"
            "Return exactly this JSON shape:\n"
            "{{\n"
            "  \"heading\": string,\n"
            "  \"edited_markdown\": string,\n"
            "  \"issues\": [\n"
            "    {{\"type\": \"grammar|spelling|clarity|consistency|style\", \"before\": string, \"after\": string, \"note\": string}}\n"
            "  ],\n"
            "  \"stats\": {{\"chars_before\": number, \"chars_after\": number, \"num_issues\": number}}\n"
            "}}\n"
            "If you cannot complete, return {{}}."
        ),
    },

    # --- Overview / long-form -------------------------------------------------
    "overview": {
        "system": "You write editorial overviews (UK English).",
        "user": (
            "Write a 2,000–2,500-word editorial overview of the manuscript below. "
            "Cover plot, character arcs, themes, style, structure, pacing, genre conventions, and continuity issues.\n\n"
            "----- BEGIN MANUSCRIPT -----\n{body}\n----- END MANUSCRIPT -----"
        ),
    },
    "detailed-summary": {
        "system": "You write detailed long-form summaries (UK English).",
        "user": (
            "Write a 5,000–10,000-word chapter-by-chapter summary of the manuscript below.\n\n"
            "----- BEGIN MANUSCRIPT -----\n{body}\n----- END MANUSCRIPT -----"
        ),
    },

    # --- Sensitivity / special analysis --------------------------------------
    "sensitivity-analysis": {
        "system": "You assess sensitive content in historical/period fiction (UK English).",
        "user": (
            "Read the sexually explicit sections of the manuscript below. "
            "Assess whether they are justified by period and context. "
            "Suggest rewordings for borderline passages.\n\n"
            "----- BEGIN MANUSCRIPT -----\n{body}\n----- END MANUSCRIPT -----"
        ),
    },

    # --- Plot checkpoints / bespoke questions --------------------------------
    "plot-points": {
        "system": "You analyse foreshadowing and reader inference (UK English).",
        "user": (
            "Analyse whether the current draft telegraphs that Mrs Nairn and the glamorous actress are Sheena in disguise; "
            "and that Marlboro cigarettes mark Katrin. Evaluate the Paddington Green scene for whether Ulrike is truly Ulrike "
            "(declining cigarettes). Consider whether readers can infer Grace's developing attraction to Ulrike at Newnham. "
            "Identify plot points that fail or could be sharpened.\n\n"
            "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
        ),
    },
}

# -----------------------------------------------------------------------------
# Prompt rendering helpers
# -----------------------------------------------------------------------------

class _Safe(dict):
    """Format-map that returns empty string for missing keys (e.g. prior_entities)."""
    def __missing__(self, key):  # type: ignore[override]
        return ""

def render_prompt_text(mode: str, heading: str, body: str) -> str:
    """
    Render PROMPTS[mode] into a single text block suitable for /v1/responses
    (also fine to send as a single chat message if you fall back).

    - Expects PROMPTS[mode] to be a dict with 'system' and 'user' (or a plain string).
    - Safely formats {body}, {heading}, and silently drops any missing placeholders.
    - Returns a merged string with a small [SYSTEM] header for clarity.
    """
    spec = PROMPTS.get(mode)
    if spec is None:
        raise KeyError(f"Unknown mode: {mode!r}")

    # Allow legacy string templates
    if isinstance(spec, str):
        return spec.format_map(_Safe({"body": body, "heading": heading}))

    if not isinstance(spec, dict):
        raise TypeError(f"PROMPTS[{mode!r}] must be str or dict with 'system'/'user'")

    fmt = _Safe({"body": body, "heading": heading})
    sys_txt = (spec.get("system") or "").format_map(fmt).strip()
    usr_txt = (spec.get("user")   or "").format_map(fmt).strip()

    parts = []
    if sys_txt:
        parts.append(f"[SYSTEM]\n{sys_txt}")
    if usr_txt:
        parts.append(usr_txt)
    return "\n\n".join(parts)

# Keep old name for existing callers
build_prompt = render_prompt_text

    # you can still keep old string-style templates if you want:
    # "old_mode": "Please summarise:\n\n{body}"

def render_prompt_text(mode: str, heading: str, body: str) -> str:
    spec = PROMPTS.get(mode)
    if spec is None:
        valid = ", ".join(sorted(PROMPTS.keys()))
        raise KeyError(f"Unknown mode '{mode}'. Valid modes: {valid}")

    if isinstance(spec, str):
        return spec.format(body=body, heading=heading)  # legacy string templates

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

# Back-compat alias used elsewhere
build_prompt = render_prompt_text

def _safe_format(t: str) -> str:
    try:
        return t.format(body=body, heading=heading)
    except KeyError as e:
        # Graceful degrade if a template references an unknown placeholder
        missing = str(e).strip("'")
        return f"{t}\n\n[warn] missing placeholder: {{{missing}}}"

    if isinstance(spec, str):
        return _safe_format(spec)

    if isinstance(spec, dict):
        sys_txt = _safe_format(spec.get("system", "").strip())
        usr_txt = _safe_format(spec.get("user", "").strip())
        parts = []
        if sys_txt:
            parts.append(f"[SYSTEM]\n{sys_txt}")
        if usr_txt:
            parts.append(usr_txt)
        return "\n\n".join(parts)

    raise TypeError(f"PROMPTS[{mode!r}] must be str or dict, got {type(spec).__name__}")