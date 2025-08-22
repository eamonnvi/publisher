# ntk_prompts.py
from typing import Dict, Any

PROMPTS = {
    # ok for string-only templates
    "concise": {
        "system": "You summarise fiction chapters for an editor.",
        "user": (
            "Summarise the following section in 4–6 crisp bullet points.\n\n"
            "{body}"
        ),
    },
    "concise_para": {
        "system": "You summarise fiction chapters for an editor.",
        "user": (
            "Summarise the following section in a single paragraph (3–5 sentences).\n\n"
            "{body}"
        ),
    },
    "concise_telegraphic": {
        "template": (
            "You are a professional story analyst.\n"
            "Summarise the following scene in **5–8 telegraphic bullet points**.\n"
            "Style rules:\n"
            "• Telegraphic (no subjects if obvious). No full sentences.\n"
            "• Each bullet on a new line starting with '- '.\n"
            "• No heading, no intro, no conclusion, no extra text.\n\n"
            "TEXT:\n{body}\n"
        )
    },
        # --- Copyediting modes ---
    # Returns the edited text ONLY (Markdown). No commentary.
    "copyedit_md": {
        "system": (
            "You are a meticulous UK-English line editor. Improve clarity, flow, grammar, "
            "and punctuation while preserving the author’s voice. Return ONLY the revised text "
            "as Markdown — no commentary. Do not summarise or shorten unless you’re removing redundancy."
        ),
        "user": (
            "# Heading\n{heading}\n\n"
            "# Text\n{body}\n\n"
            "## Edit Rules\n"
            "- Make tangible improvements (even small ones); do not return the input verbatim.\n"
            "- Preserve meaning, tone, and style; avoid Americanising spelling.\n"
            "- Keep all Markdown the author used (headings, emphasis) unless it’s clearly an error.\n"
            "- Do not add bracketed explanations or notes; output must be only the edited text."
        ),
    },
    "copyedit_suggestions": {
        "system": (
            "You are a UK-English line editor. Provide actionable copyediting suggestions "
            "without rewriting the whole passage."
        ),
        "user": (
            "# Heading\n{heading}\n\n"
            "# Text\n{body}\n\n"
            "## Output format\n"
            "Return a Markdown section with:\n"
            "### Summary\n"
            "- 2–4 bullets on overall issues (clarity, rhythm, punctuation, consistency).\n\n"
            "### Line edits\n"
            "Give 5–12 concrete suggestions. Each as a bullet:\n"
            "- **Issue:** short description\n"
            "- **Before:** quote the minimal original phrase/sentence\n"
            "- **After:** improved wording\n"
            "- **Why:** (1 short sentence)\n\n"
            "Rules:\n"
            "- Keep UK spelling and the author’s voice.\n"
            "- Don’t invent plot facts.\n"
            "- Focus on specificity: tighten wordiness, fix punctuation, untangle syntax, remove repetition.\n"
            "- If the passage truly needs no edits, write: `[No material edits]` and explain why."
        ),
    },
    # Returns a small JSON envelope with the edited markdown.
    "copyedit_json": {
        "system": (
            "You are a meticulous UK-English line editor. Improve clarity, flow, grammar, "
            "punctuation and consistency while preserving the author’s voice and meaning."
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
}

    # you can still keep old string-style templates if you want:
    # "old_mode": "Please summarise:\n\n{body}"

# ntk_prompts.py
from typing import Dict, Any

# Ensure PROMPTS is defined above this function:
# PROMPTS: Dict[str, Any] = {
#     "concise": "... {heading}\n\n{body}\n",
#     "concise_para": {...},
#     ...
# }


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