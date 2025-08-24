# ntk_prompts.py
from typing import Dict, Any

PROMPTS: Dict[str, Any] = {
    # --- Summarisation ---
    "concise": {
        "system": "You summarise fiction chapters for an editor.",
        "user": "Summarise the following section in 4–6 crisp bullet points.\n\n{body}",
    },
    "concise_para": {
        "system": "You summarise fiction chapters for an editor.",
        "user": "Summarise the following section in a single paragraph (3–5 sentences).\n\n{body}",
    },
    "concise_telegraphic": {
        "system": "You are a professional story analyst.",
        "user": (
            "Summarise the following scene in **5–8 telegraphic bullet points**.\n"
            "Style rules:\n"
            "• Telegraphic (omit subjects if obvious). No full sentences.\n"
            "• Each bullet starts with '- '.\n"
            "• No heading, intro, conclusion, or extra text.\n\n"
            "TEXT:\n{body}\n"
        ),
    },
    "discursive": {
        "system": "You are an experienced story editor.",
        "user": (
            "Provide an in-depth analytical synopsis of the section titled '{heading}'.\n"
            "Discuss motivations, conflicts, themes, subtext, structure, and seeds for later payoffs.\n\n"
            "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
        ),
    },

    # --- Developmental critique ---
    "critique_md": {
        "system": "You are an experienced developmental editor.",
        "user": (
            "Evaluate the section titled '{heading}' as follows:\n\n"
            "1. **Strengths**\n2. **Weaknesses**\n3. **Genre fit & opportunities**\n\n"
            "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
        ),
    },
    "critique_json": {
        "system": "Return STRICT JSON with keys strengths, weaknesses, improvements.",
        "user": (
            "Analyse the section '{heading}'. Return only JSON: "
            "{\"strengths\":[], \"weaknesses\":[], \"improvements\":[]}\n\n"
            "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
        ),
    },

    # --- Improvement suggestions ---
    "improvement_suggestions": {
        "system": "You suggest concrete ways to improve fiction scenes.",
        "user": (
            "For the section '{heading}', list specific, actionable revision suggestions. "
            "Focus on clarity, motivation, pacing, and ambiguity resolution. Include sample rewrites if apt.\n\n"
            "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
        ),
    },

    # --- Entities / continuity ---
    "entities_json": {
        "system": "Extract structured facts; return only JSON.",
        "user": (
            "Extract facts from '{heading}'. Return a single JSON object with keys: "
            "section_heading, characters, pov_character, themes, locations, timestamp.\n\n"
            "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
        ),
    },
    "continuity_md": {
        "system": "You are a continuity editor.",
        "user": (
            "Compare the **current section** to these **prior facts**. "
            "Flag contradictions (dates, ages, locations, actions), and note ambiguous elements.\n\n"
            "### Prior facts\n```json\n{prior_entities}\n```\n\n### Current section\n{body}\n"
        ),
    },
    "continuity_json": {
        "system": "Return only a JSON array of inconsistency strings.",
        "user": "Prior facts: {prior_entities}\n\nCurrent section:\n{body}\n",
    },

    # --- Copyediting ---
    "copyedit_md": {
        "system": (
            "You are a meticulous UK-English line editor. Improve clarity, flow, grammar and punctuation "
            "while preserving the author’s voice. Return ONLY the revised text as Markdown — no commentary."
        ),
        "user": (
            "# Heading\n{heading}\n\n"
            "# Text\n{body}\n\n"
            "## Edit Rules\n"
            "- Make tangible improvements; do not return the input verbatim.\n"
            "- Preserve meaning, tone, and style; avoid Americanising spelling.\n"
            "- Keep author’s Markdown unless clearly erroneous.\n"
            "- Output must be only the edited text."
        ),
    },
    "copyedit_suggestions": {
        "system": "You are a UK-English line editor. Provide actionable copyediting suggestions.",
        "user": (
            "# Heading\n{heading}\n\n"
            "# Text\n{body}\n\n"
            "## Output format\n"
            "### Summary\n- 2–4 bullets on overall issues.\n\n"
            "### Line edits\nGive 5–12 concrete suggestions as bullets with **Issue / Before / After / Why**.\n\n"
            "Rules: Keep UK spelling and voice; don’t invent plot facts."
        ),
    },
    "copyedit_json": {
        "system": "Return a small JSON envelope containing the edited markdown and issue list.",
        "user": (
            "{\n"
            "  \"heading\": \"{heading}\",\n"
            "  \"original\": \"\"\"{body}\"\"\",\n"
            "  \"instructions\": [\"Rewrite fully in same voice\", \"Keep content; do not summarise\", "
            "\"Fix grammar, punctuation (UK), awkward phrasing\", \"Output one JSON object only\"]\n"
            "}\n\n"
            "Return exactly this shape:\n"
            "{\n"
            "  \"heading\": string,\n"
            "  \"edited_markdown\": string,\n"
            "  \"issues\": [{\"type\": \"grammar|spelling|clarity|consistency|style\", \"before\": string, \"after\": string, \"note\": string}],\n"
            "  \"stats\": {\"chars_before\": number, \"chars_after\": number, \"num_issues\": number}\n"
            "}"
        ),
    },

    # --- Overviews ---
    "overview": {
        "system": "You write editorial overviews.",
        "user": (
            "Write a 2,000–2,500-word editorial overview of the manuscript below. "
            "Cover plot, character arcs, themes, style, structure, pacing, genre conventions, continuity issues.\n\n"
            "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
        ),
    },
    "detailed-summary": {
        "system": "You write long summaries.",
        "user": (
            "Write a 5,000–10,000-word chapter-by-chapter summary of the manuscript below.\n\n"
            "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
        ),
    },
    "sensitivity-analysis": {
        "system": "You assess sensitivity and period-appropriate tone.",
        "user": (
            "Read the sexually explicit sections of the manuscript below. "
            "Assess whether they are justified by period and context. Suggest rewordings for borderline passages.\n\n"
            "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
        ),
    },
    "plot-points": {
        "system": "You analyse plot foreshadowing and reader inference.",
        "user": (
            "Analyse whether the draft telegraphs that Mrs Nairn and the glamorous actress are Sheena in disguise; "
            "and that Marlboro cigarettes mark Katrin. Evaluate the Paddington Green scene for whether Ulrike is truly Ulrike. "
            "Consider whether readers can infer Grace’s developing attraction to Ulrike at Newnham. "
            "Identify plot points that fail or could be sharpened.\n\n"
            "----- BEGIN SECTION -----\n{body}\n----- END SECTION -----"
        ),
    },
}

def render_prompt_text(mode: str, heading: str, body: str, **extras) -> str:
    spec = PROMPTS.get(mode)
    if spec is None:
        raise KeyError(f"Unknown mode: {mode!r}")
    if isinstance(spec, str):
        return spec.format(heading=heading, body=body, **extras)
    if not isinstance(spec, dict):
        raise TypeError(f"PROMPTS[{mode!r}] must be str or dict")
    sys_txt = (spec.get("system") or "").format(heading=heading, body=body, **extras).strip()
    usr_txt = (spec.get("user") or "").format(heading=heading, body=body, **extras).strip()
    parts = []
    if sys_txt:
        parts.append(f"[SYSTEM]\n{sys_txt}")
    if usr_txt:
        parts.append(usr_txt)
    return "\n\n".join(parts)

# Back-compat alias
build_prompt = render_prompt_text