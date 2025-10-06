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
            "You are a meticulous UK-English line editor. Correct only clear errors of spelling, "
           "missing or inappropriate punctuation (e.g. omitted question marks), and misplaced words. "
           "Do not comment on optional style choices such as Oxford commas, commas before conjunctions, "
            "or commas after introductory phrases. Preserve the author’s voice and formatting. "
            "Return ONLY the corrected text as Markdown — no commentary."
        ),
        "user": (
           "# Heading\n{heading}\n\n"
           "# Text\n{body}\n\n"
            "## Guidelines\n"
            "- Fix genuine errors only: spelling, punctuation mistakes, or misplaced words.\n"
            "- Do not suggest stylistic changes to comma usage unless an error is unambiguous.\n"
           "- Preserve meaning, tone, and style; avoid Americanising spelling.\n"
            "- Keep author’s Markdown unless clearly erroneous.\n"
            "- Return ONLY the corrected text."
        ),
    },

    "copyedit_changes": {
       "system": (
           "You are a meticulous British-English copy editor.\n"
            "Analyse the text and list only the specific corrections needed.\n"
            "Do not rewrite or reproduce the passage; just list the changes.\n"
        ),
        "user": (
           "SOURCE HEADING: {heading}\n\n"
            "SOURCE TEXT:\n"
            "<<<\n"
            "{body}\n"
           ">>>\n\n"
            "OUTPUT FORMAT (MANDATORY):\n"
            "### Change notes\n"
            "For each significant issue, use this format:\n"
            "- **Issue:** brief description\n"
            "  - **Before:** original phrase\n"
            "  - **After:** corrected phrase\n"
            "  - **Why:** reason for change\n\n"
            "RULES:\n"
            "- British spelling and punctuation conventions.\n"
            "- Ignore stylistic preferences unless objectively incorrect.\n"
            "- Limit to clear errors: spelling, punctuation, grammar, word choice, or factual slips.\n"
            "- Maximum 15 bullets per section.\n"
            "- No introductory or closing remarks — output only the list above.\n"
        )
    },


    "copyedit_suggestions": {
        "system": (
            "You are a UK-English line editor. Identify typos and missing punctuation in the supplied text. Use UK-English spelling. "
        ),
        "user": (
            "# Heading\n{heading}\n\n"
            "# Text\n{body}\n\n"
            "## Output format\n"
            "- Respond in Markdown.\n"
            "- Line edits section with a bulleted list (5–12 items). Each bullet should use bold labels in the following structure:\n"
            "  - **Issue:** Briefly state the specific editing concern\n"
            "  - **Before:** Original text excerpt\n"
            "  - **After:** Revised text excerpt\n"
            "  - **Why:** Explanation for the suggested change\n\n"
    "Guidelines: Retain all existing plot facts; avoid introducing new content.\n\n"
    "After generating suggestions, quickly review your edits to ensure clarity, adherence to UK style, and accurate application of your recommendations. If any issues are detected, make a minimal correction before finalising your output."
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

    # --- Marketing ---
    "kdp_description": {
        "system": (
            "You are an experienced publishing copywriter.\n"
            "Write a compelling Amazon KDP product description for the given draft text.\n"
            "- DO NOT include plot spoilers.\n"
            "- Highlight atmosphere, themes, style, and emotional appeal.\n"
            "- Use vivid but truthful language.\n"
            "- End with a hook that encourages the reader to want more."
        ),
        "user": (
            "Book/section title: {heading}\n\n"
            "Draft:\n{body}\n\n"
            "Tone controls:\n"
            "- English variant: {english_variant}\n"
            "- Style: {style}\n"
            "- Emphasis: {emphasis}\n\n"
            "Please weave these tone instructions into the description naturally, "
            "while keeping it concise (max 200 words)."
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
    """
    Render PROMPTS[mode] into a single text block.

    Uses targeted placeholder substitution (not str.format) so that prompts
    with literal JSON braces don't break. Only {heading}, {body}, and any
    explicitly provided extras are replaced.
    """
    spec = PROMPTS.get(mode)
    if spec is None:
        raise KeyError(f"Unknown mode: {mode!r}")

    # Defaults so tone placeholders always have values unless overridden
    defaults = {
        "english_variant": "British English",
        "style": "understated",
        "emphasis": "character-driven",
    }
    merged = {**defaults, **(extras or {})}

    # Allow legacy string prompts
    if isinstance(spec, str):
        s = spec.replace("{heading}", heading).replace("{body}", body)
        for k, v in merged.items():
            s = s.replace("{" + k + "}", str(v))
        return s

    if not isinstance(spec, dict):
        raise TypeError(f"PROMPTS[{mode!r}] must be str or dict with system/user")

    def _subst(s: str) -> str:
        s = (s or "")
        s = s.replace("{heading}", heading).replace("{body}", body)
        for k, v in merged.items():
            s = s.replace("{" + k + "}", str(v))
        return s.strip()

    sys_txt = _subst(spec.get("system"))
    usr_txt = _subst(spec.get("user"))

    parts = []
    if sys_txt:
        parts.append(f"[SYSTEM]\n{sys_txt}")
    if usr_txt:
        parts.append(usr_txt)
    return "\n\n".join(parts)


# Keep compatibility alias
build_prompt = render_prompt_text