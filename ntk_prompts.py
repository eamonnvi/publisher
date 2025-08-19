# ntk_prompts.py
# -----------------------------------------------------------------------------
# Central prompt definitions for Arbuthnot Books' Novel Toolkit.
# Each mode has a small builder function. The dispatcher `build_prompt(...)`
# is the only symbol your core code needs to import.
#
# Design goals:
# - Single responsibility: only prompt text lives here.
# - Keep per-mode builders tiny and explicit.
# - Support md/json variants where it’s useful.
# - Make the model’s required output shape crystal clear in the instruction.
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Optional

# --- tiny helpers ------------------------------------------------------------

def _label_for(heading: str) -> tuple[str, str]:
    """Return ('manuscript'|'section', printable_heading)."""
    h = (heading or "").strip()
    norm = h.lower()
    label = "manuscript" if norm in {"full manuscript", "manuscript"} else "section"
    return label, h or "Untitled"

def _header_md(title: str) -> str:
    return f"# {title}\n\n"

def _fence_json(example: str) -> str:
    return "```json\n" + example.strip() + "\n```"

# --- concise -----------------------------------------------------------------

def prompt_concise(heading: str, body: str, fmt: str) -> str:
    label, H = _label_for(heading)
    if fmt == "md":
        return f"""{_header_md("Concise Summary")}You are a careful literary summariser. Write a **tight, factual** one-paragraph summary of the {label} below (no spoilers beyond what’s present). Avoid evaluative language. End with a single **Suggested title** line.

**{label.capitalize()}: {H}**

{body}

Output format:
- A paragraph of 4–7 sentences.
- Final line: "*Suggested title:* <short title>" (no quotes).
"""
    # JSON variant returns fields we already post-process nicely if needed.
    return f"""You are a careful literary summariser. Summarise the {label} below.

{label.upper()}: {H}

{body}

Return ONLY valid JSON with:
{_fence_json("""
{
  "summary": "<4-7 sentence paragraph>",
  "suggested_title": "<short title or empty string>"
}
""")}
"""

# --- discursive --------------------------------------------------------------

def prompt_discursive(heading: str, body: str, fmt: str) -> str:
    label, H = _label_for(heading)
    return f"""{_header_md("Discursive Notes")}Provide a brief discursive commentary on the {label} (tone, pacing, scene goals, POV, imagery, continuity risks). Keep it concrete; quote short fragments where needed.

**{label.capitalize()}: {H}**

{body}

Output: 5–9 bullet points, each a single sentence.
"""

# --- critique (md/json) ------------------------------------------------------

def prompt_critique(heading: str, body: str, fmt: str) -> str:
    label, H = _label_for(heading)
    if fmt == "md":
        return f"""{_header_md("Critique")}Give a constructive critique of the {label}. Focus on clarity, stakes, character intent, continuity, voice. Avoid line edits—comment at scene/paragraph level.

**{label.capitalize()}: {H}**

{body}

Output sections:
- **What works**
- **What to improve**
- **Concrete next steps** (3–5 bullets)
"""
    return f"""Critique the following {label}. Focus on clarity, stakes, intent, continuity, voice.

{label.upper()}: {H}

{body}

Return ONLY JSON:
{_fence_json("""
{
  "what_works": ["..."],
  "what_to_improve": ["..."],
  "next_steps": ["..."]
}
""")}
"""

# --- entities (JSON only) ----------------------------------------------------

def prompt_entities_json(heading: str, body: str) -> str:
    label, H = _label_for(heading)
    return f"""Extract named entities in the {label} into a structured list. Include people, orgs, places, objects of plot significance.

{label.upper()}: {H}

{body}

Return ONLY JSON matching:
{_fence_json("""
{
  "entities": [
    {
      "name": "string",
      "type": "person|organization|place|object|other",
      "aliases": ["optional", "..."],
      "description": "1-2 sentences",
      "first_appearance_line": "optional string"
    }
  ]
}
""")}
"""

# --- continuity (md/json) ----------------------------------------------------

def prompt_continuity(heading: str, body: str, fmt: str, prior_entities: Optional[str]) -> str:
    label, H = _label_for(heading)
    prior = prior_entities or "[]"
    if fmt == "md":
        return f"""{_header_md("Continuity Check")}Given the **known entities** and this {label}, list possible continuity issues (names, ages, timeline, geography, props), and note any **new facts** that should be added to the bible.

**Known entities (json)**:
{_fence_json(prior)}

**{label.capitalize()}: {H}**
{body}

Output sections:
- **Possible issues**
- **New facts to record**
"""
    return f"""Using prior entities and the {label}, produce a JSON continuity check.

PRIOR_ENTITIES (JSON):
{prior}

{label.upper()}: {H}
{body}

Return ONLY JSON:
{_fence_json("""
{
  "possible_issues": ["..."],
  "new_facts": [
    {"entity": "string", "fact": "string"}
  ]
}
""")}
"""

# --- overview (md) -----------------------------------------------------------

def prompt_overview(heading: str, body: str, fmt: str) -> str:
    label, H = _label_for(heading)
    return f"""{_header_md("Overview")}Give a brief overview: **where we are**, **who’s present**, **scene goal**, **complication**, **outcome hook**.

**{label.capitalize()}: {H}**

{body}

Output: 5 short bullets in the above order.
"""

# --- inline edit (md) --------------------------------------------------------

def prompt_inline_edit(heading: str, body: str, fmt: str) -> str:
    label, H = _label_for(heading)
    return f"""{_header_md("Inline Edits")}Lightly polish the {label} for clarity and rhythm. Keep author’s voice. Do not change plot facts. Keep British spelling if present.

**{label.capitalize()}: {H}**

Original:
---
{body}
---

Output: A revised version of the same passage (no commentary).
"""

# --- copyedit (md/json) ------------------------------------------------------

def prompt_copyedit(heading: str, body: str, fmt: str) -> str:
    label, H = _label_for(heading)
    if fmt == "md":
        return f"""{_header_md("Copyedit")}Copyedit the {label}: grammar, punctuation, consistent British spelling, typographic quotes, minimal rephrasing for flow; **no plot changes**. Preserve paragraph breaks.

**{label.capitalize()}: {H}**

Original:
---
{body}
---

Output: The corrected passage only (no preface/notes).
"""
    return f"""Copyedit this {label} and return structured JSON of edits plus a fully corrected text.

{label.upper()}: {H}

{body}

Return ONLY JSON:
{_fence_json("""
{
  "edits": [
    {
      "loc": "string (brief locator or snippet)",
      "issue": "grammar|punctuation|spelling|consistency|other",
      "before": "string",
      "after": "string",
      "note": "optional"
    }
  ],
  "corrected_text": "string"
}
""")}
"""

# --- improvement suggestions (md) --------------------------------------------

def prompt_improvement_suggestions(heading: str, body: str, fmt: str) -> str:
    label, H = _label_for(heading)
    return f"""{_header_md("Improvement Suggestions")}Suggest pragmatic, low-risk improvements (beats, transitions, clarification, sensory detail). No rewrites—just actionable notes.

**{label.capitalize()}: {H}**

{body}

Output: 6–10 bullets, each one concrete and testable.
"""

# --- detailed summary (md) ---------------------------------------------------

def prompt_detailed_summary(heading: str, body: str, fmt: str) -> str:
    label, H = _label_for(heading)
    return f"""{_header_md("Detailed Summary")}Write a detailed, beat-by-beat summary of the {label}. Keep events in order; include internal intentions if stated.

**{label.capitalize()}: {H}**

{body}

Output: 8–15 bullets, each a single beat or turn.
"""

# --- sensitivity analysis (md) -----------------------------------------------

def prompt_sensitivity_analysis(heading: str, body: str, fmt: str) -> str:
    label, H = _label_for(heading)
    return f"""{_header_md("Sensitivity Analysis")}Identify any potentially sensitive portrayals (identity, health, trauma, profession), why they may be sensitive, and low-impact mitigations while preserving intent.

**{label.capitalize()}: {H}**

{body}

Output sections:
- **Potential sensitivities**
- **Context**
- **Mitigations**
"""

# --- plot points (md/json) ---------------------------------------------------

def prompt_plot_points(heading: str, body: str, fmt: str) -> str:
    label, H = _label_for(heading)
    if fmt == "md":
        return f"""{_header_md("Plot Points")}Extract key plot points as short clauses: inciting move, reversals, reveals, decisions, stakes shifts.

**{label.capitalize()}: {H}**

{body}

Output: bullet list, 6–12 items.
"""
    return f"""Extract plot points from the {label}.

{label.upper()}: {H}

{body}

Return ONLY JSON:
{_fence_json("""
{
  "plot_points": [
    {"type": "inciting|reversal|reveal|decision|stakes|other", "text": "short clause"}
  ]
}
""")}
"""

# --- dispatcher ---------------------------------------------------------------

def build_prompt(mode: str, heading: str, body: str, fmt: str, prior: Optional[str] = None) -> str:
    """
    Public entrypoint called by ntk_core. `fmt` is either 'md' or 'json'.
    `prior` may carry JSON text (e.g., prior entities) for some modes.
    """
    m = (mode or "").strip().lower()
    f = (fmt or "md").strip().lower()

    if m == "concise":
        return prompt_concise(heading, body, f)
    if m == "discursive":
        return prompt_discursive(heading, body, f)
    if m == "critique":
        return prompt_critique(heading, body, f)
    if m == "entities":
        # entities is JSON-only in this scaffold
        return prompt_entities_json(heading, body)
    if m == "continuity":
        return prompt_continuity(heading, body, f, prior_entities=prior)
    if m == "overview":
        return prompt_overview(heading, body, f)
    if m == "inline-edit":
        return prompt_inline_edit(heading, body, f)
    if m == "copyedit":
        return prompt_copyedit(heading, body, f)
    if m == "improvement_suggestions":
        return prompt_improvement_suggestions(heading, body, f)
    if m == "detailed-summary":
        return prompt_detailed_summary(heading, body, f)
    if m == "sensitivity-analysis":
        return prompt_sensitivity_analysis(heading, body, f)
    if m == "plot-points":
        return prompt_plot_points(heading, body, f)

    # Fallback: treat as concise
    return prompt_concise(heading, body, f)
