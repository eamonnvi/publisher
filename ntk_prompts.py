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
    # you can still keep old string-style templates if you want:
    # "old_mode": "Please summarise:\n\n{body}"
}


# ntk_prompts.py
from typing import Dict, Any

# Ensure PROMPTS is defined above this function:
# PROMPTS: Dict[str, Any] = {
#     "concise": "... {heading}\n\n{body}\n",
#     "concise_para": {...},
#     ...
# }

def build_prompt(mode: str, heading: str, body: str) -> str:
    """
    Backwards-compatible prompt builder.
    - If PROMPTS[mode] is a string, .format() with {heading},{body}.
    - If it's a dict with 'system'/'user', merge them into one text block.
    - Raises a clear error if mode is unknown.
    """
    if mode not in PROMPTS:
        valid = ", ".join(sorted(PROMPTS.keys()))
        raise KeyError(f"Unknown mode '{mode}'. Valid modes: {valid}")

    spec = PROMPTS[mode]

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