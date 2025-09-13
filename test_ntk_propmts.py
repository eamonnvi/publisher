import re
from ntk_prompts import render_prompt_text

def test_kdp_defaults():
    txt = render_prompt_text(
        "kdp_description",
        heading="The Parallax View",
        body="A Cold War novel."
    )
    # It should include defaults
    assert "British English" in txt
    assert "understated" in txt
    assert "character-driven" in txt

def test_kdp_overrides():
    txt = render_prompt_text(
        "kdp_description",
        heading="The Parallax View",
        body="A Cold War novel.",
        english_variant="US English",
        style="dramatic",
        emphasis="espionage"
    )
    # The overrides should appear
    assert "US English" in txt
    assert "dramatic" in txt
    assert "espionage" in txt

def test_legacy_prompt_still_works():
    txt = render_prompt_text(
        "concise",
        heading="Chapter 1",
        body="Steve meets Grace in the library."
    )
    # It should interpolate heading/body and not crash
    assert "Chapter 1" in txt or "Steve" in txt
    # Ensure [SYSTEM] section is present
    assert txt.startswith("[SYSTEM]")

