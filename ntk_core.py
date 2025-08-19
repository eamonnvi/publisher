# ntk_core.py
import re, json
from pathlib import Path

PRICE = {
    "gpt-5":        {"in": 0.010, "out": 0.030},
    "gpt-5-mini":   {"in": 0.003, "out": 0.006},
    "gpt-4.1":      {"in": 0.005, "out": 0.015},
    "gpt-4.1-mini": {"in": 0.001, "out": 0.003},
}
CURRENCY = "USD"

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def safe_slug(s: str) -> str:
    s2 = re.sub(r"[^a-zA-Z0-9._-]+", "_", s.strip())
    return s2[:80]

def pretty_heading_from_cid(h: str) -> str:
    return re.sub(r"^sec_\d{4}_", "", h).replace("_", " ").strip()

def compile_heading_regex(rx: str|None, min_level: int):
    if rx: return re.compile(rx, re.IGNORECASE)
    # default H3+ headings
    return re.compile(rf"^\s*#{{{min_level},6}}\s+(.+?)\s*$", re.MULTILINE)

def iter_sections(text: str, heading_rx) -> list[tuple[str,str]]:
    pos = 0
    sections = []
    matches = list(heading_rx.finditer(text))
    if not matches:
        return [("Full Manuscript", text.strip())]
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((title, body))
    return sections

# stub – replace with your actual tokenizer
def num_tokens(model: str, text: str) -> int:
    # rough char/4 heuristic
    return max(1, int(len(text) / 4))

def build_prompt(mode: str, heading: str, body: str, fmt: str, prior: str|None=None) -> str:
    label = "manuscript" if heading.lower() in {"full manuscript","manuscript"} else "section"
    return f"[{mode}:{fmt}] {label}={heading}\n\n{body}"

def estimate_cost_numeric(model: str, tin: int, tout: int, fx: float|None):
    r = PRICE.get(model)
    if not r: return (0.0, 0.0, 0.0, CURRENCY)
    cin = (tin/1000.0)*r["in"]; cout = (tout/1000.0)*r["out"]; tot = cin+cout
    cur = "GBP" if (fx and fx>0) else CURRENCY
    if fx and fx>0: cin*=fx; cout*=fx; tot*=fx
    return (cin, cout, tot, cur)

def make_batch_records(sections, args):
    # choose endpoint/body
    model_lc = (args.model or "").lower()
    if model_lc.startswith("gpt-5"):
        endpoint = "/v1/responses"
        def body(prompt): return {"model": args.model, "input": prompt, "max_output_tokens": args.max_tokens_out}
    else:
        endpoint = "/v1/chat/completions"
        tok_field = "max_completion_tokens" if model_lc.startswith("gpt-5") else "max_tokens"
        def body(prompt): return {"model": args.model, "messages":[{"role":"user","content":prompt}], tok_field: args.max_tokens_out}

    records = []
    id_to_heading = {}
    for idx, (heading, text) in enumerate(sections, 1):
        if not text.strip() and not args.whole: continue
        prompt = build_prompt(args.mode, heading, text, args.format)
        cid = f"sec_{idx:04d}_{safe_slug(heading)[:40]}"
        rec = {"custom_id": cid, "method": "POST", "url": endpoint, "body": body(prompt)}
        records.append(rec)
        id_to_heading[cid] = heading
    return records, id_to_heading, endpoint

def parse_batch_output_ndjson(ndjson_bytes: bytes, id_to_heading: dict[str,str]):
    triples = []
    for raw in ndjson_bytes.splitlines():
        if not raw.strip(): continue
        try: obj = json.loads(raw)
        except Exception: continue
        cid = obj.get("custom_id") or obj.get("id")
        heading = id_to_heading.get(cid, cid or "UNKNOWN")
        text = ""
        model = ""
        if "response" in obj:
            r = obj["response"] or {}
            model = (r.get("model") or "") if isinstance(r, dict) else ""
            # responses shape
            if isinstance(r, dict):
                t = (r.get("output_text") or r.get("text") or "")
                if isinstance(t, str) and t.strip():
                    text = t.strip()
                else:
                    # look into output blocks
                    out = r.get("output") or []
                    if out and isinstance(out, list):
                        for block in out:
                            if isinstance(block, dict):
                                content = block.get("content")
                                if isinstance(content, list):
                                    for c in content:
                                        if isinstance(c, dict) and "text" in c and c["text"]:
                                            text = c["text"].strip(); break
                            if text: break
                # chat fallback
                if not text:
                    choices = r.get("choices") or []
                    if choices and isinstance(choices, list):
                        msg = (choices[0].get("message") or {})
                        text = (msg.get("content") or "").strip()
        triples.append((heading, text, model))
    return triples
