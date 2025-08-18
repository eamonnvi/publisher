#!/usr/bin/env python3
import sys, json

ALLOWED_URLS = {"/v1/responses", "/v1/chat/completions"}
FORBIDDEN_RESP_KEYS = {
    "max_tokens", "messages", "temperature", "top_p",
    "presence_penalty", "frequency_penalty"
}

def warn(i, msg):
    print(f"[warn] line {i}: {msg}", file=sys.stderr)

def main():
    if not (2 <= len(sys.argv) <= 3):
        print("Usage: strip_metadata_ndjson.py in.jsonl [out.jsonl]", file=sys.stderr)
        sys.exit(2)

    inp = sys.argv[1]
    outp = sys.argv[2] if len(sys.argv) == 3 else inp.replace(".jsonl", ".sanitized.jsonl")

    ok = 0; skipped = 0; errors = 0
    with open(inp, "r", encoding="utf-8") as f, open(outp, "w", encoding="utf-8") as g:
        for i, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                warn(i, f"invalid JSON ({e}); skipping")
                errors += 1
                continue

            # required top-level fields (don’t die if missing—just warn)
            for k in ("custom_id","method","url","body"):
                if k not in obj:
                    warn(i, f"missing top-level '{k}'")

            # strip metadata (the original purpose)
            obj.pop("metadata", None)
            if isinstance(obj.get("body"), dict):
                obj["body"].pop("metadata", None)

            # minimal schema checks
            method = obj.get("method")
            if method != "POST":
                warn(i, f"method should be 'POST', got {method!r}")

            url = obj.get("url")
            if url not in ALLOWED_URLS:
                warn(i, f"unsupported url {url!r}; allowed={sorted(ALLOWED_URLS)}")

            body = obj.get("body")
            if not isinstance(body, dict):
                warn(i, "body must be an object; skipping")
                skipped += 1
                continue

            model = (body.get("model") or "").strip()
            lower_model = model.lower()

            if url == "/v1/responses":
                # enforce required fields and forbid chat-ish keys
                if "input" not in body or not isinstance(body["input"], str):
                    warn(i, "responses.body.input must be a string")
                if "max_output_tokens" not in body or not isinstance(body["max_output_tokens"], int):
                    # auto-upgrade if max_tokens present
                    if "max_tokens" in body and isinstance(body["max_tokens"], int):
                        body["max_output_tokens"] = body.pop("max_tokens")
                        warn(i, "auto: converted max_tokens -> max_output_tokens")
                    else:
                        warn(i, "responses.body.max_output_tokens missing or invalid")
                bad = FORBIDDEN_RESP_KEYS.intersection(body.keys())
                if bad:
                    for k in sorted(bad):
                        body.pop(k, None)
                    warn(i, f"auto: removed forbidden keys for /v1/responses: {sorted(bad)}")

            elif url == "/v1/chat/completions":
                # gentle guard: if it’s a gpt-5* model, prefer /v1/responses
                if lower_model.startswith("gpt-5"):
                    # auto-migrate to responses if messages is a single user turn
                    msgs = body.get("messages")
                    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict) and "content" in msgs[0]:
                        obj["url"] = "/v1/responses"
                        obj["body"] = {
                            "model": model,
                            "input": msgs[0]["content"],
                            "max_output_tokens": (
                                body.get("max_completion_tokens") or body.get("max_tokens") or 512
                            ),
                        }
                        warn(i, "auto: migrated gpt-5 chat -> /v1/responses")
                    else:
                        warn(i, "gpt-5 with chat/completions; consider migrating to /v1/responses")

            # write sanitized object
            g.write(json.dumps(obj, ensure_ascii=False) + "\n")
            ok += 1

    print(f"[ok] wrote sanitized file: {outp}")
    print(f"[summary] ok={ok} skipped={skipped} errors={errors}", file=sys.stderr)

if __name__ == "__main__":
    main()