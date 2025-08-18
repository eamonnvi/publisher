#!/usr/bin/env python3
# validate_ndjson.py
#
# Local validator for OpenAI Batch NDJSON request files.
# - Verifies one-JSON-object-per-line (no arrays, no trailing commas)
# - Checks required fields: custom_id, method, url, body
# - Checks schema for /v1/responses (gpt-5 family) and /v1/chat/completions
# - Flags forbidden keys for gpt-5 responses (temperature/top_p/etc)
# - Warns on large payloads & duplicates
#
# Usage:
#   python3 validate_ndjson.py path/to/batch_input.jsonl
#   python3 validate_ndjson.py path/to/batch_input.jsonl --strict-gpt5
#   python3 validate_ndjson.py path/to/batch_input.jsonl --limit 50

import argparse, json, os, sys

FORBIDDEN_RESP_KEYS = {
    "max_tokens", "messages", "temperature", "top_p",
    "presence_penalty", "frequency_penalty"
}

def bad(line_no, msg):
    print(f"[bad] line {line_no}: {msg}", file=sys.stderr)

def warn(line_no, msg):
    print(f"[warn] line {line_no}: {msg}", file=sys.stderr)

def is_str(x): return isinstance(x, str)
def is_int(x): return isinstance(x, int)
def is_list(x): return isinstance(x, list)
def approx_chars(s): return len(s.encode("utf-8"))

def main():
    ap = argparse.ArgumentParser(description="Validate OpenAI Batch NDJSON file")
    ap.add_argument("path", help="NDJSON file")
    ap.add_argument("--limit", type=int, default=200, help="Max errors to display (default 200)")
    ap.add_argument("--strict-gpt5", action="store_true",
                    help="Forbid /v1/chat/completions for any gpt-5* model; enforce /v1/responses+input string.")
    ap.add_argument("--max-body-bytes-warn", type=int, default=250_000,
                    help="Warn when a single request body exceeds this many bytes (default 250k).")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"[error] file not found: {args.path}", file=sys.stderr)
        sys.exit(2)

    errors = 0
    ok = 0
    seen_ids = set()
    stats = {
        "lines": 0,
        "resp_count": 0,
        "chat_count": 0,
        "models": {},
    }

    with open(args.path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            stats["lines"] += 1
            line = line.strip()
            if not line:
                warn(i, "blank line")
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                bad(i, f"invalid JSON: {e}")
                errors += 1
                if errors >= args.limit: break
                continue

            # Top-level required fields
            for key in ("custom_id", "method", "url", "body"):
                if key not in obj:
                    bad(i, f"missing top-level '{key}'")
                    errors += 1
            if errors >= args.limit: break

            cid = obj.get("custom_id")
            if not is_str(cid) or not cid:
                bad(i, "custom_id must be a non-empty string")
                errors += 1
            elif cid in seen_ids:
                bad(i, f"duplicate custom_id: {cid!r}")
                errors += 1
            else:
                seen_ids.add(cid)

            method = obj.get("method")
            if method != "POST":
                bad(i, f"method must be 'POST', got {method!r}")
                errors += 1

            # (This block must NOT be nested under the method check)
            url = obj.get("url")
            body = obj.get("body", {})
            if not isinstance(body, dict):
                bad(i, "body must be an object")
                errors += 1
                if errors >= args.limit: break
                continue

            # Disallow metadata at top-level or inside body
            if "metadata" in obj:
                bad(i, "top-level 'metadata' is not allowed in batch NDJSON")
                errors += 1
                if errors >= args.limit: break
            if "metadata" in body:
                bad(i, "body.metadata is not allowed")
                errors += 1
                if errors >= args.limit: break

            model = body.get("model")
            if not is_str(model) or not model:
                bad(i, "body.model must be a non-empty string")
                errors += 1
            else:
                stats["models"][model] = stats["models"].get(model, 0) + 1

            url = str(url or "")
            if url == "/v1/responses":
                stats["resp_count"] += 1
                # Required for responses: input (string), max_output_tokens (int>0)
                if "input" not in body:
                    bad(i, "responses.body missing 'input'")
                    errors += 1
                elif not is_str(body["input"]):
                    bad(i, "responses.body.input must be a string")
                    errors += 1

                if "max_output_tokens" not in body:
                    bad(i, "responses.body missing 'max_output_tokens'")
                    errors += 1
                elif not is_int(body["max_output_tokens"]) or body["max_output_tokens"] <= 0:
                    bad(i, "responses.body.max_output_tokens must be a positive integer")
                    errors += 1

                # Forbidden keys for responses
                forb = FORBIDDEN_RESP_KEYS.intersection(body.keys())
                if forb:
                    bad(i, f"responses.body contains forbidden keys: {sorted(forb)}")
                    errors += 1

                # Optional strict rule
                if args.strict_gpt5 and model and model.lower().startswith("gpt-5") is False:
                    warn(i, f"strict-gpt5: /v1/responses used with non-gpt-5 model {model!r}")

                # Size warn
                size = 0
                if "input" in body and is_str(body["input"]):
                    size += approx_chars(body["input"])
                if size > args.max_body_bytes_warn:
                    warn(i, f"large body (~{size} bytes) may be rejected for size")

            elif url == "/v1/chat/completions":
                stats["chat_count"] += 1
                # Required: messages (list), max_tokens or max_completion_tokens (int>0 depending on model)
                if "messages" not in body or not is_list(body["messages"]):
                    bad(i, "chat.body.messages must be a list")
                    errors += 1

                tok_field = "max_tokens"
                if "max_completion_tokens" in body:
                    tok_field = "max_completion_tokens"
                if tok_field not in body:
                    bad(i, "chat.body missing 'max_tokens' or 'max_completion_tokens'")
                    errors += 1
                elif not is_int(body[tok_field]) or body[tok_field] <= 0:
                    bad(i, f"chat.body.{tok_field} must be a positive integer")
                    errors += 1

                if args.strict_gpt5 and model and model.lower().startswith("gpt-5"):
                    bad(i, "strict-gpt5: gpt-5* with /v1/chat/completions not allowed")
                    errors += 1

                # Heuristic: discourage temperature/top_p for batch chat with gpt-5-family
                if model and model.lower().startswith("gpt-5"):
                    extra = {"temperature", "top_p"}.intersection(body.keys())
                    if extra:
                        warn(i, f"gpt-5 chat includes {sorted(extra)}; prefer defaults or /v1/responses")

                # Size warn (sum text lengths in messages)
                try:
                    size = 0
                    for m in body.get("messages", []):
                        if isinstance(m, dict) and is_str(m.get("content")):
                            size += approx_chars(m["content"])
                    if size > args.max_body_bytes_warn:
                        warn(i, f"large messages payload (~{size} bytes) may be rejected for size")
                except Exception:
                    pass

            else:
                bad(i, f"unsupported url: {url!r}. Allowed: /v1/responses or /v1/chat/completions")
                errors += 1

            if errors >= args.limit:
                break
            else:
                ok += 1

    # Summary
    print(f"[summary] lines={stats['lines']} ok={ok} errors={errors}")
    print(f"[summary] endpoints: responses={stats['resp_count']} chat={stats['chat_count']}")
    if stats["models"]:
        models = ", ".join([f"{k}:{v}" for k,v in sorted(stats["models"].items())])
        print(f"[summary] models: {models}")

    sys.exit(1 if errors > 0 else 0)

if __name__ == "__main__":
    main()