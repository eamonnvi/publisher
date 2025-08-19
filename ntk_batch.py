# ntk_batch.py
import os, time
from pathlib import Path
from openai import OpenAI

def get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key: raise SystemExit("[error] OPENAI_API_KEY not set")
    return OpenAI(api_key=key)

def submit_batch(client, ndjson_path: Path, endpoint: str):
    file_obj = client.files.create(file=ndjson_path.open("rb"), purpose="batch")
    b = client.batches.create(input_file_id=file_obj.id, endpoint=endpoint, completion_window="24h")
    return b.id

def batch_status(client, batch_id: str) -> dict:
    return client.batches.retrieve(batch_id).model_dump()

def poll_batch(client, batch_id: str, timeout_s: int, every_s: int, verbose=True) -> dict:
    start = time.time()
    while True:
        info = batch_status(client, batch_id)
        st = info.get("status","unknown")
        if verbose:
            elapsed = int(time.time()-start)
            print(f"[poll] {batch_id} status={st} elapsed={elapsed}s")
        if st in {"completed","failed","cancelled","expired"}: return info
        if timeout_s and time.time()-start >= timeout_s: return info
        time.sleep(max(1,every_s))

def download_file(client, file_id: str) -> bytes:
    return client.files.content(file_id).read()
