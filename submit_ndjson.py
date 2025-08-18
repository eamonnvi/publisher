# submit_ndjson.py
import os, sys
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ndjson = sys.argv[1]
file_obj = client.files.create(file=open(ndjson, "rb"), purpose="batch")
# Pick endpoint to match your NDJSON ("url" values inside):
#   "/v1/responses" (gpt-5 family) or "/v1/chat/completions" (others)
b = client.batches.create(
    input_file_id=file_obj.id,
    endpoint="/v1/responses",
    completion_window="24h",
)
print("batch id:", b.id)
