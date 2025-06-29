import os
import json
import pickle
import numpy as np
from dotenv import load_dotenv
import openai

# ✅ Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Config
INPUT_FILE = "embeddings/clean_chunks.json"
OUTPUT_FILE = "embeddings/metadata.pkl"
EMBED_MODEL = "text-embedding-3-small"

# ✅ Load clean chunks
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [c["content"] for c in chunks]

# ✅ Batch embed
def get_embeddings(texts, model):
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        # Strip whitespace, ensure strings, truncate max length
        clean_batch = [str(t).strip()[:8000] for t in batch if str(t).strip()]
        if not clean_batch:
            continue

        response = openai.embeddings.create(input=clean_batch, model=model)
        all_embeddings.extend([e.embedding for e in response.data])
    return all_embeddings

# ✅ Generate embeddings
print(f"🔍 Embedding {len(texts)} chunks...")
embeddings = get_embeddings(texts, EMBED_MODEL)

# ✅ Save metadata with "messages" key
print("💾 Saving metadata.pkl...")
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump({
        "messages": chunks,                     # renamed key for compatibility
        "embeddings": np.array(embeddings)
    }, f)

print("✅ Done! metadata.pkl is ready.")
