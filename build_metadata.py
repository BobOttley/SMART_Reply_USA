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

# ✅ U.S.-specific relevance keywords
KEYWORDS = [
    "admissions", "apply", "enroll", "enrollment", "tuition", "financial aid",
    "scholarships", "curriculum", "academics", "athletics", "arts", "boarding",
    "student life", "co-curricular", "clubs", "college counseling", "international students",
    "day school", "residential life", "character", "leadership", "diversity", "inclusion"
]

# ✅ Skip URLs with these patterns
BLACKLISTED_FRAGMENTS = [
    "/privacy", "/cookies", "/terms", "/sitemap", "/404", "/login", "/admin", "/wp-json"
]

# ✅ Load chunks
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

def is_relevant(chunk):
    text = chunk.get("content", "").lower()
    url = chunk.get("url", "").lower()
    if len(text) < 100:
        return False
    if any(bad in url for bad in BLACKLISTED_FRAGMENTS):
        return False
    return any(keyword in text for keyword in KEYWORDS)

# ✅ Apply filtering
filtered_chunks = [c for c in all_chunks if is_relevant(c)]
texts = [c["content"].strip()[:8000] for c in filtered_chunks]

print(f"📦 Loaded {len(all_chunks)} chunks → {len(filtered_chunks)} relevant chunks retained.")

# ✅ Batch embed
def get_embeddings(texts, model):
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = openai.embeddings.create(input=batch, model=model)
        all_embeddings.extend([e.embedding for e in response.data])
    return all_embeddings

# ✅ Generate embeddings
print(f"🔍 Generating embeddings for {len(texts)} filtered chunks...")
embeddings = get_embeddings(texts, EMBED_MODEL)

# ✅ Save metadata with "messages" key
print("💾 Saving metadata.pkl...")
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump({
        "messages": filtered_chunks,
        "embeddings": np.array(embeddings)
    }, f)

print("✅ Done! metadata.pkl is ready.")
