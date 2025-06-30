import pickle
import numpy as np
import os

# Load metadata.pkl
with open("./embeddings/metadata.pkl", "rb") as f:
    kb = pickle.load(f)
    doc_embeddings = np.array(kb["embeddings"])
    metadata = kb["messages"]

# Filter out invalid entries
original_len = len(metadata)
new_metadata = []
new_embeddings = []
for m, e in zip(metadata, doc_embeddings):
    if isinstance(m, dict) and m.get("source") and m.get("content") and m.get("url") and m.get("name"):
        new_metadata.append(m)
        new_embeddings.append(e)
    else:
        print(f"🗑️ Removing invalid metadata entry: {m[:100]}...")

# Save cleaned metadata
with open("./embeddings/metadata.pkl", "wb") as f:
    pickle.dump({"embeddings": np.array(new_embeddings).tolist(), "messages": new_metadata}, f)

print(f"🧼 Cleaned metadata: {original_len} → {len(new_metadata)} entries")