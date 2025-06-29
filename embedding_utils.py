import os
import openai
import numpy as np
import pickle
from PyPDF2 import PdfReader

openai.api_key = os.getenv("OPENAI_API_KEY")

def embed_text(text: str) -> list:
    """Returns the embedding for a given piece of text using OpenAI Embeddings API."""
    try:
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return np.zeros(1536).tolist()  # fallback

def process_pdf_and_append_to_kb(file_path, metadata_path="embeddings/metadata.pkl"):
    """Reads PDF, embeds chunks, appends to existing metadata."""
    try:
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)

        chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 40]

        # Load existing metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            
            # Ensure embeddings list is a standard Python list
            if isinstance(metadata.get("embeddings"), np.ndarray):
                metadata["embeddings"] = metadata["embeddings"].tolist()
        else:
            metadata = {"messages": [], "embeddings": []}

        for chunk in chunks:
            embedding = embed_text(chunk)
            metadata["messages"].append(chunk)
            metadata["embeddings"].append(embedding)

        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        return len(chunks)
    except Exception as e:
        print(f"❌ PDF processing failed: {e}")
        return 0
