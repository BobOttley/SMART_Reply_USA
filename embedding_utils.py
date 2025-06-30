import os
import openai
import numpy as np
import pickle
from PyPDF2 import PdfReader

openai.api_key = os.getenv("OPENAI_API_KEY")

def embed_text(text: str) -> list:
    try:
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return np.zeros(1536).tolist()

def process_pdf_and_append_to_kb(file_path, metadata_path="embeddings/metadata.pkl"):
    try:
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            raw_text = [page.extract_text() or "" for page in reader.pages]

        text = "\n".join(raw_text).strip()
        if not text or all(t.strip() == "" for t in raw_text):
            print(f"⚠️ PDF appears blank or image-based: {file_path}")
            return -1

        chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 40]
        if not chunks:
            print(f"⚠️ No usable chunks found in: {file_path}")
            return -1

        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            if isinstance(metadata.get("embeddings"), np.ndarray):
                metadata["embeddings"] = metadata["embeddings"].tolist()
        else:
            metadata = {"messages": [], "embeddings": []}

        filename = os.path.basename(file_path)
        for chunk in chunks:
            embedding = embed_text(chunk)
            metadata_entry = {
                "source": f"uploaded_pdfs/{filename}",
                "content": chunk,
                "url": f"/uploaded_pdfs/{filename}",
                "name": filename
            }
            metadata["messages"].append(metadata_entry)
            metadata["embeddings"].append(embedding)

        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        print(f"✅ Processed {len(chunks)} chunks from {file_path}")
        return len(chunks)

    except Exception as e:
        print(f"❌ PDF processing failed: {e}")
        return -1