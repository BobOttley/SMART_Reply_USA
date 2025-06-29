import json
import os

INPUT_FILE = "scraped/content.jsonl"
OUTPUT_FILE = "embeddings/clean_chunks.json"

def split_text(text, max_length=500):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, current = [], ""

    for para in paragraphs:
        if len(current) + len(para) <= max_length:
            current += " " + para if current else para
        else:
            chunks.append(current.strip())
            current = para
    if current:
        chunks.append(current.strip())
    return chunks

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Missing input file: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_entries = [json.loads(line) for line in f if line.strip()]

    clean_chunks = []
    for entry in raw_entries:
        text = entry.get("content", "").strip()
        url = entry.get("url", "").strip()

        if not text or not url:
            continue

        for chunk in split_text(text):
            clean_chunks.append({
                "content": chunk,
                "url": url
            })

    os.makedirs("embeddings", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(clean_chunks, f, indent=2)

    print(f"✅ Saved {len(clean_chunks)} clean chunks to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
