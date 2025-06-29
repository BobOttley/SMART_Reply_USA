import json
import os

INPUT_FILE = "scraped/content.jsonl"
OUTPUT_FILE = "embeddings/clean_chunks.json"

# ✅ U.S.-focused keyword relevance filter (expanded)
KEYWORDS = [
    # Core Admissions
    "admissions", "apply", "enroll", "enrollment", "tuition", "financial aid", "scholarships",

    # Academics & Curriculum
    "curriculum", "academics", "stem", "learning support", "project-based learning",
    "advanced placement", "early childhood",

    # Student Life
    "athletics", "arts", "boarding", "student life", "co-curricular", "clubs",
    "residential program", "after-school", "college counseling",

    # Parent Experience
    "parent engagement", "parent partnership", "parent communication", "family engagement",

    # Pastoral / Values
    "wellbeing", "character education", "spiritual life", "chapel", "service learning",
    "leadership", "diversity", "inclusion",

    # Campus / Environment
    "campus life", "learning environment", "library", "labs", "innovation center", "day school"
]

# ✅ Phrasal fallback for intelligent relevance
IMPORTANT_PHRASES = [
    "our students benefit", "our approach to", "we believe", "we support", "we offer",
    "opportunities for", "personal growth", "our community", "developing character",
    "families are welcomed", "parent partnership", "we encourage"
]

# ✅ Skip URLs with these patterns
BLACKLISTED_URL_FRAGMENTS = [
    "/privacy", "/cookies", "/terms", "/404", "/sitemap", "/wp-json", "/login", "/admin"
]

def is_relevant(text, url):
    text_lower = text.lower()
    url_lower = url.lower()

    if len(text_lower) < 100:
        return False
    if any(bad in url_lower for bad in BLACKLISTED_URL_FRAGMENTS):
        return False
    if any(keyword in text_lower for keyword in KEYWORDS):
        return True
    if any(phrase in text_lower for phrase in IMPORTANT_PHRASES):
        return True
    return False

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
    skipped = 0

    for entry in raw_entries:
        text = entry.get("content", "").strip()
        url = entry.get("url", "").strip()

        if not text or not url:
            continue

        for chunk in split_text(text):
            if is_relevant(chunk, url):
                clean_chunks.append({"content": chunk, "url": url})
            else:
                skipped += 1

    os.makedirs("embeddings", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(clean_chunks, f, indent=2)

    print(f"✅ Saved {len(clean_chunks)} clean chunks to {OUTPUT_FILE}")
    print(f"🚫 Skipped {skipped} irrelevant or short chunks")

if __name__ == "__main__":
    main()
