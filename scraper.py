# ───────────── SCRAPER: Deep crawl + PDF support ─────────────

import os, json, requests, hashlib
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from PyPDF2 import PdfReader

START_URL = "https://www.smes.org/"
DOMAIN = "smes.org"
MAX_DEPTH = 5  # 🔁 Controls how deep to crawl
FORCED_URLS = [
    "https://www.cheltenhamcollege.org/admissions/international-pupils/",
    "https://www.cheltenhamcollege.org/admissions/scholarships/",
    "https://www.cheltenhamcollege.org/college/year-9/",
    "https://www.cheltenhamcollege.org/college/pastoral-care/",
    "https://www.cheltenhamcollege.org/college/sixth-form/",
    "https://www.cheltenhamcollege.org/boarding/"
]

OUTPUT_DIR = "scraped"
PDF_DIR = os.path.join(OUTPUT_DIR, "pdfs")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "content.jsonl")
URL_MAPPING_FILE = os.path.join(OUTPUT_DIR, "url_mapping.py")

visited, results, url_mapping = set(), [], {}

def is_internal(url):
    parsed = urlparse(url)
    return parsed.netloc.endswith(DOMAIN)

def clean_text(text):
    return ' '.join(text.split())

def extract_text_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    main = soup.find("main")
    body = soup.find("body")
    text = main.get_text(" ") if main else (body.get_text(" ") if body else soup.get_text(" "))
    return clean_text(text)

def extract_text_from_pdf(path):
    try:
        reader = PdfReader(path)
        return clean_text(" ".join(p.extract_text() or "" for p in reader.pages))
    except Exception as e:
        print(f"❌ PDF extraction failed: {path} — {e}")
        return None

def download_and_parse_pdf(pdf_url):
    try:
        hash_id = hashlib.sha1(pdf_url.encode()).hexdigest()[:10]
        filename = f"pdf_{hash_id}.pdf"
        local_path = os.path.join(PDF_DIR, filename)

        if not os.path.exists(local_path):
            print(f"📥 Downloading: {pdf_url}")
            r = requests.get(pdf_url, timeout=15)
            if r.status_code == 200:
                with open(local_path, "wb") as f: f.write(r.content)
            else:
                print(f"❌ PDF error {r.status_code}: {pdf_url}")
                return

        text = extract_text_from_pdf(local_path)
        if text:
            results.append({"url": pdf_url, "type": "pdf", "content": text})
        else:
            results.append({"url": pdf_url, "type": "pdf", "content": "", "error": "Unreadable PDF"})
    except Exception as e:
        print(f"❌ PDF error: {pdf_url} — {e}")

def crawl(url, depth=0):
    if depth > MAX_DEPTH: return
    if not is_internal(url) or url in visited: return
    visited.add(url)
    print(f"🔍 Crawling ({depth}): {url}")

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh) Safari/537.36"}
        res = requests.get(url, headers=headers, timeout=10)

        if res.status_code != 200 or "text/html" not in res.headers.get("Content-Type", ""):
            print(f"⚠️ Skipped non-HTML or failed: {url}")
            return

        html = res.text
        text = extract_text_from_html(html)
        if text:
            results.append({"url": url, "type": "html", "content": text})

        soup = BeautifulSoup(html, "html.parser")
        for link in soup.find_all("a", href=True):
            href = link["href"].split("#")[0]
            full_url = urljoin(url, href)
            anchor = link.get_text(strip=True)
            if is_internal(full_url):
                if full_url.lower().endswith(".pdf"):
                    download_and_parse_pdf(full_url)
                else:
                    if anchor and 3 < len(anchor) < 80:
                        url_mapping[anchor] = full_url
                    crawl(full_url, depth + 1)

    except Exception as e:
        print(f"❌ Failed to crawl: {url} — {e}")

def save_results():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in results:
            json.dump(item, f)
            f.write("\n")
    print(f"\n✅ Saved {len(results)} items to {OUTPUT_FILE}")

def save_url_mapping():
    with open(URL_MAPPING_FILE, "w", encoding="utf-8") as f:
        f.write("URL_MAPPING = {\n")
        for key, value in sorted(url_mapping.items()):
            f.write(f'    "{key}": "{value}",\n')
        f.write("}\n")
    print(f"🔗 Saved anchor → URL mapping")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)
    crawl(START_URL)
    for extra_url in FORCED_URLS:
        crawl(extra_url, depth=0)
    save_results()
    save_url_mapping()
