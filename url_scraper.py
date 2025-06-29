from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from urllib.parse import urljoin, urlparse
import time
import re

START_URL = "https://www.smes.org"
DOMAIN = "smes.org"
MAX_DEPTH = 4
MAX_URLS = 250  # Set a safe crawl cap

visited = set()
collected = {}

def clean_anchor(url):
    path = urlparse(url).path.rstrip("/")
    if not path or path == "":
        return None
    slug = path.split("/")[-1]
    slug = re.sub(r'\.html?$', '', slug)
    return slug.replace("-", " ").title()

def crawl(driver, url, depth=0):
    if depth > MAX_DEPTH or len(visited) >= MAX_URLS or url in visited or DOMAIN not in url:
        return
    visited.add(url)
    try:
        driver.get(url)
        time.sleep(1.5)

        anchor = clean_anchor(url)
        if anchor and anchor not in collected:
            collected[anchor] = url
            print(f"✅ Added: {anchor} — {url}")

        links = driver.find_elements("tag name", "a")
        hrefs = [link.get_attribute("href") for link in links if link.get_attribute("href")]

        for href in hrefs:
            if DOMAIN in href and href not in visited:
                crawl(driver, href, depth + 1)

    except Exception as e:
        print(f"❌ Error crawling {url}: {e}")

# Setup headless browser
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
driver = webdriver.Chrome(options=options)

# Start
print("🔍 Starting crawl...")
crawl(driver, START_URL)
driver.quit()

# Output
print("\n📦 Writing to url_mapping.py")
with open("url_mapping.py", "w") as f:
    f.write("URL_MAPPING = {\n")
    for k, v in sorted(collected.items()):
        f.write(f'    "{k}": "{v}",\n')
    f.write("}\n")
