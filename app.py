# ──────────────────────────────
# 🧠  SMART REPLY BACKEND — FORMATTING FIXED & ROBUST
# ──────────────────────────────

import os, json, pickle, re
import numpy as np
import threading
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from embedding_utils import process_pdf_and_append_to_kb
from flask import request
from werkzeug.utils import secure_filename
from embedding_utils import process_pdf_and_append_to_kb
from url_mapping import URL_MAPPING

print("🟢 SMART REPLY SMES USA app.py STARTED")


load_dotenv()
app = Flask(__name__, template_folder="templates")
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("🚀 PEN Reply Flask server starting…")


# ====== About the School ======
top_context = """
St. Margaret's Episcopal School (SMES) is a selective private, pre-K – 12th grade college preparatory school located in San Juan Capistrano, California. It is a member of the National Association of Episcopal Schools (NAES), the National Association for the Education of Young Children, the National Association of Independent Schools (NAIS), the Western States Association of Schools and Colleges (WASC) and the California Association of Independent Schools (CAIS).St. Margaret's offers courses in core subjects, as well as other subjects through electives. Theater, art, music and other subjects are available to be taken in addition to the main subject. There are 26 Advanced Placement courses offered in history, math, science, art, language, theatre, music and other subjects. In addition to Advanced Placement (AP) courses, St. Margeret's offers honors courses throughout their various subjects. The variety of courses gives students the opportunity to excel to the best of their ability.[4] As of 2025, the school is ranked as the #1 Christian high school in Greater Los Angeles.St. Margaret's sports teams are called the Tartans, and they compete as members of the Orange Coast League in the CIF Southern Section.

In 2006, the Tartan football team finished 14-0 and won the CIF-SS Northeast Division championship.[citation needed] and in 2008 the Girls' Varsity Tennis Team were the Academy League and CIF Champions, defeating league rival Sage Hill in the finals to finish off a 23-0 winning streak. The 2010-2011 Athletics year was one of the most successful, with 4 CIF Championships in boys' cross country, girls' volleyball, girls' tennis, and girls' soccer teams. Additionally, the boys cross country team won the CIF Division 5 State Championship in 2010, 2011, and 2018.[6] The varsity girls' soccer program won 4 CIF championships in 2006, 2011, 2012, and 2014. The St. Margaret's football team won the Southern California small-school division championship in the 2014-2015 season.

St. Margaret's boasts highly competitive men's and women's lacrosse programs, the school's only Division 1 CIF Southern Section sports. The men's varsity lacrosse team won the Division 1 CIF Southern Section title in 2005, 2013, 2015, 2019, and 2023 while the women's varsity lacrosse team won the Division 1 CIF Southern Section title in 2008, 2018, and 2019. From 2005-2014, St. Margaret's was a Trinity League men's lacrosse affiliate, winning the league title every year from 2007-2013. In 2007, St. Margaret's completed construction on the DeYoung Family Math and Science Center. In 2011, St. Margaret's constructed a performing arts center, which includes the Hurlbut Theater and houses the music, dance, and theatre departments. A new middle school was constructed in 2015. The campus covers a span of 22 acres.

St. Margaret's has 2 gyms: the Campaigne Center, used by lower schoolers, and the Pasternack Field House, used by middle and upper schoolers. The Pasternack Field House was completed in 2007. There is a middle school field and a main football field, used by members of the high school. In 2022, plans were released for a new student commons, which would include a renovated weight room and cafeteria.
"""


# ====== STRATEGY ======
STRATEGY_FILE = "strategy_advice.json"

def load_manual_strategy():
    if os.path.exists(STRATEGY_FILE):
        try:
            with open(STRATEGY_FILE, "r") as f:
                data = json.load(f)
            return data.get("strategy_text", "")
        except Exception:
            return ""
    return ""


# ====== USAGE LOGGING (for Usage Insights) ======
USAGE_LOG_FILE = "usage_log.jsonl"
USAGE_LOG_LOCK = threading.Lock()

def log_usage(event):
    import json
    from datetime import datetime
    event['timestamp'] = datetime.now().isoformat()
    with USAGE_LOG_LOCK:
        with open(USAGE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

TOPIC_LABELS = {
    "admissions": "Admissions",
    "enrollment": "Enrollment",
    "enrolment": "Enrollment",  # catch UK spelling just in case
    "tuition": "Tuition & Fees",
    "fees": "Tuition & Fees",
    "financial aid": "Financial Aid",
    "scholarship": "Scholarship",
    "after-school": "After-School Programs",
    "extended care": "After-School Programs",
    "child care": "After-School Programs",
    "bus": "Transportation",
    "transportation": "Transportation",
    "sports": "Athletics/Sports",
    "athletics": "Athletics/Sports",
    "clubs": "Clubs & Activities",
    "ap course": "AP Courses",
    "advanced placement": "AP Courses",
    "curriculum": "Curriculum",
    "calendar": "School Calendar",
    "lunch": "Lunch Program",
    "cafeteria": "Lunch Program",
    "visit": "Campus Visit",
    "tour": "Campus Visit",
    "open house": "Open House",
    "uniform": "Uniform/School Dress Code",
    "dress code": "Uniform/School Dress Code",
    "application": "Application Process",
    "deadline": "Application Deadline",
    "waitlist": "Waitlist",
    "diversity": "Diversity & Inclusion",
    "special needs": "Student Support",
    "support": "Student Support",
    "international": "International Student Program",
}

def extract_topics(msg):
    msg = msg.lower()
    topics = []
    for key, label in TOPIC_LABELS.items():
        if key in msg:
            topics.append(label)
    return list(set(topics))  # avoid duplicates

def extract_links_from_reply(reply):
    import re
    return re.findall(r'\[([^\]]+)\]\((https?://[^\)]+)\)', reply or "")

def redact_past_dates(text):
    # This removes sentences containing dates that have already passed
    # Supports formats: "October 1, 2024", "15 February 2025", "2025-01-15"
    today = datetime.now().date()
    # Pattern: Month Day, Year | Day Month Year | YYYY-MM-DD
    date_patterns = [
        r'([A-Z][a-z]+ \d{1,2}, \d{4})',   # October 1, 2024
        r'(\d{1,2} [A-Z][a-z]+ \d{4})',    # 15 February 2025
        r'(\d{4}-\d{2}-\d{2})',            # 2025-01-15
    ]
    sents = re.split(r'(?<=[.!?])\s+', text)
    filtered = []
    for sent in sents:
        keep = True
        for pat in date_patterns:
            for match in re.findall(pat, sent):
                try:
                    # Try to parse the found date
                    for fmt in ["%B %d, %Y", "%d %B %Y", "%Y-%m-%d"]:
                        try:
                            d = datetime.strptime(match, fmt).date()
                            if d < today:
                                keep = False
                        except:
                            continue
                except:
                    continue
        if keep:
            filtered.append(sent)
    return ' '.join(filtered)

# ───────────── HELPERS ─────────────

def generate_strategy_advice(insights):
    lines = []
    common = insights.get("recent_topics", {})
    template_rate = insights.get("template_reply_rate", 0)
    avg_sentiment = insights.get("avg_sentiment", 0)
    top_pdfs = insights.get("pdf_usage", {})

    # Most-asked topics
    main_topics = sorted(common.items(), key=lambda x: x[1], reverse=True)[:3] if common else []
    if main_topics:
        for topic, count in main_topics:
            if "fee" in topic.lower():
                lines.append("• Highlight your fees and financial support information on the admissions homepage and make it easy to find in all packs.")
            if "deadline" in topic.lower():
                lines.append("• Create a clear, dated 'Admissions Deadlines' section at the top of your website and in parent comms.")
            if "student support" in topic.lower() or "pastoral" in topic.lower():
                lines.append("• Publish a summary of your student support and pastoral care, and review it regularly to ensure details are current.")
            if topic.lower() not in ("fee", "deadline", "student support", "pastoral"):
                lines.append(f"• Add or update FAQs for '{topic}'—it’s now a frequent parent question.")
    else:
        lines.append("• No dominant parent queries this period. Review previous trends for recurring topics.")

    # Template use (with specific advice using main_topics)
    if template_rate < 0.07 and main_topics:
        top_topics = ', '.join([topic for topic, _ in main_topics])
        lines.append(f"• Use Smart Reply to save effective replies as standard responses for your most frequent questions ({top_topics}). Regularly review and update these templates as parent enquiries are changing, so your team always has access to the most current and relevant replies.")
    elif template_rate < 0.25:
        lines.append("• Encourage your team to use and improve standard replies in Smart Reply. This will ensure parents receive consistent, accurate answers—even as common queries change.")
    elif template_rate > 0.25:
        lines.append("• Template usage is improving—continue saving your best replies as standards and keep updating them as parent enquiries evolve.")

    # Sentiment
    if avg_sentiment < 7.5:
        lines.append("• Parent sentiment is lower than usual. Review recent replies for tone, clarity, and responsiveness—adjust as needed.")
    elif avg_sentiment > 8.5:
        lines.append("• Parent sentiment is high. Share positive examples with your team and continue reinforcing what’s working well.")

    # Document usage
    if top_pdfs:
        for pdf, cnt in list(top_pdfs.items())[:2]:
            lines.append(f"• Make '{pdf}' directly accessible from your admissions page and include it in replies where relevant.")

    # General wrap-up
    lines.append("• Revisit this strategy report monthly and use the recommendations above to keep your admissions process agile and parent-focused.")

    return "<br>".join(lines)


def get_top_context(question, doc_embeddings, metadata, SIMILARITY_THRESHOLD, RESPONSE_LIMIT):
    # Get vector for question
    q_vec = get_embedding(question)
    # Calculate similarities
    sims = [(cosine_similarity(q_vec, vec), meta) for vec, meta in zip(doc_embeddings, metadata)]
    top = sorted([m for m in sims if m[0] >= SIMILARITY_THRESHOLD], key=lambda x: x[0], reverse=True)[:RESPONSE_LIMIT]
    if not top:
        return jsonify({
            "reply": "<p>Thank you for your enquiry. A member of our admissions team will contact you shortly.</p>",
            "sentiment_score": score, "strategy_explanation": strat
        })

    context_blocks = []
    for _, m in top:
        print("DEBUG: m is type", type(m), "value:", m)  # Optional debug; remove after confirming it works
        if isinstance(m, dict):
            content = m.get('content', '')
            url = m.get('url', '')
            if url:
                context_blocks.append(f"{content}\n[Info source]({url})")
            else:
                context_blocks.append(content)
        else:
            context_blocks.append(str(m))

    top_context = "\n---\n".join(context_blocks)

def insert_links(text, url_map):
    def safe_replace(match):
        word = match.group(0)
        for anchor, url in url_map.items():
            if word.lower() == anchor.lower():
                # Force valid Markdown (quoted + parenthesis-safe)
                safe_url = url.replace(')', '%29').replace('(', '%28')  # escape problematic chars
                return f"[{word}]({safe_url})"
        return word

    sorted_anchors = sorted(url_map.keys(), key=len, reverse=True)
    pattern = r'\b(' + '|'.join(re.escape(a) for a in sorted_anchors) + r')\b'
    return re.sub(pattern, safe_replace, text, flags=re.IGNORECASE)

def remove_personal_info(text: str) -> str:
    PII_PATTERNS = [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        r"\b(?:\+44\s?7\d{3}|\(?07\d{3}\)?)\s?\d{3}\s?\d{3}\b",
        r"\b(?:\+44\s?1\d{3}|\(?01\d{3}\)?|\(?02\d{3}\)?)\s?\d{3}\s?\d{3,4}\b",
        r"\+?\d[\d\s\-().]{7,}\d",
        r"\b[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}\b",
    ]
    for pat in PII_PATTERNS:
        text = re.sub(pat, "[redacted]", text, flags=re.I)
    text = re.sub(r"\b(my name is|i am|i'm|i'm called)\s+(mr\.?|mrs\.?|ms\.?|miss)?\s*[A-Z][a-z]+\b", "my name is [redacted]", text, flags=re.I)
    text = re.sub(r"\bDear\s+(Mr\.?|Mrs\.?|Ms\.?|Miss)?\s*[A-Z][a-z]+\b", "Dear [redacted]", text, flags=re.I)
    text = re.sub(r"\b(?:regards|thanks|thank you|sincerely|best wishes|kind regards)[,]?\s+[A-Z][a-z]+\b", "[redacted]", text, flags=re.I)
    return text

def load_url_mapping():
    try:
        namespace = {}
        with open("url_mapping.py", "r") as f:
            exec(f.read(), namespace)
        return namespace.get("URL_MAPPING", {})
    except Exception as e:
        print(f"❌ Failed to load URL_MAPPING dynamically: {e}")
        return {}

def parse_url_box(url_text):
    """Parse semicolon-separated URL mappings into a dictionary."""
    url_map = {}
    if not url_text:
        return url_map
    
    # Split by semicolon and handle potential whitespace
    pairs = [p.strip() for p in url_text.split(';') if p.strip()]
    
    for pair in pairs:
        if '=' in pair:
            anchor, url = pair.split('=', 1)
            anchor = anchor.strip()
            url = url.strip()
            if anchor and url:  # Only add if both parts exist
                url_map[anchor] = url
    
    return url_map


    # Replace plain anchor mentions (case-insensitive, word-boundary)
    sorted_anchors = sorted(url_map.keys(), key=lambda a: -len(a))
    for anchor in sorted_anchors:
        pattern = r'\b' + re.escape(anchor) + r'\b'
        def repl(m):
            safe_url = url_map[anchor].replace(')', '%29').replace('(', '%28')
            return f"[{m.group(0)}]({safe_url})"
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text

def embed_text(text: str) -> np.ndarray:
    text = text.replace("\n", " ")
    res = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return np.array(res.data[0].embedding)

def markdown_to_html(text: str) -> str:
    text = re.sub(r'\[([^\]]+)\]\((https?://[^\)]+)\)', lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', '<br>', text.strip())
    paragraphs = re.split(r'\n\s*\n', text)
    return '\n'.join(f'<p>{p.strip()}</p>' for p in paragraphs if p.strip())


def markdown_to_outlook_html(md: str) -> str:
    """
    Convert Markdown to Outlook-compatible HTML with proper formatting
    Handles signatures, paragraphs, and ensures proper HTML attribute quoting
    """
    if not md.strip():
        return ""
    
    # Step 1: Handle markdown links - convert to proper HTML with quoted attributes
    md = re.sub(r'\[([^\]]+)\]\((https?://[^\)]+)\)', 
                lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', md)
    
    # Step 2: Split content into paragraphs (separated by double line breaks)
    paragraphs = re.split(r'\n\s*\n', md.strip())
    
    processed_paragraphs = []
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        lines = paragraph.split('\n')
        
        # Step 3: Detect if this is a signature block
        # Signatures typically have:
        # - Multiple lines
        # - Contains titles, names, or contact info
        # - Short lines (names, titles, school name)
        is_signature = (
            len(lines) > 1 and 
            any(re.search(r'\b(Mrs?\.?|Ms\.?|Mr\.?|Director|Manager|School|College|University|Tel:|Email:|Phone:)', 
                         line, re.I) for line in lines)
        )
        
        if is_signature:
            # For signatures: each line should be separated by <br>
            # Remove empty lines and join with <br>
            clean_lines = [line.strip() for line in lines if line.strip()]
            processed_paragraphs.append('<br>'.join(clean_lines))
        else:
            # For regular content: replace single line breaks with <br>
            # This preserves intentional line breaks within paragraphs
            paragraph_html = paragraph.replace('\n', '<br>')
            processed_paragraphs.append(paragraph_html)
    
    # Step 4: Join all paragraphs with double <br> for proper spacing
    result = '<br><br>'.join(processed_paragraphs)
    
    # Step 5: Final cleanup — ensure all href attributes are safely quoted
    result = re.sub(r'href=([^\s">]+)', r'href="\1"', result)  # ensure all href= are quoted
    result = re.sub(r'<a\s+href="([^"]+)"\s*>([^<]+)</a>', r'<a href="\1">\2</a>', result)  # ensure well-formed links

    
    return result

def clean_gpt_email_output(md: str) -> str:
    md = md.strip()

    # Remove any markdown fences
    md = re.sub(r"^```(?:markdown)?", "", md, flags=re.I).strip()
    md = re.sub(r"```$", "", md, flags=re.I).strip()

    # Remove known heading formats (Subject or fake header lines)
    lines = md.splitlines()

    if lines:
        first_line = lines[0].strip()
        if (
            len(first_line) < 80 and  # short, heading-like
            not first_line.lower().startswith("dear") and
            not first_line.endswith(".") and
            not first_line.endswith(":")
        ):
            # Remove the first line if it looks like a heading
            lines = lines[1:]

    # Rejoin and clean markdown formatting
    md = "\n".join(lines).strip()
    md = re.sub(r"\*\*(.*?)\*\*", r"\1", md)  # remove bold
    md = re.sub(r"\*(.*?)\*", r"\1", md)      # remove italic

    return md.strip()


def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0  # Or you could return float('nan') or some fallback
    return np.dot(a, b) / (norm_a * norm_b)


def get_safe_url(label: str) -> str: return load_url_mapping().get(label, "")

def generate_admissions_summary(
    overall_period,
    recent_period,
    overall_sentiment,
    recent_sentiment,
    overall_count,
    recent_count,
    overall_topics,
    recent_topics,
    sentiment_threshold=0.3,
):
    overall_topics = overall_topics or {}
    recent_topics = recent_topics or {}

    def top_topics(topic_dict, n=2):
        return ", ".join([f"{k}" for k, v in sorted(topic_dict.items(), key=lambda x: x[1], reverse=True)[:n]]) or "a range of topics"

    sentiment_change = recent_sentiment - overall_sentiment
    if abs(sentiment_change) < sentiment_threshold:
        sentiment_trend = f"has remained steady at {recent_sentiment:.1f}"
    elif sentiment_change > 0:
        sentiment_trend = f"has improved slightly to {recent_sentiment:.1f}"
    else:
        sentiment_trend = f"has dipped slightly to {recent_sentiment:.1f}"

    # crude estimate: overall_count is for month, recent is for week
    vol_change = recent_count - (overall_count / (30/7) if overall_count else 0)
    if abs(vol_change) < 2:
        vol_trend = f"a steady number of enquiries ({recent_count})"
    elif vol_change > 0:
        vol_trend = f"a recent increase in enquiries ({recent_count} in {recent_period})"
    else:
        vol_trend = f"a slight decrease in recent enquiries ({recent_count} in {recent_period})"

    topics_now = top_topics(recent_topics)
    topics_before = top_topics(overall_topics)
    if topics_now != topics_before:
        topic_trend = f"with more recent interest in {topics_now}"
    else:
        topic_trend = f"with most interest focused on {topics_now}"

    summary = (
        f"Over the {overall_period}, the admissions team handled {overall_count} parent enquiries, "
        f"with sentiment scores averaging {overall_sentiment:.1f} out of 10. "
        f"Most enquiries related to {topics_before}.\n\n"
        f"In the {recent_period}, the school has seen {vol_trend}, "
        f"{topic_trend}. The average sentiment {sentiment_trend}.\n\n"
        f"Overall, the team is maintaining strong engagement. "
    )
    if sentiment_change < -sentiment_threshold:
        summary += (
            "However, the recent dip in sentiment may warrant a quick review of recent cases to identify any common concerns or issues.\n"
        )
    return summary.strip()

def replace_vague_anchors(md):
    # Replace [here](...) and similar with a placeholder or try to infer the correct anchor
    bad_anchors = ["here", "click here", "more info", "this page", "read more"]
    for anchor in bad_anchors:
        # Find all instances of [anchor](url)
        pattern = re.compile(rf'\[{anchor}\]\(([^)]+)\)', re.IGNORECASE)
        # Replace with a generic label or leave for manual review
        md = pattern.sub(r'[School website](\1)', md)
    return md



# ───────────── APP SETUP ─────────────

EMBED_MODEL = "text-embedding-3-small"
SIMILARITY_THRESHOLD = 0.30
RESPONSE_LIMIT = 3
STANDARD_MATCH_THRESHOLD = 0.85

def safe_load_metadata():
    try:
        with open("./embeddings/metadata.pkl", "rb") as f:
            kb = pickle.load(f)
            doc_embeddings = kb.get("embeddings")
            metadata = kb.get("messages")
        if isinstance(doc_embeddings, np.ndarray):
            doc_embeddings = doc_embeddings.tolist()
        if not isinstance(doc_embeddings, list):
            print("ERROR: doc_embeddings is not a list or array! Resetting to empty list.")
            doc_embeddings = []
        if not isinstance(metadata, list):
            print("ERROR: metadata is not a list! Resetting to empty list.")
            metadata = []
        doc_embeddings = np.array(doc_embeddings)
        return doc_embeddings, metadata
    except Exception as e:
        print(f"❌ Error loading metadata.pkl: {e}")
        return np.array([]), []

doc_embeddings, metadata = safe_load_metadata()

print(f"✅ Loaded {len(metadata)} website chunks from metadata.pkl")

standard_messages, standard_embeddings, standard_replies = [], [], []

def _load_standard_library():
    path = "standard_responses.json"
    if not os.path.exists(path): return
    try:
        with open(path, "r") as f: saved = json.load(f)
        for entry in saved:
            reply = entry["reply"]
            variants = entry.get("variants", [entry.get("message")])
            for msg in variants:
                redacted = remove_personal_info(msg)
                standard_messages.append(redacted)
                standard_embeddings.append(embed_text(redacted))
                standard_replies.append(reply)
        print(f"✅ Loaded {len(standard_messages)} template reply variants.")
    except Exception as e:
        print(f"❌ Failed loading templates: {e}")

_load_standard_library()

def check_standard_match(q_vec: np.ndarray) -> str:
    if not standard_embeddings: return ""
    sims = [cosine_similarity(q_vec, emb) for emb in standard_embeddings]
    best_idx = int(np.argmax(sims))
    if sims[best_idx] >= STANDARD_MATCH_THRESHOLD:
        print(f"🔁 Using template (similarity {sims[best_idx]:.2f})")
        return standard_replies[best_idx]
    return ""

# ───────────── PDF MANAGEMENT ENDPOINTS ─────────────

UPLOAD_FOLDER = "uploaded_pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload-pdfs", methods=["POST"])
def upload_pdfs():
    files = request.files.getlist("pdfs")
    saved, flagged = [], []

    for file in files:
        fname = secure_filename(file.filename)
        if not fname.lower().endswith(".pdf"):
            continue

        path = os.path.join(UPLOAD_FOLDER, fname)
        file.save(path)

        try:
            chunks = process_pdf_and_append_to_kb(path)
            if chunks == -1 or chunks == 0:
                flagged.append(f"{fname} (unusable PDF)")
            else:
                saved.append(f"{fname} ({chunks} chunks)")
        except Exception as e:
            flagged.append(f"{fname} (error: {str(e)})")

    if not saved and not flagged:
        return "❌ No valid PDFs uploaded."

    msg = ""
    if saved:
        msg += f"✅ Uploaded: {', '.join(saved)}"
    if flagged:
        msg += f"<br><br>⚠️ Issues detected: {', '.join(flagged)}"
    return msg

@app.route("/list-pdfs")
def list_pdfs():
    try:
        files = os.listdir(UPLOAD_FOLDER)
        files = [f for f in files if f.lower().endswith(".pdf")]
        files.sort()

        file_links = [
            {"name": f, "url": f"/uploaded_pdfs/{f}"}
            for f in files
        ]
        return jsonify(file_links)
    except Exception as e:
        return jsonify([]), 500

@app.route('/uploaded_pdfs/<path:filename>')
def serve_pdf(filename):
    return send_from_directory('uploaded_pdfs', filename)

@app.route("/rename-pdf", methods=["POST"])
def rename_pdf():
    try:
        data = request.get_json(force=True, silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Request data must be a JSON object."}), 400
        url = data.get("url")
        new_name = data.get("newName")
        if not url or not new_name:
            return jsonify({"error": "URL and new name are required"}), 400

        old_filename = os.path.basename(url)
        if not old_filename.lower().endswith(".pdf"):
            return jsonify({"error": "Invalid PDF URL"}), 400

        new_filename = secure_filename(new_name)
        if not new_filename.lower().endswith(".pdf"):
            new_filename += ".pdf"

        old_path = os.path.join(UPLOAD_FOLDER, old_filename)
        new_path = os.path.join(UPLOAD_FOLDER, new_filename)

        if not os.path.exists(old_path):
            return jsonify({"error": "PDF file not found"}), 404
        if os.path.exists(new_path):
            return jsonify({"error": "A file with the new name already exists"}), 400

        os.rename(old_path, new_path)

        # Update metadata.pkl to reflect the renamed file
        try:
            with open("./embeddings/metadata.pkl", "rb") as f:
                kb = pickle.load(f)
                doc_embeds = kb.get("embeddings")
                metas = kb.get("messages")
            if isinstance(doc_embeds, np.ndarray):
                doc_embeds = doc_embeds.tolist()
            if not isinstance(metas, list): metas = []
            for item in metas:
                if item.get("url") == f"/uploaded_pdfs/{old_filename}":
                    item["url"] = f"/uploaded_pdfs/{new_filename}"
                    item["name"] = new_filename
            with open("./embeddings/metadata.pkl", "wb") as f:
                pickle.dump({"embeddings": doc_embeds, "messages": metas}, f)
        except Exception as e:
            print(f"⚠️ Failed to update metadata.pkl for rename: {e}")

        return jsonify({"status": "renamed", "new_url": f"/uploaded_pdfs/{new_filename}"})
    except Exception as e:
        print(f"❌ RENAME ERROR: {e}")
        return jsonify({"error": f"Failed to rename PDF: {str(e)}"}), 500

@app.route("/delete-pdf", methods=["POST"])
def delete_pdf():
    print("DEBUG: /delete-pdf called")
    try:
        data = request.get_json(force=True, silent=True)
        print(f"DEBUG: type(data)={type(data)}, data={data}")
        if not isinstance(data, dict) or "filename" not in data:
            raise ValueError("Missing 'filename' in request or invalid payload.")

        fname = data["filename"]
        print(f"🧹 Attempting to delete: {fname}")

        file_path = os.path.join("uploaded_pdfs", fname)
        print(f"📂 File path resolved to: {file_path}")

        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🗑️ File removed from disk.")
        else:
            print(f"⚠️ File not found on disk. Continuing to clean metadata.")

        global metadata, doc_embeddings

        original_len = len(metadata)
        new_metadata, new_embeddings = [], []

        for m, e in zip(metadata, doc_embeddings):
            if isinstance(m, dict):
                source = m.get("source", "")
            else:
                print(f"WARNING: metadata entry is not a dict: {m}")
                source = ""
            if fname not in source:
                new_metadata.append(m)
                new_embeddings.append(e)

        print(f"🧼 Original chunks: {original_len} → After delete: {len(new_metadata)}")

        for m in new_metadata:
            if isinstance(m, dict) and fname in m.get("source", ""):
                print(f"⚠️ Still contains reference to deleted file: {m.get('source')}")

        metadata.clear()
        metadata.extend(new_metadata)
        doc_embeddings = np.array(new_embeddings)

        with open("embeddings/metadata.pkl", "wb") as f:
            pickle.dump({"embeddings": doc_embeddings.tolist(), "messages": metadata}, f)

        print("DEBUG: returning success response from /delete-pdf")
        return jsonify({
            "message": f"🗑️ Deleted file: {fname}, removed {original_len - len(new_metadata)} chunks."
        }), 200

    except Exception as e:
        print(f"❌ DELETE ERROR: {type(e)} {e}")
        return jsonify({"error": str(e)}), 500

# ───────────── REPLY ENDPOINTS ─────────────

def safe_get(meta, key, default=None):
    """Safely get key from dict, else return default."""
    if isinstance(meta, dict):
        return meta.get(key, default)
    return default

@app.route("/reply", methods=["POST"])
def generate_reply():
    try:
        body = request.get_json(force=True)
        if not isinstance(body, dict):
            print("❌ Data is not a dict:", body)
            return jsonify({"error": "Invalid request format"}), 400

        question_raw = (body.get("message") or "").strip()
        include_cta = body.get("include_cta", True)
        url_box_text = (body.get("url_box") or "").strip()
        instruction_raw = (body.get("instruction") or "").strip()

        question = remove_personal_info(question_raw)
        instruction = remove_personal_info(instruction_raw)
        url_map = parse_url_box(url_box_text)

        if not question:
            return jsonify({"error": "No message received."}), 400

        q_vec = embed_text(question)

        # Static match check
        matched = check_standard_match(q_vec)
        if matched:
            reply_md = matched
            reply_html = markdown_to_html(reply_md)
            reply_outlook = markdown_to_outlook_html(reply_md)
            return jsonify({
                "reply": reply_html,
                "reply_markdown": reply_md,
                "reply_outlook": reply_outlook,
                "sentiment_score": 10,
                "strategy_explanation": "Used approved template.",
                "url": "", "link_label": ""
            })

        # Sentiment + strategy detection
                # Sentiment + strategy detection (robust, always GPT-generated)
        sent_prompt_system = (
            "You are an expert school admissions assistant. "
            "Your job is to analyze parent messages for tone and suggest a strategy to improve parent engagement and sentiment. "
            "You must always reply in strict JSON with two fields, never blank, never extra explanation."
        )
        sent_prompt_user = f"""
Analyze the following parent inquiry and return ONLY this JSON (no extra text):

{{
  "score": [integer from 1 (very negative) to 10 (very positive)],
  "strategy": [no more than 30 words, suggest a clear admissions strategy to improve sentiment or response quality]
}}

Inquiry:
\"\"\"{question}\"\"\"
"""
        sent_json = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sent_prompt_system},
                {"role": "user", "content": sent_prompt_user}
            ],
            temperature=0.3
        ).choices[0].message.content.strip()

        import re
        def try_extract_json(txt):
            # Try to extract just the first {...} block from model output
            match = re.search(r'\{.*\}', txt, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    return None
            return None

        # First attempt
        try:
            sent = json.loads(sent_json)
            score = int(sent.get("score", 5))
            strat = sent.get("strategy", "")
        except:
            sent = try_extract_json(sent_json)
            if sent:
                score = int(sent.get("score", 5))
                strat = sent.get("strategy", "")
            else:
                # Retry GPT, explicitly warn not to add ANY text
                sent_json_retry = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": sent_prompt_system},
                        {"role": "user", "content":
                         "Return ONLY the JSON with 'score' and 'strategy' for this parent email. Do NOT add any comments or extra lines. Inquiry:\n" + question}
                    ],
                    temperature=0.3
                ).choices[0].message.content.strip()
                try:
                    sent = json.loads(sent_json_retry)
                    score = int(sent.get("score", 5))
                    strat = sent.get("strategy", "")
                except:
                    sent = try_extract_json(sent_json_retry)
                    if sent:
                        score = int(sent.get("score", 5))
                        strat = sent.get("strategy", "")
                    else:
                        # As an absolute last resort, use "N/A" (should never hit this)
                        score, strat = 5, "N/A"


        # Context search
        sims = [(cosine_similarity(q_vec, vec), meta) for vec, meta in zip(doc_embeddings, metadata)]
        top = sorted([m for m in sims if m[0] >= SIMILARITY_THRESHOLD], key=lambda x: x[0], reverse=True)[:RESPONSE_LIMIT]

        if not top:
            return jsonify({
                "reply": "<p>Thank you for your inquiry. A member of our admissions team will contact you shortly.</p>",
                "sentiment_score": score,
                "strategy_explanation": strat
            })

        context_blocks = []
        for _, m in top:
            if isinstance(m, dict):
                content = m.get('content', '')
                url = m.get('url', '')
                if url:
                    context_blocks.append(f"{content}\n[Info source]({url})")
                else:
                    context_blocks.append(content)
            else:
                context_blocks.append(str(m))

        top_context = "\n---\n".join(context_blocks)
        today_date = datetime.now().strftime('%d %B %Y')

        # Topic detection
        topic = "general"
        q_lower = question.lower()
        if "visit" in q_lower or "tour" in q_lower:
            topic = "visit"
        elif "fees" in q_lower or "cost" in q_lower:
            topic = "fees"
        elif "subjects" in q_lower or "curriculum" in q_lower:
            topic = "curriculum"

        # Final GPT prompt
        prompt = f"""
TODAY'S DATE IS {today_date}.

You are the Director of Admissions at St. Margaret's Episcopal School in California, USA.

You are responding on behalf of the admissions team at an independent Preschool–Grade 12 day school in the United States.

Please write a warm, professional, and helpful email reply to the parent below. If the inquiry came from a form, treat it the same as an email — respond directly and naturally. Only use the verified school information provided below.

Only use URLs provided in this dictionary:
{json.dumps(load_url_mapping(), indent=2)}

Follow these essential rules:
- If a date is in the past, do not mention it. Only provide future dates or refer to deadlines generically (e.g., "in early February") if unsure.
- Use American spelling and terminology (e.g. enrollment, program, counselor, fall, etc.)
- DO NOT fabricate or guess any information. If something is unknown, say so honestly.
- DO include relevant links using Markdown format: [Anchor Text](https://...). Embed links naturally in the body of the reply.
- DO use approved anchor phrases like "Open Events page", "Admissions page", or "registration form"
- NEVER use vague anchors like "click here", "more info", "register here", "visit page", etc.
- NEVER show raw URLs, list links at the bottom, or use markdown formatting like bold, italics, or bullet points
- Never mention dates, always be non specific about dates, especially dates that occur in the past, always look at the current date and check if the date is in the past. If unsure, direct the parent to the relevant web page instead

End your reply with:
Kind regards,  
Admissions Team  
St. Margaret's Episcopal School

Parent Email:
\"\"\"{question}\"\"\"

School Info:
\"\"\"{top_context}\"\"\"
"""

        reply_md = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        ).choices[0].message.content.strip()

        reply_md = clean_gpt_email_output(reply_md)

        # Add CTA if toggle is on
        if include_cta:
            if topic == "visit":
                reply_md += "\n\nIf you haven’t yet had a chance to visit us, we’d be delighted to welcome you to the school."
            elif topic == "fees":
                reply_md += "\n\nIf you’d like to discuss your child’s needs further, I’d be happy to arrange a time to speak."
            elif topic == "curriculum":
                reply_md += "\n\nWe’re always happy to share more about how we support your child to thrive academically and beyond."
            elif score >= 8:
                reply_md += "\n\nIf you haven’t yet had a chance to visit us, we’d be delighted to welcome you to the school."

        reply_md = insert_links(reply_md, url_map)
        reply_html = markdown_to_html(reply_md)
        reply_outlook = markdown_to_outlook_html(reply_md)

        # Extract first link
        def extract_links_from_html(html):
            matches = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html)
            return [(text.strip(), url.strip()) for url, text in matches]

        links = extract_links_from_html(reply_html)
        matched_url = links[0][1] if links else ""
        matched_source = links[0][0] if links else ""

        return jsonify({
            "reply": reply_html,
            "reply_markdown": reply_md,
            "reply_outlook": reply_outlook,
            "sentiment_score": score,
            "strategy_explanation": strat,
            "url": matched_url,
            "link_label": matched_source
        })

    except Exception as e:
        print(f"❌ REPLY ERROR: {e}")
        return jsonify({"error": "Internal server error."}), 500


# ─────────── SMART LINK MAPPINGS ENDPOINTS ───────────

@app.route("/get-url-mappings", methods=["GET"])
def get_url_mappings():
    today_date = datetime.now().strftime("%Y-%m-%d")

    try:
        mapping = load_url_mapping()
        return jsonify(mapping)
    except Exception as e:
        print(f"❌ Error reading URL_MAPPING: {e}")
        return jsonify({}), 500

@app.route("/save-url-mappings", methods=["POST"])
def save_url_mappings():
    today_date = datetime.now().strftime("%Y-%m-%d")

    try:
        new_data = request.get_json(force=True)
        if not isinstance(new_data, dict):
            return jsonify({"error": "Invalid data format"}), 400
        with open("url_mapping.py", "w") as f:
            f.write("# Auto-generated URL mapping file\n")
            f.write("URL_MAPPING = " + json.dumps(new_data, indent=2))
        return jsonify({"status": "saved"})
    except Exception as e:
        print(f"❌ Error saving URL_MAPPING: {e}")
        return jsonify({"error": "Failed to save mappings"}), 500

# ───────────── MAIN ─────────────

@app.route("/create-replies", methods=["POST"])
def create_replies():
    today_date = datetime.now().strftime("%Y-%m-%d")

    try:
        data = request.get_json()
        raw_thread = data.get("thread", "").strip()
        if not raw_thread:
            return jsonify({"error": "Missing thread"}), 400

        prompt = f"""
You're helping extract message+reply pairs from a school admissions email thread.

From the pasted thread below, extract any distinct questions/comments from the parent and the matching replies (if any). Remove greetings and sign-offs. Return an array of JSON objects with keys:

- "message": the redacted parent question
- "reply": the matching reply (if present)

Only output a valid JSON list. Do not explain anything.

THREAD:
\"\"\"{raw_thread}\"\"\"
""".strip()

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        raw_text = res.choices[0].message.content.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]

        try:
            json_start = raw_text.find("[")
            if json_start == -1:
                raise ValueError("No JSON array found in model output.")
            json_data = raw_text[json_start:]
            parsed = json.loads(json_data)
            if not isinstance(parsed, list):
                raise ValueError("Parsed result is not a list.")
            return jsonify({"pairs": parsed})
        except Exception as inner_e:
            print("⚠️ GPT raw response:", raw_text)
            raise inner_e

    except Exception as e:
        print(f"❌ CREATE ERROR: {e}")
        return jsonify({"error": "Failed to create valid replies."}), 500



@app.route("/save-standard-batch", methods=["POST"])
def save_standard_batch():
    today_date = datetime.now().strftime("%Y-%m-%d")

    data = request.get_json()
    entries = data.get("entries", [])

    if not entries:
        return jsonify({"error": "No entries received"}), 400

    try:
        with open("standard_responses.json", "r") as f:
            existing = json.load(f)
    except:
        existing = []

    for e in entries:
        if e.get("message") and e.get("reply"):
            existing.append({
                "message": e["message"],
                "reply": e["reply"]
            })

    with open("standard_responses.json", "w") as f:
        json.dump(existing, f, indent=2)

    return jsonify({"message": "✅ Saved"}), 200

@app.route("/")
def index(): 
    return render_template("index.html")

# ───────────── LATEST Q&A EXTRACTION ENDPOINT ─────────────
def extract_latest_qa(thread_text: str):
    """
    Extracts the latest parent question and corresponding school reply from a raw email thread.
    Returns (parent_message, school_reply)
    """
    parts = re.split(r'(From:\s*(Parent|School|Admissions)[^:]*:)', thread_text, flags=re.I)
    segments = []
    i = 1
    while i < len(parts):
        sender = parts[i+1].strip().lower()
        msg = parts[i+2].strip()
        segments.append((sender, msg))
        i += 3
    last_parent_idx = None
    for idx in reversed(range(len(segments))):
        if 'parent' in segments[idx][0]:
            last_parent_idx = idx
            break
    if last_parent_idx is None:
        return None, None
    for idx in range(last_parent_idx+1, len(segments)):
        if 'school' in segments[idx][0] or 'admissions' in segments[idx][0]:
            return segments[last_parent_idx][1], segments[idx][1]
    return segments[last_parent_idx][1], None

@app.route("/extract-and-save-latest-qa", methods=["POST"])
def extract_and_save_latest_qa():
    """
    POST { "thread": "full email thread text" }
    → Extracts latest parent enquiry & reply, redacts, appends to standard_responses.json, returns what was saved
    """
    try:
        data = request.get_json(force=True)
        thread = data.get("thread", "").strip()
        if not thread:
            return jsonify({"error": "Missing thread"}), 400

        question, answer = extract_latest_qa(thread)
        if not question or not answer:
            return jsonify({"error": "Could not extract a Q&A pair from the thread"}), 400

        question = remove_personal_info(question)
        answer = remove_personal_info(answer)

        path = "standard_responses.json"
        try:
            with open(path, "r") as f:
                saved = json.load(f)
        except Exception:
            saved = []
        saved.append({
            "message": question.strip(),
            "reply": answer.strip()
        })
        with open(path, "w") as f:
            json.dump(saved, f, indent=2)

        return jsonify({
            "message": "Q&A extracted and saved",
            "question": question,
            "reply": answer
        }), 200

    except Exception as e:
        print(f"❌ EXTRACT/SAVE Q&A ERROR: {e}")
        return jsonify({"error": "Internal server error while extracting/saving Q&A"}), 500

from flask import request, jsonify

@app.route("/static-qa/list", methods=["GET"])
def list_static_qa():
    """Return all static Q&A pairs."""
    try:
        with open("standard_responses.json", "r") as f:
            data = json.load(f)
        return jsonify({"data": data})
    except Exception as e:
        return jsonify({"error": "Failed to load Q&A"}), 500

@app.route("/static-qa/update", methods=["POST"])
def update_static_qa():
    """Update a Q&A pair at a given index."""
    body = request.get_json()
    index = body.get("index")
    new_message = body.get("message")
    new_reply = body.get("reply")
    if index is None or new_message is None or new_reply is None:
        return jsonify({"error": "Missing parameters"}), 400
    try:
        with open("standard_responses.json", "r") as f:
            data = json.load(f)
        if not (0 <= index < len(data)):
            return jsonify({"error": "Invalid index"}), 400
        data[index]["message"] = new_message
        data[index]["reply"] = new_reply
        with open("standard_responses.json", "w") as f:
            json.dump(data, f, indent=2)
        return jsonify({"status": "updated"})
    except Exception as e:
        return jsonify({"error": "Update failed"}), 500

@app.route("/static-qa/delete", methods=["POST"])
def delete_static_qa():
    """Delete a Q&A pair at a given index."""
    body = request.get_json()
    index = body.get("index")
    if index is None:
        return jsonify({"error": "Missing index"}), 400
    try:
        with open("standard_responses.json", "r") as f:
            data = json.load(f)
        if not (0 <= index < len(data)):
            return jsonify({"error": "Invalid index"}), 400
        data.pop(index)
        with open("standard_responses.json", "w") as f:
            json.dump(data, f, indent=2)
        return jsonify({"status": "deleted"})
    except Exception as e:
        return jsonify({"error": "Delete failed"}), 500
# ──────────────────────────────
# ✏️  POST /revise
# ──────────────────────────────

@app.route("/revise", methods=["POST"])
def revise_reply():
    try:
        body = request.get_json(force=True)
        if not isinstance(body, dict):
            print("❌ Data is not a dict:", body)
            return jsonify({"error": "Invalid request format"}), 400
        message = (body.get("message") or "").strip()
        previous_reply = (body.get("previous_reply") or "").strip()
        instruction = (body.get("instruction") or "").strip()
        url_box_text = (body.get("url_box") or "").strip()

        if not message or not previous_reply:
            return jsonify({"error": "Missing message or previous reply."}), 400

        # Clean and process inputs
        clean_message = remove_personal_info(message)
        clean_instruction = remove_personal_info(instruction)
        url_map = parse_url_box(url_box_text)

        # --- DYNAMIC CONTEXT: Use same context logic as /reply ---
        q_vec = embed_text(clean_message)
        sims = [(cosine_similarity(q_vec, vec), meta) for vec, meta in zip(doc_embeddings, metadata)]
        top = sorted([m for m in sims if m[0] >= SIMILARITY_THRESHOLD], key=lambda x: x[0], reverse=True)[:RESPONSE_LIMIT]
        context_blocks = []
        for _, m in top:
            print("DEBUG: m is type", type(m), "value:", m)  # Optional debug; remove when happy
            if isinstance(m, dict):
                content = m.get('content', '')
                url = m.get('url', '')
                if url:
                    context_blocks.append(f"{content}\n[Info source]({url})")
                else:
                    context_blocks.append(content)
            else:
                context_blocks.append(str(m))

        top_context = "\n---\n".join(context_blocks) if context_blocks else ""

        today_date = datetime.now().strftime('%d %B %Y')

        # --- UNIFY SCHOOL INFO ---
        prompt = f"""
TODAY'S DATE IS {today_date}.

You are the Director of Admissions at St. Margaret's Episcopal School in California, USA.

Please revise the email reply below based on the parent's original enquiry and the revision instruction provided.

Only use URLs provided in this dictionary:
{json.dumps(load_url_mapping(), indent=2)}

Follow these essential rules:
- If a date is in the past, do not mention it. Only provide future dates or refer to deadlines generically (e.g., "in early February") if unsure.
- Use American spelling and terminology (e.g. enrollment, program, counselor, fall, etc.)
- DO NOT fabricate or guess any information. If something is unknown, say so honestly.
- DO include relevant links using Markdown format: [Anchor Text](https://...). Embed links naturally in the body of the reply.
- DO use approved anchor phrases like "Open Events page", "Admissions page", or "registration form"
- NEVER use vague anchors like "click here", "more info", "register here", "visit page", etc.
- NEVER show raw URLs, list links at the bottom, or use markdown formatting like bold, italics, or bullet points
- Never mention dates, always be non specific about dates, especially dates that occur in the past, always look at the current date and check if the date is in the past. If unsure, direct the parent to the relevant web page instead

Reply only with the revised email body in Markdown format, ready to send. Do not include 'Subject:', triple backticks, or code blocks.

- NEVER use markdown formatting like bold, italics, or bullet points

Original Parent Email:
\"\"\"{clean_message}\"\"\"

Previous Reply:
\"\"\"{previous_reply}\"\"\"

Revision Instruction:
\"\"\"{clean_instruction}\"\"\"

School Info:
\"\"\"{top_context}\"\"\"

End your reply with this exact sign-off, using line breaks:

Kind regards,
Admissions Team
St. Margaret's Episcopal School
""".strip()

        print("REVISE PROMPT:", prompt)  # DEBUG
        reply_md = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        ).choices[0].message.content.strip()
        print("REVISE RAW OUTPUT:", reply_md)  # DEBUG

        if not reply_md:
            return jsonify({"error": "Failed to generate revised reply."}), 500

        # Clean and process the reply
        reply_md = clean_gpt_email_output(reply_md)
        reply_md = insert_links(reply_md, url_map)

        # Convert to different formats
        reply_html = markdown_to_html(reply_md)
        reply_outlook = markdown_to_outlook_html(reply_md)

        # Extract sentiment (reuse from original if no major changes)
        try:
            sent_prompt = f"""
You are an expert school admissions assistant.

Please analyse the following parent enquiry and return a JSON object with two keys:

- "score": an integer from 1 (very negative) to 10 (very positive)
- "strategy": a maximum 30 words strategy for how to reply to the message

Only return the JSON object — no extra explanation.

Enquiry:
\"\"\"{clean_message}\"\"\"
""".strip()

            sent_json = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":sent_prompt}],
                temperature=0.3
            ).choices[0].message.content.strip()

            sent = json.loads(sent_json)
            score = int(sent.get("score", 5))
            strat = sent.get("strategy", "Revised response")
        except:
            score, strat = 5, "Revised response"

        def extract_links_from_html(html):
            matches = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html)
            return [(text.strip(), url.strip()) for url, text in matches]

        links = extract_links_from_html(reply_html)
        matched_url = links[0][1] if links else ""
        matched_source = links[0][0] if links else ""

        return jsonify({
            "reply": reply_html,
            "reply_markdown": reply_md,
            "reply_outlook": reply_outlook,
            "sentiment_score": score,
            "strategy_explanation": strat,
            "url": matched_url,
            "link_label": matched_source
        })

    except Exception as e:
        import traceback
        print(f"❌ REVISE ERROR: {e}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error during revision."}), 500


@app.route("/usage-insights")
def usage_insights():
    today_date = datetime.now().strftime("%Y-%m-%d")

    import json
    from collections import Counter, defaultdict

    # Read the log file
    rows = []
    try:
        with open(USAGE_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except:
                    continue
    except FileNotFoundError:
        pass

    by_day = defaultdict(int)
    sentiment_sum, sentiment_count = 0, 0
    sentiment_trend = defaultdict(list)
    sentiment_dist = Counter()
    topic_counts = Counter()
    link_counts = Counter()
    reply_types = Counter()
    template_replies = 0
    pdf_usage = Counter()

    for row in rows:
        dt = row.get("timestamp", "")[:10]
        by_day[dt] += 1
        t = row.get("type", "")
        reply_types[t] += 1
        s = row.get("sentiment")
        if s is not None:
            try:
                s = float(s)
                sentiment_sum += s
                sentiment_count += 1
                sentiment_trend[dt].append(s)
                sentiment_dist[round(s)] += 1
            except:
                pass
        for topic in row.get("topics", []):
            topic_counts[topic] += 1
        for anchor in row.get("links_used", []):
            if isinstance(anchor, (list, tuple)) and anchor:
                link_counts[anchor[0]] += 1
            elif isinstance(anchor, str):
                link_counts[anchor] += 1
        if row.get("used_template"):
            template_replies += 1
        for pdf in row.get("pdfs_used", []):
            pdf_usage[pdf] += 1

    sentiment_trend_avg = {d: (sum(l)/len(l)) for d, l in sentiment_trend.items()}

    all_dates = sorted(sentiment_trend_avg.keys())
    if all_dates:
        overall_period = f"{all_dates[0]} to {all_dates[-1]}"
        recent_period = "last 7 days"
        recent_dates = all_dates[-7:] if len(all_dates) >= 7 else all_dates
    else:
        overall_period = "past month"
        recent_period = "last 7 days"
        recent_dates = []

    # Recent topics
    recent_topics = Counter()
    for row in rows:
        dt = row.get("timestamp", "")[:10]
        if dt in recent_dates:
            for topic in row.get("topics", []):
                recent_topics[topic] += 1

    overall_sentiment = (sentiment_sum/sentiment_count) if sentiment_count else 0
    recent_sentiment = (
        sum([sentiment_trend_avg[d] for d in recent_dates]) / len(recent_dates)
        if recent_dates else overall_sentiment
    )
    overall_count = sum(by_day.values())
    recent_count = sum([by_day[d] for d in recent_dates]) if recent_dates else 0
    template_reply_rate = (template_replies / reply_types.get("reply", 1)) if reply_types.get("reply", 0) else 0

    def format_topics(topics):
        return ", ".join(f"{k} ({v})" for k, v in topics.most_common(3)) if topics else "no clear trends"

    def format_pdfs(pdf_usage):
        return ", ".join(f"{k} ({v})" for k, v in pdf_usage.most_common(3)) if pdf_usage else "none"

    def format_links(link_counts):
        return ", ".join(f"{k} ({v})" for k, v in link_counts.most_common(3)) if link_counts else "none"

        # Insights Summary

    start_date = all_dates[0] if all_dates else "N/A"
    end_date = all_dates[-1] if all_dates else "N/A"
    common_topics = format_topics(topic_counts)
    trending_topics = format_topics(recent_topics)
    top_pdfs = format_pdfs(pdf_usage)
    smart_links = format_links(link_counts)
    template_rate = round(template_reply_rate * 100, 1)  # percent, 1 decimal

    summary = f"""Parent Enquiries: {overall_count} replies since {start_date} to {end_date} (recent: {recent_count} in last 7 days).

    Sentiment: Average score {overall_sentiment:.1f}, recent {recent_sentiment:.1f}. Steady

    Most common topics: {common_topics}
    Trending topics (recent): {trending_topics}

    Template Reply Rate: {template_rate}% of replies used approved templates.
    Top PDFs referenced: {top_pdfs}
    Most-used Smart Links: {smart_links}

    These insights help the team understand parent interest, resource use, and support needs."""

    # Prepare the parsed insights dict for strategy function
    parsed_insights = {
        "recent_topics": dict(recent_topics),
        "template_reply_rate": template_reply_rate,
        "avg_sentiment": overall_sentiment,
        "pdf_usage": dict(pdf_usage)
    }

    # Load manual strategy if exists; otherwise generate
    manual_strategy = load_manual_strategy()
    if manual_strategy:
        strategy_text = manual_strategy
    else:
        strategy_text = generate_strategy_advice(parsed_insights)

    result = {
        "total_replies": reply_types.get("reply", 0),
        "total_revisions": reply_types.get("revise", 0),
        "avg_sentiment": (sentiment_sum/sentiment_count) if sentiment_count else None,
        "replies_per_day": dict(by_day),
        "sentiment_trend": sentiment_trend_avg,
        "sentiment_distribution": dict(sentiment_dist),
        "topic_counts": dict(topic_counts),
        "recent_topics": dict(recent_topics),
        "top_link_anchors": dict(link_counts),
        "template_reply_rate": template_reply_rate,
        "pdf_usage": dict(pdf_usage),
        "summary": summary,
        "strategy_advice": strategy_text
    }

    return jsonify(result)

    result = {
        "total_replies": reply_types.get("reply", 0),
        "total_revisions": reply_types.get("revise", 0),
        "avg_sentiment": (sentiment_sum/sentiment_count) if sentiment_count else None,
        "replies_per_day": dict(by_day),
        "sentiment_trend": sentiment_trend_avg,
        "sentiment_distribution": dict(sentiment_dist),
        "topic_counts": dict(topic_counts),
        "recent_topics": dict(recent_topics),
        "top_link_anchors": dict(link_counts),
        "template_reply_rate": template_reply_rate,
        "pdf_usage": dict(pdf_usage),
        "summary": summary,
    }
    return json.dumps(result), 200, {"Content-Type": "application/json"}

@app.route("/update-strategy", methods=["POST"])
def update_strategy():
    try:
        data = request.get_json(force=True)
        new_text = data.get("strategy_text", "").strip()
        if not new_text:
            return jsonify({"error": "Missing strategy_text"}), 400

        with open(STRATEGY_FILE, "w") as f:
            json.dump({"strategy_text": new_text}, f, indent=2)

        return jsonify({"status": "Strategy updated"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_dotenv()
    app.run(debug=True)