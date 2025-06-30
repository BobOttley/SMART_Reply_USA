# ──────────────────────────────
# 🧠  SMART REPLY BACKEND — FORMATTING FIXED & ROBUST
# ──────────────────────────────

import os, json, pickle, re
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from embedding_utils import process_pdf_and_append_to_kb
from flask import send_from_directory

# ───────────── HELPERS ─────────────

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
    url_map = {}
    parts = re.split(r'[;\n]+', url_text.strip())
    for part in parts:
        if '=' in part:
            anchor, url = part.split('=', 1)
            url_map[anchor.strip()] = url.strip()
    return url_map

def insert_links(text, url_map):
    def safe_replace(match):
        word = match.group(0)
        for anchor, url in url_map.items():
            if word.lower() == anchor.lower():
                safe_url = url.replace(')', '%29').replace('(', '%28')
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
    if not md.strip():
        return ""
    md = re.sub(r'\[([^\]]+)\]\((https?://[^\)]+)\)', 
                lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', md)
    paragraphs = re.split(r'\n\s*\n', md.strip())
    processed_paragraphs = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        lines = paragraph.split('\n')
        is_signature = (
            len(lines) > 1 and 
            any(re.search(r'\b(Mrs?\.?|Ms\.?|Mr\.?|Director|Manager|School|College|University|Tel:|Email:|Phone:)', 
                         line, re.I) for line in lines)
        )
        if is_signature:
            clean_lines = [line.strip() for line in lines if line.strip()]
            processed_paragraphs.append('<br>'.join(clean_lines))
        else:
            paragraph_html = paragraph.replace('\n', '<br>')
            processed_paragraphs.append(paragraph_html)
    result = '<br><br>'.join(processed_paragraphs)
    result = re.sub(r'href=([^\s">]+)', r'href="\1"', result)
    result = re.sub(r'<a\s+href="([^"]+)"\s*>([^<]+)</a>', r'<a href="\1">\2</a>', result)
    return result

def clean_gpt_email_output(md: str) -> str:
    md = md.strip()
    md = re.sub(r"^```(?:markdown)?", "", md, flags=re.I).strip()
    md = re.sub(r"```$", "", md, flags=re.I).strip()
    lines = md.splitlines()
    if lines:
        first_line = lines[0].strip()
        if (
            len(first_line) < 80 and
            not first_line.lower().startswith("dear") and
            not first_line.endswith(".") and
            not first_line.endswith(":")
        ):
            lines = lines[1:]
    md = "\n".join(lines).strip()
    md = re.sub(r"\*\*(.*?)\*\*", r"\1", md)
    md = re.sub(r"\*(.*?)\*", r"\1", md)
    return md.strip()

def cosine_similarity(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_safe_url(label: str) -> str: return load_url_mapping().get(label, "")

# ───────────── APP SETUP ─────────────

load_dotenv()
app = Flask(__name__, template_folder="templates")
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("🚀 PEN Reply Flask server starting…")

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
        # Defensive type checks
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

        # Extra sanity check
        for m in new_metadata:
            if isinstance(m, dict) and fname in m.get("source", ""):
                print(f"⚠️ Still contains reference to deleted file: {m.get('source')}")

        # Update global state
        metadata.clear()
        metadata.extend(new_metadata)
        doc_embeddings = np.array(new_embeddings)

        # Save updated pickle
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

@app.route("/reply", methods=["POST"])
def generate_reply():
    try:
        body = request.get_json(force=True)
        question_raw = (body.get("message") or "").strip()
        source_type = body.get("source_type", "email")
        include_cta = body.get("include_cta", True)
        url_box_text = (body.get("url_box") or "").strip()
        instruction_raw = (body.get("instruction") or "").strip()
        question = remove_personal_info(question_raw)
        instruction = remove_personal_info(instruction_raw)
        url_map = parse_url_box(url_box_text)
        if not question: return jsonify({"error":"No message received."}), 400
        q_vec = embed_text(question)

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

        sent_prompt = f"""
You are an expert school admissions assistant.

Please analyse the following parent enquiry and return a JSON object with two keys:

- "score": an integer from 1 (very negative) to 10 (very positive)
- "strategy": a maximum 30 words strategy for how to reply to the message

Only return the JSON object — no extra explanation.

Enquiry:
\"\"\"{question}\"\"\"
"""
        sent_json = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": sent_prompt}],
            temperature=0.3
        ).choices[0].message.content.strip()
        try:
            sent = json.loads(sent_json)
            score = int(sent.get("score", 5))
            strat = sent.get("strategy", "")
        except:
            score, strat = 5, ""

        sims = [(cosine_similarity(q_vec, vec), meta) for vec, meta in zip(doc_embeddings, metadata)]
        top = sorted([m for m in sims if m[0] >= SIMILARITY_THRESHOLD], key=lambda x: x[0], reverse=True)[:RESPONSE_LIMIT]
        if not top:
            return jsonify({
                "reply": "<p>Thank you for your enquiry. A member of our admissions team will contact you shortly.</p>",
                "sentiment_score": score, "strategy_explanation": strat
            })

        context_blocks = [f"{m['content']}\n[Info source]({m.get('url','')})" if m.get('url') else m['content'] for _, m in top]
        top_context = "\n---\n".join(context_blocks)
        today_date = datetime.now().strftime('%d %B %Y')

        topic = "general"
        q_lower = question.lower()
        if "visit" in q_lower or "tour" in q_lower:
            topic = "visit"
        elif "fees" in q_lower or "cost" in q_lower:
            topic = "fees"
        elif "subjects" in q_lower or "curriculum" in q_lower:
            topic = "curriculum"

        if source_type == "form":
            message_intro = "Parent Enquiry Form Submission:"
        else:
            message_intro = "Parent Email:"

        prompt = f"""
TODAY'S DATE IS {today_date}.

You are responding on behalf of the admissions team at an independent Preschool–Grade 12 day school in the United States.

Please write a warm, professional, and helpful email reply to the parent below. If the inquiry came from a form, treat it the same as an email — respond directly and naturally. Only use the verified school information provided below. If the parent is likely to benefit, include a hyperlink to the appropriate section of the website.

Only use URLs provided in this dictionary:
{json.dumps(load_url_mapping(), indent=2)}

Do not mention viewbooks, or brochures or invent any links, that do not appear in that list.

Follow these essential rules:
- Use American spelling and terminology (e.g. enrollment, program, counselor, fall, etc.)
- DO NOT make up or guess any details — if something is unclear, say you’ll follow up
- DO embed relevant links naturally using Markdown-style format: [Anchor Text](https://...)
- DO use clear, professional phrasing — don’t sound robotic or generic
- NEVER include raw URLs or vague anchors like "click here" or "more info"
- NEVER include bullet points — use short, readable paragraphs

End your reply with a simple professional closing, such as:

Kind regards,  
Admissions Team  
St. Margaret’s Episcopal School

Parent Inquiry:
\"\"\"{question}\"\"\"

School Info:
\"\"\"{top_context}\"\"\"
""".strip()

        reply_md = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.4
        ).choices[0].message.content.strip()
        reply_md = clean_gpt_email_output(reply_md)

        if include_cta:
            cta_line = ""
            if topic == "visit":
                cta_line = "We’d love to welcome your family for a campus tour—just let me know if you’d like help scheduling a visit."
            elif topic == "fees":
                cta_line = "If it would be helpful, I’d be happy to connect you with our team to talk through tuition and financial aid options."
            elif topic == "curriculum":
                cta_line = "I’d be glad to share more about our academic programs and how we support each student’s growth across all grade levels."
            elif score >= 8:
                cta_line = "Would you like someone from our admissions team to reach out personally and answer any remaining questions?"
      
            if cta_line and "Kind regards" in reply_md:
                reply_md = reply_md.replace("Kind regards", f"{cta_line}\n\nKind regards")

        reply_md = insert_links(reply_md, url_map)
        reply_html = markdown_to_html(reply_md)
        reply_outlook = markdown_to_outlook_html(reply_md)

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

@app.route("/revise", methods=["POST"])
def revise_reply():
    try:
        body = request.get_json(force=True)
        message = (body.get("message") or "").strip()
        previous_reply = (body.get("previous_reply") or "").strip()
        instruction = (body.get("instruction") or "").strip()
        url_box_text = (body.get("url_box") or "").strip()

        if not message or not previous_reply:
            return jsonify({"error": "Missing message or previous reply."}), 400

        clean_message = remove_personal_info(message)
        clean_instruction = remove_personal_info(instruction)
        url_map = parse_url_box(url_box_text)
        
        today_date = datetime.now().strftime('%d %B %Y')
        
        prompt = f"""
TODAY'S DATE IS {today_date}.

You are responding on behalf of the admissions team at an independent Preschool–Grade 12 day school in the United States.

Please revise the email reply below based on the parent's original inquiry and the revision instruction provided.

Only use URLs provided in this dictionary:
{json.dumps(load_url_mapping(), indent=2)}

Do not mention viewbooks, or brochures or invent any links, that do not appear in that list.

Follow these essential rules:
- Use American spelling and terminology (e.g. enrollment, program, counselor, fall, etc.)
- DO NOT make up or guess any details — if something is unclear, say you’ll follow up
- DO embed relevant links using Markdown-style format: [Anchor Text](https://...)
- DO use clear, natural phrasing in short paragraphs — avoid robotic or generic responses
- NEVER include raw URLs, vague link text like "click here", or bullet points
- End your revised email with a simple professional closing like:

Kind regards,  
Admissions Team  
St. Margaret’s Episcopal School

Original Parent Inquiry:
\"\"\"{clean_message}\"\"

Previous Reply:
\"\"\"{previous_reply}\"\"

Revision Instruction:
\"\"\"{clean_instruction}\"\"\"
""".strip()

        reply_md = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        ).choices[0].message.content.strip()
        
        reply_md = clean_gpt_email_output(reply_md)

        score, strat = 5, "Revised response"
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
            strat = sent.get("strategy", strat)
        except:
            pass

        if "visit" in clean_message.lower():
            reply_md += "\n\nIf you haven’t yet had a chance to visit us, we’d be delighted to welcome you to the school."
        elif "fees" in clean_message.lower():
            reply_md += "\n\nIf you’d like to discuss your child’s needs further, I’d be happy to arrange a time to speak."
        elif "curriculum" in clean_message.lower():
            reply_md += "\n\nWe’re always happy to share more about how we support girls to thrive academically and beyond."
        elif score >= 8:
            reply_md += "\n\nDo let me know if you’d like me to send a personalised prospectus tailored to your daughter’s interests."

        reply_md = insert_links(reply_md, url_map)
        reply_html = markdown_to_html(reply_md)
        reply_outlook = markdown_to_outlook_html(reply_md)
        
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
        print(f"❌ REVISE ERROR: {e}")
        return jsonify({"error": "Internal server error during revision."}), 500

@app.route("/save-standard", methods=["POST"])
def save_standard_reply():
    try:
        body = request.get_json(force=True)
        message = (body.get("message") or "").strip()
        reply = (body.get("reply") or "").strip()
        urls = body.get("urls", [])

        if not message or not reply:
            return jsonify({"error": "Missing message or reply."}), 400

        path = "standard_responses.json"
        saved = []
        if os.path.exists(path):
            with open(path, "r") as f:
                saved = json.load(f)

        entry = {
            "message": message,
            "reply": reply,
            "urls": urls
        }
        saved.append(entry)

        with open(path, "w") as f:
            json.dump(saved, f, indent=2)

        return jsonify({"status": "saved"})
    except Exception as e:
        print(f"❌ SAVE ERROR: {e}")
        return jsonify({"error": "Internal server error during save."}), 500

# ─────────── SMART LINK MAPPINGS ENDPOINTS ───────────

@app.route("/get-url-mappings", methods=["GET"])
def get_url_mappings():
    try:
        mapping = load_url_mapping()
        return jsonify(mapping)
    except Exception as e:
        print(f"❌ Error reading URL_MAPPING: {e}")
        return jsonify({}), 500

@app.route("/save-url-mappings", methods=["POST"])
def save_url_mappings():
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

if __name__ == "__main__":
    load_dotenv()
    app.run(debug=True)