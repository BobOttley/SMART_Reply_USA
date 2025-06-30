
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
"""{raw_thread}"""
""".strip()

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        raw_text = res.choices[0].message.content.strip()

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
