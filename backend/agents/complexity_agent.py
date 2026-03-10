import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------------------------------
# FAST RULE-BASED DETECTION (NO API CALL NEEDED)
# ---------------------------------------------------

SIMPLE_PATTERNS = [
    "what is",
    "summary",
    "in short",
    "define",
    "overview",
    "tell me about",
    "explain briefly",
]


def is_simple_question(question: str):
    q = question.lower()
    return any(p in q for p in SIMPLE_PATTERNS)


# ---------------------------------------------------
# MAIN CLASSIFIER
# ---------------------------------------------------

def classify_complexity(question):

    # ✅ Step 1 — rule-based shortcut
    if is_simple_question(question):
        return {
            "mode": "quick",
            "reason": "Simple informational query detected"
        }

    # ✅ Step 2 — LLM classification (Groq)
    prompt = f"""
Classify the research complexity STRICTLY.

QUICK MODE:
- definition
- summary
- overview
- short explanation
- single concept question

DEEP MODE:
- compare
- analyze
- evaluate
- weaknesses
- limitations
- implications
- methodology discussion
- multi-part reasoning

Return ONLY JSON:

{{
 "mode": "quick" or "deep",
 "reason": "short explanation"
}}

Question:
{question}
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    response_text = completion.choices[0].message.content.strip()

    # ---------------------------------------------------
    # CLEAN MARKDOWN IF MODEL RETURNS ```json BLOCK
    # ---------------------------------------------------
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
        response_text = response_text.strip()

    # ---------------------------------------------------
    # SAFE JSON PARSE
    # ---------------------------------------------------
    try:
        return json.loads(response_text)
    except Exception:
        return {
            "mode": "deep",
            "reason": "Fallback due to parsing failure"
        }