import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def evaluate_coverage(question, context):

    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
You are evaluating research depth.

Determine whether enough evidence was retrieved
to answer the research question properly.

Return ONLY JSON:

{{
 "enough_coverage": true/false,
 "reason": "...",
 "additional_query": "better search query if coverage is weak"
}}

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}
"""

    response = model.generate_content(prompt)

    text = response.text.strip()

    try:
        return json.loads(text)
    except:
        return {"enough_coverage": True}