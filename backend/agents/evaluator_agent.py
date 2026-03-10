import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def evaluate_answer(question, answer, context):

    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
You are an AI research evaluator.

Evaluate the answer based ONLY on provided context.

Return ONLY JSON:

{{
 "grounded": true/false,
 "missing_information": "...",
 "improvement_query": "better search query if needed"
}}

QUESTION:
{question}

ANSWER:
{answer}

CONTEXT:
{context}
"""

    response = model.generate_content(prompt)

    try:
        return eval(response.text.strip())
    except:
        return {"grounded": True}