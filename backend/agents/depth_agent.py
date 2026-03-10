import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def evaluate_research_depth(question, context):

    prompt = f"""
Evaluate whether the retrieved research context is sufficient.

Return ONLY JSON:

{{
 "depth": "enough" or "expand",
 "reason": "short explanation",
 "suggested_query": "better search query if expansion needed"
}}

QUESTION:
{question}

CONTEXT:
{context[:4000]}
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    response = completion.choices[0].message.content.strip()

    # clean markdown
    if response.startswith("```"):
        response = response.split("```")[1]
        if response.startswith("json"):
            response = response[4:]
        response = response.strip()

    try:
        return json.loads(response)
    except:
        return {"depth": "enough"}