import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def determine_next_hop(question, context):

    prompt = f"""
You are a research planner.

Given a research question and current evidence,
decide if more research is required.

Return ONLY JSON:

{{
 "next_hop": true or false,
 "new_query": "query to search next"
}}

Question:
{question}

Current Evidence:
{context[:2000]}
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    text = completion.choices[0].message.content.strip()

    try:
        return json.loads(text)
    except:
        return {"next_hop": False}