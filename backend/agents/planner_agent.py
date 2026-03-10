import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def plan_query(question):

    prompt = f"""
You are a research planning agent.

Break the research question into 3-5 search queries
that would help retrieve relevant academic evidence.

Rules:
- Each query should target a specific concept
- Avoid repeating the same question
- Queries should be short search phrases

Return ONLY JSON:

{{
 "queries": [
   "query1",
   "query2",
   "query3"
 ]
}}

Research Question:
{question}
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    response = completion.choices[0].message.content.strip()

    try:
        data = json.loads(response)
        return data["queries"]
    except:
        return [question]