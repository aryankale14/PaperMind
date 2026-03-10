import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def generate_answer(query, context, memory_context=""):

    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
You are an academic research assistant.

Use the research context to answer the question.

Rules:

• Cite the research paper titles when referencing evidence
• If multiple papers support the same idea mention both
• If papers disagree explain the difference
• Do not invent citations

Answer structure:

1. Summary
2. Evidence from papers
3. Comparison if relevant
4. Conclusion

Question:
{query}

Previous Knowledge:
{memory_context}

Research Context:
{context}
"""

    response = model.generate_content(prompt)

    return response.text