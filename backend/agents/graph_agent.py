import os
import json
import re
from dotenv import load_dotenv
from groq import Groq
from memory.knowledge_graph import add_triplet

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def extract_json_array(text: str):
    """
    Extract the first JSON array from LLM output.
    Works even if the model adds explanations.
    """

    # remove markdown blocks if present
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            if "[" in part and "]" in part:
                text = part
                break

    # find first JSON array
    match = re.search(r"\[.*?\]", text, re.DOTALL)

    if match:
        return match.group(0)

    return None


def extract_graph_knowledge(user_id, question, answer):

    prompt = f"""
Extract key research concept relationships.

Return a JSON array of triples ONLY.

Example format:

[
  {{"subject":"Adaptive Learning","relation":"contrasts_with","object":"Feedback Systems"}},
  {{"subject":"Adaptive Learning","relation":"uses","object":"Personalization"}}
]

QUESTION:
{question}

ANSWER:
{answer}
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw_text = completion.choices[0].message.content.strip()

    try:

        json_array = extract_json_array(raw_text)

        if not json_array:
            raise ValueError("No JSON array found")

        triples = json.loads(json_array)

        if isinstance(triples, dict):
            triples = [triples]

        seen = set()
        count = 0

        for t in triples:

            subject = t.get("subject")
            relation = t.get("relation")
            obj = t.get("object")

            if not subject or not relation or not obj:
                continue

            key = (subject, relation, obj)

            if key in seen:
                continue

            seen.add(key)

            add_triplet(user_id, subject, relation, obj)
            count += 1

        print(f"🧠 Added {count} knowledge relations")

    except Exception as e:

        print("Graph extraction failed")
        print("Error:", e)
        print("Raw output preview:", raw_text[:500])