import os
import json
import re
from dotenv import load_dotenv
from groq import Groq
from memory.research_memory import add_memory

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def extract_json(text):
    """
    Extract JSON array or object from messy LLM output
    """
    match = re.search(r"\[.*\]|\{.*\}", text, re.DOTALL)

    if match:
        return match.group(0)

    return None


def store_research_memory(user_id, question, answer):

    prompt = f"""
    Extract research knowledge.

    Return JSON list.

    Importance scale:
    3 = critical research insight
    2 = useful supporting knowledge
    1 = minor detail

    Example:

    [
    {{
        "topic": "Adaptive Learning",
        "key_finding": "Adaptive systems personalize instruction using learner models",
        "importance": 3
    }}
    ]

    QUESTION:
    {question}

    ANSWER:
    {answer}
    """

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )

    text = completion.choices[0].message.content.strip()

    # remove markdown if present
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    # extract json safely
    clean_json = extract_json(text)

    if not clean_json:
        print("❌ Memory parsing failed — no JSON found")
        print(text)
        return

    try:
        memory_data = json.loads(clean_json)

        if isinstance(memory_data, dict):
            memory_data = [memory_data]

        for entry in memory_data:
            add_memory(user_id, entry)

        print(f"✅ Stored {len(memory_data)} memories (Groq)")

    except Exception as e:
        print("❌ Memory parsing failed")
        print(clean_json)
        print(e)