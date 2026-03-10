from memory.research_memory import load_memory


def retrieve_memory(user_id, query, top_k=3):
    """
    Returns relevant stored research memories for a specific user.
    """
    memory = load_memory(user_id)

    if not memory:
        return []

    query_words = set(query.lower().split())

    scored = []

    for entry in memory:
        topic_words = set(entry["topic"].lower().split())
        overlap = len(query_words.intersection(topic_words))
        scored.append((overlap, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [m[1] for m in scored[:top_k] if m[0] > 0]