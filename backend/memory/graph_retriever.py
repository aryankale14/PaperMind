from memory.knowledge_graph import load_graph


def find_related_concepts(user_id, query):
    """Find related concepts from the user's knowledge graph."""
    graph = load_graph(user_id)

    related = set()

    for edge in graph["edges"]:
        subject = edge["subject"].lower()
        obj = edge["object"].lower()

        if subject in query.lower():
            related.add(edge["object"])

        if obj in query.lower():
            related.add(edge["subject"])

    return list(related)