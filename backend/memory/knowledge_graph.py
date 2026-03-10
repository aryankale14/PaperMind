from database import get_graph as db_get_graph, add_graph_triplet


def load_graph(user_id=None):
    """Load knowledge graph from database for a specific user."""
    if user_id:
        return db_get_graph(user_id)
    return {"nodes": [], "edges": []}


def save_graph(user_id, graph):
    """Not needed with DB — kept for compatibility."""
    pass


def add_triplet(user_id, subject, relation, obj):
    """Add a knowledge triplet to the database."""
    add_graph_triplet(user_id, subject, relation, obj)