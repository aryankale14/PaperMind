import json
from database import get_memories, add_memory_entry


def load_memory(user_id=None):
    """Load memories from database for a specific user."""
    if user_id:
        return get_memories(user_id)
    return []


def save_memory(user_id, memory):
    """Not needed with DB — kept for compatibility."""
    pass


def add_memory(user_id, entry):
    """Add a single memory entry to the database."""
    add_memory_entry(
        user_id=user_id,
        topic=entry.get("topic", ""),
        key_finding=entry.get("key_finding", ""),
        importance=entry.get("importance", 2),
    )