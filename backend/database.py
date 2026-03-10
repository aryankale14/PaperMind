"""
PostgreSQL Database Module
Handles connection pooling and schema initialization for multi-tenant data.
"""

import os
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager

# ── Connection Pool ──────────────────────────────────────────
_pool = None


def init_db():
    """Initialize the connection pool and create tables."""
    global _pool
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable is not set")

    _pool = pool.ThreadedConnectionPool(
        minconn=2,
        maxconn=10,
        dsn=database_url,
    )

    _create_tables()
    print("✅ Database initialized")


def close_db():
    """Close the connection pool."""
    global _pool
    if _pool:
        _pool.closeall()
        _pool = None


@contextmanager
def get_conn():
    """Get a connection from the pool."""
    conn = _pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


# ── Schema ───────────────────────────────────────────────────
def _create_tables():
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT,
                    display_name TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS research_history (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    mode TEXT,
                    plan JSONB,
                    sources JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS research_memory (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
                    topic TEXT NOT NULL,
                    key_finding TEXT NOT NULL,
                    importance INTEGER DEFAULT 2,
                    created_at TIMESTAMP DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS graph_nodes (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    UNIQUE(user_id, name)
                );

                CREATE TABLE IF NOT EXISTS graph_edges (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
                    subject TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    object TEXT NOT NULL,
                    UNIQUE(user_id, subject, relation, object)
                );

                CREATE TABLE IF NOT EXISTS document_embeddings (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
                    paper_id TEXT,
                    paper_title TEXT,
                    page INTEGER,
                    content TEXT,
                    embedding VECTOR(3072)
                );

                CREATE INDEX IF NOT EXISTS idx_history_user ON research_history(user_id);
                CREATE INDEX IF NOT EXISTS idx_memory_user ON research_memory(user_id);
                CREATE INDEX IF NOT EXISTS idx_nodes_user ON graph_nodes(user_id);
                CREATE INDEX IF NOT EXISTS idx_edges_user ON graph_edges(user_id);
                CREATE INDEX IF NOT EXISTS idx_document_user ON document_embeddings(user_id);
            """)

        # Register pgvector type with psycopg2
        import pgvector.psycopg2
        pgvector.psycopg2.register_vector(conn)


# ── User Operations ──────────────────────────────────────────
def upsert_user(user_id, email, display_name):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users (id, email, display_name)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    email = EXCLUDED.email,
                    display_name = EXCLUDED.display_name
            """, (user_id, email, display_name))


# ── History Operations ───────────────────────────────────────
def get_history(user_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT question, answer, mode, plan, sources, created_at
                FROM research_history
                WHERE user_id = %s
                ORDER BY created_at ASC
            """, (user_id,))
            rows = cur.fetchall()
            return [
                {
                    "question": r[0],
                    "answer": r[1],
                    "mode": r[2],
                    "plan": r[3],
                    "sources": r[4],
                }
                for r in rows
            ]


def add_history(user_id, question, answer, mode, plan, sources):
    import json
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO research_history (user_id, question, answer, mode, plan, sources)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                user_id, question, answer, mode,
                json.dumps(plan) if plan else None,
                json.dumps(sources) if sources else None,
            ))


def check_daily_limit(user_id, limit=5):
    """Returns True if user is under the limit, False if they have exceeded it."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM research_history 
                WHERE user_id = %s AND created_at > NOW() - INTERVAL '24 hours'
            """, (user_id,))
            count = cur.fetchone()[0]
            return count < limit


# ── Memory Operations ────────────────────────────────────────
def get_memories(user_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT topic, key_finding, importance
                FROM research_memory
                WHERE user_id = %s
                ORDER BY created_at ASC
            """, (user_id,))
            return [
                {"topic": r[0], "key_finding": r[1], "importance": r[2]}
                for r in cur.fetchall()
            ]


def add_memory_entry(user_id, topic, key_finding, importance=2):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO research_memory (user_id, topic, key_finding, importance)
                VALUES (%s, %s, %s, %s)
            """, (user_id, topic, key_finding, importance))


# ── Graph Operations ─────────────────────────────────────────
def get_graph(user_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM graph_nodes WHERE user_id = %s", (user_id,))
            nodes = [r[0] for r in cur.fetchall()]

            cur.execute(
                "SELECT subject, relation, object FROM graph_edges WHERE user_id = %s",
                (user_id,),
            )
            edges = [
                {"subject": r[0], "relation": r[1], "object": r[2]}
                for r in cur.fetchall()
            ]

            return {"nodes": nodes, "edges": edges}


def add_graph_triplet(user_id, subject, relation, obj):
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Upsert nodes
            cur.execute("""
                INSERT INTO graph_nodes (user_id, name) VALUES (%s, %s)
                ON CONFLICT (user_id, name) DO NOTHING
            """, (user_id, subject))
            cur.execute("""
                INSERT INTO graph_nodes (user_id, name) VALUES (%s, %s)
                ON CONFLICT (user_id, name) DO NOTHING
            """, (user_id, obj))

            # Upsert edge
            cur.execute("""
                INSERT INTO graph_edges (user_id, subject, relation, object)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, subject, relation, object) DO NOTHING
            """, (user_id, subject, relation, obj))


# ── Document Operations (pgvector) ───────────────────────────
def add_document_chunks(user_id, paper_id, paper_title, chunks, embeddings):
    """Insert text chunks and their pgvector embeddings into PostgreSQL."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            for chunk in chunks:
                page = chunk.metadata.get("page", 0)
                content = chunk.page_content
                # Get the embedding vector for this chunk
                vector = embeddings.embed_query(content)
                
                cur.execute("""
                    INSERT INTO document_embeddings (user_id, paper_id, paper_title, page, content, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (user_id, paper_id, paper_title, page, content, vector))


def get_user_papers(user_id):
    """Retrieve a unique list of papers uploaded by this user."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT paper_id, paper_title
                FROM document_embeddings
                WHERE user_id = %s
                ORDER BY paper_title ASC
            """, (user_id,))
            return [
                {
                    "paper_id": r[0],
                    "paper_title": r[1],
                }
                for r in cur.fetchall()
            ]


# ── Reset (per-user) ─────────────────────────────────────────
def reset_user_data(user_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM research_history WHERE user_id = %s", (user_id,))
            cur.execute("DELETE FROM research_memory WHERE user_id = %s", (user_id,))
            cur.execute("DELETE FROM graph_edges WHERE user_id = %s", (user_id,))
            cur.execute("DELETE FROM graph_nodes WHERE user_id = %s", (user_id,))
            cur.execute("DELETE FROM document_embeddings WHERE user_id = %s", (user_id,))


# ── Admin Operations ─────────────────────────────────────────
def get_all_users_admin_stats():
    """Aggregate all users, their uploaded papers, and their research history for the Admin Dashboard."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Get all users
            cur.execute("SELECT id, email, display_name, created_at FROM users ORDER BY created_at DESC")
            users = [
                {
                    "id": r[0],
                    "email": r[1],
                    "display_name": r[2],
                    "created_at": r[3].isoformat() if r[3] else None,
                    "papers": [],
                    "queries": [],
                }
                for r in cur.fetchall()
            ]

            user_dict = {u["id"]: u for u in users}

            # Get distinct uploaded papers per user
            cur.execute("""
                SELECT DISTINCT user_id, paper_id, paper_title 
                FROM document_embeddings
            """)
            for r in cur.fetchall():
                uid, pid, title = r[0], r[1], r[2]
                if uid in user_dict:
                    user_dict[uid]["papers"].append({"id": pid, "title": title})

            # Get all queries per user
            cur.execute("""
                SELECT user_id, question, created_at 
                FROM research_history 
                ORDER BY created_at DESC
            """)
            for r in cur.fetchall():
                uid, question, timestamp = r[0], r[1], r[2]
                if uid in user_dict:
                    user_dict[uid]["queries"].append({
                        "question": question,
                        "timestamp": timestamp.isoformat() if timestamp else None
                    })

            return users
