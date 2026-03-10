import asyncio
from database import get_conn, init_db

init_db()

with get_conn() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT atttypid::regtype FROM pg_attribute WHERE attrelid = 'document_chunks'::regclass AND attname = 'embedding';")
        print("Vector dimension in DB:", cur.fetchone())
