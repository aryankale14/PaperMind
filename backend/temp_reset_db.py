import os
from database import init_db, get_conn

# Initialize the pool
init_db()

with get_conn() as conn:
    with conn.cursor() as cur:
        print("Dropping document_chunks...")
        cur.execute("DROP TABLE IF EXISTS document_chunks;")
        print("Re-creating all tables...")
        # Now we create the tables again, document_chunks will be created with 3072!

from database import _create_tables
_create_tables()
print("Success!")
