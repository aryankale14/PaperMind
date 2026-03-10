import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cur = conn.cursor()

# Check if table exists
cur.execute("SELECT to_regclass('document_chunks');")
print("Table exists:", cur.fetchone()[0])

# Check column type
cur.execute("SELECT atttypmod FROM pg_attribute WHERE attrelid = 'document_chunks'::regclass AND attname = 'embedding';")
res = cur.fetchone()
print("Dimension (atttypmod):", res[0] if res else "Not found")

# Drop and recreate table if dimension is 768
if res and res[0] == 768:
    print("Dimension is still 768! Dropping table...")
    cur.execute("DROP TABLE document_chunks;")
    conn.commit()
    print("Table dropped. Now run the Uvicorn server to recreate it.")

conn.close()
