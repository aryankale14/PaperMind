import asyncio
import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

async def reset_db():
    conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
    
    print("Dropping table if exists...")
    await conn.execute("DROP TABLE IF EXISTS document_chunks CASCADE;")
    
    # Recreate the table exactly as it needs to be with VECTOR(3072)
    print("Creating table with VECTOR(3072)...")
    await conn.execute("""
        CREATE TABLE document_chunks (
            id SERIAL PRIMARY KEY,
            user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
            paper_id TEXT,
            paper_title TEXT,
            page INTEGER,
            content TEXT,
            embedding VECTOR(3072)
        );
        CREATE INDEX idx_document_user ON document_chunks(user_id);
    """)
    print("Done!")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(reset_db())
