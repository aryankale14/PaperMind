from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pathlib import Path
import time


def create_faiss_index(
    chunks,
    embeddings,
    save_path="data/faiss_index",
    batch_size=40,   # SAFE for Gemini free tier
    sleep_time=50    # wait to reset quota window
):
    """
    Creates FAISS index with batching to avoid API quota errors.
    """

    vectorstore = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        print(f"Embedding batch {i//batch_size + 1}")

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

        # Avoid Gemini rate limits
        if i + batch_size < len(chunks):
            print(f"Sleeping {sleep_time}s to avoid quota...")
            time.sleep(sleep_time)

    Path(save_path).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(save_path)

    print("FAISS index saved successfully")

    return vectorstore