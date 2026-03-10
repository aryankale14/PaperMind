from langchain_community.vectorstores import FAISS
from ingestion.embedder import get_embedding_model


from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

def load_vectorstore(user_id=None):

    embeddings = get_embedding_model()

    if user_id:
        index_path = str(DATA_DIR / user_id / "faiss_index")
    else:
        index_path = str(DATA_DIR / "faiss_index")

    import os
    if not os.path.exists(index_path) or not os.path.exists(os.path.join(index_path, "index.faiss")):
        import faiss
        from langchain_community.docstore.in_memory import InMemoryDocstore
        # fallback for empty store
        index = faiss.IndexFlatL2(len(embeddings.embed_query("test")))
        return FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore