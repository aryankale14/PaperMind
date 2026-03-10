from ingestion.pdf_loader import load_pdfs
from ingestion.chunker import chunk_documents
from ingestion.embedder import get_embedding_model
from ingestion.vector_store import create_faiss_index


def build_index():

    print("📄 Loading PDFs...")

    documents = load_pdfs("../data/papers")

    print(f"Loaded {len(documents)} pages")

    print("\n✂️ Chunking documents...")

    chunks = chunk_documents(documents)

    print(f"Created {len(chunks)} chunks")

    print("\n🔢 Loading embedding model...")

    embeddings = get_embedding_model()

    print("\n🧠 Creating FAISS index...")

    create_faiss_index(
    chunks,
    embeddings,
    save_path="../data/faiss_index"
    )

    print("\n✅ Index build complete!")


if __name__ == "__main__":
    build_index()