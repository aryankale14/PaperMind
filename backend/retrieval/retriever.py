from langchain_community.vectorstores import FAISS
from ingestion.embedder import get_embedding_model
from retrieval.scoring import section_score
from retrieval.bm25_retriever import BM25Retriever
from retrieval.reranker import rerank_documents
from collections import defaultdict


from database import get_conn

def search_pgvector(query, user_id, k=8):
    if not user_id:
        return []
        
    embeddings = get_embedding_model()
    query_vector = embeddings.embed_query(query)

    from langchain_core.documents import Document

    with get_conn() as conn:
        with conn.cursor() as cur:
            # <=> is the cosine distance operator in pgvector
            cur.execute("""
                SELECT content, paper_id, paper_title, page, 1 - (embedding <=> %s::vector) AS similarity
                FROM document_embeddings
                WHERE user_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_vector, user_id, query_vector, k))
            
            results = []
            for r in cur.fetchall():
                content = r[0]
                metadata = {
                    "paper_id": r[1],
                    "paper_title": r[2],
                    "page": r[3],
                    "score": r[4]
                }
                results.append(Document(page_content=content, metadata=metadata))
                
            return results


# ----------------------------------------
# Balance chunks across different papers
# ----------------------------------------
def balance_papers(docs, max_chunks_per_paper=2):

    paper_buckets = defaultdict(list)

    for doc in docs:
        paper = doc.metadata.get("paper_id", "unknown")
        paper_buckets[paper].append(doc)

    balanced = []

    for paper, chunks in paper_buckets.items():
        balanced.extend(chunks[:max_chunks_per_paper])

    return balanced


# ----------------------------------------
# Hybrid Retrieval
# ----------------------------------------
def retrieve_documents(query, k=5, user_id=None):

    bm25 = BM25Retriever(user_id=user_id)

    # pgvector ANN search
    vec_results = search_pgvector(query, user_id, k=8)

    # Keyword search
    bm25_results = bm25.search(query, k=8)

    # Merge results
    combined = list({id(doc): doc for doc in vec_results + bm25_results}.values())

    # ---------------------------------
    # Paper balancing
    # ---------------------------------
    combined = balance_papers(combined)

    # ---------------------------------
    # Cross-Encoder Reranking
    # ---------------------------------
    combined = rerank_documents(query, combined)

    # ---------------------------------
    # Section importance scoring
    # ---------------------------------
    rescored = []

    for doc in combined:

        importance = section_score(doc.page_content)

        rescored.append((doc, importance))

    rescored.sort(key=lambda x: x[1], reverse=True)

    final_docs = [doc for doc, score in rescored[:k]]

    return final_docs


# ----------------------------------------
# Context Builder with Citation Grounding
# ----------------------------------------
def build_context(docs):

    context = ""

    for i, d in enumerate(docs):

        title = d.metadata.get("paper_title", "Unknown Paper")
        page = d.metadata.get("page", "?")

        context += f"\n[Paper: {title} | Page {page}]\n"
        context += d.page_content
        context += "\n\n"

    return context