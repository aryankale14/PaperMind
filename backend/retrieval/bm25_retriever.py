from rank_bm25 import BM25Okapi
from database import get_conn
from langchain_core.documents import Document


class BM25Retriever:

    def __init__(self, user_id=None):
        self.docs = []
        if not user_id:
            self.bm25 = None
            return

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT content, paper_id, paper_title, page
                    FROM document_embeddings
                    WHERE user_id = %s
                """, (user_id,))
                
                for r in cur.fetchall():
                    content = r[0]
                    metadata = {"paper_id": r[1], "paper_title": r[2], "page": r[3]}
                    self.docs.append(Document(page_content=content, metadata=metadata))

        if not self.docs:
            self.bm25 = None
            return

        self.texts = [d.page_content for d in self.docs]
        self.tokenized = [t.lower().split() for t in self.texts]

        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query, k=5):
        if not self.bm25:
            return []

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(self.docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, _ in ranked[:k]]