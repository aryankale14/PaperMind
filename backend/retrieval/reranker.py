from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_documents(query, docs, top_k=5):

    pairs = []

    for d in docs:
        pairs.append((query, d.page_content))

    scores = reranker.predict(pairs)

    ranked = list(zip(docs, scores))
    ranked.sort(key=lambda x: x[1], reverse=True)

    reranked_docs = [doc for doc, score in ranked]

    return reranked_docs[:top_k]