from retrieval.retriever import retrieve_documents


def collect_evidence(queries, user_id=None):

    all_docs = []

    for q in queries:

        docs = retrieve_documents(q, user_id=user_id)

        all_docs.extend(docs)

    # ------------------------------------
    # Remove duplicate chunks safely
    # ------------------------------------
    seen = set()
    unique_docs = []

    for doc in all_docs:

        content = doc.page_content.strip()

        if content not in seen:
            unique_docs.append(doc)
            seen.add(content)

    return unique_docs