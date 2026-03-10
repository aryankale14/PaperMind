from langchain_text_splitters import RecursiveCharacterTextSplitter
import re


def clean_scientific_text(text: str) -> str:
    text = re.split(r"\bReferences\b", text, flags=re.IGNORECASE)[0]
    text = re.sub(r"Correspondence should be addressed.*", "", text)
    text = re.sub(r"Received .* Accepted .* Published .*", "", text)
    text = re.split(r"\bAcknowledgments\b", text, flags=re.IGNORECASE)[0]
    text = re.sub(r"Downloaded from.*", "", text)
    text = re.sub(r"Copyright .*", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def filter_chunks(chunks):

    filtered = []

    for chunk in chunks:
        text = chunk.page_content.strip()

        if len(text) < 200:
            continue

        if text.count(", vol.") > 3:
            continue

        filtered.append(chunk)

    print(f"Filtered to {len(filtered)} high-quality chunks")

    return filtered


def chunk_documents(documents):

    cleaned_docs = []

    for doc in documents:

        cleaned_text = clean_scientific_text(doc.page_content)
        doc.page_content = cleaned_text

        cleaned_docs.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = splitter.split_documents(cleaned_docs)

    # -----------------------------
    # ADD PAPER METADATA HERE
    # -----------------------------
    for i, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = i

        if "source" in chunk.metadata:
            paper_id = chunk.metadata["source"].split("/")[-1]

            chunk.metadata["paper_id"] = paper_id
            chunk.metadata["paper_title"] = paper_id.replace(".pdf", "").replace("_", " ")

    chunks = filter_chunks(chunks)

    print(f"Created {len(chunks)} cleaned chunks")

    return chunks