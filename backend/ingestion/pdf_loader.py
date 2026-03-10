from langchain_community.document_loaders import PyPDFLoader
import os 

def load_pdfs(data_folder):

    documents = []

    for file in os.listdir(data_folder):

        if file.endswith(".pdf"):

            path = os.path.join(data_folder, file)

            loader = PyPDFLoader(path)
            pages = loader.load()

            paper_id = file.replace(".pdf", "")
            paper_title = paper_id.replace("_", " ")

            for page in pages:

                page.metadata["paper_id"] = paper_id
                page.metadata["paper_title"] = paper_title

                documents.append(page)

    return documents