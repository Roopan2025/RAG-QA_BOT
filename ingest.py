import os

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import config

# PDF LOAD
def load_pdf(pdf_directory: str):
    if not os.path.exists(pdf_directory):
        print(f"PDF directory '{pdf_directory}' does not exist.")
        return []

    all_pdf_docs = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_directory, filename)
            # print("filepath : ",filepath)
            try:
                loder = PyMuPDF4LLMLoader(filepath)
                documents = loder.load()
                all_pdf_docs.extend(documents)
            except Exception as e:
                print(f"PDF file {filepath} could not be loaded. Error: {e}")

    if not all_pdf_docs:
        print("No PDF documents found or loaded in the specified directory.")
    else:
        print(f"Loaded {len(all_pdf_docs)} pages from PDF documents.")
    return all_pdf_docs


def ingest_all_docs(chunk_size: int,
                    chunk_overlap: int,
                    pdf_directory: str = "data",
                    persist_directory: str = "docs/chroma"):

    # 1. Load PDF
    pdf_docs = load_pdf(pdf_directory)

    # 2. Data Splitting(Chunk)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunked_docs = splitter.split_documents(pdf_docs)
    print(f"Split documents into {len(chunked_docs)} chunks.")

    # 3. Embeddings
    model_name = config.EMBEDDING_MODEL_NAME

    print(f"Initializing embeddings with model: {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # 4. Create and Persist Vector DB
    os.makedirs(persist_directory, exist_ok=True)

    vector_db = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )




if __name__ == "__main__":
   ingest_all_docs(chunk_size = config.CHUNK_SIZE,
                   chunk_overlap = config.CHUNK_OVERLAP,
                   pdf_directory = config.PDF_SOURCE_DIRECTORY,
                   persist_directory = config.PERSIST_DIRECTORY)

