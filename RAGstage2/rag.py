import os
import shutil
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from google import genai
from rank_bm25 import BM25Okapi
import cohere

from config.prompts import PROMPTS


# CONFIG

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIRECTORY = BASE_DIR / "chroma_db"
DOCUMENTS_DIRECTORY = BASE_DIR / "documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "gemini-2.0-flash"

def list_pdf_paths():
    pdf_paths = sorted(DOCUMENTS_DIRECTORY.glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in {DOCUMENTS_DIRECTORY}"
        )

    return pdf_paths


# CLIENTS 

@lru_cache(maxsize=1)
def get_genai_client():
    api_key = os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)

@lru_cache(maxsize=1)
def get_cohere_client():
    return cohere.Client(os.getenv("CO_API_KEY"))

@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# PDF LOADING

def load_pdf(path):
    reader = PdfReader(path)
    documents = []

    for page_num, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()

        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": Path(path).name, #show the name of the pdf 
                    "page": page_num + 1
                }
            )
        )

    if not documents:
        raise ValueError("No readable text found in PDF.")

    return documents


# CHUNKING

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)


# VECTOR DB

def build_vectorstore(force_rebuild=False):
    if force_rebuild and PERSIST_DIRECTORY.exists():
        shutil.rmtree(PERSIST_DIRECTORY)

    pdf_paths = list_pdf_paths()

    docs = []
    for path in pdf_paths:
        docs.extend(load_pdf(path))
        chunks = split_documents(docs)

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(PERSIST_DIRECTORY)
    )

def get_vectorstore():
    if PERSIST_DIRECTORY.exists():
        return Chroma(
            persist_directory=str(PERSIST_DIRECTORY),
            embedding_function=get_embeddings()
        )
    return build_vectorstore()


# HYBRID RETRIEVAL 

def setup_bm25(chunks):
    texts = [doc.page_content for doc in chunks]
    tokenized = [text.split() for text in texts]
    return BM25Okapi(tokenized), texts

def hybrid_retrieve(query, vectorstore, chunks, bm25, k=10):

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    vector_docs = vector_retriever.invoke(query)

    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]

    bm25_docs = [chunks[i] for i in top_indices]

    combined = vector_docs + bm25_docs

    unique_docs = list({doc.page_content: doc for doc in combined}.values())

    return unique_docs


# RERANK 

def rerank(query, docs, top_n=3):
    co = get_cohere_client()

    texts = [doc.page_content for doc in docs]

    results = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=texts,
        top_n=top_n
    )

    return [docs[r.index] for r in results.results]


# QUERY EXPANSION 

def expand_query(query):
    prompt = PROMPTS["query_expansion"].format(question=query)

    response = get_genai_client().models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt
    )

    return response.text.strip()


# MAIN QA FUNCTION

def ask_question(question, chunks, bm25):

    vectorstore = get_vectorstore()

    # Expand query
    expanded_query = expand_query(question)
    print("\nExpanded Query:", expanded_query)

    # Hybrid retrieval
    retrieved_docs = hybrid_retrieve(
        expanded_query,
        vectorstore,
        chunks,
        bm25
    )

    # Rerank
    docs = rerank(question, retrieved_docs)

    # Build context
    context = ""
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "Unknown")
        context += f"[Source {i+1} - Page {page}]\n{doc.page_content}\n\n"

    prompt = PROMPTS["rag_answer"].format(
        context=context,
        question=question
    )

    response = get_genai_client().models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt
    )

    return response.text

# INIT ONCE 

def initialize():
    vectorstore = get_vectorstore()

    # get chunks back from DB
    chunks = vectorstore._collection.get()["documents"]

    # convert back to Document format
    docs = [
        Document(page_content=text, metadata={})
        for text in chunks
    ]

    bm25, _ = setup_bm25(docs)

    return docs, bm25


# CHAT LOOP

if __name__ == "__main__":

    print("Initializing system...")
    chunks, bm25 = initialize()

    print("Ready!")

    while True:
        query = input("\nAsk a question (type exit to quit): ")

        if query.lower() == "exit":
            break

        answer = ask_question(query, chunks, bm25)

        print("\nAnswer:\n")
        print(answer)