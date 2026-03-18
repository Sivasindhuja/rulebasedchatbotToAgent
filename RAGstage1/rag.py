import argparse
import os
import shutil
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


load_dotenv()

RAG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RAG_DIR.parent
PERSIST_DIRECTORY = RAG_DIR / "chroma_db"
DOCUMENTS_DIRECTORY = PROJECT_ROOT / "documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "gemini-2.5-flash"


def list_pdf_paths() -> list[Path]:
    pdf_paths = sorted(DOCUMENTS_DIRECTORY.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files were found in {DOCUMENTS_DIRECTORY}."
        )
    return pdf_paths


PDF_PATHS = list_pdf_paths()


def load_pdf(path: str | Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def build_documents(pdf_paths: list[Path]) -> list[Document]:
    documents: list[Document] = []
    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "file_path": str(pdf_path),
                        "page": page_number,
                    },
                )
            )
    if not documents:
        raise ValueError("PDF files were found, but no extractable text was available.")
    return documents


def build_chunks(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    return splitter.split_documents(documents)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"local_files_only": True},
    )


@lru_cache(maxsize=1)
def get_genai_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Add it to your environment or .env file.")
    return genai.Client(api_key=api_key)


def build_vectorstore(force_rebuild: bool = False) -> Chroma:
    embeddings = get_embeddings()
    if force_rebuild and PERSIST_DIRECTORY.exists():
        shutil.rmtree(PERSIST_DIRECTORY)

    documents = build_documents(PDF_PATHS)
    chunks = build_chunks(documents)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIRECTORY),
    )


def get_vectorstore() -> Chroma:
    embeddings = get_embeddings()
    if PERSIST_DIRECTORY.exists() and any(PERSIST_DIRECTORY.iterdir()):
        return Chroma(
            persist_directory=str(PERSIST_DIRECTORY),
            embedding_function=embeddings,
        )
    return build_vectorstore()


def retrieve_context(question: str, k: int = 3) -> list:
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": k})
    return retriever.invoke(question)


def ask_question(question: str, k: int = 3) -> dict:
    docs = retrieve_context(question, k=k)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"""
Answer the question using only the context below.
Answer the question using ONLY the context.

Cite the source number like [Source 1].
If the answer is not in the context say "I don't know".

Context:
{context}

Question:
{question}
"""
    response = get_genai_client().models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )
    return {
        "answer": response.text,
        "context": docs,
    }


def describe_corpus() -> str:
    names = ", ".join(path.name for path in PDF_PATHS)
    return f"{len(PDF_PATHS)} PDF files indexed from {DOCUMENTS_DIRECTORY}: {names}"


def run_ingestion(force_rebuild: bool = False) -> None:
    vectorstore = build_vectorstore(force_rebuild=force_rebuild)
    count = vectorstore._collection.count()
    print(f"Built Chroma vector database at {PERSIST_DIRECTORY}")
    print(describe_corpus())
    print(f"Stored {count} chunks.")


def run_cli() -> None:
    while True:
        query = input("\nAsk a question (type exit to quit): ").strip()
        if query.lower() == "exit":
            break
        result = ask_question(query)
        print("\nAnswer:")
        print(result["answer"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAGStage1 PDF ingestion and CLI.")
    parser.add_argument(
        "--build-db",
        action="store_true",
        help="Build the persisted Chroma vector database from all PDFs in documents/.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete the existing Chroma database before rebuilding it.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.build_db:
        run_ingestion(force_rebuild=args.force_rebuild)
    else:
        run_cli()
