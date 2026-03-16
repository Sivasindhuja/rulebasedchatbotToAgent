import os
from dotenv import load_dotenv

from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma

from langchain_core.documents import Document

from google import genai


# Load gemini API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
#create a google gen ai client
client = genai.Client(api_key=api_key)

# Step 1: Load PDF using pypdf
def load_pdf(path):
    reader = PdfReader(path)
    documents = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path,
                    "page": page_num + 1
                }
            )
        )

    return documents

docs= load_pdf("satcom-ngp.pdf")
# print(docs)

print("Document loaded")


# Step 2: Chunking
#using Recursive splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(docs)

print("Total chunks:", len(chunks))
# print(chunks)


# Step 3: Embeddings
#other options,openAI,openrouter etc

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 4: Create ChromaDB3
vectorstore = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)



print("Vector DB created")



# Step 5: Retriever
#retrievs top k relevant chunks
#k=3 here
#dense vector retrieval

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)
print(retriever)


# Step 6: Ask Question

def ask_question(question):
    docs = retriever.invoke(question)
    # context = "\n\n".join([doc.page_content for doc in docs])
    context = ""
    for i, doc in enumerate(docs):
        context += f"[Source {i+1}]\n{doc.page_content}\n\n"
    prompt = f"""
Answer the question using ONLY the context below.
Answer the question using ONLY the context.

Cite the source number like [Source 1].
If the answer is not in the context say "I don't know".
Context:
{context}
Question:
{question}
"""
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )
    return response.text



# Chat Loop

while True:
    query = input("\nAsk a question (type exit to quit): ")
    if query.lower() == "exit":
        break
    answer = ask_question(query)
    print("\nAnswer:")
    print(answer)