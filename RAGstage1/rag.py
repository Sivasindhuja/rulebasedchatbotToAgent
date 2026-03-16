import os
from dotenv import load_dotenv

from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma

from google import genai


# Load gemini API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Step 1: Load PDF using pypdf
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

text = load_pdf("satcom-ngp.pdf")

print("Document loaded")


# Step 2: Chunking
#using Recursive splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))


# Step 3: Embeddings
#other options,openAI,openrouter etc

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 4: Create ChromaDB

vectorstore = Chroma.from_texts(
    texts=chunks,
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


# -----------------------------
# Step 6: Ask Question
# -----------------------------

def ask_question(question):
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
Answer the question using ONLY the context below.
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