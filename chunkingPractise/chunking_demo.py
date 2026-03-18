from pypdf import PdfReader

def load_pdf(path):
    reader = PdfReader(path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text

text = load_pdf("./chunkingPractise/satcom-ngp.pdf")

print("Total characters:", len(text))


#fixed length chunking with overlap

def fixed_length_chunking(text, chunk_size=500, chunk_overlap=0):
    chunks = []

    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i+chunk_size])
    
    return chunks


fixed_chunks = fixed_length_chunking(text)

# print("\n--- Fixed Length Chunking ---")
# print("Total chunks:", len(fixed_chunks))
# print(fixed_chunks[0])


#recursive charcater chunking

from langchain_text_splitters import RecursiveCharacterTextSplitter

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

recursive_chunks = recursive_splitter.split_text(text)

# print("\n--- Recursive Chunking ---")
# print("Total chunks:", len(recursive_chunks))
# print(recursive_chunks[0])


#with custom seperators

recursive_splitter_custom = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)

custom_chunks = recursive_splitter_custom.split_text(text)

# print("\n--- Recursive Splitting (Custom) ---")
# print("Total chunks:", len(custom_chunks))
# print(custom_chunks[0])

#semantic chunking

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

semantic_splitter = SemanticChunker(embeddings)

semantic_chunks = semantic_splitter.split_text(text)

# print("\n--- Semantic Chunking ---")
# print("Total chunks:", len(semantic_chunks))
# print(semantic_chunks[0])

def show_chunks(chunks, name):
    print(f"\n{name}")
    print("Total chunks:", len(chunks))

    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}")
        print(chunk[:300])


show_chunks(fixed_chunks, "Fixed Length")
show_chunks(recursive_chunks, "Recursive")
show_chunks(custom_chunks, "Recursive Custom")
show_chunks(semantic_chunks, "Semantic")