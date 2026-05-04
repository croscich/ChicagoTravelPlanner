"""
Step 1 of RAG: Load PDFs → Chunk text → Create embeddings → Save FAISS index
"""
#pip install langchain langchain-community langchain-text-splitters langchain-openai openai faiss-cpu pypdf
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# --- Config ---
PDF_FOLDER = "rag_files"
FAISS_INDEX_PATH = "faiss_index"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # set this in your environment

# --- Step 1: Load PDFs ---
print("Loading PDFs...")
documents = []
for filename in os.listdir(PDF_FOLDER):
    if filename.endswith(".pdf"):
        filepath = os.path.join(PDF_FOLDER, filename)
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        documents.extend(docs)
        print(f"  Loaded: {filename} ({len(docs)} pages)")

print(f"Total pages loaded: {len(documents)}")

# --- Step 2: Split into chunks ---
print("\nSplitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,  # characters per chunk (500 characters = ~125 tokens = 80-100 words)
    chunk_overlap=5,  # overlap between chunks to preserve context
)
chunks = text_splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")

# Print all chunks
# This is here so we can see what happened with the documents
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Source: {chunk.metadata['source']}, Page: {chunk.metadata['page']}")
    print(chunk.page_content)

# --- Step 3: Create embeddings and build FAISS index ---
print("\nCreating embeddings and building FAISS index...")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(chunks, embeddings)

# --- Step 4: Save the faiss file ---
vectorstore.save_local(FAISS_INDEX_PATH)
print(f"\nDone! FAISS index saved to: '{FAISS_INDEX_PATH}/'")

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")