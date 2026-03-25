from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

# 1. Load the PDF
print("Loading Contract...")
loader = PyPDFLoader("Muhammad_Azam_Contract_.pdf")
docs = loader.load()

# 2. Split text into small chunks (easier for AI to find specific facts)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

# 3. Create Embeddings (The 'Mathematical' version of your text)
# This runs locally on your CPU
print("Creating Embeddings (this may take a moment)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Save to ChromaDB (Local Folder)
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./zayan_db"
)

print("✅ SUCCESS: Contract data is now stored in ./zayan_db")