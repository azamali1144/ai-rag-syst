import os
import shutil
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from litellm import completion
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = FastAPI(title="Zayan AI Assistant")
templates = Jinja2Templates(directory="templates")

# 1. Setup Qdrant & Embeddings
client = QdrantClient(url="http://localhost:6333")
collection_name = "zayan_docs"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Ensure collection exists on startup
def init_db():
    try:
        collections = client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            print(f"Collection {collection_name} created!")
    except Exception as e:
        print(f"DB Init Error: {e}")

init_db()

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
   return templates.TemplateResponse(
        request=request, 
        name="index.html"
    )

# --- NEW: THE INDEXER FEATURE ---
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Save file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 1. Load and Split PDF
    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    # 2. Convert to Vectors and Upload to Qdrant
    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk.page_content)
        client.upsert(
            collection_name=collection_name,
            points=[{
                "id": i,
                "vector": vector,
                "payload": {"text": chunk.page_content, "source": file.filename}
            }]
        )
    
    os.remove(temp_path) # Clean up
    return {"status": "Success", "message": f"Indexed {len(chunks)} sections from {file.filename}"}

@app.get("/ask")
async def ask_contract(query: str):
    try:
        # Search Qdrant
        query_vector = embeddings.embed_query(query)
        
        # FIX: We use query_points which is the standard for the latest client
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=3
        )
        
        # Extract text from the search results
        context = "\n".join([res.payload["text"] for res in search_result]) if search_result else "No context found."

        def generate():
            response = completion(
                model="ollama/llama3.2", 
                messages=[
                    {"role": "system", "content": f"You are Zayan Legal AI. Use this context: {context}"},
                    {"role": "user", "content": query}
                ],
                stream=True,
                api_base="http://localhost:11434"
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        return StreamingResponse(generate(), media_type="text/plain")
    except Exception as e:
        # This will help us see exactly what went wrong in the UI if there is another error
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)