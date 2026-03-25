# RAG-Based Contract Intelligence System: Private Contract Analysis Assistant 📑🤖

This is a high-performance, privacy-focused RAG (Retrieval-Augmented Generation) system designed for legal contract analysis. It allows users to upload PDF contracts, index them into a vector database, and perform context-aware Q&A locally.

---

## 🚀 Features
- **Local-First Privacy:** Powered by **Ollama (Llama 3.2)**—no data leaves your machine.
- **High-Speed Retrieval:** Uses **Qdrant** as a vector database for millisecond document searches.
- **Contract-Aware:** Specialized prompts for identifying clauses, risks, and obligations in legal PDFs.
- **Modern Stack:** Built with **FastAPI**, **LangChain**, and **Python 3.12+**.
- **Interactive UI:** Simple web interface for document chatting and history.

---

## 🛠️ Tech Stack
| Component | Technology |
| :--- | :--- |
| **LLM** | Ollama (Llama 3.2 / Mistral) |
| **Embeddings** | FastEmbed / HuggingFace |
| **Vector DB** | Qdrant (Dockerized) |
| **Backend** | FastAPI / Python |
| **Orchestration** | LangChain |

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash  
git clone https://github.com/azamali1144/ai-rag-syst.git  
cd ai-rag-syst
```
### 2. Infrastructure (Vector DB)
```bash  
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 3. Local LLM Setup
```bash  
ollama pull llama3.2
```

### 4. Python Environment
```bash  
pip install -r requirements.txt
# OR
poetry install
```

### 5. Run the Application
```bash  
python main.py
```
Visit http://localhost:8000 in your browser.
