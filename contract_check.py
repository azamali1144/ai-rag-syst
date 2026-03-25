from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM

# 1. Load the contract PDF
print("Reading full contract pages...")
loader = PyPDFLoader("Muhammad_Azam_Contract_.pdf")
pages = loader.load()

# 2. Combine the first 3 pages into one big context string
# (This ensures we catch the header and the 'Appointment' section)
full_context = "\n".join([p.page_content for p in pages[:3]])

# 3. Initialize Llama 3.1
llm = OllamaLLM(model="llama3.1:latest")

# 4. A more specific prompt
query = "Identify the 'Employer' (Company Name) and the 'Employee' name and 'Job Title' from this contract text."
prompt = f"""
Use the following contract text to answer the question accurately.

TEXT:
{full_context}

QUESTION:
{query}
"""

print("Analyzing with Llama 3.1...")
print("-" * 30)
print(llm.invoke(prompt))
print("-" * 30)