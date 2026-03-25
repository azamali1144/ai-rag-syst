import sys
from langchain_ollama import OllamaLLM

def test_setup():
    print(f"--- Environment Check ---")
    print(f"Python Version: {sys.version}")
    
    try:
        # Initialize the model
        llm = OllamaLLM(model="llama3.1:latest")
        
        print("\nConnecting to Llama 3.1 via Ollama...")
        # A prompt based on your actual contract duties
        query = "As an AI Engineer at Zayan Technologies, what is my first priority?"
        
        response = llm.invoke(query)
        
        print("-" * 50)
        print(f"AI RESPONSE:\n{response}")
        print("-" * 50)
        print("\n✅ SUCCESS: Your AI Development environment is ready!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("Tip: Ensure the 'Ollama' application is running in your Windows System Tray.")

if __name__ == "__main__":
    test_setup()