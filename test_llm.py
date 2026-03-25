from litellm import completion

response = completion(
    model="ollama/llama3.2:3b", 
    messages=[{"content": "Respond with 'Zayan AI System Online' if you hear me.", "role": "user"}],
    api_base="http://localhost:11434"
)
print(response.choices[0].message.content)