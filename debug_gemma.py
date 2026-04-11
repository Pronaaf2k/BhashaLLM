import requests

LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"

prompt = "Translate the following English into Bangla: 'Hello world'"
payload = {
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.3,
    "top_p": 0.9,
    "max_tokens": 50
}

response = requests.post(LLAMA_SERVER_URL, json=payload)
print(response.json())
