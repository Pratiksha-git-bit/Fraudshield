import os, requests

API_KEY = "gsk_paste_your_groq_key_here"

r = requests.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"model": "llama-3.3-70b-versatile",
          "messages": [{"role": "user", "content": "Say OK."}],
          "temperature": 0},
    timeout=60,
)
print(r.status_code)
print(r.json()["choices"][0]["message"]["content"])