from openai import OpenAI

client = OpenAI(base_url="https://localhost:11/v1/", api_key="ollama")

chat_completion = client.chat.completions.create(
    messages=[{"role": "user", "content": "你好，请介绍下自己。"}],
    model="llama3",
)

chat_completion.choices[0]
