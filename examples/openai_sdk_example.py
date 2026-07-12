"""Point the official OpenAI SDK at your local localgate instance."""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="lg_your_key_here")

resp = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
