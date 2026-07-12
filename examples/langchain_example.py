"""Integrate localgate with LangChain via its OpenAI-compatible endpoint."""
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(base_url="http://localhost:8000/v1", api_key="lg_your_key_here", model="llama3")
print(llm.invoke("Hello!"))
