from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import CHAT_MODEL, OPENAI_API_KEY
from src.models.base import BaseModel


class OpenAIModel(BaseModel):
    def __init__(self):
        self.api_key = OPENAI_API_KEY

    def get_llm(self, temperature: float):
        return ChatOpenAI(model=CHAT_MODEL, temperature=temperature)

    def get_embeddings(self):
        return OpenAIEmbeddings(model="text-embedding-3-small")

    def is_configured(self) -> bool:
        return self.api_key is not None


openai_model = OpenAIModel()
