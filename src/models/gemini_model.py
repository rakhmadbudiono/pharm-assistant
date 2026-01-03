from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.config import CHAT_MODEL, GOOGLE_API_KEY
from src.models.base import BaseModel


class GeminiModel(BaseModel):
    def __init__(self):
        self.api_key = GOOGLE_API_KEY

    def get_llm(self, temperature: float):
        return ChatGoogleGenerativeAI(
            model=CHAT_MODEL,
            temperature=temperature,
        )

    def get_embeddings(self):
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def is_configured(self) -> bool:
        return self.api_key is not None


gemini_model = GeminiModel()
