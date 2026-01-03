from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEndpointEmbeddings,
)

from src.config import CHAT_MODEL, HF_API_KEY
from src.models.base import BaseModel


class HuggingFaceModel(BaseModel):
    def __init__(self):
        self.api_key = HF_API_KEY

    def get_llm(self, temperature: float):
        llm = HuggingFaceEndpoint(
            repo_id=CHAT_MODEL,
            temperature=temperature,
            huggingfacehub_api_token=self.api_key,
            task="text-generation",
        )
        return ChatHuggingFace(llm=llm)

    def get_embeddings(self):
        return HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=self.api_key,
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
        )

    def is_configured(self) -> bool:
        return self.api_key is not None


hf_model = HuggingFaceModel()
