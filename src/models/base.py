from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def get_llm(self, temperature: float):
        pass

    @abstractmethod
    def get_embeddings(self):
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        pass
