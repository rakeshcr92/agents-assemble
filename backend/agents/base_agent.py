from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def process(self, input_data: str) -> str:
        pass
