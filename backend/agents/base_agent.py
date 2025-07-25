from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def run(self, input_text: str) -> str:
        """Run the agent on the input text and return a response."""
        pass
