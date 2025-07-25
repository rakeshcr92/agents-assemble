from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def run(self, input_data: str) -> str:
        """Runs the agent logic and returns a string output."""
        pass
