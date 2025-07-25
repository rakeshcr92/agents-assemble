from google.adk.agents import BaseAgent
from abc import abstractmethod

class MyBaseAgent(BaseAgent):
    @abstractmethod
    def run(self, input: str) -> str:
        pass
