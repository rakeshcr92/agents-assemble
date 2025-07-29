from abc import ABC, abstractmethod
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict
from datetime import datetime


class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"agent.{name}")
        
        self.logger.info(f"Agent {self.name} initialized")

    @abstractmethod
    async def process(self, input_data: Dict[str, Any]):
        """
        Process input and return structured output
        
        This is the main method each agent must implement.
        
        Args:
            input_data: Dictionary containing request data
            
        Returns:
            Dictionary with processed results
        """
        pass

    def _create_response(self, data: Dict[str, Any], status: str = "success") -> Dict[str, Any]:
        """
        Create standardized response format
        
        Why needed: All agents should return responses in the same format
        so the orchestrator can handle them consistently.
        
        Args:
            data: The actual response data
            status: "success" or "error"
            
        Returns:
            Standardized response dictionary
        """
        return {
            "agent": self.name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }

    def _handle_error(self, error: Exception) -> Dict[str, Any]:
        """
        Handle errors in a consistent way
        
        Why needed: When something goes wrong, we need consistent error responses
        and proper logging so we can debug issues.
        
        Args:
            error: The exception that occurred
            
        Returns:
            Standardized error response
        """
        error_msg = str(error)
        self.logger.error(f"Error in {self.name}: {error_msg}")
        
        return self._create_response(
            {"error": error_msg}, 
            status="error"
        )

    async def health_check(self) -> bool:
        """
        Simple health check
        
        Why needed: We need to know if an agent is working properly
        before sending requests to it.
        
        Returns:
            True if agent is healthy, False otherwise
        """
        try:
            # Basic check - just return True for now
            # Subclasses can override for more complex checks
            return True
        except Exception as e:
            self.logger.error(f"Health check failed for {self.name}: {e}")
            return False