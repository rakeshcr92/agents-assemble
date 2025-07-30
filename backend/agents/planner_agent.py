from .base_agent import BaseAgent
from typing import Any, Dict, List

class PlannerAgent(BaseAgent):
    """
    The Planner Agent is the "brain" of the system.
    It analyzes user input and decides what other agents should do.
    """

    def __init__(self):
        super().__init__('planner')

        # Define patterns for intent recognition (TODO: Use a more robust NLP library in future or use AI for this)
        # Keywords that indicate user wants to STORE a memory
        self.store_keywords = [
            "remember", "save", "capture", "record", "store", "note",
            "I met", "I went", "I did", "I saw", "happened", "today",
            "yesterday", "just", "had a", "attended"
        ]
        
        # Keywords that indicate user wants to QUERY/SEARCH memories
        self.query_keywords = [
            "who", "what", "when", "where", "how", "recall", "find",
            "tell me about", "remind me", "what did", "who was",
            "do you remember", "can you find", "search for"
        ]

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main planning logic: analyze input and create execution plan
        
        This function:
        1. Looks at the user's text input
        2. Decides if they want to store or query memory
        3. Determines which agents should handle the request
        4. Creates a plan for executing the task
        """

        try:
            # Get the user's text input
            user_text = input_data.get('text', '').lower()

            #Step 1: Figure out what the user wants to do
            # intent = self._classify_intent(user_text)
            intent, confidence = self._classify_intent_with_confidence(user_text)

            # Step 2: Decide which agents are needed for this intent
            agents_needed = self._select_agents(intent, input_data)
            
            # Step 3: Create execution plan
            execution_plan = self._create_execution_plan(intent, agents_needed)

            # Return the plan
            plan_data = {
                "intent": intent,
                "confidence": confidence,
                "agents_needed": agents_needed,
                "execution_plan": execution_plan,
                "user_text": input_data.get("text", "")
            }
            
            return self._create_response(plan_data)

        except Exception as e:
            return self._handle_error(e)
    
    # def _classify_intent(self, text: str) -> str:
    #     """
    #     Classify what the user wants to do based on their text
        
    #     Why we need this: The system needs to know if user wants to:
    #     - Store a new memory ("I just met Jennifer...")
    #     - Query existing memories ("Who did I meet at the conference?")
    #     - Just have a conversation ("Hello, how are you?")
        
    #     Args:
    #         text: User's input text (in lowercase)
            
    #     Returns:
    #         One of: "store_memory", "query_memory", "general_conversation"
    #     """

    #     if not text.strip():
    #         return "general_conversation"

    #     # Count how many store vs query keywords we find (TODO: Should we use a more correct approach?)
    #     store_matches = sum(1 for keyword in self.store_keywords if keyword in text)
    #     query_matches = sum(1 for keyword in self.query_keywords if keyword in text)
        
    #     # Decision logic
    #     if store_matches > query_matches and store_matches > 0:
    #         return "store_memory"
    #     elif query_matches > 0:
    #         return "query_memory"
    #     else:
    #         return "general_conversation"
    
    def _classify_intent_with_confidence(self, text: str) -> Tuple[str, float]:
        """
        Classify what the user wants to do and return confidence score
        
        Args:
            text: User's input text (in lowercase)
            
        Returns:
            Tuple of (intent, confidence_score)
            - intent: One of "store_memory", "query_memory", "general_conversation"
            - confidence: Float between 0.0 and 1.0
        """
        # Handle empty text
        if not text.strip():
            return "general_conversation", 0.3
        
        # Count keyword matches
        store_matches = sum(1 for keyword in self.store_keywords if keyword in text)
        query_matches = sum(1 for keyword in self.query_keywords if keyword in text)
        
        total_words = len(text.split())
        
        # Calculate intent and confidence
        if store_matches > query_matches and store_matches > 0:
            # More store keywords found
            confidence = min((store_matches / max(total_words, 1)) * 5, 1.0)
            confidence = max(confidence, 0.6)  # Minimum confidence for matched keywords
            return "store_memory", confidence
            
        elif query_matches > 0:
            # Query keywords found
            confidence = min((query_matches / max(total_words, 1)) * 5, 1.0)
            confidence = max(confidence, 0.6)  # Minimum confidence for matched keywords
            return "query_memory", confidence
            
        else:
            # No specific keywords found - likely general conversation
            # Check for question patterns
            if any(q in text for q in ["?", "how", "why", "help"]):
                return "general_conversation", 0.7
            else:
                return "general_conversation", 0.5

    def _select_agents(self, intent: str, input_data: Dict[str, Any]) -> List[str]:
        """
        Decide which agents are needed based on the intent and available data
        
        Why we need this: Different tasks require different agents:
        - Storing memory might need: voice_agent (process speech) + memory_agent (save data)
        - Querying memory might need: memory_agent (search) + response_agent (format answer)
        - Photos need vision_agent to analyze them
        
        Args:
            intent: What the user wants to do
            input_data: Original input (might contain photos, audio, etc.)
            
        Returns:
            List of agent names that should handle this request
        """

        # Base agents needed for each intent
        agent_map = {
            "store_memory": ["memory_agent", "response_agent"],
            "query_memory": ["memory_agent", "response_agent"],
            "general_conversation": ["response_agent"]
        }

        # Start with base agents for this intent
        agents = agent_map.get(intent, ['response_agent']).copy()

        # Add voice agent if audio input is present
        if input_data.get("audio_data"):
            agents.insert(0, "voice_agent")  # Ensure voice agent is first for processing audio
        
        # TODO: ADD this functionality to handle photos
        if input_data.get("photo_url") or input_data.get("image_url"):
            # If there's a photo, we need vision processing
            agents.insert(-1, "vision_agent")  # Insert before the last agent
        
        return agents
    
    def _create_execution_plan(self, intent: str, agents: List[str]) -> Dict[str, Any]:
        """
        Create a detailed execution plan for the agents
        
        Why we need this: The orchestrator needs to know:
        - Which agents to run
        - In what order (some must run before others)
        - What each agent should do
        
        Args:
            intent: What the user wants to do
            agents: List of agents that will handle this
            
        Returns:
            Execution plan dictionary
        """
        
        # For most cases, agents should run sequentially (one after another)
        # because later agents often need results from earlier ones
        
        plan = {
            "type": "sequential",  # Run agents one after another
            "steps": []
        }
        
        # Create a step for each agent
        for agent_name in agents:
            step = {
                "agent": agent_name,
                "action": "process"
            }
            
            # Add specific instructions based on intent and agent
            if agent_name == "memory_agent":
                if intent == "store_memory":
                    step["instruction"] = "Save this as a new memory"
                elif intent == "query_memory":
                    step["instruction"] = "Search existing memories"
                    
            elif agent_name == "response_agent":
                step["instruction"] = "Generate conversational response"
                
            elif agent_name == "vision_agent":
                step["instruction"] = "Analyze any images in the input"
                
            elif agent_name == "voice_agent":
                step["instruction"] = "Process speech input"
            
            plan["steps"].append(step)
        
        return plan
    
    def get_supported_intents(self) -> List[str]:
        """
        Return list of intents this planner can handle
        
        Useful for debugging and system monitoring
        """
        return ["store_memory", "query_memory", "general_conversation"]
    
    def get_planning_stats(self) -> Dict[str, Any]:
        """
        Get statistics about planning (useful for debugging)
        """
        return {
            "supported_intents": self.get_supported_intents(),
            "store_keywords_count": len(self.store_keywords),
            "query_keywords_count": len(self.query_keywords)
        }