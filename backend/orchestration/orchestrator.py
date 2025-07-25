"""
Main orchestrator for coordinating multiple agents using LangGraph and LangChain.
Manages the flow of tasks between different specialized agents.
"""

from typing import Dict, Any, List, Optional, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain.schema import BaseMessage
import logging
from enum import Enum

from .event_bus import EventBus
from .task_queue import TaskQueue
from .agent_pool import AgentPool
from ..agents.base_agent import BaseAgent
from ..models.agent_models import AgentState, TaskRequest, TaskResponse


class AgentType(Enum):
    PLANNER = "planner"
    VOICE = "voice"
    VISION = "vision"
    CONTEXT = "context"
    MEMORY = "memory"
    INSIGHT = "insight"
    RESPONSE = "response"


class OrchestratorState(TypedDict):
    """State shared across all agents in the orchestration graph."""
    user_input: str
    intent: Optional[str]
    extracted_entities: Dict[str, Any]
    context_data: Dict[str, Any]
    memories: List[Dict[str, Any]]
    insights: List[Dict[str, Any]]
    response: Optional[str]
    current_agent: Optional[str]
    task_history: List[Dict[str, Any]]
    error: Optional[str]
    metadata: Dict[str, Any]


class LifeWitnessOrchestrator:
    """
    Main orchestrator that coordinates multiple specialized agents using LangGraph.
    """
    
    def __init__(self, gemini_api_key: str, config: Dict[str, Any]):
        self.gemini_api_key = gemini_api_key
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.event_bus = EventBus()
        self.task_queue = TaskQueue()
        self.agent_pool = AgentPool(gemini_api_key, config)
        
        # Initialize LangChain Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=gemini_api_key,
            temperature=0.3,
            convert_system_message_to_human=True
        )
        
        # Build the orchestration graph
        self.graph = self._build_orchestration_graph()
        
    def _build_orchestration_graph(self) -> StateGraph:
        """Build the LangGraph state graph for agent orchestration."""
        workflow = StateGraph(OrchestratorState)
        
        # Add nodes for each agent type
        workflow.add_node("planner", self._run_planner_agent)
        workflow.add_node("voice_processor", self._run_voice_agent)
        workflow.add_node("vision_processor", self._run_vision_agent)
        workflow.add_node("context_gatherer", self._run_context_agent)
        workflow.add_node("memory_manager", self._run_memory_agent)
        workflow.add_node("insight_analyzer", self._run_insight_agent)
        workflow.add_node("response_generator", self._run_response_agent)
        workflow.add_node("error_handler", self._handle_error)
        
        # Define the workflow edges
        workflow.set_entry_point("planner")
        
        # Conditional routing based on planner output
        workflow.add_conditional_edges(
            "planner",
            self._route_after_planning,
            {
                "voice": "voice_processor",
                "vision": "vision_processor",
                "context": "context_gatherer",
                "memory": "memory_manager",
                "error": "error_handler"
            }
        )
        
        # Sequential processing for multi-modal inputs
        workflow.add_edge("voice_processor", "context_gatherer")
        workflow.add_edge("vision_processor", "context_gatherer")
        workflow.add_edge("context_gatherer", "memory_manager")
        workflow.add_edge("memory_manager", "insight_analyzer")
        workflow.add_edge("insight_analyzer", "response_generator")
        workflow.add_edge("response_generator", END)
        workflow.add_edge("error_handler", END)
        
        return workflow.compile()
    
    def _route_after_planning(self, state: OrchestratorState) -> str:
        """Route to appropriate agent based on planner output."""
        try:
            intent = state.get("intent", "")
            
            if state.get("error"):
                return "error"
            elif "voice" in intent.lower():
                return "voice"
            elif "image" in intent.lower() or "photo" in intent.lower():
                return "vision"
            elif "search" in intent.lower() or "remember" in intent.lower():
                return "memory"
            else:
                return "context"
                
        except Exception as e:
            self.logger.error(f"Routing error: {e}")
            return "error"
    
    async def _run_planner_agent(self, state: OrchestratorState) -> OrchestratorState:
        """Execute the planner agent to analyze intent and plan tasks."""
        try:
            agent = await self.agent_pool.get_agent(AgentType.PLANNER)
            
            task_request = TaskRequest(
                task_type="plan",
                input_data={"user_input": state["user_input"]},
                context=state.get("metadata", {})
            )
            
            response = await agent.process_task(task_request)
            
            state["intent"] = response.result.get("intent")
            state["extracted_entities"] = response.result.get("entities", {})
            state["current_agent"] = "planner"
            
            # Emit planning event
            await self.event_bus.emit("agent.planning.completed", {
                "intent": state["intent"],
                "entities": state["extracted_entities"]
            })
            
            return state
            
        except Exception as e:
            self.logger.error(f"Planner agent error: {e}")
            state["error"] = str(e)
            return state
    
    async def _run_voice_agent(self, state: OrchestratorState) -> OrchestratorState:
        """Execute voice processing agent."""
        try:
            agent = await self.agent_pool.get_agent(AgentType.VOICE)
            
            task_request = TaskRequest(
                task_type="voice_process",
                input_data={"audio_data": state.get("user_input")},
                context=state.get("context_data", {})
            )
            
            response = await agent.process_task(task_request)
            state["context_data"]["voice_analysis"] = response.result
            state["current_agent"] = "voice"
            
            return state
            
        except Exception as e:
            self.logger.error(f"Voice agent error: {e}")
            state["error"] = str(e)
            return state
    
    async def _run_vision_agent(self, state: OrchestratorState) -> OrchestratorState:
        """Execute vision processing agent."""
        try:
            agent = await self.agent_pool.get_agent(AgentType.VISION)
            
            task_request = TaskRequest(
                task_type="vision_analyze",
                input_data={"image_data": state.get("user_input")},
                context=state.get("context_data", {})
            )
            
            response = await agent.process_task(task_request)
            state["context_data"]["vision_analysis"] = response.result
            state["current_agent"] = "vision"
            
            return state
            
        except Exception as e:
            self.logger.error(f"Vision agent error: {e}")
            state["error"] = str(e)
            return state
    
    async def _run_context_agent(self, state: OrchestratorState) -> OrchestratorState:
        """Execute context gathering agent."""
        try:
            agent = await self.agent_pool.get_agent(AgentType.CONTEXT)
            
            task_request = TaskRequest(
                task_type="gather_context",
                input_data={
                    "entities": state["extracted_entities"],
                    "intent": state["intent"]
                },
                context=state.get("context_data", {})
            )
            
            response = await agent.process_task(task_request)
            state["context_data"].update(response.result)
            state["current_agent"] = "context"
            
            return state
            
        except Exception as e:
            self.logger.error(f"Context agent error: {e}")
            state["error"] = str(e)
            return state
    
    async def _run_memory_agent(self, state: OrchestratorState) -> OrchestratorState:
        """Execute memory management agent."""
        try:
            agent = await self.agent_pool.get_agent(AgentType.MEMORY)
            
            task_request = TaskRequest(
                task_type="memory_operation",
                input_data={
                    "query": state["user_input"],
                    "entities": state["extracted_entities"],
                    "context": state["context_data"]
                }
            )
            
            response = await agent.process_task(task_request)
            state["memories"] = response.result.get("memories", [])
            state["current_agent"] = "memory"
            
            return state
            
        except Exception as e:
            self.logger.error(f"Memory agent error: {e}")
            state["error"] = str(e)
            return state
    
    async def _run_insight_agent(self, state: OrchestratorState) -> OrchestratorState:
        """Execute insight analysis agent."""
        try:
            agent = await self.agent_pool.get_agent(AgentType.INSIGHT)
            
            task_request = TaskRequest(
                task_type="analyze_patterns",
                input_data={
                    "memories": state["memories"],
                    "context": state["context_data"]
                }
            )
            
            response = await agent.process_task(task_request)
            state["insights"] = response.result.get("insights", [])
            state["current_agent"] = "insight"
            
            return state
            
        except Exception as e:
            self.logger.error(f"Insight agent error: {e}")
            state["error"] = str(e)
            return state
    
    async def _run_response_agent(self, state: OrchestratorState) -> OrchestratorState:
        """Execute response generation agent."""
        try:
            agent = await self.agent_pool.get_agent(AgentType.RESPONSE)
            
            task_request = TaskRequest(
                task_type="generate_response",
                input_data={
                    "user_input": state["user_input"],
                    "memories": state["memories"],
                    "insights": state["insights"],
                    "context": state["context_data"]
                }
            )
            
            response = await agent.process_task(task_request)
            state["response"] = response.result.get("response")
            state["current_agent"] = "response"
            
            return state
            
        except Exception as e:
            self.logger.error(f"Response agent error: {e}")
            state["error"] = str(e)
            return state
    
    async def _handle_error(self, state: OrchestratorState) -> OrchestratorState:
        """Handle errors in the orchestration flow."""
        error_msg = state.get("error", "Unknown error occurred")
        self.logger.error(f"Orchestration error: {error_msg}")
        
        state["response"] = f"I apologize, but I encountered an error: {error_msg}"
        return state
    
    async def process_user_request(self, user_input: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for processing user requests through the orchestration graph.
        """
        initial_state = OrchestratorState(
            user_input=user_input,
            intent=None,
            extracted_entities={},
            context_data={},
            memories=[],
            insights=[],
            response=None,
            current_agent=None,
            task_history=[],
            error=None,
            metadata=metadata or {}
        )
        
        try:
            # Execute the orchestration graph
            final_state = await self.graph.ainvoke(initial_state)
            
            # Emit completion event
            await self.event_bus.emit("orchestration.completed", {
                "response": final_state.get("response"),
                "agents_used": final_state.get("task_history", [])
            })
            
            return {
                "response": final_state.get("response"),
                "memories": final_state.get("memories", []),
                "insights": final_state.get("insights", []),
                "context": final_state.get("context_data", {}),
                "success": final_state.get("error") is None
            }
            
        except Exception as e:
            self.logger.error(f"Orchestration failed: {e}")
            return {
                "response": f"I'm sorry, but I encountered an error processing your request: {str(e)}",
                "memories": [],
                "insights": [],
                "context": {},
                "success": False
            }
    
    async def shutdown(self):
        """Cleanup resources."""
        await self.agent_pool.shutdown()
        await self.task_queue.shutdown()
        await self.event_bus.shutdown()