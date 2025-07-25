from typing import Dict, List, Any, Optional, TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from dataclasses import dataclass
from enum import Enum
import logging
import json
import uuid
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State passed between nodes in the graph"""
    messages: Sequence[BaseMessage]
    current_agent: str
    task_data: Dict[str, Any]
    workflow_id: str
    user_id: str
    context: Dict[str, Any]
    plan: Dict[str, Any]
    memories: List[Dict[str, Any]]
    insights: List[Dict[str, Any]]
    response: str
    error: Optional[str]
    retry_count: int


class EventType(Enum):
    """Enumeration of all possible event types in the system"""
    TRANSCRIPTION_COMPLETE = "transcription_complete"
    PLAN_CREATED = "plan_created"
    CONTEXT_ANALYZED = "context_analyzed"
    MEMORY_STORED = "memory_stored"
    INSIGHT_GENERATED = "insight_generated"
    RESPONSE_READY = "response_ready"
    VISION_ANALYZED = "vision_analyzed"


class Orchestrator:
    """Main orchestrator that coordinates all agents using LangGraph"""
    
    def __init__(self, agent_pool=None, event_bus=None, task_queue=None):
        self.agent_pool = agent_pool
        self.event_bus = event_bus
        self.task_queue = task_queue
        self.graph = self._build_graph()
        self.compiled_graph = self.graph.compile()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        # Define the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("voice_agent", self._voice_agent_node)
        workflow.add_node("planner_agent", self._planner_agent_node)
        workflow.add_node("context_agent", self._context_agent_node)
        workflow.add_node("memory_agent", self._memory_agent_node)
        workflow.add_node("insight_agent", self._insight_agent_node)
        workflow.add_node("response_agent", self._response_agent_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Set the entry point
        workflow.set_entry_point("voice_agent")
        
        # Add edges
        workflow.add_edge("voice_agent", "planner_agent")
        workflow.add_edge("planner_agent", "context_agent")
        workflow.add_edge("context_agent", "memory_agent")
        workflow.add_edge("memory_agent", "insight_agent")
        workflow.add_edge("insight_agent", "response_agent")
        workflow.add_edge("response_agent", END)
        
        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "voice_agent",
            self._check_error,
            {
                "error": "error_handler",
                "continue": "planner_agent"
            }
        )
        
        workflow.add_conditional_edges(
            "planner_agent",
            self._check_error,
            {
                "error": "error_handler",
                "continue": "context_agent"
            }
        )
        
        return workflow
    
    def _check_error(self, state: AgentState) -> str:
        """Check if there's an error in the state"""
        if state.get("error"):
            return "error"
        return "continue"
    
    async def _voice_agent_node(self, state: AgentState) -> AgentState:
        """Process voice input"""
        try:
            logger.info("Processing voice input...")
            
            # Get agent from pool if available
            agent = None
            if self.agent_pool:
                agent = await self.agent_pool.acquire("voice_agent")
            
            # Process the voice data
            audio_data = state["task_data"].get("audio_data", "")
            
            # Simulate processing (in real implementation, use actual voice agent)
            transcription = f"Transcribed: {state['task_data'].get('text', 'Hello, this is a test')}"
            
            # Update state
            state["messages"].append(HumanMessage(content=transcription))
            state["task_data"]["transcription"] = transcription
            state["current_agent"] = "voice_agent"
            
            # Publish event if event bus is available
            if self.event_bus:
                await self.event_bus.publish({
                    "type": EventType.TRANSCRIPTION_COMPLETE,
                    "data": {
                        "transcription": transcription,
                        "user_id": state["user_id"]
                    }
                })
            
            # Release agent back to pool
            if self.agent_pool and agent:
                await self.agent_pool.release("voice_agent", agent)
            
            logger.info(f"Voice transcription complete: {transcription[:50]}...")
            
        except Exception as e:
            logger.error(f"Error in voice agent: {str(e)}")
            state["error"] = str(e)
            state["retry_count"] = state.get("retry_count", 0) + 1
            
        return state
    
    async def _planner_agent_node(self, state: AgentState) -> AgentState:
        """Create a plan based on the transcription"""
        try:
            logger.info("Creating plan...")
            
            # Get agent from pool
            agent = None
            if self.agent_pool:
                agent = await self.agent_pool.acquire("planner_agent")
            
            transcription = state["task_data"].get("transcription", "")
            
            # Simulate planning (in real implementation, use actual planner agent)
            plan = {
                "steps": [
                    "Analyze user intent",
                    "Gather relevant context",
                    "Generate insights",
                    "Formulate response"
                ],
                "intent": "user_query",
                "priority": "high"
            }
            
            # Update state
            state["plan"] = plan
            state["current_agent"] = "planner_agent"
            state["messages"].append(AIMessage(content=f"Plan created: {json.dumps(plan)}"))
            
            # Publish event
            if self.event_bus:
                await self.event_bus.publish({
                    "type": EventType.PLAN_CREATED,
                    "data": {
                        "plan": plan,
                        "user_id": state["user_id"]
                    }
                })
            
            # Release agent
            if self.agent_pool and agent:
                await self.agent_pool.release("planner_agent", agent)
            
            logger.info("Plan created successfully")
            
        except Exception as e:
            logger.error(f"Error in planner agent: {str(e)}")
            state["error"] = str(e)
            
        return state
    
    async def _context_agent_node(self, state: AgentState) -> AgentState:
        """Analyze context based on the plan"""
        try:
            logger.info("Analyzing context...")
            
            # Get agent from pool
            agent = None
            if self.agent_pool:
                agent = await self.agent_pool.acquire("context_agent")
            
            plan = state.get("plan", {})
            
            # Simulate context analysis
            context = {
                "user_profile": {
                    "preferences": ["technical", "detailed"],
                    "history": ["previous_queries"]
                },
                "environment": {
                    "time": "afternoon",
                    "location": "office"
                },
                "relevant_data": ["data1", "data2"]
            }
            
            # Update state
            state["context"] = context
            state["current_agent"] = "context_agent"
            state["messages"].append(AIMessage(content=f"Context analyzed: {json.dumps(context)}"))
            
            # Publish event
            if self.event_bus:
                await self.event_bus.publish({
                    "type": EventType.CONTEXT_ANALYZED,
                    "data": {
                        "context": context,
                        "user_id": state["user_id"]
                    }
                })
            
            # Release agent
            if self.agent_pool and agent:
                await self.agent_pool.release("context_agent", agent)
            
            logger.info("Context analysis complete")
            
        except Exception as e:
            logger.error(f"Error in context agent: {str(e)}")
            state["error"] = str(e)
            
        return state
    
    async def _memory_agent_node(self, state: AgentState) -> AgentState:
        """Store relevant information in memory"""
        try:
            logger.info("Storing memory...")
            
            # Get agent from pool
            agent = None
            if self.agent_pool:
                agent = await self.agent_pool.acquire("memory_agent")
            
            context = state.get("context", {})
            
            # Simulate memory storage
            memory_entry = {
                "timestamp": "2024-01-01T00:00:00",
                "interaction": state["task_data"].get("transcription", ""),
                "context": context,
                "tags": ["voice", "query"]
            }
            
            # Update state
            state["memories"] = [memory_entry]
            state["current_agent"] = "memory_agent"
            state["messages"].append(AIMessage(content="Memory stored successfully"))
            
            # Publish event
            if self.event_bus:
                await self.event_bus.publish({
                    "type": EventType.MEMORY_STORED,
                    "data": {
                        "memory": memory_entry,
                        "user_id": state["user_id"]
                    }
                })
            
            # Release agent
            if self.agent_pool and agent:
                await self.agent_pool.release("memory_agent", agent)
            
            logger.info("Memory stored successfully")
            
        except Exception as e:
            logger.error(f"Error in memory agent: {str(e)}")
            state["error"] = str(e)
            
        return state
    
    async def _insight_agent_node(self, state: AgentState) -> AgentState:
        """Generate insights based on context and memories"""
        try:
            logger.info("Generating insights...")
            
            # Get agent from pool
            agent = None
            if self.agent_pool:
                agent = await self.agent_pool.acquire("insight_agent")
            
            context = state.get("context", {})
            memories = state.get("memories", [])
            
            # Simulate insight generation
            insights = [
                {
                    "type": "pattern",
                    "description": "User frequently asks technical questions",
                    "confidence": 0.85
                },
                {
                    "type": "recommendation",
                    "description": "Provide detailed technical explanations",
                    "confidence": 0.90
                }
            ]
            
            # Update state
            state["insights"] = insights
            state["current_agent"] = "insight_agent"
            state["messages"].append(AIMessage(content=f"Insights generated: {len(insights)} insights"))
            
            # Publish event
            if self.event_bus:
                await self.event_bus.publish({
                    "type": EventType.INSIGHT_GENERATED,
                    "data": {
                        "insights": insights,
                        "user_id": state["user_id"]
                    }
                })
            
            # Release agent
            if self.agent_pool and agent:
                await self.agent_pool.release("insight_agent", agent)
            
            logger.info(f"Generated {len(insights)} insights")
            
        except Exception as e:
            logger.error(f"Error in insight agent: {str(e)}")
            state["error"] = str(e)
            
        return state
    
    async def _response_agent_node(self, state: AgentState) -> AgentState:
        """Generate final response based on all previous processing"""
        try:
            logger.info("Generating response...")
            
            # Get agent from pool
            agent = None
            if self.agent_pool:
                agent = await self.agent_pool.acquire("response_agent")
            
            plan = state.get("plan", {})
            context = state.get("context", {})
            insights = state.get("insights", [])
            
            # Simulate response generation
            response = f"""Based on your query: {state['task_data'].get('transcription', '')}
            
I've analyzed the context and generated the following response:
- Intent: {plan.get('intent', 'unknown')}
- Relevant insights: {len(insights)} insights found
- Personalized based on your preferences

This is a sample response that would be generated by the actual response agent."""
            
            # Update state
            state["response"] = response
            state["current_agent"] = "response_agent"
            state["messages"].append(AIMessage(content=response))
            
            # Publish event
            if self.event_bus:
                await self.event_bus.publish({
                    "type": EventType.RESPONSE_READY,
                    "data": {
                        "response": response,
                        "user_id": state["user_id"]
                    }
                })
            
            # Release agent
            if self.agent_pool and agent:
                await self.agent_pool.release("response_agent", agent)
            
            logger.info("Response generated successfully")
            
        except Exception as e:
            logger.error(f"Error in response agent: {str(e)}")
            state["error"] = str(e)
            
        return state
    
    async def _error_handler_node(self, state: AgentState) -> AgentState:
        """Handle errors in the workflow"""
        error = state.get("error", "Unknown error")
        retry_count = state.get("retry_count", 0)
        
        logger.error(f"Error in workflow: {error}, retry count: {retry_count}")
        
        if retry_count < 3:
            # Clear error and retry
            state["error"] = None
            logger.info(f"Retrying workflow, attempt {retry_count + 1}")
        else:
            # Max retries reached, set final error response
            state["response"] = f"I apologize, but I encountered an error processing your request: {error}"
            
        return state
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request through the orchestration workflow"""
        initial_state = {
            "messages": [],
            "current_agent": "orchestrator",
            "task_data": request_data,
            "workflow_id": request_data.get("workflow_id", "default"),
            "user_id": request_data.get("user_id", "default_user"),
            "context": {},
            "plan": {},
            "memories": [],
            "insights": [],
            "response": "",
            "error": None,
            "retry_count": 0
        }
        
        try:
            # Run the graph
            final_state = await self.compiled_graph.ainvoke(initial_state)
            
            return {
                "success": True,
                "response": final_state.get("response", ""),
                "workflow_id": final_state.get("workflow_id"),
                "insights": final_state.get("insights", [])
            }
            
        except Exception as e:
            logger.error(f"Error in orchestration: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "workflow_id": initial_state.get("workflow_id")
            }


# Test function
async def test_orchestrator():
    """Test the orchestrator independently"""
    orchestrator = Orchestrator()
    
    test_request = {
        "audio_data": "test_audio_base64",
        "text": "What's the weather like today?",
        "user_id": "test_user",
        "workflow_id": "test_workflow"
    }
    
    result = await orchestrator.process_request(test_request)
    print(f"Test result: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_orchestrator())