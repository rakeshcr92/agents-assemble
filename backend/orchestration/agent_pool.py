import asyncio
from typing import Dict, List, Any, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import logging
from abc import ABC, abstractmethod
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentInstance:
    """Represents an instance of an agent in the pool"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: str = ""
    agent: Optional[Any] = None  # The actual agent instance
    status: str = "available"  # available, busy, error
    created_at: datetime = field(default_factory=datetime.now)
    last_used_at: Optional[datetime] = None
    usage_count: int = 0
    current_task_id: Optional[str] = None
    memory: Optional[Dict[str, Any]] = None


class BasePooledAgent(ABC):
    """Base class for agents that can be pooled"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.created_at = datetime.now()
        self.usage_count = 0
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent (load models, etc.)"""
        pass
        
    @abstractmethod
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request"""
        pass
        
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources"""
        pass
        
    async def reset(self) -> None:
        """Reset the agent state between uses"""
        pass


class MockVoiceAgent(BasePooledAgent):
    """Mock voice agent for testing"""
    
    async def initialize(self) -> None:
        logger.info(f"Initializing MockVoiceAgent {self.agent_id}")
        await asyncio.sleep(0.1)  # Simulate initialization
        
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.usage_count += 1
        audio = data.get("audio_data", "")
        # Simulate processing
        await asyncio.sleep(0.5)
        return {
            "transcription": f"Transcribed audio (agent {self.agent_id})",
            "confidence": 0.95
        }
        
    async def cleanup(self) -> None:
        logger.info(f"Cleaning up MockVoiceAgent {self.agent_id}")


class MockPlannerAgent(BasePooledAgent):
    """Mock planner agent for testing"""
    
    async def initialize(self) -> None:
        logger.info(f"Initializing MockPlannerAgent {self.agent_id}")
        self.memory = {}  # Simple dict instead of ConversationBufferMemory
        await asyncio.sleep(0.1)
        
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.usage_count += 1
        transcription = data.get("transcription", "")
        # Simulate planning
        await asyncio.sleep(0.3)
        return {
            "plan": {
                "steps": ["analyze", "search", "synthesize", "respond"],
                "intent": "information_request",
                "complexity": "medium"
            }
        }
        
    async def cleanup(self) -> None:
        logger.info(f"Cleaning up MockPlannerAgent {self.agent_id}")
        
    async def reset(self) -> None:
        """Reset memory between uses"""
        if self.memory:
            self.memory.clear()


class MockInsightAgent(BasePooledAgent):
    """Mock insight agent for testing"""
    
    async def initialize(self) -> None:
        logger.info(f"Initializing MockInsightAgent {self.agent_id}")
        await asyncio.sleep(0.1)
        
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.usage_count += 1
        context = data.get("context", {})
        # Simulate insight generation
        await asyncio.sleep(0.4)
        return {
            "insights": [
                {"type": "pattern", "description": "User prefers concise answers"},
                {"type": "recommendation", "description": "Use bullet points"}
            ]
        }
        
    async def cleanup(self) -> None:
        logger.info(f"Cleaning up MockInsightAgent {self.agent_id}")


class AgentPool:
    """Pool manager for LangChain agents"""
    
    def __init__(self):
        self.pools: Dict[str, List[AgentInstance]] = defaultdict(list)
        self.agent_configs: Dict[str, Dict[str, Any]] = {}
        self.agent_classes: Dict[str, Type[BasePooledAgent]] = {}
        self.min_pool_size: Dict[str, int] = {}
        self.max_pool_size: Dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition()
        self.statistics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_created": 0,
            "total_acquired": 0,
            "total_released": 0,
            "current_busy": 0,
            "current_available": 0
        })
        
        # Register default agent types
        self._register_default_agents()
        
    def _register_default_agents(self):
        """Register default agent types"""
        self.register_agent_type(
            "voice_agent",
            MockVoiceAgent,
            min_size=1,
            max_size=3
        )
        self.register_agent_type(
            "planner_agent",
            MockPlannerAgent,
            min_size=2,
            max_size=5
        )
        self.register_agent_type(
            "insight_agent",
            MockInsightAgent,
            min_size=1,
            max_size=3
        )
        
    def register_agent_type(
        self,
        agent_type: str,
        agent_class: Type[BasePooledAgent],
        min_size: int = 1,
        max_size: int = 5,
        config: Optional[Dict[str, Any]] = None
    ):
        """Register a new agent type"""
        self.agent_classes[agent_type] = agent_class
        self.min_pool_size[agent_type] = min_size
        self.max_pool_size[agent_type] = max_size
        self.agent_configs[agent_type] = config or {}
        logger.info(f"Registered agent type: {agent_type}")
        
    async def initialize(self):
        """Initialize all agent pools"""
        logger.info("Initializing agent pools...")
        
        # Create minimum number of agents for each type
        for agent_type in self.agent_classes:
            min_size = self.min_pool_size[agent_type]
            for _ in range(min_size):
                await self._create_agent(agent_type)
                
        logger.info("Agent pools initialized")
        
    async def _create_agent(self, agent_type: str) -> AgentInstance:
        """Create a new agent instance"""
        if agent_type not in self.agent_classes:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        agent_class = self.agent_classes[agent_type]
        agent_id = str(uuid.uuid4())
        
        # Create the agent
        agent = agent_class(agent_id)
        await agent.initialize()
        
        # Create instance wrapper
        instance = AgentInstance(
            id=agent_id,
            agent_type=agent_type,
            agent=agent,
            status="available"
        )
        
        # Add to pool
        async with self._lock:
            self.pools[agent_type].append(instance)
            self.statistics[agent_type]["total_created"] += 1
            self.statistics[agent_type]["current_available"] += 1
            
        logger.info(f"Created new {agent_type} agent: {agent_id}")
        return instance
        
    async def acquire(self, agent_type: str, timeout: float = 30.0) -> Any:
        """Acquire an agent from the pool"""
        if agent_type not in self.agent_classes:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        start_time = asyncio.get_event_loop().time()
        
        async with self._condition:
            while True:
                # Check for available agents
                available_agent = None
                async with self._lock:
                    pool = self.pools[agent_type]
                    for instance in pool:
                        if instance.status == "available":
                            instance.status = "busy"
                            instance.last_used_at = datetime.now()
                            instance.usage_count += 1
                            available_agent = instance
                            self.statistics[agent_type]["total_acquired"] += 1
                            self.statistics[agent_type]["current_busy"] += 1
                            self.statistics[agent_type]["current_available"] -= 1
                            break
                            
                if available_agent:
                    logger.info(f"Acquired {agent_type} agent: {available_agent.id}")
                    return available_agent.agent
                    
                # Check if we can create a new agent
                async with self._lock:
                    current_size = len(self.pools[agent_type])
                    max_size = self.max_pool_size[agent_type]
                    
                if current_size < max_size:
                    # Create a new agent
                    instance = await self._create_agent(agent_type)
                    async with self._lock:
                        instance.status = "busy"
                        instance.last_used_at = datetime.now()
                        instance.usage_count += 1
                        self.statistics[agent_type]["total_acquired"] += 1
                        self.statistics[agent_type]["current_busy"] += 1
                        self.statistics[agent_type]["current_available"] -= 1
                    logger.info(f"Created and acquired new {agent_type} agent: {instance.id}")
                    return instance.agent
                    
                # Wait for an agent to become available
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(f"Timeout waiting for {agent_type} agent")
                    
                try:
                    await asyncio.wait_for(
                        self._condition.wait(),
                        timeout=timeout - elapsed
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(f"Timeout waiting for {agent_type} agent")
                    
    async def release(self, agent_type: str, agent: Any) -> None:
        """Release an agent back to the pool"""
        if not isinstance(agent, BasePooledAgent):
            logger.warning(f"Attempted to release non-pooled agent: {agent}")
            return
            
        async with self._lock:
            pool = self.pools[agent_type]
            for instance in pool:
                if instance.agent == agent:
                    # Reset the agent
                    await agent.reset()
                    
                    # Mark as available
                    instance.status = "available"
                    instance.current_task_id = None
                    self.statistics[agent_type]["total_released"] += 1
                    self.statistics[agent_type]["current_busy"] -= 1
                    self.statistics[agent_type]["current_available"] += 1
                    
                    logger.info(f"Released {agent_type} agent: {instance.id}")
                    break
                    
        # Notify waiting tasks
        async with self._condition:
            self._condition.notify()
            
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics for all pools"""
        stats = {}
        async with self._lock:
            for agent_type, pool in self.pools.items():
                type_stats = self.statistics[agent_type].copy()
                type_stats["pool_size"] = len(pool)
                type_stats["min_size"] = self.min_pool_size[agent_type]
                type_stats["max_size"] = self.max_pool_size[agent_type]
                
                # Calculate average usage
                if pool:
                    total_usage = sum(instance.usage_count for instance in pool)
                    type_stats["average_usage"] = total_usage / len(pool)
                else:
                    type_stats["average_usage"] = 0
                    
                stats[agent_type] = type_stats
                
        return stats
        
    async def cleanup(self):
        """Cleanup all agent pools"""
        logger.info("Cleaning up agent pools...")
        
        async with self._lock:
            for agent_type, pool in self.pools.items():
                for instance in pool:
                    if instance.agent:
                        await instance.agent.cleanup()
                        
            self.pools.clear()
            
        logger.info("Agent pools cleaned up")
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all pools"""
        health = {
            "status": "healthy",
            "pools": {},
            "issues": []
        }
        
        async with self._lock:
            for agent_type, pool in self.pools.items():
                pool_health = {
                    "size": len(pool),
                    "available": sum(1 for i in pool if i.status == "available"),
                    "busy": sum(1 for i in pool if i.status == "busy"),
                    "error": sum(1 for i in pool if i.status == "error")
                }
                
                # Check for issues
                if pool_health["error"] > 0:
                    health["issues"].append(f"{agent_type} has {pool_health['error']} agents in error state")
                    
                if pool_health["size"] < self.min_pool_size[agent_type]:
                    health["issues"].append(f"{agent_type} pool size below minimum")
                    
                health["pools"][agent_type] = pool_health
                
        if health["issues"]:
            health["status"] = "degraded"
            
        return health


# Test function
async def test_agent_pool():
    """Test the agent pool independently"""
    pool = AgentPool()
    
    # Initialize pools
    await pool.initialize()
    
    # Get initial stats
    stats = await pool.get_pool_stats()
    print(f"Initial pool stats: {stats}")
    
    # Test acquiring and releasing agents
    agents = []
    agent_types = ["voice_agent", "planner_agent", "insight_agent"]
    
    # Acquire multiple agents
    for agent_type in agent_types:
        agent = await pool.acquire(agent_type)
        agents.append((agent_type, agent))
        print(f"Acquired {agent_type}")
        
    # Use the agents
    for agent_type, agent in agents:
        result = await agent.process({"test": "data"})
        print(f"{agent_type} result: {result}")
        
    # Release agents
    for agent_type, agent in agents:
        await pool.release(agent_type, agent)
        print(f"Released {agent_type}")
        
    # Test concurrent acquisition
    async def acquire_and_use(pool, agent_type, duration):
        agent = await pool.acquire(agent_type)
        await asyncio.sleep(duration)
        await pool.release(agent_type, agent)
        
    # Create concurrent tasks
    tasks = []
    for i in range(10):
        agent_type = agent_types[i % len(agent_types)]
        task = asyncio.create_task(acquire_and_use(pool, agent_type, 0.5))
        tasks.append(task)
        
    # Wait for all tasks
    await asyncio.gather(*tasks)
    
    # Final stats
    final_stats = await pool.get_pool_stats()
    print(f"Final pool stats: {final_stats}")
    
    # Health check
    health = await pool.health_check()
    print(f"Health check: {health}")
    
    # Cleanup
    await pool.cleanup()


if __name__ == "__main__":
    asyncio.run(test_agent_pool())