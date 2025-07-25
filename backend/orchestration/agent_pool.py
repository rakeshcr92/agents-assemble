"""
Agent pool management system for maintaining and distributing agent instances.
Handles agent lifecycle, load balancing, and resource management.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import weakref
from collections import defaultdict
import importlib

from ..agents.base_agent import BaseAgent
from ..agents.planner_agent import PlannerAgent
from ..agents.voice_agent import VoiceAgent
from ..agents.vision_agent import VisionAgent
from ..agents.context_agent import ContextAgent
from ..agents.memory_agent import MemoryAgent
from ..agents.insight_agent import InsightAgent
from ..agents.response_agent import ResponseAgent


class AgentType(Enum):
    PLANNER = "planner"
    VOICE = "voice"
    VISION = "vision"
    CONTEXT = "context"
    MEMORY = "memory"
    INSIGHT = "insight"
    RESPONSE = "response"


@dataclass
class AgentInstance:
    """Represents an agent instance in the pool."""
    id: str
    agent_type: AgentType
    agent: BaseAgent
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    is_busy: bool = False
    current_task_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PoolStats:
    """Statistics for the agent pool."""
    total_agents: int
    active_agents: int
    idle_agents: int
    agent_counts_by_type: Dict[str, int]
    average_usage: float
    total_requests: int
    pool_efficiency: float


class AgentPool:
    """
    Manages a pool of agent instances with load balancing and lifecycle management.
    """
    
    def __init__(
        self,
        gemini_api_key: str,
        config: Dict[str, Any],
        min_instances_per_type: int = 1,
        max_instances_per_type: int = 5,
        idle_timeout: int = 300,  # 5 minutes
        cleanup_interval: int = 60   # 1 minute
    ):
        self.gemini_api_key = gemini_api_key
        self.config = config
        self.min_instances_per_type = min_instances_per_type
        self.max_instances_per_type = max_instances_per_type
        self.idle_timeout = idle_timeout
        self.cleanup_interval = cleanup_interval
        
        self.logger = logging.getLogger(__name__)
        
        # Agent registry and pools
        self._agents: Dict[str, AgentInstance] = {}
        self._agent_pools: Dict[AgentType, List[str]] = defaultdict(list)
        self._agent_classes: Dict[AgentType, Type[BaseAgent]] = {
            AgentType.PLANNER: PlannerAgent,
            AgentType.VOICE: VoiceAgent,
            AgentType.VISION: VisionAgent,
            AgentType.CONTEXT: ContextAgent,
            AgentType.MEMORY: MemoryAgent,
            AgentType.INSIGHT: InsightAgent,
            AgentType.RESPONSE: ResponseAgent,
        }
        
        # Load balancing and monitoring
        self._round_robin_counters: Dict[AgentType, int] = defaultdict(int)
        self._request_queue: Dict[AgentType, asyncio.Queue] = {
            agent_type: asyncio.Queue() for agent_type in AgentType
        }
        
        # Statistics
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "agents_created": 0,
            "agents_destroyed": 0,
        }
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Weak references to avoid memory leaks
        self._weak_agents = weakref.WeakSet()
        
        # Initialize the pool
        asyncio.create_task(self._initialize_pool())
    
    async def _initialize_pool(self):
        """Initialize the agent pool with minimum instances."""
        self.logger.info("Initializing agent pool...")
        
        # Create minimum instances for each agent type
        for agent_type in AgentType:
            for _ in range(self.min_instances_per_type):
                await self._create_agent_instance(agent_type)
        
        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.info(f"Agent pool initialized with {len(self._agents)} agents")
    
    async def _create_agent_instance(self, agent_type: AgentType) -> str:
        """Create a new agent instance of the specified type."""
        try:
            agent_class = self._agent_classes[agent_type]
            
            # Create agent with configuration
            agent_config = self.config.get(agent_type.value, {})
            agent_config["gemini_api_key"] = self.gemini_api_key
            
            agent = agent_class(**agent_config)
            await agent.initialize()
            
            # Create instance wrapper
            instance_id = f"{agent_type.value}_{len(self._agents)}"
            instance = AgentInstance(
                id=instance_id,
                agent_type=agent_type,
                agent=agent,
                metadata={"config": agent_config}
            )
            
            # Register instance
            self._agents[instance_id] = instance
            self._agent_pools[agent_type].append(instance_id)
            self._weak_agents.add(instance)
            
            self._stats["agents_created"] += 1
            
            self.logger.debug(f"Created {agent_type.value} agent: {instance_id}")
            return instance_id
            
        except Exception as e:
            self.logger.error(f"Failed to create {agent_type.value} agent: {e}")
            raise
    
    async def _destroy_agent_instance(self, instance_id: str):
        """Destroy an agent instance and cleanup resources."""
        if instance_id not in self._agents:
            return
        
        instance = self._agents[instance_id]
        
        try:
            # Cleanup agent resources
            await instance.agent.cleanup()
            
            # Remove from pools
            self._agent_pools[instance.agent_type].remove(instance_id)
            del self._agents[instance_id]
            
            self._stats["agents_destroyed"] += 1
            
            self.logger.debug(f"Destroyed agent instance: {instance_id}")
            
        except Exception as e:
            self.logger.error(f"Error destroying agent {instance_id}: {e}")
    
    async def get_agent(self, agent_type: AgentType) -> BaseAgent:
        """
        Get an available agent instance of the specified type.
        
        Args:
            agent_type: Type of agent to retrieve
            
        Returns:
            BaseAgent: An available agent instance
            
        Raises:
            RuntimeError: If no agents are available and pool is at capacity
        """
        self._stats["total_requests"] += 1
        
        try:
            # Get an available instance
            instance = await self._get_available_instance(agent_type)
            
            if instance:
                # Mark as busy and update usage
                instance.is_busy = True
                instance.last_used = datetime.now()
                instance.usage_count += 1
                
                self._stats["successful_requests"] += 1
                return instance.agent
            else:
                # Try to create a new instance if under capacity
                if len(self._agent_pools[agent_type]) < self.max_instances_per_type:
                    instance_id = await self._create_agent_instance(agent_type)
                    instance = self._agents[instance_id]
                    instance.is_busy = True
                    instance.last_used = datetime.now()
                    instance.usage_count += 1
                    
                    self._stats["successful_requests"] += 1
                    return instance.agent
                else:
                    # Wait for an agent to become available
                    return await self._wait_for_available_agent(agent_type)
                    
        except Exception as e:
            self._stats["failed_requests"] += 1
            self.logger.error(f"Failed to get {agent_type.value} agent: {e}")
            raise
    
    async def _get_available_instance(self, agent_type: AgentType) -> Optional[AgentInstance]:
        """Get an available agent instance using round-robin load balancing."""
        pool = self._agent_pools[agent_type]
        
        if not pool:
            return None
        
        # Round-robin selection starting from last counter position
        start_idx = self._round_robin_counters[agent_type] % len(pool)
        
        for i in range(len(pool)):
            idx = (start_idx + i) % len(pool)
            instance_id = pool[idx]
            instance = self._agents[instance_id]
            
            if not instance.is_busy:
                self._round_robin_counters[agent_type] = idx + 1
                return instance
        
        return None  # All instances are busy
    
    async def _wait_for_available_agent(self, agent_type: AgentType, timeout: float = 30.0) -> BaseAgent:
        """Wait for an agent to become available."""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout:
            instance = await self._get_available_instance(agent_type)
            if instance:
                instance.is_busy = True
                instance.last_used = datetime.now()
                instance.usage_count += 1
                return instance.agent
            
            await asyncio.sleep(0.1)  # Small delay before retrying
        
        raise TimeoutError(f"No {agent_type.value} agents available within timeout")
    
    async def release_agent(self, agent: BaseAgent):
        """
        Release an agent back to the pool.
        
        Args:
            agent: The agent instance to release
        """
        # Find the instance wrapper
        for instance in self._agents.values():
            if instance.agent is agent:
                instance.is_busy = False
                instance.current_task_id = None
                self.logger.debug(f"Released agent: {instance.id}")
                break
    
    async def _cleanup_loop(self):
        """Background task to cleanup idle agents."""
        while not self._shutdown_event.is_set():
            try:
                await self._cleanup_idle_agents()
                await asyncio.sleep(self.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_idle_agents(self):
        """Remove agents that have been idle for too long."""
        current_time = datetime.now()
        agents_to_remove = []
        
        for agent_type, pool in self._agent_pools.items():
            # Keep minimum instances
            if len(pool) <= self.min_instances_per_type:
                continue
            
            # Find idle agents
            for instance_id in pool:
                instance = self._agents[instance_id]
                
                if (not instance.is_busy and 
                    (current_time - instance.last_used).total_seconds() > self.idle_timeout):
                    agents_to_remove.append(instance_id)
        
        # Remove idle agents
        for instance_id in agents_to_remove:
            await self._destroy_agent_instance(instance_id)
    
    async def _monitoring_loop(self):
        """Background task for monitoring pool health and performance."""
        while not self._shutdown_event.is_set():
            try:
                await self._update_pool_metrics()
                await asyncio.sleep(30)  # Update metrics every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    async def _update_pool_metrics(self):
        """Update pool performance metrics."""
        total_agents = len(self._agents)
        active_agents = sum(1 for instance in self._agents.values() if instance.is_busy)
        
        if total_agents > 0:
            pool_efficiency = active_agents / total_agents
        else:
            pool_efficiency = 0.0
        
        # Log metrics if efficiency is low
        if pool_efficiency < 0.3 and total_agents > self.min_instances_per_type:
            self.logger.warning(f"Low pool efficiency: {pool_efficiency:.2%}")
    
    def get_pool_stats(self) -> PoolStats:
        """Get current pool statistics."""
        total_agents = len(self._agents)
        active_agents = sum(1 for instance in self._agents.values() if instance.is_busy)
        idle_agents = total_agents - active_agents
        
        agent_counts = defaultdict(int)
        total_usage = 0
        
        for instance in self._agents.values():
            agent_counts[instance.agent_type.value] += 1
            total_usage += instance.usage_count
        
        average_usage = total_usage / total_agents if total_agents > 0 else 0
        pool_efficiency = active_agents / total_agents if total_agents > 0 else 0
        
        return PoolStats(
            total_agents=total_agents,
            active_agents=active_agents,
            idle_agents=idle_agents,
            agent_counts_by_type=dict(agent_counts),
            average_usage=average_usage,
            total_requests=self._stats["total_requests"],
            pool_efficiency=pool_efficiency
        )
    
    def get_agent_details(self, agent_type: Optional[AgentType] = None) -> List[Dict[str, Any]]:
        """Get detailed information about agents in the pool."""
        agents = []
        
        for instance in self._agents.values():
            if agent_type is None or instance.agent_type == agent_type:
                agents.append({
                    "id": instance.id,
                    "type": instance.agent_type.value,
                    "created_at": instance.created_at.isoformat(),
                    "last_used": instance.last_used.isoformat(),
                    "usage_count": instance.usage_count,
                    "is_busy": instance.is_busy,
                    "current_task_id": instance.current_task_id,
                    "metadata": instance.metadata
                })
        
        return agents
    
    async def scale_pool(self, agent_type: AgentType, target_count: int):
        """
        Scale the pool for a specific agent type.
        
        Args:
            agent_type: Type of agent to scale
            target_count: Target number of instances
        """
        current_count = len(self._agent_pools[agent_type])
        
        if target_count > current_count:
            # Scale up
            for _ in range(target_count - current_count):
                if len(self._agent_pools[agent_type]) < self.max_instances_per_type:
                    await self._create_agent_instance(agent_type)
                else:
                    break
        elif target_count < current_count:
            # Scale down (but respect minimum)
            target_count = max(target_count, self.min_instances_per_type)
            instances_to_remove = current_count - target_count
            
            # Remove idle instances first
            pool = self._agent_pools[agent_type]
            for i in range(instances_to_remove):
                for instance_id in pool[:]:  # Copy to avoid modification during iteration
                    instance = self._agents[instance_id]
                    if not instance.is_busy:
                        await self._destroy_agent_instance(instance_id)
                        break
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on all agents in the pool."""
        results = {
            "healthy_agents": 0,
            "unhealthy_agents": 0,
            "agent_health": {}
        }
        
        for instance in self._agents.values():
            try:
                # Perform health check on agent
                health = await instance.agent.health_check()
                results["agent_health"][instance.id] = health
                
                if health.get("status") == "healthy":
                    results["healthy_agents"] += 1
                else:
                    results["unhealthy_agents"] += 1
                    
            except Exception as e:
                results["unhealthy_agents"] += 1
                results["agent_health"][instance.id] = {
                    "status": "error",
                    "error": str(e)
                }