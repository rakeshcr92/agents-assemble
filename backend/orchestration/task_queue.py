import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from enum import Enum
import logging
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class Task:
    """Represents a task in the queue"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    retries: int = 0
    max_retries: int = 3
    result: Optional[Any] = None
    error: Optional[str] = None
    callback_handler: Optional['TaskQueueCallback'] = None


class TaskQueueCallback:
    """Callback handler for task execution tracking"""
    
    def __init__(self, task_queue: 'TaskQueue', task: Task):
        self.task_queue = task_queue
        self.task = task
        
    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain starts running."""
        logger.info(f"Task {self.task.id} chain started")
        self.task.status = TaskStatus.RUNNING
        self.task.started_at = datetime.now()
        
    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain ends running."""
        logger.info(f"Task {self.task.id} chain completed")
        self.task.status = TaskStatus.COMPLETED
        self.task.completed_at = datetime.now()
        self.task.result = outputs
        
    async def on_chain_error(self, error: Exception, **kwargs) -> None:
        """Called when a chain errors."""
        logger.error(f"Task {self.task.id} chain error: {str(error)}")
        self.task.status = TaskStatus.FAILED
        self.task.error = str(error)
        
        # Handle retry logic
        if self.task.retries < self.task.max_retries:
            self.task.retries += 1
            self.task.status = TaskStatus.RETRYING
            await self.task_queue.enqueue(self.task)


class PriorityQueue:
    """Simple priority queue implementation"""
    
    def __init__(self):
        self.queues = {1: deque(), 2: deque(), 3: deque()}  # 3 priority levels
        self.lock = asyncio.Lock()
        
    async def put(self, task: Task):
        """Add a task to the queue"""
        async with self.lock:
            priority = min(max(task.priority, 1), 3)  # Clamp between 1-3
            self.queues[priority].append(task)
            
    async def get(self) -> Optional[Task]:
        """Get the highest priority task"""
        async with self.lock:
            # Check from highest priority (3) to lowest (1)
            for priority in [3, 2, 1]:
                if self.queues[priority]:
                    return self.queues[priority].popleft()
            return None
            
    async def size(self) -> int:
        """Get total queue size"""
        async with self.lock:
            return sum(len(q) for q in self.queues.values())
            
    def empty(self) -> bool:
        """Check if queue is empty"""
        return all(len(q) == 0 for q in self.queues.values())


class TaskQueue:
    """Task queue implementation for LangChain agents"""
    
    def __init__(self, max_concurrent_tasks: int = 5):
        self.queue = PriorityQueue()
        self.max_concurrent_tasks = max_concurrent_tasks
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.workers: List[asyncio.Task] = []
        self.is_running = False
        self.event_bus = None  # Will be injected
        self.agent_pool = None  # Will be injected
        
    async def start(self, num_workers: int = 3):
        """Start the task queue workers"""
        self.is_running = True
        logger.info(f"Starting task queue with {num_workers} workers")
        
        # Create worker tasks
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
            
    async def stop(self):
        """Stop the task queue"""
        logger.info("Stopping task queue...")
        self.is_running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
            
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
    async def enqueue(self, task: Task) -> str:
        """Add a task to the queue"""
        if not isinstance(task, Task):
            raise ValueError("task must be an instance of Task")
            
        # Set callback handler
        task.callback_handler = TaskQueueCallback(self, task)
        
        await self.queue.put(task)
        logger.info(f"Enqueued task {task.id} of type {task.type} with priority {task.priority}")
        
        # Emit event if event bus is available
        if self.event_bus:
            await self.event_bus.publish({
                "type": "TASK_CREATED",
                "data": {
                    "task_id": task.id,
                    "task_type": task.type,
                    "priority": task.priority
                }
            })
            
        return task.id
        
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a task"""
        # Check running tasks
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            return {
                "id": task.id,
                "status": task.status.value,
                "type": task.type,
                "started_at": task.started_at.isoformat() if task.started_at else None
            }
            
        # Check completed tasks
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return {
                "id": task.id,
                "status": task.status.value,
                "type": task.type,
                "result": task.result,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None
            }
            
        return None
        
    async def _worker(self, worker_id: str):
        """Worker coroutine that processes tasks from the queue"""
        logger.info(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                # Check if we can run more tasks
                if len(self.running_tasks) >= self.max_concurrent_tasks:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Get a task from the queue
                task = await self.queue.get()
                if not task:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Process the task
                logger.info(f"Worker {worker_id} processing task {task.id}")
                self.running_tasks[task.id] = task
                
                try:
                    # Execute the task based on its type
                    result = await self._execute_task(task)
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    
                    # Move to completed
                    self.completed_tasks[task.id] = task
                    del self.running_tasks[task.id]
                    
                    # Emit completion event
                    if self.event_bus:
                        await self.event_bus.publish({
                            "type": "TASK_COMPLETED",
                            "data": {
                                "task_id": task.id,
                                "task_type": task.type,
                                "result": result
                            }
                        })
                        
                except Exception as e:
                    logger.error(f"Task {task.id} failed: {str(e)}")
                    task.error = str(e)
                    task.status = TaskStatus.FAILED
                    
                    # Handle retry
                    if task.retries < task.max_retries:
                        task.retries += 1
                        task.status = TaskStatus.RETRYING
                        await self.enqueue(task)
                        logger.info(f"Retrying task {task.id} (attempt {task.retries})")
                    else:
                        # Move to completed with failed status
                        self.completed_tasks[task.id] = task
                        
                    del self.running_tasks[task.id]
                    
                    # Emit failure event
                    if self.event_bus:
                        await self.event_bus.publish({
                            "type": "TASK_FAILED",
                            "data": {
                                "task_id": task.id,
                                "task_type": task.type,
                                "error": str(e),
                                "retries": task.retries
                            }
                        })
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}")
                await asyncio.sleep(1)
                
        logger.info(f"Worker {worker_id} stopped")
        
    async def _execute_task(self, task: Task) -> Any:
        """Execute a task based on its type"""
        logger.info(f"Executing task {task.id} of type {task.type}")
        
        # Simulate task execution based on type
        # In real implementation, this would call the appropriate agent
        
        if task.type == "voice_agent":
            # Simulate voice processing
            await asyncio.sleep(1)
            return {"transcription": "Simulated transcription"}
            
        elif task.type == "planner_agent":
            # Simulate planning
            await asyncio.sleep(0.5)
            return {"plan": {"steps": ["step1", "step2"]}}
            
        elif task.type == "context_agent":
            # Simulate context analysis
            await asyncio.sleep(0.3)
            return {"context": {"user_intent": "query"}}
            
        elif task.type == "memory_agent":
            # Simulate memory storage
            await asyncio.sleep(0.2)
            return {"memory_id": str(uuid.uuid4())}
            
        elif task.type == "insight_agent":
            # Simulate insight generation
            await asyncio.sleep(0.8)
            return {"insights": ["insight1", "insight2"]}
            
        elif task.type == "response_agent":
            # Simulate response generation
            await asyncio.sleep(0.5)
            return {"response": "This is a simulated response"}
            
        else:
            raise ValueError(f"Unknown task type: {task.type}")
            
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            "queue_size": await self.queue.size(),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "is_running": self.is_running,
            "num_workers": len(self.workers)
        }


# Test function
async def test_task_queue():
    """Test the task queue independently"""
    queue = TaskQueue(max_concurrent_tasks=3)
    
    # Start the queue
    await queue.start(num_workers=2)
    
    # Create test tasks
    tasks = []
    for i in range(5):
        task = Task(
            type=["voice_agent", "planner_agent", "insight_agent"][i % 3],
            data={"test_data": f"test_{i}"},
            priority=(i % 3) + 1
        )
        task_id = await queue.enqueue(task)
        tasks.append(task_id)
        
    # Wait a bit for processing
    await asyncio.sleep(3)
    
    # Check task statuses
    for task_id in tasks:
        status = await queue.get_task_status(task_id)
        print(f"Task {task_id}: {status}")
        
    # Get queue stats
    stats = await queue.get_queue_stats()
    print(f"Queue stats: {stats}")
    
    # Stop the queue
    await queue.stop()


if __name__ == "__main__":
    asyncio.run(test_task_queue())