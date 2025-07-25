"""
Async task management system with priority queues, retry logic, and monitoring.
Handles distributed task execution across multiple agents.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
from collections import defaultdict
import time


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Task:
    """Represents a task in the queue."""
    id: str
    task_type: str
    data: Dict[str, Any]
    callback: Optional[Callable] = None
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    retries: int = 0
    error: Optional[str] = None
    result: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class TaskResult:
    """Result of task execution."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskQueue:
    """
    Async task queue with priority handling, retry logic, and dependency management.
    """
    
    def __init__(
        self,
        max_workers: int = 10,
        max_queue_size: int = 1000,
        default_timeout: float = 300.0
    ):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.default_timeout = default_timeout
        self.logger = logging.getLogger(__name__)
        
        # Priority queues for different task types
        self._queues: Dict[TaskPriority, asyncio.Queue] = {
            priority: asyncio.Queue(maxsize=max_queue_size)
            for priority in TaskPriority
        }
        
        # Task registry for tracking all tasks
        self._tasks: Dict[str, Task] = {}
        self._task_results: Dict[str, TaskResult] = {}
        
        # Worker management
        self._workers: List[asyncio.Task] = []
        self._worker_semaphore = asyncio.Semaphore(max_workers)
        self._shutdown_event = asyncio.Event()
        
        # Dependency tracking
        self._task_dependencies: Dict[str, List[str]] = defaultdict(list)
        self._dependent_tasks: Dict[str, List[str]] = defaultdict(list)
        
        # Statistics and monitoring
        self._stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_cancelled": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0
        }
        
        # Performance metrics
        self._performance_history: List[Dict[str, Any]] = []
        self._max_history_size = 1000
        
        # Start worker pool
        self._start_workers()
    
    def _start_workers(self):
        """Start the worker pool for processing tasks."""
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)
    
    async def _worker(self, worker_id: str):
        """Worker coroutine that processes tasks from priority queues."""
        self.logger.debug(f"Started worker {worker_id}")
        
        while not self._shutdown_event.is_set():
            try:
                # Get next task from highest priority queue
                task = await self._get_next_task()
                if task is None:
                    continue
                
                # Acquire worker semaphore
                async with self._worker_semaphore:
                    await self._execute_task(task, worker_id)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying
        
        self.logger.debug(f"Worker {worker_id} stopped")
    
    async def _get_next_task(self) -> Optional[Task]:
        """Get the next task to execute, checking dependencies."""
        # Check queues in priority order (highest first)
        for priority in sorted(TaskPriority, key=lambda x: x.value, reverse=True):
            queue = self._queues[priority]
            
            if not queue.empty():
                try:
                    task = await asyncio.wait_for(queue.get(), timeout=1.0)
                    
                    # Check if dependencies are satisfied
                    if self._are_dependencies_satisfied(task):
                        return task
                    else:
                        # Re-queue if dependencies not met
                        await queue.put(task)
                        
                except asyncio.TimeoutError:
                    continue
        
        return None
    
    def _are_dependencies_satisfied(self, task: Task) -> bool:
        """Check if all task dependencies have completed successfully."""
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            if dep_id not in self._task_results:
                return False
            
            result = self._task_results[dep_id]
            if not result.success:
                # Dependency failed, mark this task as failed too
                task.status = TaskStatus.FAILED
                task.error = f"Dependency {dep_id} failed"
                return False
        
        return True
    
    async def _execute_task(self, task: Task, worker_id: str):
        """Execute a single task with error handling and retry logic."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        start_time = time.time()
        
        try:
            # Apply timeout if specified
            timeout = task.timeout or self.default_timeout
            
            # Execute the task callback
            if task.callback:
                if asyncio.iscoroutinefunction(task.callback):
                    result = await asyncio.wait_for(
                        task.callback(task.data),
                        timeout=timeout
                    )
                else:
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, task.callback, task.data
                        ),
                        timeout=timeout
                    )
            else:
                result = task.data  # No callback, just return data
            
            # Task completed successfully
            execution_time = time.time() - start_time
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            # Store result
            task_result = TaskResult(
                task_id=task.id,
                success=True,
                result=result,
                execution_time=execution_time,
                metadata={"worker_id": worker_id}
            )
            self._task_results[task.id] = task_result
            
            # Update statistics
            self._stats["tasks_completed"] += 1
            self._stats["total_execution_time"] += execution_time
            self._stats["average_execution_time"] = (
                self._stats["total_execution_time"] / self._stats["tasks_completed"]
            )
            
            # Process dependent tasks
            await self._process_dependent_tasks(task.id)
            
            self.logger.debug(f"Task {task.id} completed by {worker_id} in {execution_time:.2f}s")
            
        except asyncio.TimeoutError:
            await self._handle_task_failure(task, "Task timed out", worker_id)
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            self._stats["tasks_cancelled"] += 1
            raise
        except Exception as e:
            await self._handle_task_failure(task, str(e), worker_id)
    
    async def _handle_task_failure(self, task: Task, error: str, worker_id: str):
        """Handle task failure with retry logic."""
        task.error = error
        task.retries += 1
        
        if task.retries <= task.max_retries:
            # Retry the task
            task.status = TaskStatus.RETRYING
            
            # Apply exponential backoff
            delay = task.retry_delay * (2 ** (task.retries - 1))
            await asyncio.sleep(delay)
            
            # Re-queue the task
            await self._queues[task.priority].put(task)
            
            self.logger.warning(
                f"Task {task.id} failed (attempt {task.retries}/{task.max_retries}), "
                f"retrying in {delay}s: {error}"
            )
        else:
            # Task failed permanently
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            
            execution_time = time.time() - task.started_at.timestamp()
            task_result = TaskResult(
                task_id=task.id,
                success=False,
                error=error,
                execution_time=execution_time,
                metadata={"worker_id": worker_id, "retries": task.retries}
            )
            self._task_results[task.id] = task_result
            
            self._stats["tasks_failed"] += 1
            
            self.logger.error(f"Task {task.id} failed permanently: {error}")
    
    async def _process_dependent_tasks(self, completed_task_id: str):
        """Process tasks that were waiting for the completed task."""
        dependent_tasks = self._dependent_tasks.get(completed_task_id, [])
        
        for dep_task_id in dependent_tasks:
            if dep_task_id in self._tasks:
                dep_task = self._tasks[dep_task_id]
                if dep_task.status == TaskStatus.PENDING and self._are_dependencies_satisfied(dep_task):
                    # Dependencies satisfied, queue the task for execution
                    await self._queues[dep_task.priority].put(dep_task)
    
    async def submit_task(
        self,
        task_type: str,
        data: Dict[str, Any],
        callback: Optional[Callable] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: Optional[float] = None,
        dependencies: List[str] = None,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Submit a task to the queue.
        
        Args:
            task_type: Type identifier for the task
            data: Task input data
            callback: Function to execute for the task
            priority: Task priority level
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries (with exponential backoff)
            timeout: Maximum execution time in seconds
            dependencies: List of task IDs this task depends on
            tags: Tags for task categorization
            metadata: Additional metadata
            
        Returns:
            str: Unique task ID
        """
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            task_type=task_type,
            data=data,
            callback=callback,
            priority=priority,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
            dependencies=dependencies or [],
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Register task
        self._tasks[task_id] = task
        
        # Set up dependency tracking
        if task.dependencies:
            for dep_id in task.dependencies:
                self._dependent_tasks[dep_id].append(task_id)
                self._task_dependencies[task_id].append(dep_id)
        
        # Queue task if dependencies are satisfied, otherwise it will be queued later
        if self._are_dependencies_satisfied(task):
            await self._queues[priority].put(task)
        
        self._stats["tasks_submitted"] += 1
        
        self.logger.debug(f"Submitted task {task_id} with priority {priority.name}")
        return task_id
    
    async def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """
        Wait for a task to complete and return its result.
        
        Args:
            task_id: ID of the task to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            TaskResult: The task execution result
            
        Raises:
            asyncio.TimeoutError: If timeout is exceeded
            KeyError: If task ID is not found
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id} not found")
        
        start_time = time.time()
        
        while task_id not in self._task_results:
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Timeout waiting for task {task_id}")
            
            await asyncio.sleep(0.1)  # Small delay to avoid busy waiting
        
        return self._task_results[task_id]
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the current status of a task."""
        if task_id in self._tasks:
            return self._tasks[task_id].status
        return None
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task details by ID."""
        return self._tasks.get(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task if it hasn't started executing yet.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            bool: True if task was cancelled, False otherwise
        """
        if task_id not in self._tasks:
            return False
        
        task = self._tasks[task_id]
        
        if task.status in [TaskStatus.PENDING, TaskStatus.RETRYING]:
            task.status = TaskStatus.CANCELLED
            self._stats["tasks_cancelled"] += 1
            
            # Create cancelled result
            self._task_results[task_id] = TaskResult(
                task_id=task_id,
                success=False,
                error="Task was cancelled",
                metadata={"cancelled_at": datetime.now().isoformat()}
            )
            
            return True
        
        return False
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get current queue statistics."""
        return {
            "queue_sizes": {
                priority.name: queue.qsize()
                for priority, queue in self._queues.items()
            },
            "total_tasks": len(self._tasks),
            "active_workers": len([w for w in self._workers if not w.done()]),
            "statistics": self._stats.copy()
        }
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Get all tasks with a specific status."""
        return [task for task in self._tasks.values() if task.status == status]
    
    def get_tasks_by_tag(self, tag: str) -> List[Task]:
        """Get all tasks with a specific tag."""
        return [task for task in self._tasks.values() if tag in task.tags]
    
    async def wait_for_completion(self, timeout: Optional[float] = None):
        """Wait for all queued tasks to complete."""
        start_time = time.time()
        
        while True:
            # Check if all queues are empty and no tasks are running
            all_empty = all(queue.empty() for queue in self._queues.values())
            no_running = not any(
                task.status == TaskStatus.RUNNING 
                for task in self._tasks.values()
            )
            
            if all_empty and no_running:
                break
            
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError("Timeout waiting for task completion")
            
            await asyncio.sleep(0.5)
    
    async def shutdown(self, timeout: float = 30.0):
        """Shutdown the task queue and cleanup resources."""
        self.logger.info("Shutting down task queue...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel all pending tasks
        for task in self._tasks.values():
            if task.status in [TaskStatus.PENDING, TaskStatus.RETRYING]:
                await self.cancel_task(task.id)
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        # Wait for workers to finish
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._workers, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning("Some workers did not shutdown gracefully")
        
        self.logger.info("Task queue shutdown complete")