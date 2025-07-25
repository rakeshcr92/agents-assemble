"""
Event-driven communication system for coordinating between agents and components.
Provides pub/sub messaging for loose coupling between system components.
"""

import asyncio
import logging
from typing import Dict, List, Callable, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import weakref


class EventPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Event:
    """Represents an event in the system."""
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: EventPriority = EventPriority.NORMAL
    source: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventSubscription:
    """Represents a subscription to an event type."""
    callback: Callable
    event_type: str
    filter_func: Optional[Callable] = None
    once: bool = False
    created_at: datetime = field(default_factory=datetime.now)


class EventBus:
    """
    Central event bus for managing pub/sub communication between agents.
    Supports async event handling, filtering, and priority-based processing.
    """
    
    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        self.logger = logging.getLogger(__name__)
        
        # Event subscriptions organized by event type
        self._subscriptions: Dict[str, List[EventSubscription]] = {}
        
        # Event queues organized by priority
        self._event_queues: Dict[EventPriority, asyncio.Queue] = {
            priority: asyncio.Queue(maxsize=max_queue_size)
            for priority in EventPriority
        }
        
        # Event processing tasks
        self._processing_tasks: List[asyncio.Task] = []
        
        # Event history for debugging and monitoring
        self._event_history: List[Event] = []
        self._max_history_size = 1000
        
        # Statistics
        self._stats = {
            "events_published": 0,
            "events_processed": 0,
            "subscribers_count": 0,
            "processing_errors": 0
        }
        
        # Weak references to avoid memory leaks
        self._weak_subscriptions = weakref.WeakSet()
        
        # Start event processing
        self._start_event_processing()
    
    def _start_event_processing(self):
        """Start background tasks for processing events by priority."""
        for priority in EventPriority:
            task = asyncio.create_task(self._process_events(priority))
            self._processing_tasks.append(task)
    
    async def _process_events(self, priority: EventPriority):
        """Process events from a specific priority queue."""
        queue = self._event_queues[priority]
        
        while True:
            try:
                event = await queue.get()
                if event is None:  # Shutdown signal
                    break
                
                await self._handle_event(event)
                self._stats["events_processed"] += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing {priority.name} priority event: {e}")
                self._stats["processing_errors"] += 1
    
    async def _handle_event(self, event: Event):
        """Handle a single event by notifying all relevant subscribers."""
        subscriptions = self._subscriptions.get(event.event_type, [])
        
        # Also check for wildcard subscriptions
        wildcard_subscriptions = self._subscriptions.get("*", [])
        all_subscriptions = subscriptions + wildcard_subscriptions
        
        # Process subscriptions concurrently
        tasks = []
        subscriptions_to_remove = []
        
        for subscription in all_subscriptions:
            try:
                # Apply filter if present
                if subscription.filter_func:
                    if not await self._apply_filter(subscription.filter_func, event):
                        continue
                
                # Create task for async callback
                task = asyncio.create_task(
                    self._invoke_callback(subscription.callback, event)
                )
                tasks.append(task)
                
                # Mark one-time subscriptions for removal
                if subscription.once:
                    subscriptions_to_remove.append(subscription)
                    
            except Exception as e:
                self.logger.error(f"Error preparing callback for {event.event_type}: {e}")
        
        # Wait for all callbacks to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Remove one-time subscriptions
        for subscription in subscriptions_to_remove:
            self.unsubscribe(subscription.event_type, subscription.callback)
        
        # Add to history
        self._add_to_history(event)
    
    async def _apply_filter(self, filter_func: Callable, event: Event) -> bool:
        """Apply filter function to determine if event should be processed."""
        try:
            if asyncio.iscoroutinefunction(filter_func):
                return await filter_func(event)
            else:
                return filter_func(event)
        except Exception as e:
            self.logger.error(f"Error applying event filter: {e}")
            return False
    
    async def _invoke_callback(self, callback: Callable, event: Event):
        """Invoke a callback function with error handling."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)
        except Exception as e:
            self.logger.error(f"Error in event callback for {event.event_type}: {e}")
    
    def _add_to_history(self, event: Event):
        """Add event to history with size management."""
        self._event_history.append(event)
        
        # Trim history if it exceeds max size
        if len(self._event_history) > self._max_history_size:
            self._event_history = self._event_history[-self._max_history_size:]
    
    async def emit(
        self,
        event_type: str,
        data: Dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL,
        source: Optional[str] = None,
        correlation_id: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Emit an event to the bus.
        
        Args:
            event_type: Type identifier for the event
            data: Event payload data
            priority: Event processing priority
            source: Source identifier (optional)
            correlation_id: Correlation ID for request tracking (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            bool: True if event was queued successfully
        """
        try:
            event = Event(
                event_type=event_type,
                data=data,
                priority=priority,
                source=source,
                correlation_id=correlation_id,
                metadata=metadata or {}
            )
            
            queue = self._event_queues[priority]
            
            # Try to put event in queue (non-blocking)
            try:
                queue.put_nowait(event)
                self._stats["events_published"] += 1
                return True
            except asyncio.QueueFull:
                self.logger.warning(f"Event queue full for priority {priority.name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error emitting event {event_type}: {e}")
            return False
    
    def subscribe(
        self,
        event_type: str,
        callback: Callable,
        filter_func: Optional[Callable] = None,
        once: bool = False
    ) -> EventSubscription:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to subscribe to (use "*" for all events)
            callback: Function to call when event occurs
            filter_func: Optional filter function to apply before calling callback
            once: If True, subscription is automatically removed after first event
            
        Returns:
            EventSubscription: The created subscription object
        """
        subscription = EventSubscription(
            callback=callback,
            event_type=event_type,
            filter_func=filter_func,
            once=once
        )
        
        if event_type not in self._subscriptions:
            self._subscriptions[event_type] = []
        
        self._subscriptions[event_type].append(subscription)
        self._weak_subscriptions.add(subscription)
        self._stats["subscribers_count"] += 1
        
        self.logger.debug(f"New subscription for {event_type}")
        return subscription
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """
        Unsubscribe from events.
        
        Args:
            event_type: Type of events to unsubscribe from
            callback: The callback function to remove
        """
        if event_type in self._subscriptions:
            self._subscriptions[event_type] = [
                sub for sub in self._subscriptions[event_type]
                if sub.callback != callback
            ]
            
            # Clean up empty subscription lists
            if not self._subscriptions[event_type]:
                del self._subscriptions[event_type]
            
            self._stats["subscribers_count"] -= 1
            self.logger.debug(f"Unsubscribed from {event_type}")
    
    def once(
        self,
        event_type: str,
        callback: Callable,
        filter_func: Optional[Callable] = None
    ) -> EventSubscription:
        """
        Subscribe to an event type but only receive it once.
        
        Args:
            event_type: Type of events to subscribe to
            callback: Function to call when event occurs
            filter_func: Optional filter function
            
        Returns:
            EventSubscription: The created subscription object
        """
        return self.subscribe(event_type, callback, filter_func, once=True)
    
    async def wait_for_event(
        self,
        event_type: str,
        timeout: Optional[float] = None,
        filter_func: Optional[Callable] = None
    ) -> Optional[Event]:
        """
        Wait for a specific event to occur.
        
        Args:
            event_type: Type of event to wait for
            timeout: Maximum time to wait in seconds
            filter_func: Optional filter function
            
        Returns:
            Event or None if timeout occurred
        """
        future = asyncio.Future()
        
        def callback(event: Event):
            if not future.done():
                future.set_result(event)
        
        subscription = self.subscribe(event_type, callback, filter_func, once=True)
        
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self.unsubscribe(event_type, callback)
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            **self._stats,
            "queue_sizes": {
                priority.name: queue.qsize()
                for priority, queue in self._event_queues.items()
            },
            "subscription_types": list(self._subscriptions.keys()),
            "active_subscriptions": sum(len(subs) for subs in self._subscriptions.values())
        }
    
    def get_recent_events(self, limit: int = 10) -> List[Event]:
        """Get recent events from history."""
        return self._event_history[-limit:]
    
    def clear_history(self):
        """Clear event history."""
        self._event_history.clear()
    
    async def shutdown(self):
        """Shutdown the event bus and cleanup resources."""
        self.logger.info("Shutting down event bus...")
        
        # Signal all processing tasks to stop
        for priority, queue in self._event_queues.items():
            await queue.put(None)
        
        # Wait for processing tasks to complete
        if self._processing_tasks:
            await asyncio.gather(*self._processing_tasks, return_exceptions=True)
        
        # Clear all subscriptions
        self._subscriptions.clear()
        self._weak_subscriptions.clear()
        
        self.logger.info("Event bus shutdown complete")