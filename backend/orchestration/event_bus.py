import asyncio
from typing import Dict, List, Any, Callable, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import logging
import uuid
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import BaseMessage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Represents an event in the system"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventBusCallback(AsyncCallbackHandler):
    """LangChain callback handler that publishes events to the event bus"""
    
    def __init__(self, event_bus: 'EventBus', source: str = "langchain"):
        self.event_bus = event_bus
        self.source = source
        
    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain starts running."""
        await self.event_bus.publish(Event(
            type="CHAIN_STARTED",
            data={"serialized": serialized, "inputs": inputs},
            source=self.source
        ))
        
    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain ends running."""
        await self.event_bus.publish(Event(
            type="CHAIN_COMPLETED",
            data={"outputs": outputs},
            source=self.source
        ))
        
    async def on_chain_error(self, error: Exception, **kwargs) -> None:
        """Called when a chain errors."""
        await self.event_bus.publish(Event(
            type="CHAIN_ERROR",
            data={"error": str(error)},
            source=self.source
        ))
        
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts running."""
        await self.event_bus.publish(Event(
            type="LLM_STARTED",
            data={"prompts": prompts},
            source=self.source
        ))
        
    async def on_llm_end(self, response: Any, **kwargs) -> None:
        """Called when LLM ends running."""
        await self.event_bus.publish(Event(
            type="LLM_COMPLETED",
            data={"response": str(response)},
            source=self.source
        ))


class EventSubscription:
    """Represents a subscription to events"""
    
    def __init__(self, event_type: str, handler: Callable, filter_fn: Optional[Callable] = None):
        self.id = str(uuid.uuid4())
        self.event_type = event_type
        self.handler = handler
        self.filter_fn = filter_fn
        self.created_at = datetime.now()
        self.call_count = 0


class EventBus:
    """Event bus implementation for LangChain agents"""
    
    def __init__(self, max_event_history: int = 1000):
        self.subscriptions: Dict[str, List[EventSubscription]] = defaultdict(list)
        self.event_history: List[Event] = []
        self.max_event_history = max_event_history
        self.active_handlers: Set[str] = set()
        self._lock = asyncio.Lock()
        
    def create_callback_handler(self, source: str = "langchain") -> EventBusCallback:
        """Create a LangChain callback handler that publishes to this event bus"""
        return EventBusCallback(self, source)
        
    async def subscribe(self, event_type: str, handler: Callable, filter_fn: Optional[Callable] = None) -> str:
        """Subscribe to an event type"""
        if not asyncio.iscoroutinefunction(handler):
            raise ValueError("Handler must be an async function")
            
        subscription = EventSubscription(event_type, handler, filter_fn)
        
        async with self._lock:
            self.subscriptions[event_type].append(subscription)
            
        logger.info(f"Subscribed to {event_type} with handler {handler.__name__} (ID: {subscription.id})")
        return subscription.id
        
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events"""
        async with self._lock:
            for event_type, subs in self.subscriptions.items():
                for sub in subs:
                    if sub.id == subscription_id:
                        subs.remove(sub)
                        logger.info(f"Unsubscribed {subscription_id} from {event_type}")
                        return True
        return False
        
    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers"""
        if isinstance(event, dict):
            # Convert dict to Event object for backwards compatibility
            event = Event(
                type=event.get("type", "UNKNOWN"),
                data=event.get("data", {}),
                source=event.get("source", "unknown")
            )
            
        # Add to history
        async with self._lock:
            self.event_history.append(event)
            # Trim history if needed
            if len(self.event_history) > self.max_event_history:
                self.event_history = self.event_history[-self.max_event_history:]
                
        logger.info(f"Publishing event: {event.type} from {event.source}")
        
        # Get subscribers for this event type
        subscribers = self.subscriptions.get(event.type, [])
        
        # Also get wildcard subscribers
        wildcard_subscribers = self.subscriptions.get("*", [])
        
        all_subscribers = subscribers + wildcard_subscribers
        
        # Call all handlers concurrently
        tasks = []
        for subscription in all_subscribers:
            # Apply filter if present
            if subscription.filter_fn and not subscription.filter_fn(event):
                continue
                
            # Create handler task
            task = asyncio.create_task(self._call_handler(subscription, event))
            tasks.append(task)
            
        # Wait for all handlers to complete
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Handler error: {result}")
                    
    async def _call_handler(self, subscription: EventSubscription, event: Event) -> None:
        """Call a single event handler"""
        handler_id = f"{subscription.id}-{event.id}"
        
        # Prevent duplicate handling
        if handler_id in self.active_handlers:
            return
            
        self.active_handlers.add(handler_id)
        
        try:
            logger.debug(f"Calling handler {subscription.handler.__name__} for event {event.type}")
            subscription.call_count += 1
            await subscription.handler(event)
            
        except Exception as e:
            logger.error(f"Error in handler {subscription.handler.__name__}: {str(e)}")
            # Publish error event
            error_event = Event(
                type="HANDLER_ERROR",
                data={
                    "original_event": event.type,
                    "handler": subscription.handler.__name__,
                    "error": str(e)
                },
                source="event_bus"
            )
            # Don't await to avoid recursion
            asyncio.create_task(self.publish(error_event))
            
        finally:
            self.active_handlers.discard(handler_id)
            
    async def publish_batch(self, events: List[Event]) -> None:
        """Publish multiple events"""
        tasks = [self.publish(event) for event in events]
        await asyncio.gather(*tasks)
        
    async def get_event_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Event]:
        """Get event history, optionally filtered by type"""
        async with self._lock:
            if event_type:
                filtered = [e for e in self.event_history if e.type == event_type]
                return filtered[-limit:]
            else:
                return self.event_history[-limit:]
                
    async def get_subscription_stats(self) -> Dict[str, Any]:
        """Get statistics about subscriptions"""
        stats = {
            "total_subscriptions": sum(len(subs) for subs in self.subscriptions.values()),
            "event_types": list(self.subscriptions.keys()),
            "subscriptions_by_type": {}
        }
        
        for event_type, subs in self.subscriptions.items():
            stats["subscriptions_by_type"][event_type] = {
                "count": len(subs),
                "handlers": [sub.handler.__name__ for sub in subs],
                "total_calls": sum(sub.call_count for sub in subs)
            }
            
        return stats
        
    async def clear_event_history(self) -> None:
        """Clear the event history"""
        async with self._lock:
            self.event_history.clear()
            
    def create_typed_publisher(self, event_type: str, source: str) -> Callable:
        """Create a publisher function for a specific event type"""
        async def publisher(data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
            event = Event(
                type=event_type,
                data=data,
                source=source,
                metadata=metadata or {}
            )
            await self.publish(event)
        return publisher


# Global event bus instance for convenience
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance"""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


# Test function
async def test_event_bus():
    """Test the event bus independently"""
    event_bus = EventBus()
    
    # Test data collection
    received_events = []
    
    # Define test handlers
    async def handler1(event: Event):
        logger.info(f"Handler1 received: {event.type}")
        received_events.append(("handler1", event))
        
    async def handler2(event: Event):
        logger.info(f"Handler2 received: {event.type}")
        received_events.append(("handler2", event))
        await asyncio.sleep(0.1)  # Simulate some work
        
    async def wildcard_handler(event: Event):
        logger.info(f"Wildcard handler received: {event.type}")
        received_events.append(("wildcard", event))
        
    # Subscribe to events
    sub1 = await event_bus.subscribe("TEST_EVENT", handler1)
    sub2 = await event_bus.subscribe("TEST_EVENT", handler2)
    sub3 = await event_bus.subscribe("ANOTHER_EVENT", handler1)
    sub_wildcard = await event_bus.subscribe("*", wildcard_handler)
    
    # Publish events
    await event_bus.publish(Event(
        type="TEST_EVENT",
        data={"message": "Hello from test 1"},
        source="test"
    ))
    
    await event_bus.publish(Event(
        type="ANOTHER_EVENT",
        data={"message": "Another event"},
        source="test"
    ))
    
    # Test batch publish
    batch_events = [
        Event(type="TEST_EVENT", data={"batch": i}, source="batch_test")
        for i in range(3)
    ]
    await event_bus.publish_batch(batch_events)
    
    # Wait for handlers to complete
    await asyncio.sleep(0.5)
    
    # Get statistics
    stats = await event_bus.get_subscription_stats()
    print(f"Subscription stats: {stats}")
    
    # Get event history
    history = await event_bus.get_event_history()
    print(f"Event history count: {len(history)}")
    
    # Test typed publisher
    test_publisher = event_bus.create_typed_publisher("TYPED_EVENT", "typed_test")
    await test_publisher({"value": 42})
    
    # Test LangChain callback
    callback = event_bus.create_callback_handler("langchain_test")
    await callback.on_chain_start({}, {"input": "test"})
    await callback.on_chain_end({"output": "result"})
    
    # Final statistics
    print(f"Total events received: {len(received_events)}")
    for handler_name, event in received_events:
        print(f"  {handler_name}: {event.type}")
        
    # Unsubscribe
    await event_bus.unsubscribe(sub1)
    
    # Clear history
    await event_bus.clear_event_history()


if __name__ == "__main__":
    asyncio.run(test_event_bus())