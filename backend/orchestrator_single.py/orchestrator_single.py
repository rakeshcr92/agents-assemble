from typing import TypedDict, Dict, Any
from langgraph.graph import Graph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import google.generativeai as genai
import base64
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini models
gemini_model = genai.GenerativeModel('gemini-1.5-pro')
gemini_flash = genai.GenerativeModel('gemini-1.5-flash')

# Initialize LangChain Gemini for conversation
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

# Initialize conversational memory
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)

# Simple memory storage
stored_memories = []

class VoiceAgent:
    """Handles voice operations using Gemini"""
    def __init__(self):
        self.gemini_flash = gemini_flash
    
    def transcribe(self, audio_data: bytes) -> str:
        """For MVP, we'll skip actual audio transcription"""
        # In production, you would upload audio and transcribe
        # For now, return a placeholder
        return "Transcribed audio content"
    
    def synthesize(self, text: str) -> bytes:
        """Mock audio synthesis"""
        return b"mock_audio_data"

class MemoryAgent:
    """Manages conversation memory"""
    def __init__(self, memory: ConversationBufferMemory):
        self.memory = memory
        self.gemini_model = gemini_model
    
    def add_interaction(self, user_input: str, ai_response: str):
        """Add user-AI interaction to memory"""
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(ai_response)
        
        # Extract and store key information
        if "remember" in user_input.lower() or "don't forget" in user_input.lower():
            stored_memories.append({
                "id": len(stored_memories) + 1,
                "content": user_input,
                "response": ai_response,
                "type": "explicit_memory"
            })
    
    def get_context(self) -> str:
        """Get conversation history as context"""
        messages = self.memory.chat_memory.messages
        context = []
        for msg in messages[-10:]:  # Last 10 messages
            if isinstance(msg, HumanMessage):
                context.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                context.append(f"Assistant: {msg.content}")
        return "\n".join(context)
    
    def search_memories(self, query: str) -> list:
        """Simple memory search"""
        results = []
        query_lower = query.lower()
        for memory in stored_memories:
            if query_lower in memory["content"].lower():
                results.append(memory)
        return results

class ResponseAgent:
    """Generate responses using Gemini"""
    def __init__(self, memory_agent):
        self.memory_agent = memory_agent
        self.gemini_model = gemini_model
    
    def generate(self, user_input: str) -> str:
        """Generate contextual response"""
        context = self.memory_agent.get_context()
        
        # Check if user is asking about memories
        is_memory_query = any(word in user_input.lower() for word in ["what", "when", "who", "remember", "told", "said"])
        
        memories = []
        if is_memory_query:
            memories = self.memory_agent.search_memories(user_input)
        
        # Build prompt
        prompt = f"""You are a Life Witness Agent - a personal AI companion that helps people capture and remember their life experiences.

Previous conversation:
{context}

{"Relevant memories found: " + str(memories) if memories else ""}

User's current message: {user_input}

Instructions:
- If the user wants to remember something, acknowledge it warmly and confirm you'll remember it
- If they're asking about past memories, help them recall based on conversation history
- Be conversational, warm, and show genuine interest
- Keep responses concise but meaningful

Response:"""

        # Generate response using Gemini
        response = self.gemini_model.generate_content(prompt)
        return response.text.strip()

# Initialize agents
voice_agent = VoiceAgent()
memory_agent = MemoryAgent(memory)
response_agent = ResponseAgent(memory_agent)

# Define state schema
class AgentState(TypedDict):
    audio_input: bytes
    transcribed_text: str
    response_text: str
    audio_output: bytes
    error: str

# Node functions
def transcribe_audio(state: AgentState) -> AgentState:
    """Convert audio to text"""
    try:
        if state.get("audio_input"):
            text = voice_agent.transcribe(state["audio_input"])
            state["transcribed_text"] = text
        else:
            # For testing without audio
            state["transcribed_text"] = state.get("test_text", "Hello")
    except Exception as e:
        state["error"] = f"Transcription error: {str(e)}"
        state["transcribed_text"] = "Could not understand audio"
    return state

def generate_response(state: AgentState) -> AgentState:
    """Generate AI response with memory context"""
    try:
        user_input = state["transcribed_text"]
        response = response_agent.generate(user_input)
        state["response_text"] = response
        
        # Update memory with this interaction
        memory_agent.add_interaction(user_input, response)
        
    except Exception as e:
        state["error"] = f"Response generation error: {str(e)}"
        state["response_text"] = "I'm sorry, I couldn't process that request."
    return state

def synthesize_speech(state: AgentState) -> AgentState:
    """Convert response text to speech"""
    try:
        audio_data = voice_agent.synthesize(state["response_text"])
        state["audio_output"] = audio_data
    except Exception as e:
        state["error"] = f"Speech synthesis error: {str(e)}"
        state["audio_output"] = b""
    return state

# Build the LangGraph workflow
workflow = Graph()

# Add nodes
workflow.add_node("transcribe", transcribe_audio)
workflow.add_node("generate", generate_response)
workflow.add_node("synthesize", synthesize_speech)

# Define edges
workflow.add_edge("transcribe", "generate")
workflow.add_edge("generate", "synthesize")
workflow.add_edge("synthesize", END)

# Set entry point
workflow.set_entry_point("transcribe")

# Compile the graph
app = workflow.compile()

# Main orchestration function
async def process_voice_input(audio_data: bytes = None, test_text: str = None) -> Dict[str, Any]:
    """Process voice input through the workflow"""
    initial_state = {
        "audio_input": audio_data,
        "transcribed_text": "",
        "response_text": "",
        "audio_output": b"",
        "error": "",
        "test_text": test_text
    }
    
    # Run the workflow
    result = await app.ainvoke(initial_state)
    
    return {
        "transcribed_text": result.get("transcribed_text"),
        "response_text": result.get("response_text"),
        "audio_output": base64.b64encode(result.get("audio_output", b"")).decode('utf-8'),
        "error": result.get("error"),
        "conversation_length": len(memory.chat_memory.messages),
        "total_memories": len(stored_memories)
    }

# Synchronous wrapper for Flask
def process_voice_sync(audio_data: bytes = None, test_text: str = None) -> Dict[str, Any]:
    """Synchronous wrapper for the async orchestration"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(process_voice_input(audio_data, test_text))