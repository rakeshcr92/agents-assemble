import requests
import time
import json

base_url = "http://localhost:8000"

print("=== Life Witness Agent - Enhanced Gemini Test ===\n")

# Test 1: Store a memory with automatic function calling
print("Test 1: Storing a detailed memory")
response = requests.post(
    f"{base_url}/api/text/process",
    json={"text": "I just had an amazing lunch with my colleague Sarah at Chez Pierre restaurant. We discussed the new AI project and she gave me great advice about leadership. I'm feeling really inspired!"}
)
result = response.json()
print(f"Response: {result['response_text']}")
print(f"Function calls: {json.dumps(result.get('function_calls', []), indent=2)}\n")
time.sleep(2)

# Test 2: Store another memory
print("Test 2: Family memory")
response = requests.post(
    f"{base_url}/api/text/process",
    json={"text": "Remember that my nephew Jake's birthday is next week. He's turning 9 and loves robotics. I should get him that Arduino kit we saw."}
)
result = response.json()
print(f"Response: {result['response_text']}")
print(f"Function calls: {json.dumps(result.get('function_calls', []), indent=2)}\n")
time.sleep(2)

# Test 3: Query memories
print("Test 3: Searching for memories")
response = requests.post(
    f"{base_url}/api/text/process",
    json={"text": "What did Sarah tell me about?"}
)
result = response.json()
print(f"Response: {result['response_text']}")
print(f"Function calls: {json.dumps(result.get('function_calls', []), indent=2)}\n")
time.sleep(2)

# Test 4: Complex query
print("Test 4: Complex memory query")
response = requests.post(
    f"{base_url}/api/text/process",
    json={"text": "What upcoming events do I need to remember?"}
)
result = response.json()
print(f"Response: {result['response_text']}")
print(f"Function calls: {json.dumps(result.get('function_calls', []), indent=2)}\n")
time.sleep(2)

# Test 5: Emotional memory
print("Test 5: Emotional context")
response = requests.post(
    f"{base_url}/api/text/process",
    json={"text": "I'm feeling a bit overwhelmed with work. My manager just assigned me three new projects and I don't know how I'll manage everything."}
)
result = response.json()
print(f"Response: {result['response_text']}")
print(f"Function calls: {json.dumps(result.get('function_calls', []), indent=2)}\n")

# Test 6: Memory statistics
print("\nTest 6: Conversation summary")
response = requests.post(
    f"{base_url}/api/text/process",
    json={"text": "Can you summarize what we've talked about today?"}
)
result = response.json()
print(f"Response: {result['response_text']}")
print(f"\nTotal memories stored: {result.get('total_memories', 0)}")
print(f"Conversation length: {result.get('conversation_length', 0)} messages")