# Technical Explanation

## 1. Agent Workflow

Describe step-by-step how your agent processes an input:
1. Receive user input  
2. (Optional) Retrieve relevant memory  
3. Plan sub-tasks (e.g., using ReAct / BabyAGI pattern)  
4. Call tools or APIs as needed  
5. Summarize and return final output  

## 2. Key Modules

- **Planner** (`planner.py`): …  
- **Executor** (`executor.py`): …  
- **Memory Store** (`memory.py`): …  

## 3. Tool Integration

List each external tool or API and how you call it:
- **Search API**: function `search(query)`  
- **Calculator**: LLM function calling  

## 4. Observability & Testing

Explain your logging and how judges can trace decisions:
- Logs saved in `logs/` directory  
- `TEST.sh` exercises main path  

## 5. Known Limitations

Be honest about edge cases or performance bottlenecks:
- Long-running API calls  
- Handling of ambiguous user inputs  

