from langchain.llms.base import LLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from base_agent import BaseAgent  # your custom base class
from langchain_google_vertexai import ChatVertexAI

class ResponseAgent(BaseAgent):
    def __init__(self):
        # Initialize the Google Vertex AI chat model (Gemini)
        self.llm = ChatVertexAI(
            model_name="gemini-2.5-pro",          # update model if needed
            project="your-project-id",             # replace with your GCP project ID
            location="us-central1"                  # update region if needed
        )
        
        # Prompt template designed for Life Witness app with memory context
        prompt_template = PromptTemplate(
            input_variables=["input", "context"],
            template="""
You are Life Witness, a personal AI memory assistant that listens to life stories, captures moments, and helps recall memories with rich context.

You have access to the user's life memories, including events, photos, calendar entries, and relationships.

Respond in a warm, empathetic, and conversational tone — like a trusted friend who remembers important details.

Use the provided context to ground your replies, including dates, locations, people, and emotional cues.

If the user asks about past events, recall details clearly and helpfully.

If the user wants to set reminders or follow up, help create clear action items.

Do NOT say you are an AI or mention technical details.

---

Context:
{context}

User:
{input}

Life Witness:
"""
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=prompt_template)

    def run(self, input_text: str, context: str = "") -> str:
        """
        Generate a conversational response using the input text and optional memory context.

        Args:
            input_text (str): The user query or statement.
            context (str, optional): A string summarizing relevant memories or context. Defaults to "".

        Returns:
            str: The agent's generated response.
        """
        return self.chain.run(input=input_text, context=context)

#Example Usage:

"""
response_agent = ResponseAgent()
response = response_agent.run(user_input, memory_context)
"""


"""
if __name__ == "__main__":
    # Make sure you have authenticated your GCP credentials via:
    # gcloud auth application-default login
    
    print("Initializing ResponseAgent...")
    agent = ResponseAgent()
    print("Agent initialized.")

    prompt = "Hello, how can you assist me today?"
    print(f"Sending prompt: \"{prompt}\"")
    try:
        reply = agent.run(prompt)
        print(f"Agent reply: {reply}")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Check your Google Cloud authentication and project/location settings.")
"""
