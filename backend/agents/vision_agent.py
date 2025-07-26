from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from base_agent import BaseAgent
from dotenv import load_dotenv
import base64


load_dotenv()


class VisionAgent(BaseAgent):
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initializes the VisionAgent with a specified Google Generative AI model.

        Args:
            model_name (str): The name of the model to use.
        """
        super().__init__()
        self.model = ChatGoogleGenerativeAI(model=model_name)
        print(f"VisionAgent initialized with model: {model_name}")

    def describe(self, image_path: str) -> str:
        """
        Creates a detailed textual description of the image at the given file path.

        Args:
            image_path (str): The file path to the image to be described.

        Returns:
            str: A textual description of the image.
        """
        # Read the image file and encode it in base64
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        except FileNotFoundError:
            return f"Error: Image file not found at '{image_path}'"

        # Prepare the message for the model
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image in detail."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )

        # Invoke the model and return the content of the response
        response = self.model.invoke([message])
        return response.content

if __name__=="__main__":

    vision_agent = VisionAgent()

    image_file_path = "path_to_image"
  
    description = vision_agent.describe(image_file_path)

    print("\n--- Image Description ---")
    print(description)
