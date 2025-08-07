from .llm import ModelConfig, LlamaModel
from .nlu import convert
import logging

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class BanglaLLM:
    def __init__(self):
        """
        Initialize the BanglaLLM with model configuration and Llama model.
        """
        try:
            logger.info("Initializing BanglaLLM with model configuration")
            self.config = ModelConfig()
            self.llm = LlamaModel(self.config)
            logger.info("BanglaLLM initialization completed successfully")
        except Exception as e:
            logger.exception("Failed to initialize BanglaLLM")
            raise RuntimeError(f"BanglaLLM initialization failed: {str(e)}")

    def invoke(self, text: str) -> str:
        """
        Process input text: convert to English, generate response, and convert back to Bengali.

        Args:
            text (str): The input text to process.

        Returns:
            str: The final response in Bengali.

        Raises:
            ValueError: If the input text is invalid.
            RuntimeError: If conversion or generation fails.
        """
        if not isinstance(text, str) or not text.strip():
            logger.error("Invalid input: text must be a non-empty string")
            raise ValueError("Input text must be a non-empty string")

        try:
            # Convert input text to English
            logger.info("Converting input text to English")
            converted = convert(text, target='en')
            logger.info(f"Converted text to English: {converted}")

            # Generate response using Llama model
            logger.info("Generating response with Llama model")
            res = self.llm.generate(converted)
            logger.info(f"Generated response: {res}")

            # Convert generated response to Bengali
            logger.info("Converting generated response to Bengali")
            response = convert(res, target='bn')
            logger.info(f"Final response in Bengali: {response}")

            return response

        except Exception as e:
            logger.exception("Error during text processing in invoke method")
            raise RuntimeError(f"Text processing failed: {str(e)}")
