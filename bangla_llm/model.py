from .llm import ModelConfig, LlamaModel, get_chat_prompt, get_chat_messages
from .nlu import convert
import logging
from typing import List, Dict, Optional, Generator, Any

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BanglaLLM:
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the BanglaLLM with model configuration and Llama model.

        Args:
            config (Optional[ModelConfig]): Custom model configuration. If None, uses default.
        """
        try:
            logger.info("Initializing BanglaLLM with model configuration")
            self.config = config or ModelConfig()
            self.llm = LlamaModel(self.config)
            logger.info("BanglaLLM initialization completed successfully")
        except Exception as e:
            logger.exception("Failed to initialize BanglaLLM")
            raise RuntimeError(f"BanglaLLM initialization failed: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dict[str, Any]: Model information dictionary.
        """
        try:
            return self.llm.get_model_info()
        except Exception as e:
            logger.exception("Failed to get model info")
            return {"error": str(e)}

    def invoke(self, text: str, **kwargs) -> str:
        """
        Process input text: convert to English, generate response, and convert back to Bengali.
        Uses formatted prompt method.

        Args:
            text (str): The input text to process.
            **kwargs: Additional generation parameters (max_tokens, temperature, etc.)

        Returns:
            str: The final response in Bengali.

        Raises:
            ValueError: If the input text is invalid.
            RuntimeError: If conversion or generation fails.
        """
        logger.info(f'Processing input text: {text}', flush=True)
        logger.info(f'Processing input text: {text}')
        if not isinstance(text, str) or not text.strip():
            logger.error("Invalid input: text must be a non-empty string")
            raise ValueError("Input text must be a non-empty string")

        try:
            # Convert input text to English
            logger.info("Converting input text to English")
            converted = convert(text, target='en')
            logger.info(f"Converted text to English: {converted}")

            # Create formatted prompt
            prompt = get_chat_prompt(converted)

            # Generate response using Llama model
            logger.info("Generating response with Llama model using formatted prompt")
            res = self.llm.generate_with_chat_template(
                prompt,
                max_tokens=kwargs.get('max_tokens'),
                temperature=kwargs.get('temperature'),
                top_p=kwargs.get('top_p'),
                top_k=kwargs.get('top_k'),
                repeat_penalty=kwargs.get('repeat_penalty'),
                stop=kwargs.get('stop')
            )
            logger.info(f"Generated response: {res}")

            # Convert generated response to Bengali
            logger.info("Converting generated response to Bengali")
            response = convert(res, target='bn')
            logger.info(f"Final response in Bengali: {response}")

            return response

        except Exception as e:
            logger.exception("Error during text processing in invoke method")
            raise RuntimeError(f"Text processing failed: {str(e)}")

    def chat_completion(self, text: str, **kwargs) -> str:
        """
        Process input text using chat completion method.

        Args:
            text (str): The input text to process.
            **kwargs: Additional generation parameters

        Returns:
            str: The final response in Bengali.
        """
        if not isinstance(text, str) or not text.strip():
            logger.error("Invalid input: text must be a non-empty string")
            raise ValueError("Input text must be a non-empty string")

        try:
            # Convert input text to English
            logger.info("Converting input text to English for chat completion")
            converted = convert(text, target='en')
            logger.info(f"Converted text to English: {converted}")

            # Create chat messages
            messages = get_chat_messages(converted)

            # Generate response using chat completion
            logger.info("Generating response with chat completion")
            res = self.llm.generate_with_chat_template(messages, **kwargs)
            logger.info(f"Generated response: {res}")

            # Convert generated response to Bengali
            logger.info("Converting generated response to Bengali")
            response = convert(res, target='bn')
            logger.info(f"Final response in Bengali: {response}")

            return response

        except Exception as e:
            logger.exception("Error during chat completion")
            raise RuntimeError(f"Chat completion failed: {str(e)}")

    def stream(self, text: str, **kwargs) -> Generator[str, None, str]:
        """
        Process input text with streaming generation.

        Args:
            text (str): The input text to process.
            **kwargs: Additional generation parameters

        Yields:
            str: Streaming tokens in Bengali.

        Returns:
            str: Complete response in Bengali.
        """
        if not isinstance(text, str) or not text.strip():
            logger.error("Invalid input: text must be a non-empty string")
            raise ValueError("Input text must be a non-empty string")

        try:
            # Convert input text to English
            logger.info("Converting input text to English for streaming")
            converted = convert(text, target='en')
            logger.info(f"Converted text to English: {converted}")

            # Create formatted prompt
            prompt = get_chat_prompt(converted)

            # Generate response with streaming
            logger.info("Starting streaming generation")
            english_response = ""

            # Collect all tokens from streaming
            for token in self.llm.generate(
                    prompt,
                    stream=True,
                    max_tokens=kwargs.get('max_tokens'),
                    temperature=kwargs.get('temperature'),
                    top_p=kwargs.get('top_p'),
                    top_k=kwargs.get('top_k'),
                    repeat_penalty=kwargs.get('repeat_penalty'),
                    stop=kwargs.get('stop')
            ):
                english_response += token

            # Convert complete response to Bengali
            logger.info("Converting streaming response to Bengali")
            bengali_response = convert(english_response, target='bn')
            logger.info(f"Final streaming response in Bengali: {bengali_response}")

            # Yield the complete Bengali response (since we can't translate token by token)
            yield bengali_response
            return bengali_response

        except Exception as e:
            logger.exception("Error during streaming generation")
            raise RuntimeError(f"Streaming generation failed: {str(e)}")

    def generate_raw(self, text: str, **kwargs) -> str:
        """
        Generate response without language conversion (direct English input/output).

        Args:
            text (str): The input text in English.
            **kwargs: Additional generation parameters

        Returns:
            str: The generated response in English.
        """
        if not isinstance(text, str) or not text.strip():
            logger.error("Invalid input: text must be a non-empty string")
            raise ValueError("Input text must be a non-empty string")

        try:
            # Create formatted prompt
            prompt = get_chat_prompt(text)

            # Generate response directly
            logger.info("Generating raw response (no language conversion)")
            response = self.llm.generate(
                prompt,
                max_tokens=kwargs.get('max_tokens'),
                temperature=kwargs.get('temperature'),
                top_p=kwargs.get('top_p'),
                top_k=kwargs.get('top_k'),
                repeat_penalty=kwargs.get('repeat_penalty'),
                stop=kwargs.get('stop')
            )
            logger.info(f"Raw response generated: {response}")
            return response

        except Exception as e:
            logger.exception("Error during raw generation")
            raise RuntimeError(f"Raw generation failed: {str(e)}")

    def chat_completion_raw(self, text: str, **kwargs) -> str:
        """
        Chat completion without language conversion (direct English input/output).

        Args:
            text (str): The input text in English.
            **kwargs: Additional generation parameters

        Returns:
            str: The generated response in English.
        """
        if not isinstance(text, str) or not text.strip():
            logger.error("Invalid input: text must be a non-empty string")
            raise ValueError("Input text must be a non-empty string")

        try:
            # Create chat messages
            messages = get_chat_messages(text)

            # Generate response using chat completion
            logger.info("Generating raw chat completion (no language conversion)")
            response = self.llm.generate_with_chat_template(messages, **kwargs)
            logger.info(f"Raw chat completion response: {response}")
            return response

        except Exception as e:
            logger.exception("Error during raw chat completion")
            raise RuntimeError(f"Raw chat completion failed: {str(e)}")

    def multi_turn_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Handle multi-turn conversation with automatic language conversion.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'.
            **kwargs: Additional generation parameters

        Returns:
            str: The final response in Bengali.
        """
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list")

        try:
            # Convert all user messages to English
            converted_messages = []
            for msg in messages:
                if msg.get('role') == 'user':
                    converted_content = convert(msg['content'], target='en')
                    converted_messages.append({'role': 'user', 'content': converted_content})
                else:
                    converted_messages.append(msg)

            logger.info("Processing multi-turn chat with converted messages")

            # Generate response using chat completion
            res = self.llm.generate_with_chat_template(converted_messages, **kwargs)
            logger.info(f"Multi-turn chat response: {res}")

            # Convert response to Bengali
            response = convert(res, target='bn')
            logger.info(f"Final multi-turn response in Bengali: {response}")

            return response

        except Exception as e:
            logger.exception("Error during multi-turn chat")
            raise RuntimeError(f"Multi-turn chat failed: {str(e)}")

    def batch_process(self, texts: List[str], method: str = 'invoke', **kwargs) -> List[str]:
        """
        Process multiple texts in batch.

        Args:
            texts (List[str]): List of input texts to process.
            method (str): Processing method ('invoke', 'chat_completion', 'raw')
            **kwargs: Additional generation parameters

        Returns:
            List[str]: List of processed responses.
        """
        if not texts or not isinstance(texts, list):
            raise ValueError("Texts must be a non-empty list")

        if method not in ['invoke', 'chat_completion', 'raw']:
            raise ValueError("Method must be 'invoke', 'chat_completion', or 'raw'")

        results = []
        method_func = getattr(self, method)

        try:
            logger.info(f"Starting batch processing of {len(texts)} texts using {method}")

            for i, text in enumerate(texts):
                logger.info(f"Processing text {i + 1}/{len(texts)}")
                result = method_func(text, **kwargs)
                results.append(result)

            logger.info(f"Batch processing completed successfully")
            return results

        except Exception as e:
            logger.exception(f"Error during batch processing")
            raise RuntimeError(f"Batch processing failed: {str(e)}")

    def __call__(self, text: str, **kwargs) -> str:
        """
        Make the class callable, defaults to invoke method.

        Args:
            text (str): The input text to process.
            **kwargs: Additional generation parameters

        Returns:
            str: The final response in Bengali.
        """
        return self.invoke(text, **kwargs)
