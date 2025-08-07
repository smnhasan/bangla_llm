import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from langchain_community.llms import LlamaCpp
from huggingface_hub import hf_hub_download
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the LlamaCpp model."""
    model_id: str = os.getenv("LLAMA_MODEL_ID", "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")
    model_basename: str = os.getenv("LLAMA_MODEL_BASENAME", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
    context_window_size: int = int(os.getenv("CONTEXT_WINDOW_SIZE", 8096))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", 8096))
    n_gpu_layers: int = int(os.getenv("N_GPU_LAYERS", 100))
    n_batch: int = int(os.getenv("N_BATCH", 512))
    temperature: float = float(os.getenv("TEMPERATURE", 0.8))
    verbose: bool = os.getenv("VERBOSE", "False").lower() == "true"
    model_directory: str = os.getenv("MODEL_DIRECTORY", "models/llm/models")
    resume_download: bool = True


class LlamaModel:
    """Manages loading and interaction with the LlamaCpp model."""

    def __init__(self, config: ModelConfig):
        """
        Initialize the LlamaCpp model with the given configuration.

        Args:
            config (ModelConfig): Configuration for the model.

        Raises:
            RuntimeError: If model loading fails.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.llm = self._load_model()

    def _load_model(self) -> Optional[LlamaCpp]:
        """
        Load the LlamaCpp model from Hugging Face Hub.

        Returns:
            Optional[LlamaCpp]: Loaded LlamaCpp model instance or None if loading fails.

        Raises:
            RuntimeError: If model downloading or initialization fails.
        """
        try:
            model_path = hf_hub_download(
                repo_id=self.config.model_id,
                filename=self.config.model_basename,
                resume_download=self.config.resume_download,
                cache_dir=self.config.model_directory,
            )
            logger.info(f"Model downloaded to: {model_path}")

            kwargs = {
                "model_path": model_path,
                "temperature": self.config.temperature,
                "n_ctx": self.config.context_window_size,
                "max_tokens": self.config.max_new_tokens,
                "verbose": self.config.verbose,
                "n_batch": self.config.n_batch,
                "streaming": True,
            }

            if self.device.type == "cuda":
                kwargs["n_gpu_layers"] = self.config.n_gpu_layers
                logger.debug(f"Using {self.config.n_gpu_layers} GPU layers")

            return LlamaCpp(**kwargs)
        except Exception as e:
            logger.error(f"Failed to load LlamaCpp model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def generate(self, text: str) -> str:
        """
        Generate a response for a given prompt using the LlamaCpp model.

        Args:
            text (str): Input prompt for the model.

        Returns:
            str: Generated response text.

        Raises:
            ValueError: If the prompt is empty or invalid.
            RuntimeError: If generation fails.
        """
        if not isinstance(text, str) or not text.strip():
            logger.error(f"Invalid text: must be a non-empty string. Got : {text}")
            raise ValueError("Text must be a non-empty string")

        try:
            prompt = get_chat_prompt(text)
            response = self.llm.generate([prompt])
            text = response.flatten()
            generated_text = text[0].generations[0][0].text
            logger.info("Successfully generated response")
            return generated_text
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise RuntimeError(f"Generation failed: {str(e)}")


def get_chat_prompt(user_input: str) -> str:
    """
    Create a formatted chat prompt with system instructions and user input.

    Args:
        user_input (str): User's input text.

    Returns:
        str: Formatted prompt string.

    Raises:
        ValueError: If user_input is empty or invalid.
    """
    if not isinstance(user_input, str) or not user_input.strip():
        logger.error("Invalid user input: must be a non-empty string")
        raise ValueError("User input must be a non-empty string")

    today_date = datetime.today().strftime("%d %B %Y")
    prompt = (
        "<|start_header_id|>system<|end_header_id>\n\n"
        f"Cutting Knowledge Date: December 2023\n"
        f"Today Date: {today_date}\n\n"
        "You are an assistant. You always provide correct information.\n"        
        "<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id>\n\n{user_input}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id>\n\n"
    )
    logger.debug("Generated chat prompt")
    return prompt


def main():
    """Example usage of the LlamaModel and prompt generation."""
    config = ModelConfig()
    try:
        # Initialize model
        llama_model = LlamaModel(config)

        # Example prompt
        user_input = "Can you help me with Python coding?"
        prompt = get_chat_prompt(user_input)

        # Generate response
        response = llama_model.generate(prompt)
        print(f"User: {user_input}")
        print(f"Assistant: {response}")
    except (ValueError, RuntimeError) as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"Error: {str(e)}")
