from typing import List
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from normalizer import normalize
import torch
import logging

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# === Device Setup ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# === Preload Models and Tokenizers ===
try:
    logger.info("Loading bn→en model and tokenizer...")
    _bn_en_model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/banglat5_nmt_bn_en").to(DEVICE)
    _bn_en_tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglat5_nmt_bn_en", use_fast=False)

    logger.info("Loading en→bn model and tokenizer...")
    _en_bn_model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/banglat5_nmt_en_bn").to(DEVICE)
    _en_bn_tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglat5_nmt_en_bn", use_fast=False)

except Exception as e:
    logger.exception("Failed to preload translation models.")
    raise RuntimeError(f"Model/tokenizer preloading failed: {str(e)}")


# === Translation Class ===
class TranslationModel:
    """Translation wrapper using preloaded model and tokenizer."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def translate(self, text: str, max_tokens: int = 128) -> str:
        """
        Translate a single text using the preloaded model and tokenizer.

        Args:
            text (str): The input text to translate.
            max_tokens (int): Maximum number of tokens to generate (default: 128).

        Returns:
            str: The translated text.

        Raises:
            ValueError: If the input text is empty or not a string.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")

        logger.info("Processing single text for translation")
        try:
            normalized_text = normalize(text)
            inputs = self.tokenizer(normalized_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_tokens
                )
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation
        except Exception as e:
            logger.exception("Translation failed for text")
            raise RuntimeError(f"Translation failed: {str(e)}")

    def translate_batch(self, sentences: List[str], max_tokens: int = 128, batch_size: int = 32) -> List[str]:
        """
        Translate a batch of texts using the preloaded model and tokenizer.

        Args:
            sentences (List[str]): List of input texts to translate.
            max_tokens (int): Maximum number of tokens to generate (default: 128).
            batch_size (int): Number of texts to process per batch (default: 32).

        Returns:
            List[str]: List of translated texts.

        Raises:
            ValueError: If the input list is empty or contains invalid texts.
        """
        if not sentences:
            raise ValueError("Input sentence list cannot be empty")
        if not all(isinstance(s, str) and s.strip() for s in sentences):
            raise ValueError("All sentences must be non-empty strings")

        normalized = [normalize(s) for s in sentences]
        translations = []

        for i in range(0, len(normalized), batch_size):
            batch = normalized[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1} with {len(batch)} sentences")
            try:
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_tokens
                    )
                batch_translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                translations.extend(batch_translations)
            except Exception as e:
                logger.exception(f"Translation failed for batch starting at index {i}")
                raise RuntimeError(f"Translation failed: {str(e)}")

        return translations


# === Singleton Instances ===
bn_to_en_model = TranslationModel(_bn_en_model, _bn_en_tokenizer, DEVICE)
en_to_bn_model = TranslationModel(_en_bn_model, _en_bn_tokenizer, DEVICE)


# === Public Functions ===
def translate_bn_to_en(text: str, max_tokens: int = 128) -> str:
    """
    Translate a single Bengali text to English.

    Args:
        text (str): The Bengali text to translate.
        max_tokens (int): Maximum number of tokens to generate (default: 128).

    Returns:
        str: The translated English text.
    """
    return bn_to_en_model.translate(text, max_tokens)


def translate_en_to_bn(text: str, max_tokens: int = 128) -> str:
    """
    Translate a single English text to Bengali.

    Args:
        text (str): The English text to translate.
        max_tokens (int): Maximum number of tokens to generate (default: 128).

    Returns:
        str: The translated Bengali text.
    """
    return en_to_bn_model.translate(text, max_tokens)


def translate_bn_to_en_batch(sentences: List[str], max_tokens: int = 128, batch_size: int = 32) -> List[str]:
    """
    Translate a batch of Bengali texts to English.

    Args:
        sentences (List[str]): List of Bengali texts to translate.
        max_tokens (int): Maximum number of tokens to generate (default: 128).
        batch_size (int): Number of texts to process per batch (default: 32).

    Returns:
        List[str]: List of translated English texts.
    """
    return bn_to_en_model.translate_batch(sentences, max_tokens, batch_size)


def translate_en_to_bn_batch(sentences: List[str], max_tokens: int = 128, batch_size: int = 32) -> List[str]:
    """
    Translate a batch of English texts to Bengali.

    Args:
        sentences (List[str]): List of English texts to translate.
        max_tokens (int): Maximum number of tokens to generate (default: 128).
        batch_size (int): Number of texts to process per batch (default: 32).

    Returns:
        List[str]: List of translated Bengali texts.
    """
    return en_to_bn_model.translate_batch(sentences, max_tokens, batch_size)


def is_bengali_text(text: str, threshold: float = 0.5) -> bool:
    """
    Check if the given text is primarily in Bengali based on Unicode character ranges.

    Args:
        text (str): The input text to check.
        threshold (float): Minimum proportion of Bengali characters required (default: 0.5).

    Returns:
        bool: True if the text is primarily Bengali, False otherwise.
    """
    if not text or not isinstance(text, str):
        return False

    bengali_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
    total_chars = sum(1 for char in text if char.isalpha())

    if total_chars == 0:
        return False

    return (bengali_chars / total_chars) >= threshold


def convert(text: str, target: str) -> str:
    """
    Convert text to the target language (Bengali or English) based on its current language.

    Args:
        text (str): The input text to convert.
        target (str): The target language ('bn' for Bengali, 'en' for English).

    Returns:
        str: The translated text if conversion is needed, otherwise the original text.

    Raises:
        ValueError: If target is not 'bn' or 'en', or if text is invalid.
    """
    if target not in ['bn', 'en']:
        raise ValueError("Target language must be 'bn' or 'en'")

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string")

    if target == "bn" and not is_bengali_text(text):
        return translate_en_to_bn(text)
    elif target == "en" and is_bengali_text(text):
        return translate_bn_to_en(text)
    else:
        return text
