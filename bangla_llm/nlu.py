from typing import List, Tuple
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from normalizer import normalize
import torch
import logging
import re
import time

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# === Device Setup ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# === Constants ===
MAX_CHUNK_TOKENS = 100  # Even safer limit below 512
MIN_CHUNK_TOKENS = 0  # Minimum chunk size for efficiency
BATCH_SIZE = 16  # Reduced batch size for longer chunks
MODEL_MAX_LENGTH = 512  # Hard model limit

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


# === Text Chunking Utilities ===
def split_into_sentences(text: str, is_bengali: bool = False) -> List[str]:
    """
    Split text into sentences based on language-specific patterns.

    Args:
        text (str): Input text to split.
        is_bengali (bool): Whether the text is in Bengali.

    Returns:
        List[str]: List of sentences.
    """
    if is_bengali:
        # Bengali sentence delimiters - more comprehensive
        pattern = r'([।!?।।]+)\s*'
        sentences = re.split(pattern, text)

        # Rejoin sentences with their delimiters
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            delimiter = sentences[i + 1] if i + 1 < len(sentences) else ''
            if sentence:
                result.append(sentence + delimiter)

        # Add last sentence if it doesn't have delimiter
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())

    else:
        # English sentence delimiters
        pattern = r'([.!?]+)\s+'
        sentences = re.split(pattern, text)

        # Rejoin sentences with their delimiters
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            delimiter = sentences[i + 1] if i + 1 < len(sentences) else ''
            if sentence:
                result.append(sentence + delimiter + ' ')

        # Add last sentence if it doesn't have delimiter
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())

    # Clean and filter sentences
    result = [s.strip() for s in result if s.strip()]
    return result


def count_tokens(text: str, tokenizer: PreTrainedTokenizer) -> int:
    """Count tokens in text using the given tokenizer."""
    try:
        tokens = tokenizer.encode(text, add_special_tokens=True, truncation=False)
        return len(tokens)
    except Exception:
        # Fallback: rough estimation (1 token ≈ 4 characters for English, 6 for Bengali)
        return len(text) // 4


def create_chunks(text: str, tokenizer: PreTrainedTokenizer, is_bengali: bool = False) -> List[str]:
    """
    Split text into chunks that fit within token limits while preserving sentence boundaries.

    Args:
        text (str): Input text to chunk.
        tokenizer: Tokenizer to count tokens.
        is_bengali (bool): Whether text is in Bengali.

    Returns:
        List[str]: List of text chunks.
    """
    # First check if text fits in one chunk
    total_tokens = count_tokens(text, tokenizer)
    if total_tokens <= MAX_CHUNK_TOKENS:
        return [text]

    logger.info(f"Text has {total_tokens} tokens, splitting into chunks...")

    # Split into sentences
    sentences = split_into_sentences(text, is_bengali)

    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence, tokenizer)

        # If single sentence exceeds limit, split it by words/phrases
        if sentence_tokens > MAX_CHUNK_TOKENS:
            logger.warning(f"Single sentence has {sentence_tokens} tokens, splitting by words...")

            # Save current chunk if not empty
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_tokens = 0

            # Split long sentence by words/phrases
            word_chunks = split_long_sentence(sentence, tokenizer, is_bengali)
            chunks.extend(word_chunks)
            continue

        # Check if adding this sentence would exceed limit
        if current_tokens + sentence_tokens > MAX_CHUNK_TOKENS:
            # Save current chunk and start new one
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = sentence_tokens
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_tokens += sentence_tokens

    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Final safety check - verify all chunks are within limits
    verified_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_tokens = count_tokens(chunk, tokenizer)
        if chunk_tokens > MAX_CHUNK_TOKENS:
            logger.warning(f"Chunk {i} still has {chunk_tokens} tokens, re-splitting...")
            # Emergency re-split
            sub_chunks = split_long_sentence(chunk, tokenizer, is_bengali)
            verified_chunks.extend(sub_chunks)
        else:
            verified_chunks.append(chunk)

    logger.info(f"Created {len(verified_chunks)} verified chunks")
    return verified_chunks


def split_long_sentence(sentence: str, tokenizer: PreTrainedTokenizer, is_bengali: bool) -> List[str]:
    """
    Split a very long sentence into smaller chunks by words/phrases.

    Args:
        sentence (str): Long sentence to split.
        tokenizer: Tokenizer for counting tokens.
        is_bengali (bool): Whether text is Bengali.

    Returns:
        List[str]: List of sentence chunks.
    """
    # Split by comma, semicolon, or other natural breaks first
    if is_bengali:
        parts = re.split(r'[,;।]', sentence)
    else:
        parts = re.split(r'[,;]', sentence)

    chunks = []
    current_chunk = ""
    current_tokens = 0

    for part in parts:
        part = part.strip()
        if not part:
            continue

        part_tokens = count_tokens(part, tokenizer)

        if part_tokens > MAX_CHUNK_TOKENS:
            # Split by words as last resort
            words = part.split()
            word_chunk = ""
            word_tokens = 0

            for word in words:
                word_token_count = count_tokens(word, tokenizer)
                if word_tokens + word_token_count > MAX_CHUNK_TOKENS:
                    if word_chunk:
                        chunks.append(word_chunk.strip())
                    word_chunk = word
                    word_tokens = word_token_count
                else:
                    word_chunk += " " + word if word_chunk else word
                    word_tokens += word_token_count

            if word_chunk:
                chunks.append(word_chunk.strip())
            continue

        if current_tokens + part_tokens > MAX_CHUNK_TOKENS:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = part
            current_tokens = part_tokens
        else:
            current_chunk += ", " + part if current_chunk else part
            current_tokens += part_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# === Enhanced Translation Class ===
class TranslationModel:
    """Translation wrapper with chunking support for long texts."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def translate(self, text: str, max_tokens: int = 128) -> str:
        """
        Translate a single text, handling long texts by chunking with verification.

        Args:
            text (str): The input text to translate.
            max_tokens (int): Maximum number of tokens to generate per chunk.

        Returns:
            str: The translated text.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")

        try:
            # Log original text stats
            original_length = len(text)
            logger.info(f"Starting translation of text with {original_length} characters")

            # Determine if input is Bengali
            is_bengali = is_bengali_text(text)
            logger.info(f"Detected language: {'Bengali' if is_bengali else 'English'}")

            # Create chunks
            chunks = create_chunks(text, self.tokenizer, is_bengali)
            logger.info(f"Split into {len(chunks)} chunks")

            # Log chunk details
            total_chunk_chars = 0
            for i, chunk in enumerate(chunks):
                chunk_tokens = count_tokens(chunk, self.tokenizer)
                chunk_chars = len(chunk)
                total_chunk_chars += chunk_chars
                logger.info(f"Chunk {i + 1}: {chunk_tokens} tokens, {chunk_chars} chars")
                logger.debug(f"Chunk {i + 1} preview: {chunk[:100]}...")

            # Verify we didn't lose any content during chunking
            if total_chunk_chars < original_length * 0.95:  # Allow 5% loss for whitespace
                logger.warning(f"Potential content loss: {original_length} -> {total_chunk_chars} chars")

            if len(chunks) == 1:
                # Single chunk - use original method
                logger.info("Using single chunk translation")
                result = self._translate_single_chunk(chunks[0], max_tokens)
            else:
                # Multiple chunks - use batch translation
                logger.info(f"Translating {len(chunks)} chunks in batches")
                translated_chunks = self.translate_batch(chunks, max_tokens, batch_size=BATCH_SIZE)

                # Join translated chunks with appropriate spacing
                if is_bengali:
                    # For Bengali, use single space
                    result = " ".join(translated_chunks)
                else:
                    # For English, ensure proper sentence spacing
                    result = " ".join(translated_chunks)

                # Clean up extra spaces
                result = re.sub(r'\s+', ' ', result).strip()

            # Log final result stats
            result_length = len(result)
            logger.info(f"Translation complete: {original_length} -> {result_length} characters")

            return result

        except Exception as e:
            logger.exception("Translation failed for text")
            raise RuntimeError(f"Translation failed: {str(e)}")

    def _translate_single_chunk(self, text: str, max_tokens: int) -> str:
        """Translate a single chunk with strict token limit enforcement."""
        normalized_text = normalize(text)

        # Double-check token count before sending to model
        token_count = count_tokens(normalized_text, self.tokenizer)
        if token_count > MODEL_MAX_LENGTH:
            logger.warning(f"Chunk still has {token_count} tokens, force truncating...")
            # Emergency truncation by character count
            char_limit = MODEL_MAX_LENGTH * 3  # Rough estimate: 3 chars per token
            normalized_text = normalized_text[:char_limit]
            logger.info(f"Truncated to {len(normalized_text)} characters")

        # Use stricter tokenization with max_length enforcement
        inputs = self.tokenizer(
            normalized_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MODEL_MAX_LENGTH  # Explicit max length
        ).to(self.device)

        # Log actual input size for debugging
        actual_input_length = inputs["input_ids"].shape[1]
        logger.debug(f"Actual input length: {actual_input_length} tokens")

        if actual_input_length > MODEL_MAX_LENGTH:
            logger.error(f"Input still exceeds limit: {actual_input_length} > {MODEL_MAX_LENGTH}")
            raise ValueError(f"Input too long after truncation: {actual_input_length} tokens")

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_tokens
            )
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation

    def translate_batch(self, sentences: List[str], max_tokens: int = 128, batch_size: int = 32) -> List[str]:
        """
        Translate a batch of texts using the preloaded model and tokenizer.

        Args:
            sentences (List[str]): List of input texts to translate.
            max_tokens (int): Maximum number of tokens to generate (default: 128).
            batch_size (int): Number of texts to process per batch (default: 32).

        Returns:
            List[str]: List of translated texts.
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
                # Verify each item in batch doesn't exceed limits
                for j, item in enumerate(batch):
                    item_tokens = count_tokens(item, self.tokenizer)
                    if item_tokens > MODEL_MAX_LENGTH:
                        logger.warning(f"Batch item {j} has {item_tokens} tokens, truncating...")
                        # Truncate by character count as emergency measure
                        char_limit = MODEL_MAX_LENGTH * 3
                        batch[j] = item[:char_limit]

                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MODEL_MAX_LENGTH  # Explicit max length
                ).to(self.device)

                # Log batch dimensions for debugging
                batch_max_length = inputs["input_ids"].shape[1]
                logger.debug(f"Batch max length: {batch_max_length} tokens")

                if batch_max_length > MODEL_MAX_LENGTH:
                    logger.error(f"Batch exceeds limit: {batch_max_length} > {MODEL_MAX_LENGTH}")
                    raise ValueError(f"Batch too long: {batch_max_length} tokens")

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

    def translate_long_text(self, text: str, max_tokens: int = 128) -> str:
        """
        Explicitly handle long text translation with detailed logging.

        Args:
            text (str): Long input text to translate.
            max_tokens (int): Maximum tokens per chunk.

        Returns:
            str: Complete translated text.
        """
        logger.info("Starting long text translation...")

        # Check input length
        input_tokens = count_tokens(text, self.tokenizer)
        logger.info(f"Input text has {input_tokens} tokens")

        if input_tokens <= MAX_CHUNK_TOKENS:
            logger.info("Text fits in single chunk, using direct translation")
            return self._translate_single_chunk(text, max_tokens)

        # Determine language and create chunks
        is_bengali = is_bengali_text(text)
        logger.info(f"Detected language: {'Bengali' if is_bengali else 'English'}")

        chunks = create_chunks(text, self.tokenizer, is_bengali)
        logger.info(f"Split into {len(chunks)} chunks:")

        for i, chunk in enumerate(chunks):
            chunk_tokens = count_tokens(chunk, self.tokenizer)
            logger.info(f"  Chunk {i + 1}: {chunk_tokens} tokens")

        # Translate chunks in batches
        translations = self.translate_batch(chunks, max_tokens, batch_size=BATCH_SIZE)

        # Join results
        result = " ".join(translations)
        output_tokens = count_tokens(result, self.tokenizer)
        logger.info(f"Translation complete. Output: {output_tokens} tokens")

        return result


# === Singleton Instances ===
bn_to_en_model = TranslationModel(_bn_en_model, _bn_en_tokenizer, DEVICE)
en_to_bn_model = TranslationModel(_en_bn_model, _en_bn_tokenizer, DEVICE)


# === Enhanced Public Functions with Full Translation Guarantee ===
def translate_bn_to_en(text: str, max_tokens: int = 128) -> str:
    """Translate Bengali text to English, handling long texts automatically."""
    return bn_to_en_model.translate(text, max_tokens)


def translate_en_to_bn(text: str, max_tokens: int = 128) -> str:
    """Translate English text to Bengali, handling long texts automatically."""
    return en_to_bn_model.translate(text, max_tokens)


def translate_bn_to_en_complete(text: str, max_tokens: int = 128, verify: bool = True) -> dict:
    """
    Complete Bengali to English translation with verification and stats.

    Args:
        text (str): Bengali text to translate
        max_tokens (int): Max tokens per chunk
        verify (bool): Whether to run verification checks

    Returns:
        dict: Translation result with stats and verification info
    """
    logger.info("Starting complete Bengali to English translation")

    # Pre-translation validation
    original_stats = get_text_stats(text)
    logger.info(f"Original text stats: {original_stats}")

    if verify:
        logger.info("Running pre-translation validation...")
        validation_passed = validate_chunks(text)
        if not validation_passed:
            logger.warning("Validation failed - proceeding anyway but results may be incomplete")

    # Perform translation
    start_time = time.time() if 'time' in globals() else 0
    translated_text = bn_to_en_model.translate(text, max_tokens)
    end_time = time.time() if 'time' in globals() else 0

    # Post-translation stats
    result_stats = {
        "original_chars": len(text),
        "translated_chars": len(translated_text),
        "original_words": len(text.split()),
        "translated_words": len(translated_text.split()),
        "translation_time": end_time - start_time if start_time > 0 else 0,
        "validation_passed": validation_passed if verify else None
    }

    logger.info(f"Translation completed: {result_stats}")

    return {
        "translated_text": translated_text,
        "original_text": text,
        "stats": result_stats,
        "chunks_info": preview_chunks(text) if verify else None
    }


def translate_en_to_bn_complete(text: str, max_tokens: int = 128, verify: bool = True) -> dict:
    """
    Complete English to Bengali translation with verification and stats.

    Args:
        text (str): English text to translate
        max_tokens (int): Max tokens per chunk
        verify (bool): Whether to run verification checks

    Returns:
        dict: Translation result with stats and verification info
    """
    logger.info("Starting complete English to Bengali translation")

    # Pre-translation validation
    original_stats = get_text_stats(text)
    logger.info(f"Original text stats: {original_stats}")

    if verify:
        logger.info("Running pre-translation validation...")
        validation_passed = validate_chunks(text)
        if not validation_passed:
            logger.warning("Validation failed - proceeding anyway but results may be incomplete")

    # Perform translation
    start_time = time.time() if 'time' in globals() else 0
    translated_text = en_to_bn_model.translate(text, max_tokens)
    end_time = time.time() if 'time' in globals() else 0

    # Post-translation stats
    result_stats = {
        "original_chars": len(text),
        "translated_chars": len(translated_text),
        "original_words": len(text.split()),
        "translated_words": len(translated_text.split()),
        "translation_time": end_time - start_time if start_time > 0 else 0,
        "validation_passed": validation_passed if verify else None
    }

    logger.info(f"Translation completed: {result_stats}")

    return {
        "translated_text": translated_text,
        "original_text": text,
        "stats": result_stats,
        "chunks_info": preview_chunks(text) if verify else None
    }


def translate_bn_to_en_batch(sentences: List[str], max_tokens: int = 128, batch_size: int = 32) -> List[str]:
    """Translate a batch of Bengali texts to English."""
    return bn_to_en_model.translate_batch(sentences, max_tokens, batch_size)


def translate_en_to_bn_batch(sentences: List[str], max_tokens: int = 128, batch_size: int = 32) -> List[str]:
    """Translate a batch of English texts to Bengali."""
    return en_to_bn_model.translate_batch(sentences, max_tokens, batch_size)


def translate_long_text_bn_to_en(text: str, max_tokens: int = 128) -> str:
    """Explicitly translate long Bengali text to English."""
    return bn_to_en_model.translate_long_text(text, max_tokens)


def translate_long_text_en_to_bn(text: str, max_tokens: int = 128) -> str:
    """Explicitly translate long English text to Bengali."""
    return en_to_bn_model.translate_long_text(text, max_tokens)


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
    Now handles long texts automatically.

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

    # Check if translation is needed
    text_is_bengali = is_bengali_text(text)

    if target == "bn" and not text_is_bengali:
        # English to Bengali
        res = translate_en_to_bn_complete(text)
        return res['translated_text']
    elif target == "en" and text_is_bengali:
        # Bengali to English
        res = translate_bn_to_en_complete(text)
        return res['translated_text']
    else:
        # No translation needed
        return text


# === Utility Functions ===
def get_text_stats(text: str) -> dict:
    """Get statistics about a text (tokens, language, etc.)."""
    is_bengali = is_bengali_text(text)
    tokenizer = _bn_en_tokenizer if is_bengali else _en_bn_tokenizer

    return {
        "character_count": len(text),
        "word_count": len(text.split()),
        "token_count": count_tokens(text, tokenizer),
        "is_bengali": is_bengali,
        "needs_chunking": count_tokens(text, tokenizer) > MAX_CHUNK_TOKENS,
        "estimated_chunks": max(1, count_tokens(text, tokenizer) // MAX_CHUNK_TOKENS)
    }


def preview_chunks(text: str) -> List[Tuple[str, int]]:
    """Preview how text would be chunked (for debugging)."""
    is_bengali = is_bengali_text(text)
    tokenizer = _bn_en_tokenizer if is_bengali else _en_bn_tokenizer

    chunks = create_chunks(text, tokenizer, is_bengali)
    return [(chunk, count_tokens(chunk, tokenizer)) for chunk in chunks]


def test_long_text_translation(text: str) -> dict:
    """
    Test function for long text translation with comprehensive reporting.

    Args:
        text (str): Long text to test translation

    Returns:
        dict: Complete test results
    """
    print("=" * 80)
    print("LONG TEXT TRANSLATION TEST")
    print("=" * 80)

    # Step 1: Analyze input text
    print(f"\n1. INPUT ANALYSIS:")
    print(f"   Characters: {len(text)}")
    print(f"   Words: {len(text.split())}")

    is_bengali = is_bengali_text(text)
    print(f"   Language: {'Bengali' if is_bengali else 'English'}")

    # Step 2: Validate chunking strategy
    print(f"\n2. CHUNKING VALIDATION:")
    validation_result = validate_chunks(text)

    # Step 3: Preview chunks
    print(f"\n3. CHUNK PREVIEW:")
    chunks_info = preview_chunks(text)
    for i, (chunk, tokens) in enumerate(chunks_info):
        print(f"   Chunk {i + 1}: {tokens} tokens, {len(chunk)} chars")
        print(f"   Preview: {chunk[:80]}...")
        print()

    # Step 4: Perform translation
    print(f"\n4. TRANSLATION:")
    if is_bengali:
        result = translate_bn_to_en_complete(text, verify=True)
    else:
        result = translate_en_to_bn_complete(text, verify=True)

    # Step 5: Display results
    print(f"\n5. RESULTS:")
    print(f"   Original: {result['stats']['original_chars']} chars, {result['stats']['original_words']} words")
    print(f"   Translated: {result['stats']['translated_chars']} chars, {result['stats']['translated_words']} words")
    print(f"   Translation time: {result['stats']['translation_time']:.2f}s")
    print(f"   Validation passed: {result['stats']['validation_passed']}")

    print(f"\n6. TRANSLATED TEXT:")
    print("-" * 60)
    print(result['translated_text'])
    print("-" * 60)

    return result


def validate_chunks(text: str) -> bool:
    """Validate that all chunks are within token limits."""
    chunks_info = preview_chunks(text)
    max_tokens = max(token_count for _, token_count in chunks_info)

    print(f"Text validation:")
    print(f"  Total chunks: {len(chunks_info)}")
    print(f"  Max chunk tokens: {max_tokens}")
    print(f"  Within limits: {max_tokens <= MAX_CHUNK_TOKENS}")

    for i, (chunk, tokens) in enumerate(chunks_info):
        status = "✓" if tokens <= MAX_CHUNK_TOKENS else "✗"
        print(f"  Chunk {i + 1}: {tokens} tokens {status}")
        if tokens > MAX_CHUNK_TOKENS:
            print(f"    Preview: {chunk[:100]}...")

    return max_tokens <= MAX_CHUNK_TOKENS


