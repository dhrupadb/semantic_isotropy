import tiktoken
import time
import requests
from requests.exceptions import RequestException
from datetime import datetime
import threading
import logging


logger = logging.getLogger(__name__)

class TokenRateLimiter:
    DEFAULT_TOKENS_PER_MINUTE = 200000

    def __init__(self, tokens_per_minute):
        self.tokens_per_minute = tokens_per_minute if (tokens_per_minute is not None and tokens_per_minute > 0) else self.DEFAULT_TOKENS_PER_MINUTE
        self.token_count = 0
        self.last_reset = datetime.now()
        self._lock = threading.Lock()  # Add lock for thread safety

    def add_tokens(self, token_count: int):
        with self._lock:  # Ensure thread-safe access to shared resources
            current_time = datetime.now()
            # Reset counter if more than a minute has passed
            if (current_time - self.last_reset).total_seconds() >= 60:
                self.token_count = 0
                self.last_reset = current_time

            self.token_count += token_count

            # If we're over limit, sleep until next minute
            if self.token_count >= self.tokens_per_minute:
                seconds_to_wait = 60 - (current_time - self.last_reset).total_seconds()
                if seconds_to_wait > 0:
                    logger.info(f"Rate limit reached. Sleeping for {seconds_to_wait:.1f} seconds")
                    time.sleep(seconds_to_wait)
                    self.token_count = token_count
                    self.last_reset = datetime.now()

def estimate_tokens(text, model="gpt2"):
    """
    Estimate the number of tokens for a given text and model.

    Args:
        text (str): The input text to tokenize.
        model (str): The model name, e.g., "gpt-4", "gpt-3.5-turbo".

    Returns:
        int: Estimated number of tokens.
    """
    try:
        # Get the tokenizer for the specified model
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        raise ValueError(f"Model '{model}' is not supported by tiktoken.")

    # Encode the text and count tokens
    tokens = encoding.encode(text)
    return len(tokens)

def check_server_availability(url: str, timeout: int = 5) -> bool:
    """
    Check if the server is running at the specified URL.

    Args:
        url (str): The URL to check.
        timeout (int): Timeout in seconds for the request.

    Returns:
        bool: True if server is running, False otherwise.
    """
    try:
        # Remove /v1 if present for the health check
        check_url = url.replace('/v1', '') if '/v1' in url else url
        check_url = check_url.rstrip('/') + '/health'
        response = requests.get(check_url, timeout=timeout)
        return response.status_code == 200
    except RequestException as e:
        raise RuntimeError(f"Server not available at {url}: {str(e)}")

def detect_api_model(model: str) -> bool:
    """
    Detect the API model from the model name.
    """
    if 'gemini' in model:
        return True
    elif 'openai' in model:
        return True
    elif 'claude' in model:
        return True
    elif 'deepseek' in model:
        return True
    elif 'cohere' in model:
        return True
    else:
        return False
