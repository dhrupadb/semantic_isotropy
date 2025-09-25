import os
from typing import Any, Dict, Optional, Union, List, Tuple, Callable
from openai import OpenAI
from retry import retry

from google import generativeai as genai
from openai import OpenAI
import cohere

from longform_uq.llm.utils import TokenRateLimiter, estimate_tokens


def apply_rate_limiting(texts: List[str], rate_limiter: Optional[Tuple[TokenRateLimiter, str]], token_estimator: Callable[[str], int] = estimate_tokens) -> None:
    """
    Apply rate limiting based on the rate limiter configuration.

    Parameters:
        texts (List[str]): The texts to be processed for rate limiting calculation.
        rate_limiter (Optional[Tuple[TokenRateLimiter, str]]): Rate limiter configuration.
    """
    if rate_limiter and rate_limiter[1] == 'tokens':
        est_tokens = sum([token_estimator(text) for text in texts])
        rate_limiter[0].add_tokens(est_tokens)
    elif rate_limiter and rate_limiter[1] == 'words':
        est_words = sum([len(text.split()) for text in texts])
        rate_limiter[0].add_tokens(est_words)
    elif rate_limiter and rate_limiter[1] == 'requests':
        rate_limiter[0].add_tokens(1)


@retry(delay=20, tries=2)
def embed_api(texts: List[str], api: str = "gemini", **kwargs: Any) -> Union[Dict[str, Any], str]:
    if api == "gemini":
        return embed_gemini(texts, **kwargs)
    elif api == "openai":
        return embed_openai(texts, **kwargs)
    elif api == "cohere":
        return embed_cohere(texts, **kwargs)
    else:
        raise ValueError(f"Unsupported API {api}")

def embed_openai(texts: List[str], model: str = 'text-embedding-3-small', rate_limiter: Optional[Tuple[TokenRateLimiter, str]] = None, **kwargs: Any) -> Dict[str, Optional[Union[str, Any]]]:
    """
    Queries the OpenAI API to generate a response from a language model.

    Parameters:
        texts (List[str]): The texts to be embedded.
        model (str): The identifier of the model to be queried.
        **kwargs: Additional optional parameters to customize the API request.

    Returns:
        dict: A dictionary containing the generated text response and log probabilities.
    Raises:
        RuntimeError: If the API key is not provided or found in the environment variables.
    """
    api_key = kwargs.get('api_key', None)
    if not api_key:
        api_key = os.environ.get('OPENAI_API_KEY', None)

    if api_key is None:
        raise RuntimeError("OpenAI API key not set / given.")

    apply_rate_limiting(texts, rate_limiter)

    client = OpenAI(api_key=api_key)

    model = "text-embedding-3-small" if model == 'openai-v3-small'\
            else "text-embedding-3-large" if model == 'openai-v3-large' \
            else "text-embedding-ada-002"
    try:
        response = client.embeddings.create(
            input=texts,
            model=model
        )

        embeddings = [item.embedding for item in response.data]
        return {'embedding': embeddings, 'response': response}
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {'embedding': None, 'logprobs': None, 'message': str(e)}

def embed_gemini(texts: List[str], model: str = 'models/text-embedding-004', rate_limiter: Optional[Tuple[TokenRateLimiter, str]] = None, **kwargs: Any) -> Dict[str, Optional[Union[str, Any]]]:
    """
    Queries the Gemini API to generate a response from a language model.

    Parameters:
        texts (List[str]): The texts to be embedded.
        model (str): The identifier of the model to be queried.
        **kwargs: Additional optional parameters to customize the API request.

    Returns:
        dict: A dictionary containing the generated text response and log probabilities.
    Raises:
        RuntimeError: If the API key is not provided or found in the environment variables.
    """
    api_key = kwargs.get('api_key', None)
    if not api_key:
        api_key = os.environ.get('GOOGLE_API_KEY', None)
    if api_key is None:
        raise RuntimeError("Google API key not set / given.")

    apply_rate_limiting(texts, rate_limiter)

    task_type = kwargs.get('task_type', 'SEMANTIC_SIMILARITY')

    model = model.replace('gemini/', '')
    assert len(model) > 0, "Model name must be provided."

    try:
        result = genai.embed_content(
            model=model,
            content=texts,
            task_type=task_type
        )

        return result

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {'embedding': None, 'logprobs': None, 'message': str(e)}


def embed_cohere(texts: List[str], model: str = 'embed-english-v3.0', rate_limiter: Optional[Tuple[TokenRateLimiter, str]] = None, **kwargs: Any) -> Dict[str, Optional[Union[str, Any]]]:
    """
    Queries the Cohere API to generate a response from a language model.

    Parameters:
        texts (List[str]): The texts to be embedded.
        model (str): The identifier of the model to be queried.
        **kwargs: Additional optional parameters to customize the API request.

    Returns:
        dict: A dictionary containing the generated text response and log probabilities.
    Raises:
        RuntimeError: If the API key is not provided or found in the environment variables.
    """
    api_key = kwargs.get('api_key', None)
    if not api_key:
        api_key = os.environ.get('COHERE_API_KEY', None)
    if api_key is None:
        raise RuntimeError("Cohere API key not set / given.")

    apply_rate_limiting(texts, rate_limiter)

    task_type = kwargs.get('task_type', 'clustering')

    model = model.replace('cohere/', '')
    assert len(model) > 0, "Model name must be provided."

    co = cohere.ClientV2(api_key=api_key)

    try:
        result = co.embed(
            texts=texts,
            model=model,
            input_type=task_type,
            embedding_types=['float']
        )

        retval = {'embedding': result.embeddings.float, 'response': result}
        return retval

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {'embedding': None, 'message': str(e)}
