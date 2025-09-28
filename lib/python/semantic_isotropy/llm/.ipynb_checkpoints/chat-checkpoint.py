import os
import requests
from typing import Any, Dict, Optional, Union
from openai import OpenAI
from requests.exceptions import RequestException
from retry import retry

def _check_server_availability(url: str, timeout: int = 5) -> bool:
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

@retry(delay=1, tries=2)
def chat_api(request: str, api: str = "anthropic", **kwargs: Any) -> Union[Dict[str, Any], str]:
    if api == "anthropic":
        return chat_claude(request, **kwargs)
    elif api == "openai":
        return chat_gpt(request, **kwargs)
    elif api == "ollama":
        return chat_ollama(request, **kwargs)
    elif api == "vllm":
        return chat_vllm(request, **kwargs)
    elif api == "llamacpp":
        return chat_llamacpp(request, **kwargs)
    elif api == "deepseek":
        return chat_deepseek(request, **kwargs)
    else:
        raise ValueError(f"Unsupported API {api}")

def chat_gpt(request: str, model: str = 'gpt-4o-mini', **kwargs: Any) -> Dict[str, Optional[Union[str, Any]]]:
    """
    Queries the OpenAI API to generate a response from a language model.

    Parameters:
        request (str): The user input or query that will be sent to the model.
        model (str): The identifier of the model to be queried.
        **kwargs: Additional optional parameters to customize the API request.

    Returns:
        dict: A dictionary containing the generated text response and log probabilities.
    Raises:
        RuntimeError: If the API key is not provided or found in the environment variables.
    """
    api_key = kwargs.get('api_key', os.environ.get('OPENAI_API_KEY', None))
    if api_key is None:
        raise RuntimeError("OpenAI API key not set / given.")

    client = OpenAI(api_key=api_key)

    messages = [{"role": "user", "content": request}]

    system = kwargs.get('system', '')
    if system:
        messages.insert(0, {"role": "system", "content": system})
        
    prefill = kwargs.get('prefill', None)
    if prefill:
        messages.append({"role": "assistant", "content": prefill})

    optional_kwargs = {'logprobs': kwargs.get('logprobs', False)}
    if optional_kwargs['logprobs']:
        optional_kwargs['top_logprobs'] = kwargs.get('top_logprobs', 0)

    try:
        message = client.chat.completions.create(
            model=model,
            max_tokens=kwargs.get('max_tokens', 4096),
            temperature=kwargs.get('temperature', 0.0),
            messages=messages,
            **optional_kwargs
        )

        response_text = message.choices[0].message.content
        logprobs = message.choices[0].logprobs

        return {
            'response': response_text,
            'logprobs': logprobs,
            'message': message
        }

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {'response': None, 'logprobs': None, 'message': None}

def chat_deepseek(request: str, model: str = 'deepseek-chat', **kwargs: Any) -> Dict[str, Optional[Union[str, Any]]]:
    """
    Queries the DeepSeek API to generate a response from a language model.

    Parameters:
        request (str): The user input or query that will be sent to the model.
        model (str): The identifier of the model to be queried.
        **kwargs: Additional optional parameters to customize the API request.

    Returns:
        dict: A dictionary containing the generated text response and log probabilities.
    Raises:
        RuntimeError: If the API key is not provided or found in the environment variables.
    """
    api_key = kwargs.get('api_key', os.environ.get('DEEPSEEK_API_KEY', None))
    if api_key is None:
        raise RuntimeError("DeepSeek API key not set / given.")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    messages = [{"role": "user", "content": request}]

    system = kwargs.get('system', '')
    if system:
        messages.insert(0, {"role": "system", "content": system})
        
    prefill = kwargs.get('prefill', None)
    if prefill:
        messages.append({"role": "assistant", "content": prefill})

    optional_kwargs = {'logprobs': kwargs.get('logprobs', False)}
    if optional_kwargs['logprobs']:
        optional_kwargs['top_logprobs'] = kwargs.get('top_logprobs', 0)

    try:
        message = client.chat.completions.create(
            model=model,
            max_tokens=kwargs.get('max_tokens', 4096),
            temperature=kwargs.get('temperature', 0.0),
            messages=messages,
            **optional_kwargs
        )

        response_text = message.choices[0].message.content
        logprobs = message.choices[0].logprobs

        return {
            'response': response_text,
            'logprobs': logprobs,
            'message': message
        }

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {'response': None, 'logprobs': None, 'message': None}

def chat_claude(request: str, model: str = 'claude-3-5-sonnet-20240620', **kwargs: Any) -> str:
    import anthropic
    """
    Queries the Anthropic API to generate a response from a language model.

    Parameters:
        request (str): The user input or query that will be sent to the model.
        model (str): The identifier of the model to be queried.
        **kwargs: Additional optional parameters to customize the API request.

    Returns:
        str: The generated text response from the model.

    Raises:
        RuntimeError: If the API key is not provided or found in the environment variables.
    """
    api_key = kwargs.get('api_key', os.environ.get('ANTHROPIC_API_KEY', None))
    if api_key is None:
        raise RuntimeError("Anthropic API key not set / given.")

    system = kwargs.get('system', '')
    ac = anthropic.Anthropic(api_key=api_key)

    messages = [{"role": "user", "content": request}]
    prefill = kwargs.get('prefill', None)
    if prefill:
        messages.append({"role": "assistant", "content": prefill})

    message = ac.messages.create(
        model=model,
        max_tokens=kwargs.get('max_tokens', 4096),
        temperature=kwargs.get('temperature', 0.0),
        system=system,
        messages=messages
    )

    return {'response': message.content[0].text, 'message': message}

@retry(delay=5, tries=2)
def chat_ollama(request: str, model: str = "llama3.1:8b", **kwargs: Any) -> str:
    import ollama
    system = kwargs.get('system', None)
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": request})

    prefill = kwargs.get('prefill', None)
    if prefill:
        messages.append({"role": "assistant", "content": prefill})

    ollama_kwargs = {}
    if 'temperature' in kwargs:
        ollama_kwargs['options'] = {'temperature': float(kwargs['temperature'])}

    return {'response': res['message']['content'], 'message': res}

@retry(delay=5, tries=2)
def chat_vllm(request: str, url: str, model: str = "microsoft/Phi-3-mini-4k-instruct", **kwargs: Any) -> str:
    """
    Queries a language model being served at a specified URL endpoint.

    Parameters:
        request (str): The user input or query to send to the model.
        url (str): The URL endpoint where the LLM is being served.
        model (str): The model to use.
        **kwargs: Additional optional parameters.

    Returns:
        str: The generated text response from the model.

    Raises:
        requests.exceptions.RequestException: If the HTTP request fails.
    """
    _check_server_availability(url)

    if 'v1' not in url:
        print(f"Warning: {url} does not contain /v1, adding it. Assuming it's a port is 8000.")
        url = url.replace(":8000", ":8000/v1")

    system = kwargs.get('system', None)
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": request})

    prefill = kwargs.get('prefill', None)
    if prefill:
        messages.append({"role": "assistant", "content": prefill})

    stream = kwargs.get('stream', False)

    payload = {
        "messages": messages,
        "model": model,
        "stream": stream,
    }

    max_tokens = kwargs.get('max_tokens', -1)
    if max_tokens > 0:
        payload['max_tokens'] = max_tokens

    if 'temperature' in kwargs:
        payload['temperature'] = float(kwargs['temperature'])

    client = OpenAI(base_url=url, api_key="foobar")
    message = client.chat.completions.create(**payload)
    return {'response': message.choices[0].message.content, 'message': message}

@retry(delay=5, tries=2)
def chat_llamacpp(request: str, url: str, model: str = "llama.cpp/models/microsoft/Phi-3-mini-4k-instruct-fp16.gguf", **kwargs: Any) -> str:
    """
    Queries a language model being served at a specified URL endpoint.

    Parameters:
        request (str): The user input or query to send to the model.
        url (str): The URL endpoint where the LLM is being served.
        model (str): The model to use.
        **kwargs: Additional optional parameters.

    Returns:
        str: The generated text response from the model.

    Raises:
        requests.exceptions.RequestException: If the HTTP request fails.
    """
    _check_server_availability(url)

    system = kwargs.get('system', None)
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": request})

    prefill = kwargs.get('prefill', None)
    if prefill:
        messages.append({"role": "assistant", "content": prefill})

    stream = kwargs.get('stream', False)

    payload = {
        "messages": messages,
        "model": model,
        "stream": stream,
    }

    max_tokens = kwargs.get('max_tokens', -1)
    if max_tokens > 0:
        payload['max_tokens'] = max_tokens

    if 'temperature' in kwargs:
        payload['temperature'] = float(kwargs['temperature'])

    client = OpenAI(base_url=url, api_key="foobar")
    message = client.chat.completions.create(**payload)
    return {'response': message.choices[0].message.content, 'message': message}
