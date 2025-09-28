import os
import anthropic
import ollama

from retry import retry


@retry(delay=1, tries=2)
def query_api(request, api="anthropic", **kwargs):
    if api == "anthropic":
        if 'model' not in kwargs:
            kwargs['model'] = 'claude-3-5-sonnet-20240620'
        return query_anthropic(request, **kwargs)
    else:
        raise ValueError(f"Unsupported API {api}")


def query_anthropic(request, model, **kwargs):
    """
    Queries the Anthropic API to generate a response from a language model.

    This method sends a request to the specified Anthropic model and retrieves a response based on the provided
    parameters. It allows for customization of various parameters, such as the model, system message, and other
    settings related to the API call.

    Parameters:
        request (str): The user input or query that will be sent to the model.
        model (str): The identifier of the model to be queried (e.g., "claude-3-5-sonnet-20240620").
        **kwargs: Additional optional parameters to customize the API request:
            - api_key (str): The API key for accessing the Anthropic API. If not provided, it defaults to the
              `ANTHROPIC_API_KEY` environment variable.
            - system (str): An optional system message to guide the model's behavior.
            - prefill (str): Optional pre-filled response content to be included in the conversation context.
            - max_tokens (int): The maximum number of tokens allowed in the response (default is 4096).
            - temperature (float): Sampling temperature to control the randomness of the response
              (default is 0.0, for deterministic responses).

    Returns:
        str: The generated text response from the model.

    Raises:
        RuntimeError: If the API key is not provided or found in the environment variables.
    """
    api_key = kwargs.get('api_key', os.environ.get('ANTHROPIC_API_KEY', None))
    if api_key is None:
        raise RuntimeError("Anthropic API key not set / given.")

    system = kwargs.get('system', '')
    ac = anthropic.Anthropic(
        api_key=api_key,
    )

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

    return message.content[0].text


@retry(delay=5, tries=2)
def query_llm(request, model="llama3.1:8b", **kwargs):
    system = kwargs.get('system', None)
    messages = [{"role": "system", "content": system}] if system else []
    messages.append(
        {"role": "user", "content": request}
    )

    prefill = kwargs.get('prefill', None)
    if prefill:
        messages.append({"role": "assistant", "content": prefill})

    ollama_kwargs = {}
    if 'temperature' in kwargs:
        ollama_kwargs['options'] = {
            'temperature': float(kwargs['temperature'])
        }

    res = ollama.chat(model=model, messages=messages, **ollama_kwargs)
    return res['message']['content']
