import logging
from typing import List
import tiktoken
import json

from semantic_isotropy.llm.utils import estimate_tokens, TokenRateLimiter, detect_api_model
from semantic_isotropy.llm.chat import chat_api
from semantic_isotropy.llm.openai import process_batch_completions as openai_process_batch_completions

from functools import partial


class GenResponse:
    def __init__(self, text: str, logprobs: List[float] = None):
        self.text = text
        self.logprobs = logprobs

class GenOutput:
    def __init__(self, outputs: List[GenResponse]):
        self.outputs = outputs

class SamplingParams:
    """
    A simple container class for sampling parameters, similar to vLLM's SamplingParams.
    """
    def __init__(
        self,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs
    ):
        self.temperature = temperature
        self.top_p = top_p
        # Store any additional parameters
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        """
        Convert the sampling parameters to a dictionary.
        """
        return self.__dict__.copy()


class LLM:
    """
    Wrapper class for vllm.generate. Takes the same arguments and implements the same methods,
    but overrides the generate method.
    """
    def __init__(self, model: str, *args, **kwargs):
        self.api_model = detect_api_model(model)
        if not self.api_model:
            from vllm import LLM as vLLM
            self._vllm = vLLM(model=model, *args, **kwargs)
        else:
            self.api_endpoint = model.split('/')[0].lower()
            self.api_model_str = '/'.join(model.split('/')[1:])
            self._api = partial(chat_api, api=self.api_endpoint, model=self.api_model_str)

    def __getattr__(self, name):
        # Delegate attribute access to the underlying LLM instance
        if not self.api_model:
            return getattr(self._vllm, name)
        else:
            raise AttributeError(f"LLM initialized in API model mode. Attribute {name} not found.")

    def generate(self, batch_prompts: List[str], sampling_params: SamplingParams, **kwargs):
        """
        Override the generate method.
        Default to vLLM generate
        Otherwise implement custom generation loops for API only models (batch and sequential mode).
        """
        if not self.api_model:
            from vllm import SamplingParams as vLLMSamplingParams
            sampling_params_dict_copy = sampling_params.to_dict()
            _ = sampling_params_dict_copy.pop('top_logprobs', None)
            vllm_sampling_params = vLLMSamplingParams(**sampling_params_dict_copy)
            results = self._vllm.generate(batch_prompts, vllm_sampling_params, **kwargs)
        else:
            results = []
            if self.api_endpoint == "openai-batch":
                results_file_path, batch_job = openai_process_batch_completions(batch_prompts, model=self.api_model_str, sampling_params=sampling_params.to_dict())
                if not results_file_path:
                    raise RuntimeError(f"Results file path is empty for batch job {batch_job.id}. Batch Failed!")

                with open(results_file_path, "r") as f:
                    for line in f:
                        result = json.loads(line)
                        results.append(GenOutput([
                            GenResponse(
                                result['response']['body']['choices'][0]['message']['content'],
                                [x['logprob'] for x in result['response']['body']['choices'][0]['logprobs']['content']] if sampling_params.logprobs else []
                            )
                        ]))
            else:
                token_rate_limiter = TokenRateLimiter(getattr(sampling_params, "tokens_per_minute", TokenRateLimiter.DEFAULT_TOKENS_PER_MINUTE))
                for prompt in batch_prompts:
                    token_rate_limiter.add_tokens(estimate_tokens(prompt))
                    result = self._api(prompt, **sampling_params.to_dict())
                    res = GenOutput([GenResponse(result['response'], result.get('logprobs', None))])
                    results.append(res)
        return results

    def get_tokenizer(self):
        """
        Return the tokenizer associated with the underlying LLM or API model.
        For vLLM, delegate to the vLLM instance.
        For API models, use tiktoken for OpenAI, otherwise raise NotImplementedError.
        """
        if not self.api_model:
            return self._vllm.get_tokenizer()
        else:
            if "openai" in self.api_endpoint:
                return tiktoken.encoding_for_model(self.api_model_str)
            else:
                raise NotImplementedError(f"Tokenizer retrieval not implemented for this API model: {self.api_model_str}")
