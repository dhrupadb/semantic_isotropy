import numpy as np
import torch

from torch import Tensor

from semantic_isotropy.llm.utils import estimate_tokens
from semantic_isotropy.llm.embed import embed_api


def header_prompt(entity, text, model_name=''):
    res = f"""The following is an excerpt from a text about: '{entity}' \n\n {text}"""
    if 'nomic' in model_name:
        res = f"""clustering: {res}"""
    return res

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# Implementation of core method from https://github.com/D2I-ai/eigenscore
# @article{chen2024inside,
# title={INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection},
# author={Chen, Chao and Liu, Kai and Chen, Ze and Gu, Yi and Wu, Yue and Tao, Mingyuan and Fu, Zhihang and Ye, Jieping},
# booktitle={The Twelfth International Conference on Learning Representations},
# year={2024}
# }
def calculate_eigenscore(Z, alpha=1e-3):
    K, d = Z.shape
    Jd = np.eye(Z.shape[1]) - 1/d
    Sigma = Z @ Jd @ Z.T
    eigenscore = (1/K)*(np.log(np.linalg.det(Sigma + alpha*np.eye(Z.shape[0]))))
    return eigenscore
###################

def get_embedding_density(entity, responses, model, tokenizer, pooling_method="mean", max_length=1000, device=torch.device("mps"),
                       use_multi_gpu=False, model_name='', rate_limiter=None, api_key=None, task_type=None):
    structured_responses = [header_prompt(entity, response['response'], model_name) for response in responses]
    calc_eigenscore = True

    if 'gemini' in model_name:
        results = embed_api(structured_responses, 'gemini', model=model_name, task_type=task_type, rate_limiter=rate_limiter, api_key=api_key)
        pooled_state = torch.tensor(results['embedding'])
        calc_eigenscore = False
    elif 'openai' in model_name:
        results = embed_api(structured_responses, 'openai', model=model_name, rate_limiter=rate_limiter, api_key=api_key)
        pooled_state = torch.tensor(results['embedding'])
        calc_eigenscore = False
    elif 'cohere' in model_name:
        results = embed_api(structured_responses, 'cohere', model=model_name, rate_limiter=rate_limiter, api_key=api_key)
        pooled_state = torch.tensor(results['embedding'])
        calc_eigenscore = False
    else:
        max_length_computed = np.min([max_length, np.max([estimate_tokens(sr) for sr in structured_responses])]) # performance optimization

        inputs = tokenizer([response for response in structured_responses], return_tensors="pt", max_length=max_length_computed, truncation=True, padding=True)

        if use_multi_gpu:
            # When using device_map="auto", inputs should go to the device where the first layer is
            first_device = next(model.parameters()).device
            inputs = inputs.to(first_device)
        else:
            inputs = inputs.to(device)

        # Perform inference
        with torch.no_grad():
            out = model(**inputs)

        if pooling_method == "mean":
            pooled_state = torch.mean(out.last_hidden_state, dim=1)
        elif pooling_method == "cls":
            pooled_state = out.last_hidden_state[:, 0, :]
        elif pooling_method == "max":
            pooled_state = torch.max(out.last_hidden_state, dim=1)
        elif pooling_method == "last":
            pooled_state = last_token_pool(out.last_hidden_state, inputs.attention_mask)
        else:
            raise ValueError(f"Invalid pooling method: {pooling_method}")

    if calc_eigenscore:
        if out.hidden_states and len(out.hidden_states) > 0:
            eigenscore_vec = last_token_pool(out.hidden_states[len(out.hidden_states)//2], inputs.attention_mask).cpu().numpy()
        else:
            eigenscore_vec = pooled_state.cpu().numpy()
    else:
        eigenscore_vec = pooled_state.cpu().numpy()

    return pooled_state.cpu().numpy(), eigenscore_vec

def embedding_density(responses, model, tokenizer, entity, max_length=1000, pooling_method="mean", device=torch.device("mps"),
                  use_multi_gpu=False, model_name='', rate_limiter=None, api_key=None, task_type=None):
    pooled_state, eigenscore_vec,  = get_embedding_density(entity, responses, model, tokenizer, pooling_method, max_length, device,
                                                                                               use_multi_gpu, model_name, rate_limiter, api_key, task_type)
    return pooled_state, eigenscore_vec
