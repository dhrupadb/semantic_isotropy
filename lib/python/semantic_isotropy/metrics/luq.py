from typing import List, Callable, Iterable
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch
import gc


# Implementation of https://arxiv.org/abs/2403.20279 (https://github.com/caiqizh/LUQ)
# @misc{zhang2024luqlongtextuncertaintyquantification,
#       title={LUQ: Long-text Uncertainty Quantification for LLMs},
#       author={Caiqi Zhang and Fangyu Liu and Marco Basaldella and Nigel Collier},
#       year={2024},
#       eprint={2403.20279},
#       archivePrefix={arXiv},
#       primaryClass={cs.CL},
#       url={https://arxiv.org/abs/2403.20279},
# }

def entailment_score_func(model_name: str = "potsawee/deberta-v3-large-mnli", max_length: int = 1000, model_args: dict = {}, device: str = "mps") -> Callable[[Iterable[str], str], torch.Tensor]:
    """
    Calculate entailment scores for a list of sentences and a reference response.
    """

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **{k:v for k,v in model_args.items() if k != 'dtype'})

    if 'dtype' in model_args and model_args['dtype'] == 'half':
        model.half()
    model.gradient_checkpointing_enable()
    model.to(device)
    model.eval()

    # Perform inference
    # Use TorchScript for faster inference if possible
    scripted_model = None
    try:
        # Try to script the model for faster inference
        scripted_model = torch.jit.script(model)
    except Exception:
        # Fallback to eager mode if scripting fails (e.g., for some HuggingFace models)
        scripted_model = model

    def inner(sentences: Iterable[str], ref_response: str) -> torch.Tensor:
        premises = sentences
        hypothesis = [ref_response] * len(sentences)

        # Tokenize the inputs
        inputs = tokenizer(premises, hypothesis, return_tensors="pt", max_length=max_length, truncation=True, padding="longest").to(device)

        with torch.no_grad():
            logits = scripted_model(**inputs).logits

        # Compute probabilities
        probs = torch.softmax(logits, dim=1)
        entailment_score = probs[:, 0] / probs.sum(dim=1)
        result = np.sum(entailment_score.detach().cpu().numpy())

        if device == "mps":
            # Clear memory cache
            torch.mps.empty_cache()

        return result
    return inner

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def luq_simscore(responses: List[str], entailment_func: Callable[[str, str], float], max_sentences: int = 20) -> float:
    """
    Calculate Long-Text Uncertainty Quantification (LUQ) Similarity score.

    Args:
        reference_response: The reference response
        sampled_responses: A list of sampled responses

    Returns:
        The LUQ score
    """
    simscores = np.zeros((len(responses), len(responses))) # This is not a symmetric matrix as S(r_i, r_j) != S(r_j, r_i)
    for reference_idx in range(len(responses)):
        reference_response = responses[reference_idx]
        sentences = [s['text'] for s in reference_response['statements']]
        for i, r in enumerate(responses):
            if i == reference_idx:
                simscores[reference_idx, i] = 1
                continue

            for chunk in chunker(sentences, max_sentences):
                simscores[reference_idx, i] += entailment_func(chunk, r['response'])

            simscores[reference_idx, i] /= len(sentences)

        # Delete unused variables and run garbage collection
        gc.collect()

    return simscores


def luq(responses: List[str], entailment_func: Callable[[Iterable[str], str], float], chunk_size: int = 20) -> float:
    """
    Calculate Long-Text Uncertainty Quantification (LUQ) score.

    Args:
        responses: A list of responses
    """
    simscores = luq_simscore(responses, entailment_func, chunk_size)

    confidences = (np.sum(simscores, axis=1) - 1) / (len(responses) - 1)
    uncertainty = np.mean(1 - confidences)
    return confidences, uncertainty
