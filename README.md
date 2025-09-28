# Embedding Trust: Semantic Isotropy Predicts Nonfactuality in Long‑Form Text Generation

&#x20;&#x20;

---

## 🧠 Overview

This repository accompanies the paper ***Embedding Trust: Semantic Isotropy Predicts Nonfactuality in Long‑Form Text Generation*** (under review at ICLR 2026).

The project introduces **Semantic Isotropy**, a lightweight, model‑agnostic measure that quantifies the *dispersion* of normalized text embeddings across multiple LLM generations. High isotropy (uniform dispersion on the unit sphere) correlates strongly with **nonfactuality**, while low isotropy (tight clustering) signals **trustworthy, factually consistent** generations.

Unlike claim‑by‑claim fact‑checking methods, Semantic Isotropy requires:

- No labeled data or fine‑tuning
- No hyperparameter tuning
- Only a few generated samples per prompt

It provides a computationally inexpensive proxy for factual consistency in long‑form text generation.

---

## 📂 Repository Structure

```
├── src/                     # Core implementation
│   ├── isotropy.py          # Semantic isotropy computation
│   ├── segment_score.py     # Segment‑Score factuality evaluation
│   ├── utils.py             # Helper functions
│
├── experiments/             # Reproduction of paper experiments
│   ├── run_triviaqa.sh
│   ├── run_factscorebio.sh
│
├── notebooks/               # Example Jupyter notebooks
│   ├── demo_isotropy.ipynb
│   └── visualization.ipynb
│
├── data/                    # Links and scripts for dataset preparation
│   ├── download_factscorebio.py
│   ├── download_triviaqa.py
│
├── figures/                 # Plots and visualizations from the paper
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ⚙️ Installation

**Requirements:**

- Python ≥ 3.10
- PyTorch ≥ 2.1
- sentence‑transformers ≥ 2.3
- vLLM ≥ 0.3
- numpy, pandas, matplotlib

```bash
# Clone repository
git clone https://github.com/<your‑org>/semantic‑isotropy.git
cd semantic‑isotropy

# Create environment
conda create -n isotropy python=3.10
conda activate isotropy

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

Compute **Semantic Isotropy** for a given prompt:

```python
from src.isotropy import semantic_isotropy_score
from sentence_transformers import SentenceTransformer

prompts = ["Write a few paragraphs about Paris, France."]
responses = generate_responses(model="llama-3-8b-instruct", prompt=prompts[0], n=10)

embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v1")
score = semantic_isotropy_score(responses, embedder)
print(f"Semantic Isotropy Score: {score:.3f}")
```

Evaluate **factuality** using **Segment‑Score**:

```python
from src.segment_score import segment_score
fact_score = segment_score(response=responses[0], reference="<wiki_text>")
print(f"Segment‑Score factuality: {fact_score:.2f}")
```

---

## 🔁 Reproducing Paper Results

To reproduce the main ICLR 2026 results:

```bash
# Step 1. Generate responses (Llama 3.1 8B, Phi‑3.5 Mini, GPT‑4.1 Mini)
python experiments/generate_responses.py --dataset triviaqa --model llama-3.1-8b

# Step 2. Compute isotropy scores
python src/isotropy.py --input data/triviaqa_responses.json --embedder nomic-ai/nomic-embed-text-v1

# Step 3. Compute factuality scores (Segment‑Score)
python src/segment_score.py --input data/triviaqa_responses.json

# Step 4. Evaluate correlation
python experiments/evaluate.py --metric isotropy --scoring segment-score
```

All experiments can be run on a single GPU (e.g., NVIDIA V100). Average runtime per topic ≈1.8s vs. >300s for LUQ‑Atomic.

---

## 📊 Key Results

Semantic Isotropy outperforms all baseline uncertainty and factuality metrics (R² of linear regression Factuality ∼ Isotropy):

| Dataset  | Model        | Best Baseline (LUQ‑Atomic) | Semantic Isotropy (ours) |
| -------- | ------------ | -------------------------- | ------------------------ |
| TriviaQA | Llama‑3.1 8B | 0.31                       | **0.43**                 |
| FS‑BIO   | Phi‑3.5 Mini | 0.29                       | **0.39**                 |
| FS‑BIO   | GPT‑4.1 Mini | 0.36                       | **0.46**                 |

Semantic Isotropy remains robust across:

- Response lengths (125–1000 words)
- Number of samples (≈6–8 sufficient)
- Embedding models (OpenAI, Cohere, Nomic, Gemini)

---

## 📚 Datasets and Models

- **Datasets:**

  - TriviaQA Entities (1,000 entities)
  - FactScore‑BIO (182 entities)
  - Segment‑Score dataset (\~65,450 labeled responses)

- **Embedding Models:**

  - Nomic v1, OpenAI Embeddings (Small/Large), Cohere v3.0/4.0, Gemini v1

- **Generative Models:**

  - Meta Llama 3.1 8B Instruct, Microsoft Phi 3.5 Mini, OpenAI GPT 4.1 Mini

---

## 🧩 Citation

```bibtex
@inproceedings{bhardwaj2026embeddingtrust,
  title={Embedding Trust: Semantic Isotropy Predicts Nonfactuality in Long‑Form Text Generation},
  author={Anonymous Authors},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

---

## 📜 License

This project is released under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

This work builds upon prior studies on factuality and uncertainty in LLMs, including **FactScore**, **LUQ**, **INSIDE**, and **Semantic Entropy**. We thank the open‑source community for datasets, embedding models, and inference tools that made this research possible.

---

**Maintainer:** [Anonymous Author(s)]\
**Contact:** TBD (post‑review release)

