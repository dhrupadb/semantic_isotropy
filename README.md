# Semantic Isotropy

Research on Uncertainty Quantification in Long Form Generations using Semantic Isotropy

---

## Overview

This project introduces **Semantic Isotropy**, a lightweight, model-agnostic measure that quantifies the *dispersion* of normalized text embeddings across multiple LLM generations. High isotropy (uniform dispersion on the unit sphere) correlates strongly with **nonfactuality**, while low isotropy (tight clustering) signals **trustworthy, factually consistent** generations.

Unlike claim-by-claim fact-checking methods, Semantic Isotropy requires:

- No labeled data or fine-tuning
- No hyperparameter tuning
- Only a few generated samples per prompt

It provides a computationally inexpensive proxy for factual consistency in long-form text generation.

---

## Repository Structure

```
├── lib/python/longform_uq/          # Core implementation libraries
│              ├── datasets          # Data loading and manipulation
│              ├── llm               # LLM APIs for queries and embeddings
│              ├── metrics           # Uncertainty Metrics
│              ├── pipeline          # Pipeline utilities for experiment generation
│              ├── prompts           # Prompt Library
│              └── tests             # Unit tests for pipeline utilities
│
├── scripts/               # Driver Scripts
│   ├── oeq_sample.py
│   ├── factscore_open_ended_gen.py
│   ├── triviaqa_open_ended_gen.py
│   ├── booksummaries_open_ended_gen.py
│   ├── segscore/
│       ├── oeq_seg_score.py
│       └── gen_metric.py
│   └── runner.py
│
├── data/
│   └── bio_entities.txt     # List of entities for FS-BIO Dataset
│
├── figures/                # Plots and visualizations
│
├── requirements.txt
├── README.md
└── LICENSE
```
---

## Installation

```bash
# Clone repository
git clone <repository-url>
cd semantic_isotropy

# Create environment
python -m venv /path/to/venvs/semantic_isotropy
source /path/to/venvs/semantic_isotropy/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Local Libraries
pip install -e .
```
---

## Reproducing Results

### 1. Generate Set of Prompt Entities
```
python scripts/triviaqa_open_ended_gen.py --output_dir ~/datasets/experiments/triviaqa/
```

### 2. Generate Samples for prompts

```
python scripts/oeq_sample.py --model microsoft/Phi-3.5-mini-instruct --input-path ~/datasets/experiments/triviaqa/triviaqa_oe_prompts.csv --n 20 --batch-size 120 --dtype half
        --output-path ~/datasets/experiments/triviaqa/oeq_sample_msft_phi3.5-mini-instruct/responses.jsonl --tensor_parallel_size 4 --temperature 0.7 --group-batch-size 120 --word-count 500
```

Assumes 4 GPU node. Set `tensor_parallel_size` accordingly.

### 3. Segment and Score responses
```
python scripts/segscore/oeq_seg_score.py --input-path ~/datasets/experiments/triviaqa/oeq_sample_msft_phi3.5-mini-instruct/responses.jsonl
    --output-path ~/datasets/experiments/triviaqa/oeq_sample_msft_phi3.5-mini-instruct/seg_score.jsonl --group-batch-size 50 --model gpt-4.1-mini --dataset triviaqa
```

### 4. Generate Metrics on Segmented Dataset (using API model to score)
```
python scripts/segscore/gen_metric.py --input-path ~/datasets/experiments/triviaqa/oeq_sample_msft_phi3.5-mini-instruct/seg_score.jsonl --output-path ~/datasets/experiments/triviaqa/oeq_sample_msft_phi3.5-mini-instruct/si_gemini.pkl
```

### 5. Generate Metrics on Segmented Dataset (using Open Weight Model)
```
python scripts/segscore/gen_metric.py --input-path ~/datasets/experiments/triviaqa/oeq_sample_msft_phi3.5-mini-instruct/seg_score.jsonl --output-path ~/datasets/experiments/triviaqa/oeq_sample_msft_phi3.5-mini-instruct/si_deberta.pkl
```

### 6. Running embedding pipelines using Job Runner Utility [OPTIONAL]
See the template `config/sample.cfg` file to generate a configuration of scoring tasks. These can be run in a parallelized fashion using the command below:
```
python scripts/runner -c config/sample.cfg [-d,--dryrun,-q]
```

All sampling and experiments can be run on a single node with 4 x NVIDIA V100 GPUs or equivalent.

---

## Supported Datasets

- **TriviaQA**: Open-domain question answering
- **FactScore-Bio**: Biographical fact verification
- **BookSummaries**: Book plot summarization
