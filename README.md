# Semantic Isotropy

Research on Uncertainty Quantification in Long Form Generations using Semantic Isotropy

## Setup
```
pip install -r requirements.txt
```
Install local libraries
```
pip install -e .
```

## New Experiments Quickstart workflow

### 1. Generate Set of Prompt Entities
```
python triviaqa_open_ended_gen.py --output_dir ~/datasets/experiments/triviaqa_entities/
```

### 2. Generate Samples for prompts

```
python scripts/oeq_sample.py --model microsoft/Phi-3.5-mini-instruct --input-path ~/datasets/experiments/triviaqa_entities/triviaqa_oe_prompts.csv --k 20 --batch-size 120 --dtype half
        --output-path ~/datasets/experiments/triviaqa_entities/oeq_sample_msft_phi3.5-mini-instruct/responses.json --tensor_parallel_size 4 --temperature 0.6 --group-batch-size 120 --word-count 500
```

Assumes 4 GPU node. Set `tensor_parallel_size` accordingly.

### 3. Segment and Score responses
```
python scripts/segscore/oeq_seg_score.py --input-path ~/datasets/experiments/triviaqa_entities/oeq_sample_msft_phi3.5-mini-instruct/responses.json
    --output-path ~/datasets/experiments/triviaqa_entities/oeq_sample_msft_phi3.5-mini-instruct/seg_score.json --group-batch-size 50 --model gpt-4.1-mini --dataset triviaqa
```

### 4. Generate Metrics on Segmented Dataset (using API model to score)
```
python scripts/segscore/gen_metric.py --input-path ~/datasets/experiments/triviaqa_entities/oeq_sample_msft_phi3.5-mini-instruct/seg_score.json --output-path ~/datasets/experiments/triviaqa_entities/oeq_sample_msft_phi3.5-mini-instruct/si_gemini.pkl
    --metric si --embedding-model gemini_004 --device cpu --group-batch-size 100 --response-count 500 --response-max 500
```

### 5. Generate Metrics on Segmented Dataset (using Open Weight Model)
```
python scripts/segscore/gen_metric.py --input-path ~/datasets/experiments/triviaqa_entities/oeq_sample_msft_phi3.5-mini-instruct/seg_score.json --output-path ~/datasets/experiments/triviaqa_entities/oeq_sample_msft_phi3.5-mini-instruct/si_deberta.pkl
    --metric si --embedding-model nomic-ai/nomic-embed-text-v1 --device cuda:0 --group-batch-size 100 --response-count 500 --response-max 500
```

### 6. Running embedding pipelines using Job Runner Utility
```
python scripts/runner -c config/sample.cfg [-d,--dryrun,-q]
```
