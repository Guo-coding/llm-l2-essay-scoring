# LLM Scoring of L2 Essays
***Shiyi Guo***

Code and data used for my master's thesis on LLM-based scoring of essays from
the ELLIPSE corpus. The repository includes prompt-based scoring, linguistic
feature extraction, Ridge calibration and fusion, statistical tests, and the
figures reported in the thesis.

## Contents

```text
data/
  raw/                original ELLIPSE files
  interim/            files created during processing
  processed/          sampled essays and model scores
  resources/          lexical resources
external/             location for TAACO
results/              tables, tests, and figures
src/                  Python scripts
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

TAACO is only needed to extract the cohesion features again. Place it in
`external/TAACO`, or set `TAACO_PATH` to its location.

## Data preparation

The processed dataset and results are already included. To rebuild the sample
from the original files, run:

```bash
python src/prepare_dataset.py
python src/sample_dataset.py
cp data/processed/ellipse_sampled_1200.jsonl data/processed/ellipse_results.jsonl
```

Feature extraction:

```bash
python src/extract_readability_lexical.py
python src/extract_syntactic.py
python src/extract_cohesion.py
```

## LLM scoring

API keys, model names, and prompt conditions are read from environment
variables. For example:

```bash
export OPENAI_API_KEY="..."
export ELLIPSE_OPENAI_MODELS="gpt-5.4"
export ELLIPSE_CONDITIONS="essay_only,essay_no_rubric"
python src/run_llm_scoring.py
```

The four conditions are `essay_only`, `essay_no_rubric`,
`essay_plus_analytic_rubric`, and `essay_plus_all_metrics`.

## Analysis

```bash
python src/summarize_results.py
python src/run_fusion_ablation.py
python src/test_baseline.py
python src/test_rubric_conditions.py
python src/test_analytic_dimensions.py
python src/test_linguistic_prompt.py
python src/test_fusion.py
python src/test_ablation.py
python src/generate_figures.py
```

Provider keys and temporary batch files should not be committed.
