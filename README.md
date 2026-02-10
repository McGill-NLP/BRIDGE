<div align="center">

# BRIDGE: Predicting Human Task Completion Time From Model Performance

[Fengyuan Liu*](https://fy-liu.github.io/), [Jay Gala*](https://jaygala24.github.io/), [Nilaksh](https://hskalin.github.io/), [Dzmitry Bahdanau](https://rizar.github.io/), [Siva Reddy](https://sivareddy.in/), [Hugo Larochelle](https://mila.quebec/en/directory/hugo-larochelle)

<sub>
*Equal Contribution
</sub>

<br>
<br>

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.07267) [![Website](https://img.shields.io/badge/Project%20Page-4285F4?style=for-the-badge&logo=googlechrome&logoColor=white)](https://mcgill-nlp.github.io/BRIDGE/)

</div>

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Running the Pipeline](#running-the-pipeline)
  - [Step 1: Parse Cybench Logs](#step-1-parse-cybench-logs)
  - [Step 2: Prepare SWE-bench IRT Input](#step-2-prepare-swe-bench-irt-input)
  - [Step 3: Build Sparse Combined py-IRT Dataset](#step-3-build-sparse-combined-py-irt-dataset)
  - [Step 4: Fit IRT Model](#step-4-fit-irt-model)
  - [Step 5: Compute Baseline Estimates](#step-5-compute-baseline-estimates)
  - [Step 6: Combine Human Time Estimates](#step-6-combine-human-time-estimates)
  - [Step 7: Merge Human Minutes into IRT Output](#step-7-merge-human-minutes-into-irt-output)
  - [Step 8: Add Release Times](#step-8-add-release-times)
- [Baselines](#baselines)
  - [Compute Baseline (Logit Success-Rate)](#compute-baseline-logit-success-rate)
  - [LLM-Based Time Estimation](#llm-based-time-estimation)
- [Analysis](#analysis)
- [License](#license)
- [Citation](#citation)

## Overview

> Evaluating the real-world capabilities of AI systems requires grounding benchmark performance in human-interpretable measures of task difficulty. Existing approaches that rely on direct human task completion time annotations are costly, noisy, and difficult to scale across benchmarks. In this work, we propose BRIDGE, a unified psychometric framework that learns the latent difficulty scale from model responses and anchors it to human task completion time. Using a two-parameter logistic Item Response Theory model, we jointly estimate latent task difficulty and model capability from model performance data across multiple benchmarks. We demonstrate that latent task difficulty varies linearly with the logarithm of human completion time, allowing human task completion time to be inferred for new benchmarks from model performance alone. Leveraging this alignment, we forecast frontier model capabilities in terms of human task length and independently reproduce METR's exponential scaling results, with the 50\% solvable task horizon doubling approximately every 6 months.

### TL;DR

- We fit a 2-parameter logistic IRT model on binary (pass/fail) outcomes from multiple benchmarks (METR, SWE-bench, MLE-bench, GDPval, Cybench) to jointly estimate latent task difficulty and model ability.
- Latent task difficulty correlates strongly with the log of human completion time (R² = 0.81 on METR tasks), enabling prediction of human task duration from model performance alone.
- On out-of-distribution benchmarks, BRIDGE outperforms both logit success-rate heuristics and LLM-based time estimators (Gemini 3 Pro, GPT-5.2), placing 92.3% of Cybench predictions within a 0.5x--2x tolerance band of actual human times.
- Frontier model capabilities are growing exponentially: the 50% solvable task-length horizon doubles approximately every 6 months, consistent with METR's findings -- but derived entirely from model performance data without human time annotations.
- Current SOTA models (as of late 2025) achieve 50% success on tasks estimated to require ~1-2.5 hours of human effort.

## Installation

```bash
# Clone the repository and navigate to the project directory
git clone https://github.com/McGill-NLP/BRIDGE.git
cd BRIDGE

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync
```

Note: Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

## Data Preparation

The pipeline expects benchmark results and evaluation logs to be present under the `data/` directory. Each benchmark source has its own normalized JSONL format, and models are mapped to canonical names via `data/model_run_mapping.json`.

Here is the expected directory structure of the project:

```
BRIDGE/
├── data/
│   ├── model_run_mapping.json            # Curated model -> run_id mapping with metadata
│   ├── all_runs.jsonl                    # Per-task run logs
│   ├── swebench_normalized_results.jsonl # SWE-bench results (normalized)
│   ├── gdpval_normalized_results.jsonl   # GDPVal benchmark results
│   ├── mlebench_normalized_results.jsonl # MLEBench benchmark results
│   ├── human_minutes_by_task.jsonl       # SWE-bench human time estimates
│   ├── cybench_fst_master.jsonl          # Cybench FST reference data
│   ├── cybench_data_by_challenges/       # Raw Cybench log files
│   ├── cybench/benchmark/               # CyBench benchmark tasks (used by LLM estimation)
│   └── experiments/evaluation/           # SWE-bench verified logs from the public leaderboard
│       ├── verified/
│       └── bash-only/
├── parse_cybench_logs.py
├── prepare_irt.py
├── prepare_sparse_pyirt.py
├── fit_irt.py
├── compute_baseline.py
├── merge_human_minutes.py
├── add_release_time.py
├── swebench_human_time_llm_estimate.py
├── cybench_human_time_llm_estimate.py
├── results_analysis.ipynb
├── run_pipeline.sh
├── pyproject.toml
└── README.md
```

## Running the Pipeline

The full analysis pipeline is orchestrated by `run_pipeline.sh`:

```bash
bash run_pipeline.sh
```

This runs the following steps in order:

### Step 1: Parse Cybench Logs

```bash
uv run python parse_cybench_logs.py --verbose
```

Parses raw Cybench evaluation log files from `data/cybench_data_by_challenges/` and produces:
- `data/cybench_normalized_results.jsonl` -- normalized per-task, per-model scores (unguided mode only, filtered to tasks with at least one model success)
- `data/cybench_human_minutes_by_task.jsonl` -- human time estimates matched from `data/cybench_fst_master.jsonl`

### Step 2: Prepare SWE-bench IRT Input

```bash
uv run python prepare_irt.py --verbose
```

Builds `data/swe_a_pyirt.jsonl` by combining:
- SWE-bench results (`data/swebench_results.jsonl` and `data/swebench_normalized_results.jsonl`)
- Verified evaluation runs from `data/experiments/evaluation/verified/` (and `bash-only/`)
- A curated model-to-run mapping (`data/model_run_mapping.json`)

Outputs a py-IRT JSONL file with one row per model (subject), containing binary pass/fail responses across all SWE-bench task IDs.

### Step 3: Build Sparse Combined py-IRT Dataset

```bash
uv run python prepare_sparse_pyirt.py \
  --model-mapping data/model_run_mapping.json \
  --pyirt-input data/swe_a_pyirt.jsonl \
  --runs-input data/all_runs.jsonl \
  --gdpval-input data/gdpval_normalized_results.jsonl \
  --mlebench-input data/mlebench_normalized_results.jsonl \
  --cybench-input data/cybench_normalized_results.jsonl \
  --output data/all_a_pyirt.jsonl \
  --print-subject-counts \
  --keep-unmapped-pyirt-subjects \
  --verbose
```

Merges data from multiple benchmarks (SWE-bench, GDPVal, MLEBench, Cybench) into a single sparse py-IRT dataset at `data/all_a_pyirt.jsonl`. Models are canonicalized via the model mapping, and duplicate responses are resolved by majority vote.

### Step 4: Fit IRT Model

```bash
uv run python fit_irt.py --input_path data/all_a_pyirt.jsonl
```

Fits a two-parameter logistic (2PL) IRT model on the combined dataset. Outputs:
- `params/all_a_pyirt.csv` -- item parameters (discrimination `a`, difficulty `b`)
- `params/all_a_pyirt_abilities.csv` -- subject (model) ability estimates

### Step 5: Compute Baseline Estimates

```bash
uv run python compute_baseline.py --input_path data/all_a_pyirt.jsonl
```

Computes simple averaging-based baselines for comparison with IRT (see [Baselines](#baselines) for details). Outputs:
- `params/all_a_pyirt_baseline.csv` -- item difficulty (success rate, logit-transformed)
- `params/all_a_pyirt_baseline_abilities.csv` -- subject ability (success rate, logit-transformed)

### Step 6: Combine Human Time Estimates

```bash
cat data/human_minutes_by_task.jsonl data/cybench_human_minutes_by_task.jsonl > data/combined_human_minutes.jsonl
```

Concatenates SWE-bench and Cybench human-minute annotations into a single file.

### Step 7: Merge Human Minutes into IRT Output

```bash
uv run python merge_human_minutes.py --csv params/all_a_pyirt.csv --jsonl data/combined_human_minutes.jsonl
```

Attaches `human_minutes` values to the IRT item parameters CSV by matching on `task_id`.

### Step 8: Add Release Times

```bash
uv run python add_release_time.py
```

Annotates the subject abilities CSV (`params/all_a_pyirt_abilities.csv`) with model release dates sourced from `data/model_run_mapping.json`.

## Baselines

In addition to the IRT-based difficulty estimates, we provide two types of baselines for comparison: a compute baseline that uses logit-transformed success rates, and an LLM-based baseline that directly prompts a language model to estimate human task completion time.

### Compute Baseline (Logit Success-Rate)

A simple averaging-based baseline that estimates task difficulty and model ability from raw pass/fail rates without fitting an IRT model. For each task, the difficulty is the logit-transformed failure rate across all models that attempted it; for each model, the ability is the logit-transformed success rate across all tasks it attempted.

```bash
uv run python compute_baseline.py \
  --input_path data/all_a_pyirt.jsonl \
  --verbose
```

| Argument | Description |
|----------|-------------|
| `--input_path` | Path to the py-IRT JSONL file (required) |
| `--output_dir` | Output directory for CSVs (default: `params/`) |
| `--verbose` | Print summary statistics (sparsity, mean/std of estimates) |

Outputs:
- `params/all_a_pyirt_baseline.csv` -- per-task success rate, difficulty, and logit-transformed difficulty
- `params/all_a_pyirt_baseline_abilities.csv` -- per-model success rate and logit-transformed ability

### LLM-Based Time Estimation

Uses an LLM to directly estimate human task completion time from task descriptions. Each task's problem statement is sent to the model with a structured meta-prompt that asks for a point estimate in minutes along with a justification. This serves as a non-psychometric baseline for comparison with IRT-derived time predictions.

Both scripts require API keys set as environment variables (`OPENAI_API_KEY` or `GOOGLE_API_KEY`), or in a `.env` file in the project root.

**SWE-bench tasks** (`swebench_human_time_llm_estimate.py`):

Loads the SWE-bench Verified dataset from HuggingFace, extracts each issue's problem statement and repository context, and prompts the LLM to estimate how long a skilled human developer would take to resolve it.

```bash
# Gemini 3 Pro
uv run python swebench_human_time_llm_estimate.py \
  --provider google --model gemini-3-pro-preview \
  --batch-size 20 --verbose

# GPT 5.2
uv run python swebench_human_time_llm_estimate.py \
  --provider openai --model gpt-5.2-2025-12-11 \
  --batch-size 20 --verbose
```

| Argument | Description |
|----------|-------------|
| `--provider` | LLM provider: `openai` or `google` (default: `openai`) |
| `--model` | Model name (defaults: `gpt-4o` for OpenAI, `gemini-2.5-flash` for Google) |
| `--output-file` | Output JSONL path (default: `data/swebench_time_estimations_{provider}_{model}.jsonl`) |
| `--max-samples` | Limit number of tasks to process |
| `--start-idx` | Resume processing from a given index |
| `--include-patch-info` | Include gold patch size in the prompt (may bias estimates) |
| `--reasoning-effort` | Reasoning effort for OpenAI o-series/GPT-5 models: `low`, `medium`, `high` (default: `medium`) |
| `--batch-size` | Number of concurrent API requests (default: `10`) |
| `--analyze-only` | Only print summary statistics from an existing results file |
| `--results-file` | Path to results file (used with `--analyze-only`) |
| `--verbose` | Print detailed progress information |

**Cybench CTF tasks** (`cybench_human_time_llm_estimate.py`):

Reads task metadata (category, difficulty, subtasks, prompts) from the local CyBench benchmark directory and prompts the LLM to estimate how long an experienced CTF player would take to solve each challenge.

```bash
# Gemini 3 Pro
uv run python cybench_human_time_llm_estimate.py \
  --provider google --model gemini-3-pro-preview \
  --benchmark-path data/cybench/benchmark --no-subtasks \
  --batch-size 10 --verbose

# GPT 5.2
uv run python cybench_human_time_llm_estimate.py \
  --provider openai --model gpt-5.2-2025-12-11 \
  --benchmark-path data/cybench/benchmark --no-subtasks \
  --batch-size 10 --verbose
```

| Argument | Description |
|----------|-------------|
| `--benchmark-path` | Path to CyBench benchmark directory (default: `data/cybench/benchmark`) |
| `--provider` | LLM provider: `openai` or `google` (default: `openai`) |
| `--model` | Model name (same defaults as SWE-bench script) |
| `--output-file` | Output JSONL path (default: `data/cybench_time_estimations_{provider}_{model}.jsonl`) |
| `--max-tasks` | Limit number of tasks to process |
| `--category` | Filter by CTF category (e.g., `crypto`, `web`, `pwn`, `reverse`, `forensics`) |
| `--competition` | Filter by competition name (e.g., `hackthebox`, `hkcert-ctf`) |
| `--use-hard-prompt` | Use hard prompt (less context) instead of easy prompt |
| `--no-subtasks` | Exclude subtask information from the prompt |
| `--reasoning-effort` | Reasoning effort for OpenAI o-series/GPT-5 models (default: `medium`) |
| `--batch-size` | Number of concurrent API requests (default: `10`) |
| `--analyze-only` | Only print summary statistics from an existing results file |
| `--results-file` | Path to results file (used with `--analyze-only`) |
| `--verbose` | Print detailed progress information |

After running the full pipeline, the following files are generated:

| File | Description |
|------|-------------|
| `data/cybench_normalized_results.jsonl` | Cybench benchmark results (step 1) |
| `data/cybench_human_minutes_by_task.jsonl` | Cybench human time estimates (step 1) |
| `data/swe_a_pyirt.jsonl` | SWE-bench py-IRT input (step 2) |
| `data/all_a_pyirt.jsonl` | Combined sparse py-IRT dataset (step 3) |
| `data/combined_human_minutes.jsonl` | Combined human time estimates (step 6) |
| `params/all_a_pyirt.csv` | IRT item parameters with human_minutes (steps 4, 7) |
| `params/all_a_pyirt_abilities.csv` | IRT subject abilities with release dates (steps 4, 8) |
| `params/all_a_pyirt_baseline.csv` | Baseline item difficulty estimates (step 5) |
| `params/all_a_pyirt_baseline_abilities.csv` | Baseline subject ability estimates (step 5) |
| `data/swebench_time_estimations_{provider}_{model}.jsonl` | LLM time estimates for SWE-bench |
| `data/cybench_time_estimations_{provider}_{model}.jsonl` | LLM time estimates for Cybench |

## Analysis

After running the pipeline and the LLM baseline scripts, you can reproduce all figures and tables from the paper using the provided Jupyter notebook:

```bash
uv run jupyter notebook results_analysis.ipynb
```

The notebook loads the fitted IRT parameters, baseline estimates, LLM time estimations, and ground-truth human times, then walks through the following analyses:

1. **Model Fitting** -- Fits two linear regressions mapping task difficulty to log(human minutes): one using IRT difficulty `b` and one using the logit success-rate baseline. Both are trained on METR benchmark tasks that have ground-truth human time annotations.
2. **Task Difficulty vs Human Time** -- Scatter plot of IRT difficulty against actual human completion time for METR tasks, with the fitted regression line (R² = 0.81).
3. **Task Length Estimation Distributions** -- Histograms of predicted human completion times for SWE-bench, GDPval, MLE-bench, and Cybench tasks on a log scale.
4. **SWE-bench Time Bucket Classification** -- Compares accuracy, macro F1, and weighted Cohen's kappa across four methods (Logit Success Rate, Gemini 3 Pro, GPT-5.2, and BRIDGE) for predicting SWE-bench time buckets.
5. **Cybench Task Length Prediction** -- Scatter plots of predicted vs actual human time for each method on Cybench, with R² and within-2x accuracy metrics.
6. **Success Probability vs Task Length** -- 2PL IRT success-probability curves for frontier models (best per 5-month release window) across all benchmarks, showing how success decays with increasing task length.
7. **Task Length Frontier Forecasting** -- Exponential fits of the 50% and 80% solvable task-length horizons over time, with bootstrap confidence intervals and doubling-time estimates.

All plots are saved to the `plots/` directory as PDF files.

## License

[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

## Citation

```bibtex
@article{liu2026bridge0,
  title   = {BRIDGE: Predicting Human Task Completion Time From Model Performance},
  author  = {Fengyuan Liu and Jay Gala and Nilaksh and Dzmitry Bahdanau and Siva Reddy and Hugo Larochelle},
  year    = {2026},
  journal = {arXiv preprint arXiv: 2602.07267}
}
```
