.venv/bin/python parse_cybench_logs.py --verbose

.venv/bin/python prepare_irt.py --verbose

.venv/bin/python prepare_sparse_pyirt.py \
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

.venv/bin/python fit_irt.py --input_path data/all_a_pyirt.jsonl

.venv/bin/python compute_baseline.py --input_path data/all_a_pyirt.jsonl

cat data/human_minutes_by_task.jsonl data/cybench_human_minutes_by_task.jsonl > data/combined_human_minutes.jsonl

.venv/bin/python merge_human_minutes.py --csv params/all_a_pyirt.csv --jsonl data/combined_human_minutes.jsonl

.venv/bin/python run_swebench_baseline_analysis.py

.venv/bin/python run_cybench_analysis.py