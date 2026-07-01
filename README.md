# Characterizing Human-Agent Dynamics in Agentic Pull Requests

This folder contains the ICSE 2027 artifact package for reproducing the paper analyses on a deterministic 10% PR-level sample of the dataset.

## Contents

- `data/combined_dataset/` contains the sampled CSV dataset.
- `data/derived/` contains sampled paper-derived retained-feedback labels and the seniority score table used by manuscript figure generation.
- `scripts/run_reproduction.py` runs the analysis pipeline and bundles the results.
- `scripts/generate_manuscript_figures.py` generates the current manuscript figure set from the sample run.
- `scripts/package_submission_dataset.py` rebuilds the 10% sample from a full `combined_dataset` source, if the full data are available.
- `src/agent_audience_submission/` contains the analysis pipeline code.
- `resources/vader_lexicon.txt` is the local sentiment lexicon used by the pipeline.
- `results/submission_sample/` contains a previously generated sample run.
- `requirements.txt` lists the Python dependencies.

## Dataset

The released CSVs are a deterministic random sample at the PR level with `sample_fraction=0.10` and `seed=7`.
After PRs are sampled, comments, reviews, commits, and repository rows are filtered to the sampled PR IDs.
The issue tables are omitted from this release.
`users.csv` is kept intact because the activity-score construction uses the full public user metadata table.
The derived retained-feedback files are filtered to the sampled agent-authored PRs.
`data/derived/seniority_scores.csv` is kept intact because the manuscript defines activity-score quartiles over the full agent-scope scoreable account set.

Row counts for the source and sampled tables are recorded in `data/combined_dataset/sample_manifest.json`.
Because this package contains a 10% sample, regenerated numeric results will not exactly match the full-paper values.
The package reproduces the analysis workflow, table construction, and current manuscript figure families on the released sample.
When `data/derived/` is present, manuscript figures use the same retained-feedback labels, refined actor classes, and seniority-score definitions as the paper.

## Requirements

Use Python 3.10 or newer.
Install dependencies from this folder:

```bash
python -m pip install -r requirements.txt
```

## Run the Sample Analysis

From this folder, run:

```bash
./run_sample_analysis.sh
```

The script writes a fresh run to:

- `results/submission_sample/outputs/`
- `results/submission_sample/figures/`
- `results/submission_sample/manuscript_tables/`
- `results/submission_sample/results_bundle.json`

The `figures/` directory contains the manuscript analysis/result figures generated from the sample run.
The illustrative workflow/example figure is omitted because it is not a sample-derived result.
Text-only results are kept in CSV outputs rather than converted into extra plots.
The mapping from manuscript labels to artifact files is written to `results/submission_sample/manuscript_tables/manuscript_figure_map.csv`.

The generated figure files are:

- `fig_rq1_activity_score_strip`
- `fig_rq1_human_triager_quartiles`
- `fig_rq2_feedback_body_syntax`
- `fig_rq2_feedback_actor_intent`
- `fig_results_actor_intent_cycle_latency`
- `fig_results_post_feedback_revision`
- `fig_results_integration_by_pathway`
- `fig_results_rq4_endpoint_delta`

The figure directory is cleaned before manuscript figures are regenerated, so old pipeline plots do not linger.

## Manual Command

The shell script is a thin wrapper around:

```bash
python scripts/run_reproduction.py \
  --dataset-dir data/combined_dataset \
  --human-dataset-dir data/combined_dataset \
  --optional-dataset-dir data \
  --workdir results/submission_sample \
  --run-name submission_sample \
  --drop-issues \
  --paper-title "Characterizing Human-Agent Dynamics in Agentic Pull Requests"

python scripts/generate_manuscript_figures.py \
  --data-dir data/combined_dataset \
  --run-dir results/submission_sample \
  --figure-dir results/submission_sample/figures \
  --table-dir results/submission_sample/manuscript_tables
```

## Rebuilding the Sample from Full Data

If a full `combined_dataset` directory is available, rebuild the released sample with:

```bash
python scripts/package_submission_dataset.py \
  --source-dir /path/to/full/combined_dataset \
  --dest-dir data/combined_dataset \
  --derived-source-dir /path/to/paper/derived_tables \
  --sample-fraction 0.10 \
  --seed 7
```

This sampling step is deterministic.
If `--derived-source-dir` is provided, the script also writes sampled derived tables to `data/derived/`.
