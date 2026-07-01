# ICSE 2027 10% Sample (No Issue Tables)

This dataset is the 10% PR-level sampled artifact package for:

`Characterizing Human-Agent Dynamics in Agentic Pull Requests`

Contents:

- `prs.csv`, `comments.csv`, `reviews.csv`, `commits.csv`, `repos.csv`
- `human_prs.csv`, `human_comments.csv`, `human_reviews.csv`, `human_commits.csv`, `human_repos.csv`
- `users.csv`
- `sample_manifest.json`

Notes:

- Sampling is deterministic PR-level sampling with `sample_fraction=0.10` and `seed=7`.
- Issue tables are intentionally omitted.
- `users.csv` is kept intact because it is required for the paper's seniority construction.
- Row counts for both the source dataset and the sampled dataset are recorded in `sample_manifest.json`.
