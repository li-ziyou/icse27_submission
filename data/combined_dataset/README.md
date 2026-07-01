# ICSE 2027 50% Sample (No Issue Tables)

This dataset is the 50% PR-level sampled artifact package for:

`Characterizing Human-Agent Dynamics in Agentic Pull Requests`

Contents:

- `prs.csv`, `comments.csv`, `reviews.csv`, `commits.csv`, `repos.csv`
- `human_prs.csv`, `human_comments.csv`, `human_reviews.csv`, `human_commits.csv`, `human_repos.csv`
- `users.csv`

Notes:

- Sampling is deterministic stratified systematic PR-level sampling with `sample_fraction=0.50` and `seed=7`.
- Issue tables are intentionally omitted.
- `users.csv` is kept intact because it is required for the paper's seniority construction.
