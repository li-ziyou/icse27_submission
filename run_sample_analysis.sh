#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif [[ -x /opt/anaconda3/bin/python ]]; then
  PYTHON_BIN="/opt/anaconda3/bin/python"
else
  PYTHON_BIN="python3"
fi
PAPER_TITLE="Characterizing Human-Agent Dynamics in Agentic Pull Requests"

"$PYTHON_BIN" "$ROOT_DIR/scripts/run_reproduction.py" \
  --dataset-dir "$ROOT_DIR/data/combined_dataset" \
  --human-dataset-dir "$ROOT_DIR/data/combined_dataset" \
  --optional-dataset-dir "$ROOT_DIR/data" \
  --workdir "$ROOT_DIR/results/submission_sample" \
  --run-name submission_sample \
  --drop-issues \
  --paper-title "$PAPER_TITLE" \
  --python-executable "$PYTHON_BIN"

rm -f \
  "$ROOT_DIR/results/submission_sample/.pipeline_cache/pathway_share_by_seniority.csv" \
  "$ROOT_DIR/results/submission_sample/outputs/pathway_mix_by_seniority_pivot.csv" \
  "$ROOT_DIR/results/submission_sample/outputs/pathway_share_by_seniority.csv" \
  "$ROOT_DIR/results/submission_sample/outputs/pdf_tables/pathway_mix_by_seniority.pdf" \
  "$ROOT_DIR/results/submission_sample/outputs/seniority_robustness_full_tables.csv" \
  "$ROOT_DIR/results/submission_sample/outputs/seniority_robustness_note.md" \
  "$ROOT_DIR/results/submission_sample/outputs/seniority_robustness_summary.csv" \
  "$ROOT_DIR/results/submission_sample/outputs/seniority_robustness_summary.tex"

rm -rf \
  "$ROOT_DIR/results/submission_sample/figures" \
  "$ROOT_DIR/results/submission_sample/manuscript_tables"
"$PYTHON_BIN" "$ROOT_DIR/scripts/generate_manuscript_figures.py" \
  --data-dir "$ROOT_DIR/data/combined_dataset" \
  --run-dir "$ROOT_DIR/results/submission_sample" \
  --figure-dir "$ROOT_DIR/results/submission_sample/figures" \
  --table-dir "$ROOT_DIR/results/submission_sample/manuscript_tables"
