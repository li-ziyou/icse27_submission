#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent_audience_submission.result_utils import bundle_results, dataset_row_counts, write_comparison

DEFAULT_PAPER_TITLE = "Characterizing Human-Agent Dynamics in Agentic Pull Requests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the submission reproduction pipeline and bundle outputs into JSON.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing combined_dataset-style CSVs.")
    parser.add_argument("--human-dataset-dir", type=Path, help="Optional directory for human_*.csv files. Defaults to --dataset-dir.")
    parser.add_argument("--optional-dataset-dir", type=Path, help="Directory for optional cached tables such as linked_issues.csv.")
    parser.add_argument("--workdir", type=Path, help="Run directory where outputs/, figures/, and results_bundle.json are created.")
    parser.add_argument("--run-name", default="submission_run", help="Run name used when --workdir is not provided.")
    parser.add_argument("--run-fraction", type=float, default=1.0, help="PR sampling fraction passed into the notebook-derived pipeline.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed used for deterministic PR sampling.")
    parser.add_argument("--start-date", default="2025-04-01", help="Inclusive start date for the paper scope.")
    parser.add_argument("--end-date", default="2026-01-31", help="Inclusive end date for the paper scope.")
    parser.add_argument("--drop-issues", action="store_true", help="Skip issue tables even if they exist.")
    parser.add_argument("--compare-to", type=Path, help="Existing run/output directory to compare CSV outputs against.")
    parser.add_argument("--python-executable", default=sys.executable, help="Python interpreter used to execute the core pipeline.")
    parser.add_argument("--paper-title", default=DEFAULT_PAPER_TITLE, help="Paper title stored in the JSON bundle metadata.")
    parser.add_argument("--no-clean", action="store_true", help="Preserve existing files in the workdir instead of cleaning outputs first.")
    return parser.parse_args()


def _default_optional_dataset_dir(dataset_dir: Path) -> Path:
    sibling = dataset_dir.parent / "dataset"
    if sibling.exists():
        return sibling
    return ROOT / "data"


def _build_env(args: argparse.Namespace, workdir: Path) -> dict[str, str]:
    dataset_dir = args.dataset_dir.expanduser().resolve()
    human_dataset_dir = (args.human_dataset_dir or args.dataset_dir).expanduser().resolve()
    optional_dataset_dir = (args.optional_dataset_dir or _default_optional_dataset_dir(dataset_dir)).expanduser().resolve()

    env = os.environ.copy()
    env["SUBMISSION_PIPELINE_ROOT"] = str(ROOT)
    env["SUBMISSION_DATASET_DIR"] = str(dataset_dir)
    env["SUBMISSION_HUMAN_DATASET_DIR"] = str(human_dataset_dir)
    env["SUBMISSION_OPTIONAL_DATASET_DIR"] = str(optional_dataset_dir)
    env["SUBMISSION_RESOURCE_DIR"] = str((ROOT / "resources").resolve())
    env["SUBMISSION_PIPELINE_CACHE_DIR"] = str((workdir / ".pipeline_cache").resolve())
    env["SUBMISSION_RUN_FRACTION"] = str(args.run_fraction)
    env["SUBMISSION_RANDOM_SEED"] = str(args.seed)
    env["SUBMISSION_START_DATE"] = args.start_date
    env["SUBMISSION_END_DATE"] = args.end_date
    env["SUBMISSION_DROP_ISSUES"] = "1" if args.drop_issues else "0"
    env["SUBMISSION_PAPER_ONLY"] = "1"
    env["SUBMISSION_AUTO_EXPORT_ALL_PLOTS"] = "0"
    env["SUBMISSION_SAVE_INTERMEDIATE_PNGS"] = "0"
    env["SUBMISSION_TIMELINE_PR_KEY"] = "microsoft/testfx#5633"
    env["MPLCONFIGDIR"] = str((workdir / ".mplconfig").resolve())
    env["MPLBACKEND"] = "Agg"
    env["XDG_CACHE_HOME"] = str((workdir / ".cache").resolve())
    return env


def _clean_workdir(workdir: Path) -> None:
    for relative in ["outputs", "figures", ".pipeline_cache", ".mplconfig", ".cache"]:
        target = workdir / relative
        if target.exists():
            shutil.rmtree(target)
    for relative in ["results_bundle.json", "comparison_to_reference.json"]:
        target = workdir / relative
        if target.exists():
            target.unlink()


def _portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def main() -> int:
    args = parse_args()
    workdir = (args.workdir or (ROOT / "results" / args.run_name)).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    if not args.no_clean:
        _clean_workdir(workdir)

    env = _build_env(args, workdir)
    command = [
        args.python_executable,
        str((ROOT / "src" / "agent_audience_submission" / "core_pipeline.py").resolve()),
    ]
    subprocess.run(command, cwd=workdir, env=env, check=True)

    metadata = {
        "paper_title": args.paper_title,
        "run_name": args.run_name,
        "workdir": _portable_path(workdir),
        "dataset_dir": _portable_path(args.dataset_dir),
        "human_dataset_dir": _portable_path(args.human_dataset_dir or args.dataset_dir),
        "optional_dataset_dir": _portable_path(args.optional_dataset_dir or _default_optional_dataset_dir(args.dataset_dir.expanduser().resolve())),
        "run_fraction": args.run_fraction,
        "seed": args.seed,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "drop_issues": args.drop_issues,
        "python_executable": args.python_executable,
        "dataset_row_counts": dataset_row_counts(args.dataset_dir.expanduser().resolve()),
    }
    bundle_path = bundle_results(workdir, metadata)
    print(f"Result bundle written to {bundle_path}")

    if args.compare_to:
        comparison_path = workdir / "comparison_to_reference.json"
        write_comparison(args.compare_to.expanduser().resolve(), workdir, comparison_path)
        print(f"Comparison written to {comparison_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
