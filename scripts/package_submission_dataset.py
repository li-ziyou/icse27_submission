#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a PR-level sampled submission dataset.")
    parser.add_argument("--source-dir", type=Path, required=True, help="Source combined_dataset directory.")
    parser.add_argument("--dest-dir", type=Path, required=True, help="Destination directory for the sampled dataset.")
    parser.add_argument("--sample-fraction", type=float, default=0.10, help="Fraction of PRs to retain in each PR table.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for deterministic PR sampling.")
    parser.add_argument(
        "--derived-source-dir",
        type=Path,
        default=None,
        help="Optional directory containing paper derived CSVs such as engagement_feedback_events_filtered.csv.",
    )
    parser.add_argument(
        "--derived-dest-dir",
        type=Path,
        default=None,
        help="Destination for sampled derived CSVs. Defaults to DEST_DIR/../derived.",
    )
    parser.add_argument("--drop-issues", dest="drop_issues", action="store_true", default=True, help="Exclude issues.csv and human_issues.csv.")
    parser.add_argument("--keep-issues", dest="drop_issues", action="store_false", help="Retain issue tables.")
    return parser.parse_args()


def sample_prs(df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if df.empty or fraction >= 1.0:
        return df.copy()
    if fraction <= 0.0:
        return df.iloc[0:0].copy()
    sample_n = max(1, int(round(len(df) * fraction)))
    sample_n = min(sample_n, len(df))
    return df.sample(n=sample_n, random_state=seed).copy()


def filter_event_table(df: pd.DataFrame, pr_ids: set[str]) -> pd.DataFrame:
    if df.empty or "pr_id" not in df.columns:
        return df.copy()
    pr_series = df["pr_id"].fillna("").astype(str)
    return df.loc[pr_series.isin(pr_ids)].copy()


def count_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def paper_pr_keys(prs: pd.DataFrame) -> set[str]:
    text = prs["url"].fillna("").astype(str)
    parts = text.str.extract(r"github\.com/[^/]+/([^/]+)/pull/(\d+)", expand=True)
    keys = parts[0].fillna("") + "#" + parts[1].fillna("")
    return set(keys.loc[parts[0].notna()])


def write_sampled_derived_tables(source_dir: Path, dest_dir: Path, sampled_agent_prs: pd.DataFrame) -> dict[str, int]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    keys = paper_pr_keys(sampled_agent_prs)
    written_counts: dict[str, int] = {}
    for filename in ["engagement_feedback_events_filtered.csv", "comment_feedback_intent_labels.csv"]:
        source_path = source_dir / filename
        if not source_path.exists():
            continue
        df = pd.read_csv(source_path, low_memory=False)
        keep = df["pr_key"].fillna("").astype(str).isin(keys)
        if "pr_author_type" in df.columns:
            keep &= df["pr_author_type"].fillna("").astype(str).eq("agent_authored")
        sampled = df.loc[keep].copy()
        sampled.to_csv(dest_dir / filename, index=False)
        written_counts[filename] = len(sampled)

    scores_path = source_dir / "seniority_scores.csv"
    if scores_path.exists():
        scores = pd.read_csv(scores_path, low_memory=False)
        scores.to_csv(dest_dir / "seniority_scores.csv", index=False)
        written_counts["seniority_scores.csv"] = len(scores)
    return written_counts


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    dest_dir = args.dest_dir.expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    prs = pd.read_csv(source_dir / "prs.csv")
    human_prs = pd.read_csv(source_dir / "human_prs.csv")

    sampled_prs = sample_prs(prs, args.sample_fraction, args.seed)
    sampled_human_prs = sample_prs(human_prs, args.sample_fraction, args.seed + 1)

    agent_pr_ids = set(sampled_prs["id"].fillna("").astype(str))
    human_pr_ids = set(sampled_human_prs["id"].fillna("").astype(str))

    files_to_write: dict[str, pd.DataFrame] = {
        "prs.csv": sampled_prs,
        "human_prs.csv": sampled_human_prs,
        "comments.csv": filter_event_table(pd.read_csv(source_dir / "comments.csv"), agent_pr_ids),
        "reviews.csv": filter_event_table(pd.read_csv(source_dir / "reviews.csv"), agent_pr_ids),
        "commits.csv": filter_event_table(pd.read_csv(source_dir / "commits.csv"), agent_pr_ids),
        "repos.csv": filter_event_table(pd.read_csv(source_dir / "repos.csv"), agent_pr_ids),
        "human_comments.csv": filter_event_table(pd.read_csv(source_dir / "human_comments.csv"), human_pr_ids),
        "human_reviews.csv": filter_event_table(pd.read_csv(source_dir / "human_reviews.csv"), human_pr_ids),
        "human_commits.csv": filter_event_table(pd.read_csv(source_dir / "human_commits.csv"), human_pr_ids),
        "human_repos.csv": filter_event_table(pd.read_csv(source_dir / "human_repos.csv"), human_pr_ids),
        "users.csv": pd.read_csv(source_dir / "users.csv"),
    }

    if not args.drop_issues:
        for optional_name in ["issues.csv", "human_issues.csv"]:
            optional_path = source_dir / optional_name
            if optional_path.exists():
                files_to_write[optional_name] = pd.read_csv(optional_path)

    for filename, df in files_to_write.items():
        df.to_csv(dest_dir / filename, index=False)

    source_counts = {
        path.name: count_rows(path)
        for path in sorted(source_dir.glob("*.csv"))
        if (not args.drop_issues) or path.name not in {"issues.csv", "human_issues.csv"}
    }
    sampled_counts = {path.name: count_rows(path) for path in sorted(dest_dir.glob("*.csv"))}

    derived_counts: dict[str, int] = {}
    if args.derived_source_dir is not None:
        derived_source_dir = args.derived_source_dir.expanduser().resolve()
        derived_dest_dir = (
            args.derived_dest_dir.expanduser().resolve()
            if args.derived_dest_dir is not None
            else dest_dir.parent / "derived"
        )
        derived_counts = write_sampled_derived_tables(derived_source_dir, derived_dest_dir, sampled_prs)

    manifest = {
        "source_dir": str(source_dir),
        "dest_dir": str(dest_dir),
        "sample_fraction": args.sample_fraction,
        "seed": args.seed,
        "drop_issues": args.drop_issues,
        "source_row_counts": source_counts,
        "sampled_row_counts": sampled_counts,
        "sampled_derived_row_counts": derived_counts,
    }
    (dest_dir / "sample_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
