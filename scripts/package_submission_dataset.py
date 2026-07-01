#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def package_relative_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PACKAGE_ROOT))
    except ValueError:
        return "full combined_dataset source, not included in this release"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a PR-level sampled submission dataset.")
    parser.add_argument("--source-dir", type=Path, required=True, help="Source combined_dataset directory.")
    parser.add_argument("--dest-dir", type=Path, required=True, help="Destination directory for the sampled dataset.")
    parser.add_argument("--sample-fraction", type=float, default=0.50, help="Fraction of PRs to retain in each PR table.")
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


def _strata_frame(df: pd.DataFrame) -> pd.DataFrame:
    strata = pd.DataFrame(index=df.index)
    for column in ["agent", "state", "review_decision", "author_association"]:
        if column in df.columns:
            strata[column] = df[column].fillna("[missing]").astype(str)
    if "created_at" in df.columns:
        created = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        strata["created_month"] = created.dt.strftime("%Y-%m").fillna("[missing]")
    if strata.empty:
        strata["all"] = "all"
    return strata


def _size_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    for column in ["additions", "deletions", "changed_files", "commits_count", "reviews_count", "comments_count"]:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce").fillna(0).clip(lower=0)
            score += values
    return score


def _allocate_group_sample_sizes(group_sizes: pd.Series, fraction: float, sample_n: int, seed: int) -> dict[object, int]:
    targets = group_sizes.astype(float) * fraction
    allocation = np.floor(targets).astype(int)
    remaining = sample_n - int(allocation.sum())
    if remaining > 0:
        remainders = targets - allocation
        tie_breaker = pd.Series(
            np.random.default_rng(seed).random(len(group_sizes)),
            index=group_sizes.index,
        )
        order = (
            pd.DataFrame({"remainder": remainders, "tie": tie_breaker})
            .sort_values(["remainder", "tie"], ascending=[False, True], kind="mergesort")
            .index
        )
        for key in order[:remaining]:
            allocation.loc[key] += 1
    elif remaining < 0:
        removable = allocation[allocation > 0]
        remainders = targets.loc[removable.index] - removable
        tie_breaker = pd.Series(
            np.random.default_rng(seed).random(len(removable)),
            index=removable.index,
        )
        order = (
            pd.DataFrame({"remainder": remainders, "tie": tie_breaker})
            .sort_values(["remainder", "tie"], ascending=[True, True], kind="mergesort")
            .index
        )
        for key in order[: abs(remaining)]:
            allocation.loc[key] -= 1
    return allocation.astype(int).to_dict()


def _systematic_take(group: pd.DataFrame, n_sample: int, seed: int) -> pd.DataFrame:
    if n_sample <= 0:
        return group.iloc[0:0].copy()
    if n_sample >= len(group):
        return group.copy()

    sortable = group.copy()
    sortable["_sample_size_score"] = _size_score(sortable)
    sort_columns = ["_sample_size_score"]
    if "created_at" in sortable.columns:
        sort_columns.append("created_at")
    if "id" in sortable.columns:
        sort_columns.append("id")
    sortable = sortable.sort_values(sort_columns, kind="mergesort")

    half_floor = len(sortable) // 2
    half_ceil = len(sortable) - half_floor
    if n_sample in {half_floor, half_ceil}:
        scores = sortable["_sample_size_score"].to_numpy()
        positions = [start for start in range(0, half_floor * 2, 2)]
        selected_score = float(scores[positions].sum())
        if n_sample > half_floor:
            positions.append(len(sortable) - 1)
            selected_score += float(scores[-1])
        target_score = float(scores.sum() * (n_sample / len(sortable)))
        pair_deltas = [
            (float(scores[start + 1] - scores[start]), pair_number)
            for pair_number, start in enumerate(range(0, half_floor * 2, 2))
        ]
        tie_breaker = np.random.default_rng(seed).random(len(pair_deltas))
        pair_deltas = [
            (delta, pair_number, float(tie_breaker[pair_number]))
            for delta, pair_number in pair_deltas
        ]
        pair_deltas.sort(key=lambda item: (-abs(item[0]), item[2]))
        for delta, pair_number, _ in pair_deltas:
            if abs((selected_score + delta) - target_score) < abs(selected_score - target_score):
                positions[pair_number] += 1
                selected_score += delta
        sampled = sortable.iloc[positions].drop(columns=["_sample_size_score"])
        return sampled.copy()

    step = len(sortable) / n_sample
    offset = np.random.default_rng(seed).random() * step
    positions = [min(int(math.floor(offset + i * step)), len(sortable) - 1) for i in range(n_sample)]
    sampled = sortable.iloc[positions].drop(columns=["_sample_size_score"])
    return sampled.copy()


def sample_prs(df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if df.empty or fraction >= 1.0:
        return df.copy()
    if fraction <= 0.0:
        return df.iloc[0:0].copy()
    sample_n = max(1, int(round(len(df) * fraction)))
    sample_n = min(sample_n, len(df))

    strata = _strata_frame(df)
    group_keys = list(strata.columns)
    group_sizes = strata.groupby(group_keys, dropna=False, sort=False).size()
    allocation = _allocate_group_sample_sizes(group_sizes, fraction, sample_n, seed)
    keyed = df.join(strata.add_prefix("_stratum_"))
    sampled_parts = []
    for group_number, (key, group) in enumerate(keyed.groupby([f"_stratum_{col}" for col in group_keys], dropna=False, sort=False)):
        n_group = allocation.get(key, 0)
        sampled_parts.append(_systematic_take(group.drop(columns=[f"_stratum_{col}" for col in group_keys]), n_group, seed + group_number + 1))
    if not sampled_parts:
        return df.iloc[0:0].copy()
    return pd.concat(sampled_parts, ignore_index=True)


def filter_event_table(df: pd.DataFrame, pr_ids: set[str]) -> pd.DataFrame:
    if df.empty or "pr_id" not in df.columns:
        return df.copy()
    pr_series = df["pr_id"].fillna("").astype(str)
    return df.loc[pr_series.isin(pr_ids)].copy()


def count_rows(csv_path: Path) -> int:
    total = 0
    for chunk in pd.read_csv(csv_path, usecols=[0], chunksize=200_000, low_memory=False):
        total += len(chunk)
    return total


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
        sampled.to_csv(dest_dir / filename, index=False, quoting=csv.QUOTE_ALL, lineterminator="\n")
        written_counts[filename] = len(sampled)

    scores_path = source_dir / "footprint_scores.csv"
    if not scores_path.exists():
        scores_path = source_dir / "seniority_scores.csv"
    if scores_path.exists():
        scores = pd.read_csv(scores_path, low_memory=False)
        scores.to_csv(dest_dir / "footprint_scores.csv", index=False, quoting=csv.QUOTE_ALL, lineterminator="\n")
        written_counts["footprint_scores.csv"] = len(scores)
    return written_counts


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    dest_dir = args.dest_dir.expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    prs = pd.read_csv(source_dir / "prs.csv", low_memory=False)
    human_prs = pd.read_csv(source_dir / "human_prs.csv", low_memory=False)

    sampled_prs = sample_prs(prs, args.sample_fraction, args.seed)
    sampled_human_prs = sample_prs(human_prs, args.sample_fraction, args.seed + 1)

    agent_pr_ids = set(sampled_prs["id"].fillna("").astype(str))
    human_pr_ids = set(sampled_human_prs["id"].fillna("").astype(str))

    files_to_write: dict[str, pd.DataFrame] = {
        "prs.csv": sampled_prs,
        "human_prs.csv": sampled_human_prs,
        "comments.csv": filter_event_table(pd.read_csv(source_dir / "comments.csv", low_memory=False), agent_pr_ids),
        "reviews.csv": filter_event_table(pd.read_csv(source_dir / "reviews.csv", low_memory=False), agent_pr_ids),
        "commits.csv": filter_event_table(pd.read_csv(source_dir / "commits.csv", low_memory=False), agent_pr_ids),
        "repos.csv": filter_event_table(pd.read_csv(source_dir / "repos.csv", low_memory=False), agent_pr_ids),
        "human_comments.csv": filter_event_table(pd.read_csv(source_dir / "human_comments.csv", low_memory=False), human_pr_ids),
        "human_reviews.csv": filter_event_table(pd.read_csv(source_dir / "human_reviews.csv", low_memory=False), human_pr_ids),
        "human_commits.csv": filter_event_table(pd.read_csv(source_dir / "human_commits.csv", low_memory=False), human_pr_ids),
        "human_repos.csv": filter_event_table(pd.read_csv(source_dir / "human_repos.csv", low_memory=False), human_pr_ids),
        "users.csv": pd.read_csv(source_dir / "users.csv", low_memory=False),
    }

    if not args.drop_issues:
        for optional_name in ["issues.csv", "human_issues.csv"]:
            optional_path = source_dir / optional_name
            if optional_path.exists():
                files_to_write[optional_name] = pd.read_csv(optional_path, low_memory=False)

    for filename, df in files_to_write.items():
        df.to_csv(dest_dir / filename, index=False, quoting=csv.QUOTE_ALL, lineterminator="\n")

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
        "source_dir": package_relative_path(source_dir),
        "dest_dir": package_relative_path(dest_dir),
        "sample_fraction": args.sample_fraction,
        "sampling_method": "deterministic stratified systematic PR-level sample",
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
