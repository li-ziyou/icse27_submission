from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RESULT_TABLES = {
    "outputs/rq1_intensity.csv": "rq1_intensity",
    "outputs/baseline_first_action_table.csv": "baseline_first_action",
    "outputs/baseline_latency_summary.csv": "baseline_latency",
    "outputs/baseline_merge_proxy.csv": "baseline_merge_proxy",
    "outputs/sentiment_vader_robustness.csv": "sentiment_summary",
    "outputs/comment_intent_signals_by_pathway.csv": "intent_signals_by_pathway",
    "outputs/comment_intent_signals_stats.csv": "intent_signals_stats",
    "outputs/comment_intent_signal_examples.csv": "intent_signal_examples",
    "outputs/pathway_summary_metrics.csv": "pathway_summary_metrics",
    "outputs/baseline_sanity_checks.csv": "baseline_sanity_checks",
    "outputs/baseline_rq2_sanity_checks.csv": "baseline_rq2_sanity_checks",
    "outputs/per_agent_summary.csv": "per_agent_summary",
    "outputs/agent_sensitivity_drop_top.csv": "agent_sensitivity_drop_top",
}

EXPECTED_ARTIFACTS = [
    "outputs/rq1_intensity.csv",
    "outputs/pdf_tables/rq1_intensity.pdf",
    "outputs/pdf_tables/rq2_first_action.pdf",
    "outputs/baseline_first_action_table.csv",
    "outputs/baseline_latency_summary.csv",
    "outputs/pdf_tables/rq2_latency.pdf",
    "outputs/baseline_merge_proxy.csv",
    "outputs/pdf_tables/rq2_merge_proxy.pdf",
    "figures/fig_baseline_latency_log.pdf",
    "figures/fig_first_action.pdf",
    "figures/rq2_timeline_example.pdf",
    "figures/fig_pathway_summary.pdf",
    "outputs/pathway_summary_metrics.csv",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        if getattr(value, "tzinfo", None) is None:
            return value.isoformat()
        return value.astimezone(timezone.utc).isoformat()
    if pd.isna(value):
        return None
    return value


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {column: _json_safe(value) for column, value in row.items()}
        for row in df.replace({np.nan: None}).to_dict(orient="records")
    ]


def dataset_row_counts(dataset_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not dataset_dir.exists():
        return counts
    for csv_path in sorted(dataset_dir.glob("*.csv")):
        with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
            line_count = sum(1 for _ in handle)
        counts[csv_path.name] = max(line_count - 1, 0)
    return counts


def collect_manifest(run_dir: Path) -> list[dict[str, Any]]:
    manifest = []
    for relative_path in EXPECTED_ARTIFACTS:
        artifact_path = run_dir / relative_path
        manifest.append(
            {
                "path": relative_path,
                "exists": artifact_path.exists(),
                "size_bytes": artifact_path.stat().st_size if artifact_path.exists() else 0,
            }
        )
    return manifest


def bundle_results(run_dir: Path, metadata: dict[str, Any]) -> Path:
    tables: dict[str, Any] = {}
    schemas: dict[str, Any] = {}
    for relative_path, table_name in RESULT_TABLES.items():
        csv_path = run_dir / relative_path
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        tables[table_name] = dataframe_to_records(df)
        schemas[table_name] = {
            "source_csv": relative_path,
            "rows": int(len(df)),
            "columns": list(df.columns),
        }

    bundle = {
        "metadata": {key: _json_safe(value) for key, value in metadata.items()},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": collect_manifest(run_dir),
        "table_schemas": schemas,
        "tables": tables,
    }
    bundle_path = run_dir / "results_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    return bundle_path


def _stable_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.reset_index(drop=True)
    sortable = df.copy()
    sort_key = sortable.fillna("").astype(str).agg("||".join, axis=1)
    sortable = sortable.assign(__sort_key=sort_key).sort_values("__sort_key", kind="mergesort")
    return sortable.drop(columns="__sort_key").reset_index(drop=True)


def _column_match(left: pd.Series, right: pd.Series, tolerance: float) -> tuple[bool, float | None]:
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    numeric_like = left.notna().sum() == left_num.notna().sum() and right.notna().sum() == right_num.notna().sum()
    if numeric_like:
        left_values = left_num.fillna(0.0).to_numpy(dtype=float)
        right_values = right_num.fillna(0.0).to_numpy(dtype=float)
        if left_values.size == 0:
            return True, 0.0
        max_abs_diff = float(np.max(np.abs(left_values - right_values)))
        return bool(np.allclose(left_values, right_values, atol=tolerance, rtol=0.0)), max_abs_diff
    return bool(left.fillna("").astype(str).equals(right.fillna("").astype(str))), None


def compare_outputs(reference_dir: Path, candidate_dir: Path, tolerance: float = 1e-9) -> dict[str, Any]:
    comparisons = []
    overall_match = True
    for relative_path in RESULT_TABLES:
        reference_path = reference_dir / relative_path
        candidate_path = candidate_dir / relative_path
        entry: dict[str, Any] = {
            "path": relative_path,
            "reference_exists": reference_path.exists(),
            "candidate_exists": candidate_path.exists(),
        }
        if not reference_path.exists() or not candidate_path.exists():
            overall_match = False
            entry["match"] = False
            comparisons.append(entry)
            continue

        ref_df = _stable_sort(pd.read_csv(reference_path))
        cand_df = _stable_sort(pd.read_csv(candidate_path))
        entry["reference_rows"] = int(len(ref_df))
        entry["candidate_rows"] = int(len(cand_df))
        entry["reference_columns"] = list(ref_df.columns)
        entry["candidate_columns"] = list(cand_df.columns)

        if list(ref_df.columns) != list(cand_df.columns) or len(ref_df) != len(cand_df):
            overall_match = False
            entry["match"] = False
            comparisons.append(entry)
            continue

        column_results = []
        file_match = True
        for column in ref_df.columns:
            column_match, max_abs_diff = _column_match(ref_df[column], cand_df[column], tolerance)
            if not column_match:
                file_match = False
            column_results.append(
                {
                    "column": column,
                    "match": column_match,
                    "max_abs_diff": max_abs_diff,
                }
            )
        entry["columns"] = column_results
        entry["match"] = file_match
        overall_match = overall_match and file_match
        comparisons.append(entry)

    return {
        "reference_dir": str(reference_dir),
        "candidate_dir": str(candidate_dir),
        "tolerance": tolerance,
        "overall_match": overall_match,
        "comparisons": comparisons,
    }


def write_comparison(reference_dir: Path, candidate_dir: Path, output_path: Path, tolerance: float = 1e-9) -> Path:
    comparison = compare_outputs(reference_dir, candidate_dir, tolerance=tolerance)
    output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return output_path
