#!/usr/bin/env python
"""Generate manuscript-style figures from the released ICSE 2027 sample.

The full paper figures are generated from the full dataset. This script uses
the released 10% PR-level sample and reproduces the same figure families,
filenames, axes, and visual grammar on the sampled data.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch


PAPER_TITLE = "Characterizing Human-Agent Dynamics in Agentic Pull Requests"
START_DATE = "2025-04-01"
END_DATE = "2026-01-31"
AGENTS_IN_SCOPE = {
    "amazon",
    "claude",
    "codegen",
    "codex",
    "copilot",
    "cosine",
    "cursor",
    "devin",
    "jules",
    "junie",
    "junieap",
    "openhands",
    "openhandsb",
    "tembo",
}

COLORS = {
    "dark": "#1f2937",
    "gray": "#6b7280",
    "dark_gray": "#4b5563",
    "light_gray": "#d9dde3",
    "pale_gray": "#eef1f4",
    "blue": "#3b82c4",
    "deep_blue": "#1f5f99",
    "light_blue": "#b9d6ec",
    "green": "#4c956c",
    "dark_green": "#2f6f4e",
    "orange": "#e6a23c",
    "red": "#c95f52",
}
INTENT_ORDER = [
    "Non-actionable",
    "Functional",
    "Maintainability",
    "Documentation",
    "Discussion",
    "Process",
    "Validation",
]
INTENT_SHORT = {
    "Non-actionable": "Non-action.",
    "Functional": "Functional",
    "Maintainability": "Maint.",
    "Documentation": "Docs",
    "Discussion": "Discussion",
    "Process": "Process",
    "Validation": "Validation",
}
INTENT_COLOR = {
    "Non-actionable": COLORS["light_gray"],
    "Functional": COLORS["red"],
    "Maintainability": COLORS["green"],
    "Documentation": COLORS["light_blue"],
    "Discussion": COLORS["gray"],
    "Process": COLORS["blue"],
    "Validation": COLORS["orange"],
}
PATHWAY_ORDER = [
    "no_visible_feedback",
    "feedback_without_code_response",
    "single_feedback_response_cycle",
    "multi_cycle_feedback_response",
    "late_feedback_after_final_code_change",
]
PATHWAY_LABEL = {
    "no_visible_feedback": "No feedback",
    "feedback_without_code_response": "No response",
    "single_feedback_response_cycle": "1 cycle",
    "multi_cycle_feedback_response": "Multi-cycle",
    "late_feedback_after_final_code_change": "Late final",
}
SCORE_COL = "activity_score"
PAPER_SCORE_COL = "variant_50_tenure_50_activity_score"
PAPER_INTENT_LABELS = {
    "false_positive_or_non_actionable": "Non-actionable",
    "functional_concern": "Functional",
    "refactoring_or_maintainability": "Maintainability",
    "documentation_or_explanation": "Documentation",
    "discussion_or_clarification": "Discussion",
    "process_or_integration": "Process",
    "validation_or_evidence_request": "Validation",
}
AI_REVIEW_AGENT_PATTERNS = [
    r"coderabbit",
    r"gemini[-_]?code[-_]?assist",
    r"(^|[-_])korbit($|[-_])",
    r"(^|[-_])qodo($|[-_])",
    r"codium",
    r"greptile",
    r"ellipsis[-_]?dev",
    r"cubic[-_]?dev[-_]?ai",
    r"pullpal[-_]?ai",
    r"codeant[-_]?ai",
    r"hyperlint[-_]?ai",
    r"codara[-_]?ai",
    r"gitauto[-_]?ai",
    r"(^|[-_])sourcery($|[-_])",
    r"amazon[-_]?q(?:[-_]?developer)?",
    r"chatgpt[-_]?codex",
    r"copilot[-_]?pull[-_]?request[-_]?reviewer",
    r"(^|[-_])(?:mcp)?claude(?:[-_]?ai)?($|[-_])",
    r"(^|[-_])windsurf(?:[-_]?bot)?($|[-_])",
    r"(^|[-_])devin(?:[-_]?ai(?:[-_]?integration)?)?($|[-_])",
    r"(^|[-_])openhands($|[-_])",
    r"google[-_]?labs[-_]?jules|(^|[-_])jules($|[-_])",
    r"(^|[-_])sweep(?:[-_]?ai)?($|[-_])",
    r"mateacademy[-_]?ai",
    r"(^|[-_])codegen(?:[-_]?sh)?($|[-_])",
]
HUMAN_TYPED_AI_ALLOWLIST = {
    "coderabbit",
    "gemini",
    "korbit",
    "qodo",
    "codium",
    "greptile",
    "ellipsis",
    "cubic",
    "pullpal",
    "codeant",
    "hyperlint",
    "codara",
    "gitauto",
    "sourcery",
    "mateacademy",
}
AUTOMATION_BOT_PATTERNS = [
    r"\[bot\]|(^|[-_])bot($|[-_])|bot$",
    r"(^|[-_])robot($|[-_])|robot$",
    r"codecov",
    r"dependabot",
    r"renovate",
    r"github[-_]?actions",
    r"continuous[-_]?integration|(^|[-_])ci($|[-_])",
    r"pipeline",
    r"mergify|mergeable|merge[-_]?queue",
    r"pkg[-_]?pr[-_]?new",
    r"coveralls|coverage[-_]?bot",
    r"sonar|snyk|deepsource",
    r"vercel|netlify|cloudflare[-_]?pages",
    r"(^|[-_])cla($|[-_])|cla[-_]?assistant",
    r"(^|[-_])bors($|[-_])",
]
FIGURE_MAP = [
    {
        "latex_label": "fig:rq1_human_activity",
        "figure_file": "fig_rq1_activity_score_strip.pdf; fig_rq1_human_triager_quartiles.pdf",
        "source_csv": "fig_rq1_human_activity_source.csv; sample_activity_scores.csv",
        "artifact_note": "Two manuscript panels in one LaTeX figure environment.",
    },
    {
        "latex_label": "fig:rq2_feedback_body_syntax",
        "figure_file": "fig_rq2_feedback_body_syntax.pdf",
        "source_csv": "fig_rq2_feedback_body_syntax_source.csv",
        "artifact_note": "Manuscript figure generated from the released sample.",
    },
    {
        "latex_label": "fig:rq2_feedback_actor_intent",
        "figure_file": "fig_rq2_feedback_actor_intent.pdf",
        "source_csv": "fig_rq2_feedback_actor_intent_source.csv",
        "artifact_note": "Manuscript figure generated from the released sample.",
    },
    {
        "latex_label": "fig:results_actor_intent_cycle_latency",
        "figure_file": "fig_results_actor_intent_cycle_latency.pdf",
        "source_csv": "fig_results_actor_intent_cycle_latency_source.csv",
        "artifact_note": "Manuscript figure generated from the released sample.",
    },
    {
        "latex_label": "fig:results_post_feedback_revision",
        "figure_file": "fig_results_post_feedback_revision.pdf",
        "source_csv": "fig_results_post_feedback_revision_source.csv",
        "artifact_note": "Manuscript figure generated from the released sample.",
    },
    {
        "latex_label": "fig:results_integration_by_pathway",
        "figure_file": "fig_results_integration_by_pathway.pdf",
        "source_csv": "fig_results_integration_by_pathway_source.csv",
        "artifact_note": "Manuscript figure generated from the released sample.",
    },
    {
        "latex_label": "fig:results_rq4_endpoint_delta",
        "figure_file": "fig_results_rq4_endpoint_delta.pdf",
        "source_csv": "fig_results_rq4_endpoint_delta_source.csv",
        "artifact_note": "Manuscript figure generated from the released sample.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 7.6,
            "axes.titlesize": 8.4,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7.2, pad=1)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, figure_dir: Path, stem: str, *, pad_inches: float = 0.01) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=pad_inches)
    fig.savefig(figure_dir / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)


def fmt_n(value: int | float) -> str:
    return f"{int(round(value)):,}"


def fmt_short(value: int | float) -> str:
    value = int(round(value))
    return f"{value / 1000:.1f}k" if value >= 1000 else str(value)


def pct(part: float, whole: float) -> float:
    return part / whole * 100 if whole else 0.0


def pr_key_from_url(url: pd.Series) -> pd.Series:
    text = url.fillna("").astype(str)
    parts = text.str.extract(r"github\.com/[^/]+/([^/]+)/pull/(\d+)", expand=True)
    keys = parts[0].fillna("") + "#" + parts[1].fillna("")
    return keys.where(parts[0].notna(), "")


def extract_login(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.lower()
    login = text.str.extract(r"'login'\s*:\s*'([^']*)'", expand=False)
    login = login.fillna(text.str.extract(r'"login"\s*:\s*"([^"]*)"', expand=False))
    login = login.fillna(text.str.extract(r"login\s*[:=]\s*([a-z0-9_.\-\[\]]+)", expand=False))
    return login.fillna(text).str.strip().str.lower()


def classify_actor(login: pd.Series, author_raw: pd.Series | None = None) -> pd.Series:
    login = login.fillna("").astype(str).str.lower()
    raw = author_raw.fillna("").astype(str).str.lower() if author_raw is not None else login
    agent_pat = (
        r"copilot|codex|cursor|devin|jules|junie|claude|openhands|codegen|"
        r"amazon-q|q-developer|cosine|tembo|swe-agent"
    )
    bot_pat = (
        r"\[bot\]|bot$|dependabot|renovate|github-actions|vercel|netlify|"
        r"cloudflare|codecov|sonar|merge.queue|pre-commit|cla|typename': 'bot|typename\": \"bot"
    )
    out = pd.Series("Human", index=login.index, dtype="object")
    out = out.mask(login.str.contains(bot_pat, regex=True) | raw.str.contains(bot_pat, regex=True), "Bot")
    out = out.mask(login.str.contains(agent_pat, regex=True) | raw.str.contains(agent_pat, regex=True), "Agent")
    return out


def body_features(body: pd.Series) -> pd.DataFrame:
    text = body.fillna("").astype(str)
    lower = text.str.lower()
    line_count = text.str.count(r"\n") + 1
    word_count = text.str.findall(r"[A-Za-z0-9_]+").map(len)
    return pd.DataFrame(
        {
            "nonempty_body": text.str.strip().ne(""),
            "word_count": word_count,
            "line_count": line_count,
            "has_code_block": text.str.contains(r"```|`[^`]+`", regex=True),
            "has_markdown_table": text.str.contains(r"\|.+\|", regex=True),
            "has_bullets": text.str.contains(r"(?m)^\s*[-*+]\s+", regex=True),
            "is_boilerplate_or_status_like": lower.str.contains(
                r"deploy|preview|workflow|ci|build|check|coverage|signed-off|cla|assigned|label|release|generated"
            ),
            "is_summary_like": lower.str.contains(r"summary|overview|analysis|findings|changes|review"),
        }
    )


def classify_intent(body: pd.Series) -> pd.Series:
    text = body.fillna("").astype(str).str.lower()
    out = pd.Series("Non-actionable", index=body.index, dtype="object")
    out = out.mask(text.str.contains(r"\b(?:test|verify|validate|repro|evidence|confirm|prove)\b", regex=True), "Validation")
    out = out.mask(text.str.contains(r"\b(?:doc|readme|comment|explain|description)\b", regex=True), "Documentation")
    out = out.mask(text.str.contains(r"\b(?:refactor|cleanup|style|maintain|simplify|rename|dedup)\b", regex=True), "Maintainability")
    out = out.mask(text.str.contains(r"\b(?:bug|fix|fail|error|incorrect|broken|regression|logic)\b", regex=True), "Functional")
    out = out.mask(text.str.contains(r"\b(?:merge|ci|workflow|release|deploy|approve|lgtm|ready|conflict)\b", regex=True), "Process")
    out = out.mask(text.str.contains(r"\?|discuss|question|clarify|why\b|how\b", regex=True), "Discussion")
    return out


def data_derived_dir(data_dir: Path) -> Path:
    return data_dir.parent / "derived"


def derived_inputs_available(data_dir: Path) -> bool:
    derived = data_derived_dir(data_dir)
    return all(
        (derived / name).exists()
        for name in [
            "engagement_feedback_events_filtered.csv",
            "comment_feedback_intent_labels.csv",
            "seniority_scores.csv",
        ]
    )


def _matches_any(account: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, account) for pattern in patterns)


def refine_feedback_actor(actor_account: object, feedback_actor_type: object) -> str:
    account = "" if pd.isna(actor_account) else str(actor_account).strip().lower()
    original = "" if pd.isna(feedback_actor_type) else str(feedback_actor_type).strip().lower()
    if not account or original == "unknown":
        return "unknown"
    ai_match = _matches_any(account, AI_REVIEW_AGENT_PATTERNS)
    if original == "human_account":
        ai_match = ai_match and any(token in account for token in HUMAN_TYPED_AI_ALLOWLIST)
    if ai_match:
        return "ai_review_agent"
    if _matches_any(account, AUTOMATION_BOT_PATTERNS):
        return "automation_bot"
    if original == "agent_bot_account":
        return "unresolved_agent_or_bot"
    if original == "human_account":
        return "human_account"
    return "unknown"


def add_refined_feedback_actor(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["refined_feedback_actor_class"] = [
        refine_feedback_actor(account, actor_type)
        for account, actor_type in zip(events["actor_account"], events["feedback_actor_type"], strict=False)
    ]
    return events


def load_paper_scores(data_dir: Path) -> pd.DataFrame:
    scores = pd.read_csv(data_derived_dir(data_dir) / "seniority_scores.csv", low_memory=False)
    scores["login"] = scores["user_login_norm"].fillna("").astype(str).str.strip().str.lower()
    scores[SCORE_COL] = pd.to_numeric(scores[PAPER_SCORE_COL], errors="coerce")
    scoped = scores.loc[
        scores["included_in_agent_scope_seniority"].eq(True) & scores[SCORE_COL].notna()
    ].copy()
    q1, q2, q3 = scoped[SCORE_COL].quantile([0.25, 0.50, 0.75]).to_numpy()

    def quartile(value: float) -> str:
        if value <= q1:
            return "Q1"
        if value <= q2:
            return "Q2"
        if value <= q3:
            return "Q3"
        return "Q4"

    scores["quartile"] = scores[SCORE_COL].map(lambda value: quartile(value) if pd.notna(value) else np.nan)
    return scores[["login", SCORE_COL, "quartile", "included_in_agent_scope_seniority"]]


def load_derived_retained_events(data_dir: Path) -> pd.DataFrame:
    derived = data_derived_dir(data_dir)
    labels = pd.read_csv(
        derived / "comment_feedback_intent_labels.csv",
        usecols=[
            "event_id",
            "source_event_file",
            "event_type",
            "pr_author_type",
            "primary_label",
            "is_boilerplate_or_status",
            "is_multipoint_or_summary_like",
        ],
        low_memory=False,
    )
    filtered = pd.read_csv(
        derived / "engagement_feedback_events_filtered.csv",
        usecols=[
            "pr_key",
            "event_id",
            "source_event_file",
            "event_type",
            "event_time",
            "pr_author_type",
            "actor_account",
            "feedback_actor_type",
        ],
        low_memory=False,
    )
    filtered = add_refined_feedback_actor(filtered)
    filtered = filtered[
        [
            "pr_key",
            "event_id",
            "source_event_file",
            "event_type",
            "event_time",
            "pr_author_type",
            "actor_account",
            "refined_feedback_actor_class",
        ]
    ].drop_duplicates(["event_id", "source_event_file"])
    events = labels.merge(
        filtered,
        on=["event_id", "source_event_file"],
        how="inner",
        suffixes=("_label", ""),
    )
    events["event_type"] = events["event_type"].fillna(events["event_type_label"]).astype(str).str.lower()
    events = events.loc[events["pr_author_type"].eq("agent_authored")].copy()
    events = events.loc[
        events["refined_feedback_actor_class"].isin(["human_account", "ai_review_agent", "automation_bot"])
    ].copy()
    events["intent"] = events["primary_label"].map(PAPER_INTENT_LABELS)
    events = events.loc[events["intent"].notna()].copy()
    events["actor_login"] = events["actor_account"].fillna("").astype(str).str.strip().str.lower()
    events["actor_class"] = events["refined_feedback_actor_class"].map(
        {
            "human_account": "Human",
            "ai_review_agent": "Agent",
            "automation_bot": "Bot",
        }
    )
    events["event_time"] = pd.to_datetime(events["event_time"], utc=True, errors="coerce")
    return events.sort_values(["pr_key", "event_time", "event_type", "event_id"], kind="mergesort").reset_index(drop=True)


def attach_raw_bodies(data_dir: Path, events: pd.DataFrame) -> pd.DataFrame:
    body_frames = []
    for filename in ["comments.csv", "reviews.csv"]:
        path = data_dir / filename
        if not path.exists():
            continue
        body_frames.append(
            pd.read_csv(path, usecols=["id", "body"], low_memory=False).assign(source_event_file=filename)
        )
    bodies = pd.concat(body_frames, ignore_index=True).rename(columns={"id": "event_id", "body": "body"})
    return events.merge(bodies, on=["event_id", "source_event_file"], how="left")


def build_first_human_from_derived(events: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    first = (
        events.sort_values(["pr_key", "event_time", "event_type", "event_id"], kind="mergesort")
        .drop_duplicates("pr_key", keep="first")
        .copy()
    )
    first_human = first.loc[first["actor_class"].eq("Human")].copy()
    first_human = first_human.merge(
        scores.loc[scores["included_in_agent_scope_seniority"].eq(True), ["login", SCORE_COL, "quartile"]],
        left_on="actor_login",
        right_on="login",
        how="inner",
    )
    return first_human


def prepare_derived_body_events(data_dir: Path, events: pd.DataFrame) -> pd.DataFrame:
    events = attach_raw_bodies(data_dir, events)
    features = body_features(events["body"])
    events = pd.concat([events, features], axis=1)
    events["analysis_body_available"] = events["nonempty_body"]
    events["is_boilerplate_or_status_like"] = events["is_boilerplate_or_status"].fillna(False).astype(bool)
    events["is_summary_like"] = events["is_multipoint_or_summary_like"].fillna(False).astype(bool)
    return events


def load_agent_prs(data_dir: Path) -> pd.DataFrame:
    prs = pd.read_csv(data_dir / "prs.csv", low_memory=False)
    prs["created_at_ts"] = pd.to_datetime(prs["created_at"], utc=True, errors="coerce")
    prs["agent_norm"] = prs["agent"].fillna("").astype(str).str.strip().str.lower()
    prs["pr_key"] = pr_key_from_url(prs["url"])
    start = pd.Timestamp(START_DATE, tz="UTC")
    end = pd.Timestamp(END_DATE, tz="UTC") + pd.Timedelta(days=1)
    scoped = prs.loc[
        prs["created_at_ts"].ge(start)
        & prs["created_at_ts"].lt(end)
        & prs["agent_norm"].isin(AGENTS_IN_SCOPE)
        & prs["pr_key"].ne("")
    ].copy()
    scoped = scoped.sort_values(["pr_key", "created_at_ts"], kind="mergesort")
    scoped = scoped.drop_duplicates("pr_key", keep="first")
    scoped["merged"] = pd.to_datetime(scoped["merged_at"], utc=True, errors="coerce").notna()
    scoped["closed"] = pd.to_datetime(scoped["closed_at"], utc=True, errors="coerce").notna()
    scoped["endpoint"] = np.where(
        scoped["merged"],
        "Merged",
        np.where(scoped["closed"], "Closed", "Open @ Observation"),
    )
    return scoped


def load_feedback_events(data_dir: Path, agent_prs: pd.DataFrame) -> pd.DataFrame:
    keep_prs = set(agent_prs["id"].astype(str))
    parts: list[pd.DataFrame] = []
    for filename, event_type in [("comments.csv", "comment"), ("reviews.csv", "review")]:
        df = pd.read_csv(data_dir / filename, low_memory=False)
        df = df.loc[df["pr_id"].astype(str).isin(keep_prs)].copy()
        df["event_type"] = event_type
        df["event_time"] = pd.to_datetime(
            df.get("created_at", df.get("submitted_at")), utc=True, errors="coerce"
        )
        df["actor_login"] = extract_login(df["author"])
        df["actor_class"] = classify_actor(df["actor_login"], df["author"])
        features = body_features(df["body"])
        df = pd.concat([df, features], axis=1)
        df["intent"] = classify_intent(df["body"])
        parts.append(
            df[
                [
                    "id",
                    "pr_id",
                    "event_type",
                    "event_time",
                    "body",
                    "actor_login",
                    "actor_class",
                    "intent",
                    "word_count",
                    "line_count",
                    "has_code_block",
                    "has_markdown_table",
                    "has_bullets",
                    "is_boilerplate_or_status_like",
                    "is_summary_like",
                    "nonempty_body",
                ]
            ].rename(columns={"id": "event_id"})
        )
    events = pd.concat(parts, ignore_index=True)
    events = events.loc[events["event_time"].notna()].copy()
    events = events.merge(agent_prs[["id", "pr_key"]], left_on="pr_id", right_on="id", how="left")
    events = events.loc[events["pr_key"].notna()].copy()
    # Retained-feedback approximation for the sample release.
    events = events.loc[events["nonempty_body"]].copy()
    return events.sort_values(["pr_key", "event_time"], kind="mergesort").reset_index(drop=True)


def load_commits(data_dir: Path, agent_prs: pd.DataFrame) -> pd.DataFrame:
    keep_prs = set(agent_prs["id"].astype(str))
    commits = pd.read_csv(data_dir / "commits.csv", low_memory=False)
    commits = commits.loc[commits["pr_id"].astype(str).isin(keep_prs)].copy()
    commits["commit_time"] = pd.to_datetime(commits["committed_date"], utc=True, errors="coerce")
    commits["changed_lines"] = pd.to_numeric(commits["additions"], errors="coerce").fillna(0) + pd.to_numeric(
        commits["deletions"], errors="coerce"
    ).fillna(0)
    commits = commits.merge(agent_prs[["id", "pr_key"]], left_on="pr_id", right_on="id", how="left")
    return commits.loc[commits["commit_time"].notna() & commits["pr_key"].notna()].copy()


def attach_next_commit(events: pd.DataFrame, commits: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["next_commit_time"] = pd.Series(pd.NaT, index=events.index, dtype="datetime64[ns, UTC]")
    events["latency_minutes"] = np.nan
    commit_map = {
        pr_key: part["commit_time"].sort_values().to_numpy(dtype="datetime64[ns]")
        for pr_key, part in commits.groupby("pr_key", sort=False)
    }
    for pr_key, idx in events.groupby("pr_key", sort=False).groups.items():
        times = commit_map.get(pr_key)
        if times is None or len(times) == 0:
            continue
        feedback_times = events.loc[idx, "event_time"].to_numpy(dtype="datetime64[ns]")
        pos = np.searchsorted(times, feedback_times, side="left")
        valid = pos < len(times)
        valid_index = np.asarray(list(idx))[valid]
        next_times = times[pos[valid]]
        events.loc[valid_index, "next_commit_time"] = pd.to_datetime(next_times, utc=True)
        delta = (next_times - feedback_times[valid]).astype("timedelta64[s]").astype(float) / 60.0
        events.loc[valid_index, "latency_minutes"] = delta
    return events


def build_pr_metrics(agent_prs: pd.DataFrame, events: pd.DataFrame, commits: pd.DataFrame) -> pd.DataFrame:
    pr = agent_prs[["id", "pr_key", "endpoint", "merged"]].copy()
    feedback = events.groupby("pr_key").agg(
        feedback_events=("event_id", "count"),
        first_feedback_time=("event_time", "min"),
        final_feedback_time=("event_time", "max"),
        first_actor=("actor_class", "first"),
        first_intent=("intent", "first"),
    )
    cycles = events.loc[events["latency_minutes"].notna()].groupby("pr_key").agg(
        response_cycles=("event_id", "count"),
        median_latency_minutes=("latency_minutes", "median"),
    )
    commit_summary = commits.groupby("pr_key").agg(final_commit_time=("commit_time", "max"))
    pr = pr.join(feedback, on="pr_key").join(cycles, on="pr_key").join(commit_summary, on="pr_key")
    pr["feedback_events"] = pr["feedback_events"].fillna(0).astype(int)
    pr["response_cycles"] = pr["response_cycles"].fillna(0).astype(int)

    first_feedback = pr.set_index("pr_key")["first_feedback_time"].dropna().to_dict()
    post_lines = []
    for pr_key, part in commits.groupby("pr_key", sort=False):
        first_time = first_feedback.get(pr_key)
        if pd.isna(first_time) or first_time is None:
            continue
        post_lines.append(
            {
                "pr_key": pr_key,
                "post_feedback_changed_lines": float(part.loc[part["commit_time"].ge(first_time), "changed_lines"].sum()),
            }
        )
    post = pd.DataFrame(post_lines)
    pr = pr.merge(post, on="pr_key", how="left")
    pr["post_feedback_changed_lines"] = pr["post_feedback_changed_lines"].fillna(0.0)

    pr["pathway"] = "no_visible_feedback"
    pr.loc[(pr["feedback_events"] > 0) & (pr["response_cycles"] == 0), "pathway"] = "feedback_without_code_response"
    pr.loc[pr["response_cycles"] == 1, "pathway"] = "single_feedback_response_cycle"
    pr.loc[pr["response_cycles"] >= 2, "pathway"] = "multi_cycle_feedback_response"
    late = (
        (pr["response_cycles"] > 0)
        & pr["final_feedback_time"].notna()
        & pr["final_commit_time"].notna()
        & pr["final_feedback_time"].gt(pr["final_commit_time"])
    )
    pr.loc[late, "pathway"] = "late_feedback_after_final_code_change"
    return pr


def build_activity_scores(data_dir: Path) -> pd.DataFrame:
    users = pd.read_csv(data_dir / "users.csv", low_memory=False)
    users["login"] = users["user_login"].fillna("").astype(str).str.strip().str.lower()

    def count_field(value: object, field: str) -> float:
        match = re.search(rf"'{field}'\s*:\s*([0-9.]+)", str(value))
        if not match:
            match = re.search(rf'"{field}"\s*:\s*([0-9.]+)', str(value))
        return float(match.group(1)) if match else 0.0

    for field in ["commits", "issues", "pull_requests", "reviews", "total"]:
        users[field] = users["contribution_counts"].map(lambda value, f=field: count_field(value, f))
    users["created_ts"] = pd.to_datetime(users["account_created_at"], utc=True, errors="coerce")
    ref = pd.Timestamp("2026-02-25", tz="UTC")
    users["tenure_days"] = (ref - users["created_ts"]).dt.days.clip(lower=0)
    contrib = np.log1p(users["commits"] + users["issues"] + users["pull_requests"] + users["reviews"])
    tenure = np.log1p(users["tenure_days"].fillna(0))
    def norm(series: pd.Series) -> pd.Series:
        lo, hi = series.quantile([0.05, 0.95])
        clipped = series.clip(lo, hi)
        return (clipped - clipped.min()) / max(clipped.max() - clipped.min(), 1e-9)
    users[SCORE_COL] = 0.5 * norm(tenure) + 0.5 * norm(contrib)
    users = users.loc[users["login"].ne("")].drop_duplicates("login", keep="first")
    q1, q2, q3 = users[SCORE_COL].quantile([0.25, 0.50, 0.75]).to_numpy()
    users["quartile"] = pd.cut(
        users[SCORE_COL],
        bins=[-np.inf, q1, q2, q3, np.inf],
        labels=["Q1", "Q2", "Q3", "Q4"],
        include_lowest=True,
    ).astype(str)
    return users[["login", SCORE_COL, "quartile", "tenure_days", "commits", "issues", "pull_requests", "reviews"]]


def draw_donut(
    ax: plt.Axes,
    values: list[int],
    labels: list[str],
    colors: list[str],
    center: str,
    *,
    show_wedge_labels: bool = True,
) -> None:
    total = sum(values)
    wedges, _ = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.34, "edgecolor": "white", "linewidth": 0.6},
    )
    ax.text(0, 0, center, ha="center", va="center", fontsize=7.0, color=COLORS["dark"])
    for wedge, value, label in zip(wedges, values, labels, strict=False):
        if not show_wedge_labels or total == 0 or value / total * 100 < 8:
            continue
        angle = (wedge.theta1 + wedge.theta2) / 2
        x = 1.04 * np.cos(np.deg2rad(angle))
        y = 1.04 * np.sin(np.deg2rad(angle))
        ax.text(x, y, label, ha="center", va="center", fontsize=6.2, color=COLORS["dark"])
    ax.set_aspect("equal")
    ax.axis("off")


def draw_interval_rows(
    fig: plt.Figure,
    ax: plt.Axes,
    rows: list[tuple[str, np.ndarray, str]],
    *,
    xscale: str,
    xlim: tuple[float, float],
    xlabel: str,
    xticks: list[float],
    xticklabels: list[str],
) -> None:
    positions = np.arange(len(rows))
    for yi, (_, values, color) in zip(positions, rows, strict=False):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        p10, q1, median, q3, p90 = np.percentile(values, [10, 25, 50, 75, 90])
        ax.hlines(yi, p10, p90, color=color, linewidth=0.75, alpha=0.95)
        ax.hlines(yi, q1, q3, color=color, linewidth=5.2, alpha=0.70)
        ax.vlines(median, yi - 0.18, yi + 0.18, color="white", linewidth=1.0)
        ax.text(1.012, yi, f"n={fmt_short(len(values))}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=6.2, color=COLORS["dark_gray"], clip_on=False)
        label_x = xlim[1] * 0.86 if xscale == "log" else xlim[1] * 0.80
        ax.text(label_x, yi, f"{median:.1f}" if median < 100 else f"{median:.0f}",
                ha="right", va="center", fontsize=6.2, color=COLORS["dark"])
    ax.set_yticks(positions, [row[0] for row in rows])
    ax.invert_yaxis()
    ax.set_xscale(xscale)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks, xticklabels)
    ax.set_xlabel(xlabel, fontsize=7.6, labelpad=1)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.5)
    style_ax(ax)


def fig_rq1_activity(scores: pd.DataFrame, first_human: pd.DataFrame, figure_dir: Path, table_dir: Path) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(table_dir / "sample_activity_scores.csv", index=False)
    values = scores[SCORE_COL].dropna().to_numpy()
    q1, q2, q3 = np.quantile(values, [0.25, 0.50, 0.75])
    fig, ax = plt.subplots(figsize=(3.43, 1.05))
    ax.hist(values, bins=np.linspace(0, 1, 28), color=COLORS["light_blue"], edgecolor=COLORS["dark"], linewidth=0.35)
    for q, color in [(q1, COLORS["orange"]), (q2, COLORS["gray"]), (q3, COLORS["green"])]:
        ax.axvline(q, color=color, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Activity score", fontsize=7.6)
    ax.set_ylabel("Accounts", fontsize=7.6)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.5)
    style_ax(ax)
    save(fig, figure_dir, "fig_rq1_activity_score_strip")

    counts = first_human["quartile"].value_counts().reindex(["Q1", "Q2", "Q3", "Q4"]).fillna(0)
    total = counts.sum()
    rq1_source = [
        {"panel": "activity_score_strip", "metric": "accounts", "level": "all", "value": len(scores)},
        {"panel": "activity_score_strip", "metric": "q1_threshold", "level": "all", "value": q1},
        {"panel": "activity_score_strip", "metric": "median_threshold", "level": "all", "value": q2},
        {"panel": "activity_score_strip", "metric": "q3_threshold", "level": "all", "value": q3},
        {"panel": "human_triager_quartiles", "metric": "scoreable_first_human_triagers", "level": "all", "value": total},
    ]
    rq1_source.extend(
        {
            "panel": "human_triager_quartiles",
            "metric": "quartile_share_pct",
            "level": quartile,
            "value": pct(counts[quartile], total),
        }
        for quartile in ["Q1", "Q2", "Q3", "Q4"]
    )
    pd.DataFrame(rq1_source).to_csv(table_dir / "fig_rq1_human_activity_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.43, 0.82))
    left = 0.0
    colors = ["#d7e9f7", "#bfcad4", "#8fae9a", "#4c956c"]
    for label, color in zip(["Q1", "Q2", "Q3", "Q4"], colors, strict=False):
        value = pct(counts[label], total)
        ax.barh([0], [value], left=left, color=color, edgecolor="white", height=0.52)
        if value >= 7:
            ax.text(left + value / 2, 0, f"{value:.1f}%", ha="center", va="center", fontsize=7.0, color=COLORS["dark"])
        left += value
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100], ["0", "50", "100%"])
    ax.set_xlabel(f"Scoreable first-human triagers, n={fmt_n(total)}", fontsize=7.6)
    ax.legend([Patch(facecolor=c) for c in colors], ["Q1", "Q2", "Q3", "Q4"], ncol=4, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, 1.35), fontsize=6.8)
    style_ax(ax)
    save(fig, figure_dir, "fig_rq1_human_triager_quartiles")


def fig_rq2_body_syntax(events: pd.DataFrame, figure_dir: Path, table_dir: Path) -> None:
    groups = [
        ("comment", "Human", "Comment Human", COLORS["green"]),
        ("comment", "Agent", "Comment Agent", COLORS["blue"]),
        ("comment", "Bot", "Comment Bot", COLORS["orange"]),
        ("review", "Human", "Review Human", COLORS["green"]),
        ("review", "Agent", "Review Agent", COLORS["blue"]),
        ("review", "Bot", "Review Bot", COLORS["orange"]),
    ]
    features = [
        ("is_boilerplate_or_status_like", "Status"),
        ("has_code_block", "Code\nblock"),
        ("has_markdown_table", "Table"),
        ("has_bullets", "Bullets"),
        ("is_summary_like", "Summary\nlike"),
    ]
    matrix, words, lines, labels = [], [], [], []
    source_rows = []
    for event_type, actor, label, _ in groups:
        part = events.loc[events["event_type"].eq(event_type) & events["actor_class"].eq(actor)]
        labels.append(label)
        body_available = part["analysis_body_available"] if "analysis_body_available" in part else part["nonempty_body"]
        feature_values = []
        for col, _ in features:
            if len(part) == 0:
                feature_values.append(0.0)
            elif col in {"has_code_block", "has_markdown_table", "has_bullets"}:
                feature_values.append(float((part[col].astype(bool) & body_available.astype(bool)).mean() * 100))
            else:
                feature_values.append(float(part[col].astype(bool).mean() * 100))
        body_part = part.loc[body_available.astype(bool)].copy()
        matrix.append(feature_values)
        words.append(np.clip(body_part["word_count"].to_numpy(dtype=float), 1, None))
        lines.append(np.clip(body_part["line_count"].to_numpy(dtype=float), 1, None))
        source_rows.append({"group": label, "events": len(part), **{name: val for (_, name), val in zip(features, matrix[-1], strict=False)}})
    pd.DataFrame(source_rows).to_csv(table_dir / "fig_rq2_feedback_body_syntax_source.csv", index=False)

    fig = plt.figure(figsize=(7.15, 2.15))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 0.33, 1.38], left=0.035, right=0.965, bottom=0.23, top=0.90, wspace=0.045)
    ax_heat = fig.add_subplot(gs[0, 0])
    arr = np.array(matrix)
    cmap = LinearSegmentedColormap.from_list("syntax", ["#fff7bc", "#fdae61", "#b2182b"])
    ax_heat.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=max(90, float(arr.max()) if arr.size else 90))
    ax_heat.set_xticks(np.arange(len(features)), [label for _, label in features])
    ax_heat.set_yticks(np.arange(len(labels)), [""] * len(labels))
    ax_heat.tick_params(axis="x", labelsize=7.0, pad=1)
    ax_heat.tick_params(axis="y", left=False)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax_heat.text(j, i, f"{arr[i, j]:.1f}%", ha="center", va="center", fontsize=6.2,
                         color="white" if arr[i, j] >= 38 else COLORS["dark"])
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_gap = fig.add_subplot(gs[0, 1])
    ax_gap.set_xlim(0, 1)
    ax_gap.set_ylim(len(labels) - 0.5, -0.5)
    for yi, label in enumerate(labels):
        ax_gap.text(0.5, yi, label, ha="center", va="center", fontsize=7.0, color=COLORS["dark"])
    ax_gap.axis("off")
    ax = fig.add_subplot(gs[0, 2])
    positions = np.arange(len(labels))
    for vals, offset, color in [(words, -0.15, COLORS["deep_blue"]), (lines, 0.15, COLORS["orange"])]:
        bp = ax.boxplot(vals, positions=positions + offset, vert=False, widths=0.20, patch_artist=True,
                        showfliers=False, whis=(10, 90))
        for patch in bp["boxes"]:
            patch.set(facecolor=color, alpha=0.72, edgecolor=color, linewidth=0.8)
        for key in ["medians", "whiskers", "caps"]:
            for artist in bp[key]:
                artist.set(color=color, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.85, 4200)
    ax.set_ylim(len(labels) - 0.62, -0.62)
    ax.set_yticks(positions, [""] * len(labels))
    ax.tick_params(axis="y", left=False)
    ax.set_xlabel("Count", fontsize=7.6)
    ax.text(0.70, 1.035, "Words", transform=ax.transAxes, color=COLORS["deep_blue"], fontsize=7.0, fontweight="bold")
    ax.text(0.84, 1.035, "Lines", transform=ax.transAxes, color=COLORS["orange"], fontsize=7.0, fontweight="bold")
    for yi, w, l in zip(positions, words, lines, strict=False):
        if len(w) and len(l):
            ax.text(1.00, yi, f"{np.median(w):.0f}/{np.median(l):.0f}", transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=6.2, clip_on=False)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.55)
    style_ax(ax)
    ax.spines["left"].set_visible(False)
    save(fig, figure_dir, "fig_rq2_feedback_body_syntax")


def fig_rq2_actor_intent(events: pd.DataFrame, first_human_scores: pd.DataFrame, figure_dir: Path, table_dir: Path) -> None:
    activity = first_human_scores[["login", "quartile"]].drop_duplicates("login")
    e = events.merge(activity, left_on="actor_login", right_on="login", how="left")
    e["actor_slice"] = np.where(e["actor_class"].eq("Human"), e["quartile"].fillna("No metadata"), e["actor_class"])
    slice_order = [
        ("Q1", "Q1", "#cfd6de"),
        ("Q2", "Q2", "#b8c8ca"),
        ("Q3", "Q3", "#8fae9a"),
        ("Q4", "Q4", COLORS["green"]),
        ("No metadata", "No metadata", COLORS["pale_gray"]),
        ("Agent", "Agent", COLORS["blue"]),
        ("Bot", "Bot", COLORS["orange"]),
    ]
    scopes = [("All", e), ("Comments", e.loc[e["event_type"].eq("comment")]), ("Reviews", e.loc[e["event_type"].eq("review")])]
    fig = plt.figure(figsize=(3.43, 3.05))
    axes = [fig.add_axes([0.020, 0.600, 0.295, 0.340]), fig.add_axes([0.352, 0.600, 0.295, 0.340]), fig.add_axes([0.684, 0.600, 0.295, 0.340])]
    for ax, (title, part) in zip(axes, scopes, strict=False):
        counts = part["actor_slice"].value_counts()
        values = [int(counts.get(key, 0)) for key, _, _ in slice_order]
        draw_donut(
            ax,
            values,
            [lab for _, lab, _ in slice_order],
            [c for _, _, c in slice_order],
            f"{title}\nn={fmt_short(len(part))}",
            show_wedge_labels=False,
        )
    key = fig.add_axes([0.025, 0.525, 0.95, 0.045])
    key.axis("off")
    human_colors = [color for label, _, color in slice_order if label.startswith("Q")]
    for idx, color in enumerate(human_colors):
        key.scatter([0.055 + idx * 0.025], [0.50], s=18, marker="s", color=color, transform=key.transAxes)
    key.text(0.180, 0.50, "Human Q1-4", transform=key.transAxes, ha="left", va="center", fontsize=6.5)
    for x, label, color in [
        (0.380, "No meta.", COLORS["pale_gray"]),
        (0.585, "Agent", COLORS["blue"]),
        (0.790, "Bot", COLORS["orange"]),
    ]:
        key.scatter([x], [0.50], s=18, marker="s", color=color, transform=key.transAxes)
        key.text(x + 0.035, 0.50, label, transform=key.transAxes, ha="left", va="center", fontsize=6.5)
    ax = fig.add_axes([0.020, 0.145, 0.295, 0.340])
    intent_counts = e["intent"].value_counts().reindex(INTENT_ORDER).fillna(0).astype(int)
    draw_donut(
        ax,
        intent_counts.tolist(),
        [INTENT_SHORT[x] for x in INTENT_ORDER],
        [INTENT_COLOR[x] for x in INTENT_ORDER],
        f"Intent all\nn={fmt_short(len(e))}",
        show_wedge_labels=False,
    )
    comments = e.loc[e["event_type"].eq("comment")]
    reviews = e.loc[e["event_type"].eq("review")]
    diff = reviews["intent"].value_counts(normalize=True).reindex(INTENT_ORDER).fillna(0) * 100
    diff -= comments["intent"].value_counts(normalize=True).reindex(INTENT_ORDER).fillna(0) * 100
    order = diff.sort_values(ascending=False).index.tolist()
    ax = fig.add_axes([0.555, 0.195, 0.405, 0.300])
    y = np.arange(len(order))
    for yi, label in zip(y, order, strict=False):
        value = float(diff[label])
        color = COLORS["green"] if value > 0 else COLORS["gray"]
        ax.hlines(yi, 0, value, color=color, linewidth=2.0)
        ax.scatter([value], [yi], s=12, color=color, zorder=3)
        ax.text(value + (0.06 if value >= 0 else -0.06), yi, f"{value:+.1f}",
                ha="left" if value >= 0 else "right", va="center", fontsize=6.2)
    ax.axvline(0, color=COLORS["dark"], linewidth=0.65)
    ax.set_yticks(y, [""] * len(order))
    lim = max(2.35, float(np.nanmax(np.abs(diff.to_numpy(dtype=float)))) + 5.0)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("pp", fontsize=7.6, labelpad=1)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.55)
    style_ax(ax)
    intent_key = fig.add_axes([0.345, 0.195, 0.190, 0.300])
    intent_key.axis("off")
    for row, label in enumerate(order):
        y_pos = 0.93 - row * (0.86 / max(len(order) - 1, 1))
        intent_key.scatter([0.05], [y_pos], s=14, marker="s", color=INTENT_COLOR[label], transform=intent_key.transAxes)
        intent_key.text(0.13, y_pos, INTENT_SHORT[label], transform=intent_key.transAxes, ha="left", va="center", fontsize=6.2)
    e.groupby(["event_type", "actor_class", "intent"]).size().reset_index(name="events").to_csv(
        table_dir / "fig_rq2_feedback_actor_intent_source.csv", index=False
    )
    save(fig, figure_dir, "fig_rq2_feedback_actor_intent")


def fig_latency(events: pd.DataFrame, figure_dir: Path, table_dir: Path) -> None:
    data = events.loc[events["latency_minutes"].notna() & events["latency_minutes"].gt(0)].copy()
    rows: list[tuple[str, np.ndarray, str]] = [
        ("Human", data.loc[data["actor_class"].eq("Human"), "latency_minutes"].to_numpy(), COLORS["dark_green"]),
        ("Agent/bot", data.loc[data["actor_class"].isin(["Agent", "Bot"]), "latency_minutes"].to_numpy(), COLORS["deep_blue"]),
    ]
    rows.extend((INTENT_SHORT[label], data.loc[data["intent"].eq(label), "latency_minutes"].to_numpy(), COLORS["blue"]) for label in INTENT_ORDER)
    rows = [(a, b, c) for a, b, c in rows if len(b)]
    fig, ax = plt.subplots(figsize=(3.43, 1.88))
    fig.subplots_adjust(left=0.315, right=0.835, bottom=0.190, top=0.960)
    draw_interval_rows(fig, ax, rows, xscale="log", xlim=(0.8, 650), xlabel="Minutes", xticks=[1, 10, 100], xticklabels=["1", "10", "100"])
    if len(rows) > 2:
        ax.axhline(1.5, color="#d1d5db", linewidth=0.55)
    data.to_csv(table_dir / "fig_results_actor_intent_cycle_latency_source.csv", index=False)
    save(fig, figure_dir, "fig_results_actor_intent_cycle_latency")


def fig_post_feedback(pr: pd.DataFrame, figure_dir: Path, table_dir: Path) -> None:
    rows = [
        ("Raw visible", pr.loc[pr["feedback_events"].gt(0), "post_feedback_changed_lines"].to_numpy(), "#6f8fae"),
        ("Cycle-bearing", pr.loc[pr["response_cycles"].gt(0), "post_feedback_changed_lines"].to_numpy(), COLORS["deep_blue"]),
    ]
    for label, mask in [
        ("1 cycle", pr["response_cycles"].eq(1)),
        ("2 cycles", pr["response_cycles"].eq(2)),
        ("3 cycles", pr["response_cycles"].eq(3)),
        ("4-5 cycles", pr["response_cycles"].between(4, 5)),
        ("6+ cycles", pr["response_cycles"].ge(6)),
    ]:
        rows.append((label, pr.loc[mask, "post_feedback_changed_lines"].to_numpy(), COLORS["deep_blue"]))
    rows = [(a, b, c) for a, b, c in rows if len(b)]
    fig, ax = plt.subplots(figsize=(3.55, 1.68))
    fig.subplots_adjust(left=0.335, right=0.835, bottom=0.205, top=0.970)
    positions = np.arange(len(rows))
    for yi, (_, values, color) in zip(positions, rows, strict=False):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        p10, q1, median, q3, p90 = np.percentile(values, [10, 25, 50, 75, 90])
        ax.hlines(yi, p10, p90, color=color, linewidth=0.75, alpha=0.95)
        ax.hlines(yi, q1, q3, color=color, linewidth=5.2, alpha=0.70)
        ax.vlines(median, yi - 0.18, yi + 0.18, color="white", linewidth=1.0)
        label_x = 82000
        label_ha = "right"
        if median >= 300:
            label_x = 2.0
            label_ha = "left"
        ax.text(
            label_x,
            yi,
            f"{median:.0f}",
            ha=label_ha,
            va="center",
            fontsize=6.2,
            color=COLORS["dark"],
            clip_on=True,
        )
        ax.text(
            1.012,
            yi,
            f"n={fmt_short(len(values))}",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=6.2,
            color=COLORS["dark_gray"],
            clip_on=False,
        )
    ax.set_yticks(positions, [label for label, _, _ in rows])
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, 100000)
    ax.set_xticks([0, 10, 100, 1000, 10000, 100000], ["0", "10", "100", "1k", "10k", "100k"])
    ax.set_xlabel("Changed lines", fontsize=7.6, labelpad=1)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.5)
    style_ax(ax)
    if len(rows) > 2:
        ax.axhline(1.5, color="#d1d5db", linewidth=0.55)
    for tick in ax.get_yticklabels():
        tick.set_rotation(30)
        tick.set_rotation_mode("anchor")
        tick.set_ha("right")
    pr.to_csv(table_dir / "fig_results_post_feedback_revision_source.csv", index=False)
    save(fig, figure_dir, "fig_results_post_feedback_revision")


def fig_integration(pr: pd.DataFrame, figure_dir: Path, table_dir: Path) -> None:
    summary = (
        pr.groupby(["pathway", "endpoint"]).size().unstack(fill_value=0).reindex(PATHWAY_ORDER).fillna(0).astype(int)
    )
    for col in ["Merged", "Closed", "Open @ Observation"]:
        if col not in summary:
            summary[col] = 0
    pct_df = summary.div(summary.sum(axis=1).replace(0, np.nan), axis=0) * 100
    fig, ax = plt.subplots(figsize=(3.43, 1.72))
    fig.subplots_adjust(left=0.290, right=0.835, bottom=0.240, top=0.900)
    y = np.arange(len(summary))
    left = np.zeros(len(summary))
    cols = [("Merged", COLORS["green"]), ("Closed", COLORS["orange"]), ("Open @ Observation", COLORS["gray"])]
    for label, color in cols:
        vals = pct_df[label].fillna(0).to_numpy()
        ax.barh(y, vals, left=left, color=color, height=0.52, edgecolor="white", linewidth=0.7)
        for i, (value, offset) in enumerate(zip(vals, left, strict=False)):
            if value >= 8:
                ax.text(offset + value / 2, i, f"{value:.1f}", ha="center", va="center", fontsize=7.0, color="white")
            elif label == "Open @ Observation" and value > 0:
                ax.text(offset + value / 2, i, f"{value:.1f}", ha="center", va="center", fontsize=6.0, color="white")
        left += vals
    ax.set_xlim(0, 108)
    ax.set_yticks(y, [PATHWAY_LABEL[x] for x in summary.index])
    for tick in ax.get_yticklabels():
        tick.set_rotation(30)
        tick.set_rotation_mode("anchor")
        tick.set_ha("right")
    ax.invert_yaxis()
    ax.set_xticks([0, 50, 100], ["0", "50", "100%"])
    for i, n in enumerate(summary.sum(axis=1)):
        ax.text(1.012, i, f"n={fmt_short(n)}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=6.2, color=COLORS["dark_gray"], clip_on=False)
    ax.set_xlabel("Percent", fontsize=7.6, labelpad=1)
    ax.legend([Patch(facecolor=c) for _, c in cols], [a for a, _ in cols], ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.23), frameon=False, fontsize=6.2)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.5)
    style_ax(ax)
    summary.to_csv(table_dir / "fig_results_integration_by_pathway_source.csv")
    save(fig, figure_dir, "fig_results_integration_by_pathway", pad_inches=0.015)


def fig_endpoint_delta(pr: pd.DataFrame, first_human: pd.DataFrame, figure_dir: Path, table_dir: Path) -> None:
    rows: list[tuple[str, str, float]] = []
    actor_subset = pr.loc[pr["first_actor"].notna()]
    actor_base = actor_subset["merged"].mean() * 100 if len(actor_subset) else 0
    for actor in ["Human", "Agent", "Bot"]:
        part = actor_subset.loc[actor_subset["first_actor"].eq(actor)]
        if len(part):
            rows.append(("Actor", actor, part["merged"].mean() * 100 - actor_base))
    intent_subset = pr.loc[pr["first_intent"].notna()]
    intent_base = intent_subset["merged"].mean() * 100 if len(intent_subset) else 0
    for intent in ["Process", "Maintainability", "Documentation", "Discussion"]:
        part = intent_subset.loc[intent_subset["first_intent"].eq(intent)]
        if len(part):
            rows.append(("Intent", INTENT_SHORT[intent], part["merged"].mean() * 100 - intent_base))
    fh = first_human.merge(pr[["pr_key", "merged"]], on="pr_key", how="left")
    quart_base = fh["merged"].mean() * 100 if len(fh) else 0
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        part = fh.loc[fh["quartile"].eq(q)]
        if len(part):
            rows.append(("Quartile", q, part["merged"].mean() * 100 - quart_base))
    labels, y_positions, prev, current_y = [], [], None, 0.0
    for group, label, _ in rows:
        if prev is not None and group != prev:
            current_y += 0.35
        labels.append(f"{group}: {label}" if group != prev else label)
        y_positions.append(current_y)
        current_y += 1.0
        prev = group
    values = np.array([row[2] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(3.43, 1.68))
    fig.subplots_adjust(left=0.355, right=0.965, bottom=0.250, top=0.965)
    y = np.array(y_positions, dtype=float)
    colors = [COLORS["green"] if v >= 0 else COLORS["gray"] for v in values]
    lim = max(10, float(np.nanmax(np.abs(values))) + 7 if len(values) else 10)
    for yi, value, color in zip(y, values, colors, strict=False):
        ax.hlines(yi, 0, value, color=color, linewidth=2.2, alpha=0.82)
        ax.scatter([value], [yi], s=17, color=color, zorder=3)
        ax.text(value + (0.35 if value >= 0 else -0.35), yi, f"{value:+.1f}",
                ha="left" if value >= 0 else "right", va="center", fontsize=6.5, color=COLORS["dark"])
    group_changes = [y[i] - 0.5 for i in range(1, len(rows)) if rows[i][0] != rows[i - 1][0]]
    for sep in group_changes:
        ax.axhline(sep, color="#d1d5db", linewidth=0.5)
    ax.axvline(0, color=COLORS["dark"], linewidth=0.75)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Percentage points", fontsize=7.6, labelpad=1)
    ax.set_xlim(-lim, lim)
    if lim <= 12:
        ax.set_xticks([-10, -5, 0, 5, 10], ["-10", "-5", "0", "+5", "+10"])
    else:
        tick = 10
        bound = int(np.ceil(lim / tick) * tick)
        ticks = np.arange(-bound, bound + tick, tick)
        ax.set_xticks(ticks, [f"+{int(t)}" if t > 0 else str(int(t)) for t in ticks])
    for tick_label in ax.get_yticklabels():
        tick_label.set_rotation(30)
        tick_label.set_rotation_mode("anchor")
        tick_label.set_ha("right")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.5)
    style_ax(ax)
    pd.DataFrame(rows, columns=["construct", "level", "merged_rate_delta_pp"]).to_csv(
        table_dir / "fig_results_rq4_endpoint_delta_source.csv", index=False
    )
    save(fig, figure_dir, "fig_results_rq4_endpoint_delta")


def main() -> int:
    args = parse_args()
    setup_style()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    agent_prs = load_agent_prs(args.data_dir)
    events = load_feedback_events(args.data_dir, agent_prs)
    commits = load_commits(args.data_dir, agent_prs)
    events = attach_next_commit(events, commits)
    pr = build_pr_metrics(agent_prs, events, commits)
    scores = build_activity_scores(args.data_dir)

    figure_events = events
    figure_scores = scores
    derived_events = None
    used_derived_inputs = derived_inputs_available(args.data_dir)
    if used_derived_inputs:
        derived_events = load_derived_retained_events(args.data_dir)
        figure_events = prepare_derived_body_events(args.data_dir, derived_events)
        paper_scores = load_paper_scores(args.data_dir)
        figure_scores = paper_scores.loc[
            paper_scores["included_in_agent_scope_seniority"].eq(True) & paper_scores[SCORE_COL].notna()
        ].copy()
        first_human = build_first_human_from_derived(derived_events, paper_scores)
    else:
        first_event = (
            events.sort_values(["pr_key", "event_time"], kind="mergesort")
            .drop_duplicates("pr_key", keep="first")
            .copy()
        )
        first_human = first_event.loc[first_event["actor_class"].eq("Human")].copy()
        first_human = first_human.merge(
            scores[["login", SCORE_COL, "quartile"]],
            left_on="actor_login",
            right_on="login",
            how="inner",
        )

    pd.DataFrame(FIGURE_MAP).to_csv(args.table_dir / "manuscript_figure_map.csv", index=False)

    fig_rq1_activity(figure_scores, first_human, args.figure_dir, args.table_dir)
    fig_rq2_body_syntax(figure_events, args.figure_dir, args.table_dir)
    fig_rq2_actor_intent(figure_events, figure_scores, args.figure_dir, args.table_dir)
    fig_latency(events, args.figure_dir, args.table_dir)
    fig_post_feedback(pr, args.figure_dir, args.table_dir)
    fig_integration(pr, args.figure_dir, args.table_dir)
    fig_endpoint_delta(pr, first_human, args.figure_dir, args.table_dir)

    manifest = {
        "paper_title": PAPER_TITLE,
        "data_dir": str(args.data_dir),
        "run_dir": str(args.run_dir),
        "figures": sorted(path.name for path in args.figure_dir.glob("*.pdf")),
        "figure_map_csv": str(args.table_dir / "manuscript_figure_map.csv"),
        "figure_map": FIGURE_MAP,
        "used_derived_inputs": used_derived_inputs,
        "agent_prs": int(len(agent_prs)),
        "retained_feedback_events_approx": int(len(figure_events)),
        "raw_visible_feedback_events": int(len(events)),
        "feedback_response_events": int(events["latency_minutes"].notna().sum()),
        "scoreable_first_human_triagers": int(len(first_human)),
    }
    (args.run_dir / "manuscript_figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
