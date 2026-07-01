"""Shared EXP-01 cohort helpers.

These helpers mirror the paper pipeline's PR-key construction and coarse
author grouping so the EXP-01 outputs can be regenerated without importing the
side-effect-heavy core pipeline module.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


PATHWAY_ORDER = [
    "No human handling",
    "Discussion only",
    "Silent edit",
    "Early takeover",
]

BOT_AUTHOR_PATTERN = re.compile(
    r"dependabot|renovate|codecov|greenkeeper|gitguardian|github-actions|\[bot\]|-bot$|bot\b",
    flags=re.IGNORECASE,
)
PR_URL_RE = re.compile(r"github\.com[:/]([^/]+/[^/#?]+)/pull/(\d+)", flags=re.IGNORECASE)
MERGE_ARTIFACT_MESSAGE_PATTERNS = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in [
        r"^\s*merge pull request\b",
        r"^\s*merge branch\b",
        r"^\s*merge remote-tracking branch\b",
        r"^\s*merge tag\b",
        r"^\s*squash(?:ed)? commit\b",
        r"^\s*squash and merge\b",
        r"^\s*rebase\b",
        r"^\s*auto-merge\b",
    ]
]


def norm_login(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().lower()


def _literal(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text[0] in "[{(":
        try:
            return ast.literal_eval(text)
        except Exception:
            return value
    return value


def extract_repo_name(value: object) -> str:
    """Match the EXP-00 artifact PR key convention for repository dicts."""
    parsed = _literal(value)
    if isinstance(parsed, dict):
        for key in ["name", "nameWithOwner", "full_name", "repository"]:
            if parsed.get(key):
                return str(parsed.get(key)).strip().lower()
        if parsed.get("url"):
            match = re.search(r"github\.com[:/]([^/]+/[^/#?]+)", str(parsed.get("url")), flags=re.IGNORECASE)
            if match:
                return match.group(1).strip().lower()
        if parsed.get("name"):
            return str(parsed.get("name")).strip().lower()
    text = "" if parsed is None else str(parsed)
    match = re.search(r"github\.com[:/]([^/]+/[^/#?]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return text.strip().lower()


def extract_logins(value: object) -> list[str]:
    """Extract one or more login-like identifiers from GitHub actor fields."""
    parsed = _literal(value)
    out: list[str] = []

    def add(candidate: object) -> None:
        text = norm_login(candidate)
        if text and text not in out:
            out.append(text)

    def walk(obj: object) -> None:
        if isinstance(obj, dict):
            for key in ["login", "name", "email"]:
                if obj.get(key):
                    add(obj.get(key))
            if obj.get("url"):
                match = re.search(r"github\.com/(?:apps/)?([^/?#]+)", str(obj.get("url")), flags=re.IGNORECASE)
                if match:
                    add(match.group(1))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item)
        elif obj is not None:
            text = str(obj)
            for pattern in [
                r"['\"]login['\"]\s*:\s*['\"]([^'\"]+)",
                r"['\"]name['\"]\s*:\s*['\"]([^'\"]+)",
                r"[a-z0-9._%-]+\+([a-z0-9-]+)@users\.noreply\.github\.com",
            ]:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    add(match.group(1))
            if not out:
                add(text)

    walk(parsed)
    return out


def first_login(value: object) -> str:
    logins = extract_logins(value)
    return logins[0] if logins else ""


def canonical_account(value: object) -> str:
    """Normalize visible account aliases for account-level comparisons."""
    account = norm_login(value)
    if account.endswith("[bot]"):
        account = account[: -len("[bot]")]
    if account.endswith("@users.noreply.github.com"):
        match = re.search(r"[a-z0-9._%-]+\+([a-z0-9-]+)(?:\[bot\])?@users\.noreply\.github\.com", account)
        if match:
            account = match.group(1)
    return account.strip()


def is_agent_bot_account(account: object, selected_agents: Iterable[str]) -> bool:
    raw = norm_login(account)
    canonical = canonical_account(raw)
    if not canonical:
        return False
    if BOT_AUTHOR_PATTERN.search(raw):
        return True
    if canonical in {"github", "github-actions", "github-actions[bot]"}:
        return True
    compact = re.sub(r"[^a-z0-9]+", "", canonical)
    return any(agent and agent in compact for agent in selected_agents)


def merge_artifact_reasons(
    *,
    sha: object,
    message_headline: object = "",
    message_body: object = "",
    commit_time: object = None,
    merged_at: object = None,
    merge_commit_sha: object = "",
    committer_account: object = "",
) -> list[str]:
    """Return conservative merge-artifact reasons for one commit row."""
    reasons: list[str] = []
    sha_text = norm_login(sha)
    merge_sha = norm_login(merge_commit_sha)
    if sha_text and merge_sha and sha_text == merge_sha:
        reasons.append("matches_pull_request_merge_commit_sha")

    headline = "" if message_headline is None or (isinstance(message_headline, float) and pd.isna(message_headline)) else str(message_headline)
    body = "" if message_body is None or (isinstance(message_body, float) and pd.isna(message_body)) else str(message_body)
    message = f"{headline}\n{body}".strip()
    if any(pattern.search(message) for pattern in MERGE_ARTIFACT_MESSAGE_PATTERNS):
        reasons.append("routine_merge_message")

    if pd.notna(commit_time) and pd.notna(merged_at):
        try:
            if pd.Timestamp(commit_time) >= pd.Timestamp(merged_at):
                reasons.append("at_or_after_merged_at")
        except Exception:
            pass

    committer = canonical_account(committer_account)
    if committer in {"github", "github-actions"} and "at_or_after_merged_at" in reasons:
        reasons.append("github_merge_button_or_queue_boundary")

    return reasons


def make_pr_key(repo_values: pd.Series, number_values: pd.Series) -> pd.Series:
    repo = repo_values.map(extract_repo_name).fillna("").astype(str)
    number = pd.to_numeric(number_values, errors="coerce").astype("Int64").astype(str).replace("<NA>", "")
    key = repo + "#" + number
    return key.mask(repo.eq("") | number.eq(""), "").astype(str)


def parse_pr_key_from_url(url_values: pd.Series) -> pd.Series:
    text = url_values.fillna("").astype(str)
    extracted = text.str.extract(PR_URL_RE, expand=True)
    if extracted.shape[1] != 2:
        return pd.Series("", index=url_values.index, dtype="object")
    return make_pr_key(extracted[0], extracted[1])


def load_bundle_table(bundle_path: Path, table_name: str) -> pd.DataFrame:
    bundle = json.loads(bundle_path.read_text())
    return pd.DataFrame(bundle.get("tables", {}).get(table_name, []))


def selected_agents_from_bundle(bundle_path: Path) -> list[str]:
    sanity = load_bundle_table(bundle_path, "baseline_sanity_checks")
    row = sanity.loc[sanity["metric"].eq("selected_agents_effective_n")].head(1)
    if row.empty:
        return []
    detail = str(row.iloc[0].get("detail", ""))
    return [norm_login(x) for x in detail.split(",") if norm_login(x)]


def pathway_label(has_discussion: bool, has_code: bool) -> str:
    if has_discussion and has_code:
        return "Early takeover"
    if has_discussion:
        return "Discussion only"
    if has_code:
        return "Silent edit"
    return "No human handling"


def summarize_pathways(flags: pd.DataFrame, denominator: int) -> pd.DataFrame:
    rows = []
    for label in PATHWAY_ORDER:
        count = int(flags["pathway"].eq(label).sum())
        rows.append(
            {
                "pathway": label,
                "visible_human_discussion": label in {"Discussion only", "Early takeover"},
                "visible_human_code_intervention": label in {"Silent edit", "Early takeover"},
                "N_prs": count,
                "share_of_prs": round(count / max(denominator, 1) * 100.0, 6),
            }
        )
    return pd.DataFrame(rows)


def is_human_actor(actor_logins: Iterable[str], pr_author_login: str, selected_agents: set[str]) -> bool:
    for actor in actor_logins:
        login = norm_login(actor)
        if not login:
            continue
        if login == norm_login(pr_author_login):
            continue
        if login in selected_agents or BOT_AUTHOR_PATTERN.search(login):
            continue
        return True
    return False
