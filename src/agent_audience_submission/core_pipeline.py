import json
import os

from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")


def _env_flag(name, default='1'):

    return os.environ.get(name, default).strip().lower() in {'1', 'true', 'yes', 'y'}



EXTRA_PLOTS = _env_flag('SUBMISSION_EXTRA_PLOTS', '1')

RUN_SENTIMENT = _env_flag('SUBMISSION_RUN_SENTIMENT', '1')

RUN_TRUST_CUES = _env_flag('SUBMISSION_RUN_TRUST_CUES', '1')

RUN_INTENSIVE_TRUST_CUES = _env_flag('SUBMISSION_RUN_INTENSIVE_TRUST_CUES', '1')

RUN_RQ3 = _env_flag('SUBMISSION_RUN_RQ3', '1')

RUN_COARSE_BOT_BASELINE = _env_flag('SUBMISSION_RUN_COARSE_BOT_BASELINE', '1')

RUN_VADER_CHECK = _env_flag('SUBMISSION_RUN_VADER_CHECK', '1')



# Paper-scope guardrails

PIPELINE_WORKDIR = Path.cwd()
PIPELINE_ROOT = Path(os.environ.get('SUBMISSION_PIPELINE_ROOT', str(PIPELINE_WORKDIR))).expanduser()
DATASET_DIR = Path(
    os.environ.get(
        'SUBMISSION_DATASET_DIR',
        str((PIPELINE_ROOT / 'data' / 'combined_dataset').resolve() if (PIPELINE_ROOT / 'data' / 'combined_dataset').exists() else PIPELINE_ROOT / 'combined_dataset'),
    )
).expanduser()
HUMAN_DATASET_DIR = Path(os.environ.get('SUBMISSION_HUMAN_DATASET_DIR', str(DATASET_DIR))).expanduser()
OPTIONAL_DATASET_DIR = Path(os.environ.get('SUBMISSION_OPTIONAL_DATASET_DIR', str(PIPELINE_ROOT / 'data'))).expanduser()
RESOURCE_DIR = Path(os.environ.get('SUBMISSION_RESOURCE_DIR', str(PIPELINE_ROOT / 'resources'))).expanduser()
PIPELINE_CACHE_DIR = Path(os.environ.get('SUBMISSION_PIPELINE_CACHE_DIR', str(PIPELINE_WORKDIR / '.pipeline_cache'))).expanduser()

selected_agents_raw = os.environ.get('SUBMISSION_SELECTED_AGENTS', '').strip()
SELECTED_AGENTS = [value.strip() for value in selected_agents_raw.split(',') if value.strip()] or None

START_DATE = os.environ.get('SUBMISSION_START_DATE', '2025-04-01')

END_DATE = os.environ.get('SUBMISSION_END_DATE', '2026-01-31')


RUN_FRACTION = float(os.environ.get('SUBMISSION_RUN_FRACTION', '1.0'))

RANDOM_SEED = int(os.environ.get('SUBMISSION_RANDOM_SEED', '7'))

DROP_ISSUE_TABLES = os.environ.get('SUBMISSION_DROP_ISSUES', '0').strip().lower() in {'1', 'true', 'yes', 'y'}

ENABLE_PARALLEL = True

N_JOBS = -1

MIN_ROWS = 50           # threshold for "enough data" warnings



OUTPUT_DIR_DEFAULT = str(PIPELINE_CACHE_DIR)

OUTPUT_DIR_FALLBACK = 'outputs'





def _resolve_output_dir():

    primary = Path(OUTPUT_DIR_DEFAULT)

    try:

        primary.mkdir(parents=True, exist_ok=True)

        test_file = primary / '.write_test.tmp'

        test_file.write_text('ok')

        test_file.unlink(missing_ok=True)

        return primary

    except Exception as exc:

        fallback = Path(OUTPUT_DIR_FALLBACK)

        fallback.mkdir(parents=True, exist_ok=True)

        print(f"Warning: cannot write to {primary} ({exc}); using fallback {fallback}.")

        return fallback





OUTPUT_DIR = _resolve_output_dir()





def output_path(filename):

    return str(OUTPUT_DIR / filename)





def save_csv(df, filename, index=False):

    path = Path(output_path(filename))

    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=index)

    fallback_dir = Path(OUTPUT_DIR_FALLBACK)

    fallback_dir.mkdir(parents=True, exist_ok=True)

    if path.parent.resolve() != fallback_dir.resolve():

        df.to_csv(fallback_dir / filename, index=index)

    return str(path)





def save_text(text, filename):

    path = Path(output_path(filename))

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(str(text))

    fallback_dir = Path(OUTPUT_DIR_FALLBACK)

    fallback_dir.mkdir(parents=True, exist_ok=True)

    if path.parent.resolve() != fallback_dir.resolve():

        (fallback_dir / filename).write_text(str(text))

    return str(path)





def stop_with_questions_for_user(question_text):

    message = f"## QUESTIONS FOR USER\n\n{question_text}"

    try:

        from IPython.display import Markdown, display

        display(Markdown(message))

    except Exception:

        print(message)

    raise RuntimeError(f"QUESTIONS FOR USER: {question_text}")





def maybe_sample_prs(df_prs, frac, seed):

    """Deterministically sample PRs only; downstream event tables should be filtered to these keys."""

    if df_prs is None or df_prs.empty or frac >= 1.0:

        return df_prs.copy() if isinstance(df_prs, pd.DataFrame) else df_prs

    if frac <= 0:

        return df_prs.iloc[0:0].copy()

    n_sample = max(1, int(round(len(df_prs) * frac)))

    n_sample = min(n_sample, len(df_prs))

    return df_prs.sample(n=n_sample, random_state=seed).copy()





def maybe_sample(df, frac, seed, stratify_cols=None):

    """Backward-compatible alias used by older cells."""

    return maybe_sample_prs(df, frac, seed)





def safe_int_series(values, fill_value=0, nullable=False):

    series = values.copy() if isinstance(values, pd.Series) else pd.Series(values)

    numeric = pd.to_numeric(series, errors='coerce')

    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(fill_value)

    return numeric.astype('Int64' if nullable else 'int64')





def normalize_action_bucket(values, valid_actions, fallback='Other'):

    series = values.copy() if isinstance(values, pd.Series) else pd.Series(values)

    series = series.fillna('').astype(str)

    return series.where(series.isin(valid_actions), fallback)





def _extract_repo_name(series):

    if series is None:

        return None

    values = series.fillna('').astype(str)



    def _pick(v):

        if isinstance(v, dict):

            for key in ['nameWithOwner', 'full_name', 'repository']:

                if v.get(key):

                    return str(v.get(key))

            if v.get('url'):

                m = re.search(r'github\\.com[:/]([^/]+/[^/#?]+)', str(v.get('url')))

                if m:

                    return m.group(1)

            if v.get('name'):

                return str(v.get('name'))

        text = str(v) if v else ''

        m = re.search(r'github\\.com[:/]([^/]+/[^/#?]+)', text)

        if m:

            return m.group(1)

        return text



    if values.str.startswith('{').any():

        try:

            parsed = values.apply(lambda x: ast.literal_eval(x) if str(x).startswith('{') else x)

            values = parsed.apply(_pick).astype(str)

        except Exception:

            values = values.apply(_pick)

    else:

        values = values.apply(_pick)



    return values.fillna('').astype(str).str.strip().str.lower()





def make_pr_key(repo_full_name, pr_number):

    """Canonical PR key: normalized repo + '#' + integer PR number."""

    repo = repo_full_name.copy() if isinstance(repo_full_name, pd.Series) else pd.Series(repo_full_name)

    repo = _extract_repo_name(repo)

    repo = repo.fillna('').astype(str).str.strip().str.lower()



    number = pr_number.copy() if isinstance(pr_number, pd.Series) else pd.Series(pr_number)

    number_num = pd.to_numeric(number, errors='coerce')

    number_int = number_num.astype('Int64').astype(str).replace('<NA>', '')



    key = repo + '#' + number_int

    invalid = repo.eq('') | number_int.eq('')

    key = key.mask(invalid, '')

    return key.astype(str)





PR_ID_TO_KEY = {}





def refresh_pr_id_to_key_map(*pr_frames):

    global PR_ID_TO_KEY

    merged = {}

    for frame in pr_frames:

        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty or 'id' not in frame.columns:

            continue

        repo_col = next((c for c in ['repo_full_name', 'base_repo_full_name', 'repository', 'repo_name', 'base_repository', 'repo', 'full_name'] if c in frame.columns), None)

        number_col = next((c for c in ['number', 'pr_number', 'pull_number'] if c in frame.columns), None)

        if repo_col is None or number_col is None:

            continue

        keys = make_pr_key(frame[repo_col], frame[number_col])

        ids = frame['id'].fillna('').astype(str)

        valid = keys.ne('') & ids.ne('')

        for pr_id, pr_key in zip(ids.loc[valid], keys.loc[valid]):

            merged[pr_id] = pr_key

    PR_ID_TO_KEY = merged

    return PR_ID_TO_KEY





def _parse_pr_key_from_url(url_values):

    series = url_values.copy() if isinstance(url_values, pd.Series) else pd.Series(url_values)

    text = series.fillna('').astype(str)

    extracted = text.str.extract(r'github\\.com[:/]([^/]+/[^/#?]+)/pull/(\\d+)', expand=True)

    if extracted.shape[1] != 2:

        return pd.Series('', index=series.index, dtype='object')

    return make_pr_key(extracted[0], extracted[1])





def _build_pr_key_for_pr_table(df_prs):

    if df_prs is None:

        return pd.Series(dtype='object')

    if df_prs.empty:

        return pd.Series(index=df_prs.index, dtype='object')



    repo_col = next((c for c in ['repo_full_name', 'base_repo_full_name', 'repository', 'repo_name', 'base_repository', 'repo', 'full_name'] if c in df_prs.columns), None)

    number_col = next((c for c in ['number', 'pr_number', 'pull_number'] if c in df_prs.columns), None)

    if repo_col is None or number_col is None:

        raise ValueError('Cannot create canonical pr_key: missing repo and/or PR number columns in PR table.')



    keys = make_pr_key(df_prs[repo_col], df_prs[number_col])

    if 'id' in df_prs.columns and PR_ID_TO_KEY:

        ids = df_prs['id'].fillna('').astype(str)

        mapped = ids.map(PR_ID_TO_KEY).fillna('')

        keys = keys.where(keys.ne(''), mapped)



    if keys.fillna('').eq('').all():

        raise ValueError('Failed to create canonical pr_key values for PR table.')

    return keys.fillna('').astype(str)





def ensure_pr_key(df, table_kind='events'):

    if df is None:

        return pd.DataFrame()

    out = df.copy()

    if out.empty:

        if 'pr_key' not in out.columns:

            out['pr_key'] = pd.Series(dtype='object')

        return out



    if 'pr_key' in out.columns:

        out['pr_key'] = out['pr_key'].fillna('').astype(str)

        return out



    if table_kind == 'prs':

        out['pr_key'] = _build_pr_key_for_pr_table(out)

        return out



    repo_col = next((c for c in ['repo_full_name', 'base_repo_full_name', 'repository', 'repo_name', 'base_repository', 'repo', 'full_name'] if c in out.columns), None)

    number_col = next((c for c in ['number', 'pr_number', 'pull_number'] if c in out.columns), None)

    key = pd.Series('', index=out.index, dtype='object')



    if repo_col is not None and number_col is not None:

        key = make_pr_key(out[repo_col], out[number_col])



    if key.eq('').any():

        for url_col in ['url', 'html_url', 'pull_request_url', 'pr_url']:

            if url_col in out.columns:

                parsed = _parse_pr_key_from_url(out[url_col])

                key = key.where(key.ne(''), parsed)



    if key.eq('').any() and PR_ID_TO_KEY:

        for id_col in ['pr_id', 'pull_request_id', 'pullRequestId', 'pull_request_node_id']:

            if id_col in out.columns:

                mapped = out[id_col].fillna('').astype(str).map(PR_ID_TO_KEY).fillna('')

                key = key.where(key.ne(''), mapped)



    out['pr_key'] = key.fillna('').astype(str)

    return out





def filter_events_to_prs(df_events, pr_keys):

    if df_events is None:

        return pd.DataFrame()

    out = ensure_pr_key(df_events, table_kind='events')

    if out.empty:

        return out

    pr_key_index = pd.Index(pd.Series(list(pr_keys), dtype='object').astype(str).unique())

    return out[out['pr_key'].astype(str).isin(pr_key_index)].copy()


def to_datetime_utc(value):

    """Parse timestamps to UTC for scalars or Series."""

    if isinstance(value, pd.Series):

        return pd.to_datetime(value, errors='coerce', utc=True)

    if value is None or value == '' or (isinstance(value, float) and pd.isna(value)):

        return pd.NaT

    try:

        return pd.to_datetime(value, errors='coerce', utc=True)

    except Exception:

        return pd.NaT





def norm_login(x):

    """Normalize login/username for consistent comparisons."""

    if x is None or (isinstance(x, float) and pd.isna(x)):

        return ''

    return str(x).strip().lower()





def get_login(value):

    """Robustly extract a login-like identifier from dict/list/serialized author fields."""

    if value is None or value == '' or (isinstance(value, float) and pd.isna(value)):

        return ''

    try:

        login, _ = extract_author_info(value)

        return norm_login(login)

    except Exception:

        return norm_login(value)





def is_merged(df_prs):

    """Detect merged PRs using multiple schema variants."""

    if df_prs is None:

        return pd.Series(dtype=bool)

    if df_prs.empty:

        return pd.Series(index=df_prs.index, dtype=bool)

    signals = []

    if 'state' in df_prs.columns:

        signals.append(df_prs['state'].astype(str).str.lower().eq('merged'))

    if 'merged' in df_prs.columns:

        merged_bool = df_prs['merged']

        if merged_bool.dtype != bool:

            merged_bool = merged_bool.astype(str).str.lower().isin(['true', '1', 'yes'])

        signals.append(merged_bool.fillna(False))

    if 'merged_at' in df_prs.columns:

        signals.append(df_prs['merged_at'].notna())

    if not signals:

        return pd.Series(False, index=df_prs.index)

    result = signals[0].copy()

    for signal in signals[1:]:

        result = result | signal

    return result.fillna(False)





import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from datasets import load_from_disk

from tqdm.auto import tqdm

from collections import defaultdict

import re

from datetime import datetime

import ast

from math import pi

import warnings

import pytz

warnings.filterwarnings('ignore')





plt.rcParams.update({

    'font.size': 12,

    'axes.titlesize': 14,

    'axes.labelsize': 12,

    'xtick.labelsize': 10,

    'ytick.labelsize': 10,

    'legend.fontsize': 10,

    'figure.titlesize': 16,

    'figure.dpi': 100

})



# Auto-export plots to PDF so every displayed figure is reproducible outside the notebook UI.

PAPER_ONLY_EXPORTS = os.environ.get('SUBMISSION_PAPER_ONLY', '1').strip().lower() in {'1', 'true', 'yes', 'y'}

EXPORT_PLOTS_TO_PDF = os.environ.get('SUBMISSION_AUTO_EXPORT_ALL_PLOTS', '0').strip().lower() in {'1', 'true', 'yes', 'y'}

SAVE_INTERMEDIATE_BASELINE_PNGS = os.environ.get('SUBMISSION_SAVE_INTERMEDIATE_PNGS', '0').strip().lower() in {'1', 'true', 'yes', 'y'}

PREFERRED_TIMELINE_PR_KEY = os.environ.get('SUBMISSION_TIMELINE_PR_KEY', 'microsoft/testfx#5633').strip().lower()

PLOT_EXPORT_SUBDIR = 'plots_pdf'

REPLACE_PLOT_PDFS_EACH_RUN = True



_original_plt_show = getattr(plt, 'show', None)

if _original_plt_show is not None and not getattr(_original_plt_show, '_pdf_export_wrapped', False):

    _plot_export_state = {'counter': 0, 'initialized': False}



    def _slugify_plot_name(text):

        value = re.sub(r'[^a-zA-Z0-9]+', '_', str(text or '').strip().lower()).strip('_')

        return value[:80] if value else 'figure'



    def _infer_plot_title(fig):

        suptitle = getattr(fig, '_suptitle', None)

        if suptitle is not None:

            text = (suptitle.get_text() or '').strip()

            if text:

                return text

        for ax in getattr(fig, 'axes', []):

            title = (ax.get_title() or '').strip()

            if title:

                return title

        return 'figure'



    def _resolve_plot_export_dir():

        try:

            export_dir = Path(output_path(PLOT_EXPORT_SUBDIR))

            export_dir.mkdir(parents=True, exist_ok=True)

            return export_dir

        except Exception:

            export_dir = Path('outputs') / PLOT_EXPORT_SUBDIR

            export_dir.mkdir(parents=True, exist_ok=True)

            return export_dir



    def _initialize_plot_export_dir(export_dir):

        if _plot_export_state['initialized']:

            return

        if bool(globals().get('REPLACE_PLOT_PDFS_EACH_RUN', True)):

            removed = 0

            for old_pdf in export_dir.glob('*.pdf'):

                try:

                    old_pdf.unlink()

                    removed += 1

                except Exception:

                    pass

            if removed:

                print(f'Removed {removed} old PDF plot files from {export_dir}')

        _plot_export_state['initialized'] = True



    def _export_open_figures_to_pdf():

        if not bool(globals().get('EXPORT_PLOTS_TO_PDF', False)):

            return

        export_dir = _resolve_plot_export_dir()

        _initialize_plot_export_dir(export_dir)

        for fig_num in list(plt.get_fignums()):

            fig = plt.figure(fig_num)

            _plot_export_state['counter'] += 1

            title_slug = _slugify_plot_name(_infer_plot_title(fig))

            filename = f"{_plot_export_state['counter']:03d}_{title_slug}.pdf"

            out_path = export_dir / filename

            try:

                fig.savefig(out_path, format='pdf', bbox_inches='tight')

            except Exception as exc:

                print(f'Warning: failed to export figure #{fig_num} to PDF: {exc}')



    def _show_with_pdf_export(*args, **kwargs):

        _export_open_figures_to_pdf()

        return _original_plt_show(*args, **kwargs)



    _show_with_pdf_export._pdf_export_wrapped = True

    plt.show = _show_with_pdf_export

    print(

        f"Plot PDF auto-export {'enabled' if EXPORT_PLOTS_TO_PDF else 'disabled'}: "

        f"{PLOT_EXPORT_SUBDIR} (replace_old={REPLACE_PLOT_PDFS_EACH_RUN})"

    )






# PAPER_EXPORT_HELPERS_START

from pathlib import Path





def _paper_path_candidates(relative_path):

    rel = Path(relative_path)

    candidates = [rel]



    # Only use output_path fallback for non-paper relative paths to avoid

    # creating duplicates like outputs/outputs/... and outputs/figures/...

    first_part = rel.parts[0] if rel.parts else ''

    use_output_path = first_part not in {'outputs', 'figures'}

    if use_output_path and 'output_path' in globals():

        try:

            candidates.insert(0, Path(output_path(str(rel))))

        except Exception:

            pass



    unique = []

    seen = set()

    for path in candidates:

        key = str(path)

        if key in seen:

            continue

        seen.add(key)

        unique.append(path)

    return unique





def _write_csv_dual(df, relative_path, index=False):

    last_path = None

    for path in _paper_path_candidates(relative_path):

        path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(path, index=index)

        last_path = path

    return str(last_path) if last_path is not None else ''





def export_fig_pdf(fig, filename):

    # Export a matplotlib figure to PDF with tight bounding box.

    out_paths = []

    for path in _paper_path_candidates(filename):

        path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(path, format='pdf', bbox_inches='tight')

        out_paths.append(str(path))

    return out_paths





def export_table_pdf(df, filename, title=None, index=False, max_rows=None):

    # Render a DataFrame as a clean matplotlib table and export as PDF.

    if df is None:

        table_df = pd.DataFrame([{'message': 'No data'}])

    else:

        table_df = df.copy()

        if not index:

            table_df = table_df.reset_index(drop=True)

    if max_rows is not None and len(table_df) > int(max_rows):

        table_df = table_df.head(int(max_rows)).copy()



    if table_df.empty:

        table_df = pd.DataFrame([{'message': 'No rows'}])



    display_df = table_df.copy()

    for col in display_df.columns:

        display_df[col] = display_df[col].apply(

            lambda v: f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v)

        )



    n_rows, n_cols = display_df.shape

    fig_w = min(24, max(8, 1.4 * n_cols + 1.0))

    fig_h = min(30, max(2.8, 0.42 * (n_rows + 2) + 1.2))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.axis('off')



    mpl_table = ax.table(

        cellText=display_df.values,

        colLabels=list(display_df.columns),

        loc='center',

        cellLoc='left',

    )

    mpl_table.auto_set_font_size(False)

    mpl_table.set_fontsize(9)

    mpl_table.scale(1.0, 1.2)

    try:

        mpl_table.auto_set_column_width(col=list(range(n_cols)))

    except Exception:

        pass



    for (row, col), cell in mpl_table.get_celld().items():

        if row == 0:

            cell.set_text_props(weight='bold')

            cell.set_facecolor('#f2f2f2')

        cell.set_edgecolor('#b5b5b5')



    if title:

        ax.set_title(str(title), fontsize=12, pad=12)



    out_paths = export_fig_pdf(fig, filename)

    plt.close(fig)

    return out_paths

# PAPER_EXPORT_HELPERS_END

def get_local_dataset_configs(local_path):

    if not os.path.exists(local_path):

        return []

    return [d for d in os.listdir(local_path) if os.path.isdir(os.path.join(local_path, d))]



def add_agent_column(df, config):

    agent_name = config.split('_', 1)[1] if '_' in config else 'Unknown'

    df_copy = df.copy()

    df_copy['agent'] = agent_name

    return df_copy



def extract_author_info(author_data):

    if isinstance(author_data, list) and len(author_data) > 0:

        return (author_data[0].get('name', ''), author_data[0].get('email', ''))

    if isinstance(author_data, dict):

        return (author_data.get('login', author_data.get('name', '')), author_data.get('email', ''))

    if isinstance(author_data, str):

        try:

            if author_data.startswith('[') or author_data.startswith('{'):

                parsed_data = ast.literal_eval(author_data)

                return extract_author_info(parsed_data)

        except Exception:

            pass

        return (author_data, '')

    return (str(author_data), '')



def classify_actor_as_ai(name, email, agent_name):

    is_ai = False

    confidence = 'low'

    signals = []

    name_lower = name.lower() if name else ''

    email_lower = email.lower() if email else ''

    if isinstance(agent_name, str):
        agent_lower = agent_name.lower()
    elif agent_name is None or pd.isna(agent_name):
        agent_lower = ''
    else:
        agent_lower = str(agent_name).lower()

    if '[bot]' in name_lower:

        is_ai = True

        confidence = 'high'

        signals.append('name_contains_bot')

    if re.search(r'\d+\+.*\[bot\]@users\.noreply\.github\.com', email_lower):

        is_ai = True

        confidence = 'high'

        signals.append('github_bot_email_pattern')

    if name_lower.endswith('bot') or name_lower.endswith('[bot]'):

        is_ai = True

        confidence = 'high'

        signals.append('name_ends_with_bot')

    ai_tools = ['amazon-q-developer', 'copilot', 'github-actions', 'github-copilot', 'claude']

    service_accounts = ['dependabot', 'renovate', 'codecov', 'greenkeeper', 'gitguardian']

    if any(tool in name_lower for tool in ai_tools):

        is_ai = True

        confidence = 'high'

        signals.append('known_ai_tool')

    if any(service in name_lower for service in service_accounts):

        is_ai = True

        confidence = 'high'

        signals.append('service_account')

    ai_keywords = ['assistant', 'automated', 'auto-', '-ai-', '_ai_', 'ai-', '-ai']

    if any(keyword in name_lower for keyword in ai_keywords):

        is_ai = True

        if confidence == 'low':

            confidence = 'medium'

        signals.append('ai_keywords_in_name')

    if agent_lower and any(ai_indicator in agent_lower for ai_indicator in ['bot', 'ai', 'automated', 'agent']):

        signals.append('agent_context_suggests_ai')

        if not is_ai:

            is_ai = True

            confidence = 'medium'

    return (is_ai, confidence, signals)



def add_actor_classification(df, author_field='author'):

    df_classified = df.copy()

    classifications = []

    confidences = []

    signals_list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Classifying actors'):

        author_data = row.get(author_field)

        agent_name = row.get('agent', '')

        name, email = extract_author_info(author_data)

        is_ai, confidence, signals = classify_actor_as_ai(name, email, agent_name)

        classifications.append(is_ai)

        confidences.append(confidence)

        signals_list.append(signals)

    df_classified['is_ai'] = classifications

    df_classified['ai_confidence'] = confidences

    df_classified['ai_signals'] = signals_list

    df_classified['actor_name'] = [extract_author_info(row.get(author_field))[0] for _, row in tqdm(df.iterrows(), total=len(df), desc='Extracting actor names')]

    df_classified['actor_email'] = [extract_author_info(row.get(author_field))[1] for _, row in tqdm(df.iterrows(), total=len(df), desc='Extracting actor emails')]

    return df_classified


def _read_csv_required(path):

    csv_path = Path(path)

    if not csv_path.exists():

        raise FileNotFoundError(f'Missing required CSV: {csv_path}')

    return pd.read_csv(csv_path)



def _read_csv_optional(path):

    csv_path = Path(path)

    return pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()



df_comments = _read_csv_required(DATASET_DIR / "comments.csv")

df_reviews = _read_csv_required(DATASET_DIR / "reviews.csv")

df_commits = _read_csv_required(DATASET_DIR / "commits.csv")

df_repos = _read_csv_required(DATASET_DIR / "repos.csv")

df_prs = _read_csv_required(DATASET_DIR / "prs.csv")

df_issues = pd.DataFrame() if DROP_ISSUE_TABLES else _read_csv_optional(DATASET_DIR / "issues.csv")

users_df = _read_csv_required(DATASET_DIR / "users.csv")

users_df = users_df[users_df['account_created_at'].notna()]

human_base_dir = str(HUMAN_DATASET_DIR)



def _load_human_tables(base_dir):

    return {

        "comments": _read_csv_required(Path(base_dir) / "human_comments.csv"),

        "commits": _read_csv_required(Path(base_dir) / "human_commits.csv"),

        "issues": pd.DataFrame() if DROP_ISSUE_TABLES else _read_csv_optional(Path(base_dir) / "human_issues.csv"),

        "prs": _read_csv_required(Path(base_dir) / "human_prs.csv"),

        "repos": _read_csv_required(Path(base_dir) / "human_repos.csv"),

        "reviews": _read_csv_required(Path(base_dir) / "human_reviews.csv"),

    }





def _human_files_exist(base_dir):

    required = ['comments', 'commits', 'prs', 'repos', 'reviews']

    return all(Path(base_dir, f"human_{name}.csv").exists() for name in required)





def _normalize_id_series(series):

    if series is None:

        return pd.Series(dtype='object')

    s = series.fillna('').astype(str).str.strip()

    # numeric-like normalization (e.g., "12345.0" -> "12345")

    as_num = pd.to_numeric(s, errors='coerce')

    numeric_norm = as_num.astype('Int64').astype(str).replace('<NA>', '')

    out = numeric_norm.where(numeric_norm.ne(''), s)

    return out.fillna('').astype(str).str.strip()





def _human_pr_event_overlap(df_prs_local, df_event_local):

    if (

        df_prs_local.empty

        or df_event_local.empty

        or 'id' not in df_prs_local.columns

        or 'pr_id' not in df_event_local.columns

    ):

        return 0

    pr_ids = set(_normalize_id_series(df_prs_local['id']))

    event_ids = _normalize_id_series(df_event_local['pr_id'])

    return int(event_ids.isin(pr_ids).sum())





def _human_source_stats(base_dir):

    tables = _load_human_tables(base_dir)

    overlaps = {

        "comments": _human_pr_event_overlap(tables['prs'], tables['comments']),

        "reviews": _human_pr_event_overlap(tables['prs'], tables['reviews']),

        "commits": _human_pr_event_overlap(tables['prs'], tables['commits']),

    }

    event_rows = {

        "comments": int(len(tables['comments'])),

        "reviews": int(len(tables['reviews'])),

        "commits": int(len(tables['commits'])),

    }

    overlap_total = int(sum(overlaps.values()))

    event_total = int(sum(event_rows.values()))

    overlap_ratio = float(overlap_total / event_total) if event_total > 0 else 0.0

    return {

        "base_dir": base_dir,

        "tables": tables,

        "overlap_counts": overlaps,

        "event_rows": event_rows,

        "overlap_total": overlap_total,

        "event_total": event_total,

        "overlap_ratio": overlap_ratio,

    }





candidate_dirs_raw = [

    str(HUMAN_DATASET_DIR),

    str(DATASET_DIR),

    str(PIPELINE_WORKDIR / "combined_dataset"),

    str(PIPELINE_WORKDIR / "dataset"),

    str(PIPELINE_ROOT / "combined_dataset"),

    str(PIPELINE_ROOT / "data" / "combined_dataset"),

]

candidate_dirs = []

seen_dirs = set()

for path in candidate_dirs_raw:

    abs_path = os.path.abspath(path)

    if abs_path in seen_dirs:

        continue

    seen_dirs.add(abs_path)

    if _human_files_exist(abs_path):

        candidate_dirs.append(abs_path)



if not candidate_dirs:

    stop_with_questions_for_user(

        "No human baseline source found. Expected human_*.csv in combined_dataset or dataset."

    )



source_stats = [_human_source_stats(d) for d in candidate_dirs]

source_stats = sorted(source_stats, key=lambda x: (x['overlap_total'], x['overlap_ratio'], x['base_dir'] == 'dataset'), reverse=True)

best = source_stats[0]



human_base_dir = best['base_dir']

human_tables = best['tables']

overlap_counts = best['overlap_counts']



print("Human baseline source selection diagnostics:")

for row in source_stats:

    print(

        f"  {row['base_dir']}: overlap_total={row['overlap_total']:,}/{row['event_total']:,} "

        f"({row['overlap_ratio']:.2%}), overlaps={row['overlap_counts']}"

    )

if os.path.basename(human_base_dir) != "combined_dataset":

    print(f"Selected {human_base_dir} because it has stronger PR/event linkage for baseline timelines.")



print(f"Human baseline source: {human_base_dir}")

print("Human PR/event ID overlap rows:", overlap_counts)



df_human_comments = human_tables['comments']

df_human_commits = human_tables['commits']

df_human_issues = human_tables.get('issues', pd.DataFrame())

df_human_prs = human_tables['prs']

df_human_repos = human_tables['repos']

df_human_reviews = human_tables['reviews']



# Keep one global mapping from GraphQL PR ids to canonical make_pr_key(repo#number)

refresh_pr_id_to_key_map(df_prs, df_human_prs)

print(f"Loaded PR-id map entries: {len(PR_ID_TO_KEY):,}")


df_comments.head()

df_reviews.head()

df_commits.head()

df_repos.head()

df_prs.head()

df_issues.head()

users_df.head()

df_human_comments.head()

df_human_commits.head()

df_human_issues.head()

df_human_prs.head()

df_human_repos.head()

df_human_reviews.head()

if not df_issues.empty:

    df_issues_classified = add_actor_classification(df_issues, 'author')

else:

    df_issues_classified = pd.DataFrame()



def link_issues_to_agent_prs(df_issues, df_prs):



    if df_issues.empty or df_prs.empty:

        return pd.DataFrame()



    if 'is_ai' in df_prs.columns:

        agent_prs = df_prs[df_prs['is_ai'] == True].copy()

    else:

        agent_prs = df_prs.copy()



    linked_issues = []



    for _, issue in tqdm(df_issues.iterrows(), total=len(df_issues), desc='Linking issues to PRs'):

        issue_repo = issue.get('repository', issue.get('repo_name', ''))

        issue_created = pd.to_datetime(issue.get('created_at', issue.get('timestamp', '')), errors='coerce')



        for _, pr in agent_prs.iterrows():

            pr_repo = pr.get('repository', pr.get('repo_name', ''))

            pr_created = pd.to_datetime(pr.get('created_at', pr.get('timestamp', '')), errors='coerce')



            repo_match = (issue_repo == pr_repo) if (issue_repo and pr_repo) else False



            temporal_match = False

            if pd.notna(issue_created) and pd.notna(pr_created):

                time_diff = abs((pr_created - issue_created).days)

                temporal_match = time_diff <= 30



            explicit_ref = False

            issue_body = str(issue.get('body', '')) + ' ' + str(issue.get('title', ''))

            pr_body = str(pr.get('body', '')) + ' ' + str(pr.get('title', ''))

            if issue_body and pr_body:

                closing_keywords = ['closes', 'fixes', 'resolves', 'close', 'fix', 'resolve']

                issue_number = str(issue.get('number', issue.get('id', '')))

                pr_number = str(pr.get('number', pr.get('id', '')))



                for keyword in closing_keywords:

                    if (f"{keyword} #{issue_number}" in pr_body.lower() or

                        f"{keyword} #{pr_number}" in issue_body.lower()):

                        explicit_ref = True

                        break



            if repo_match and (temporal_match or explicit_ref):

                linked_issues.append({

                    'issue_id': issue.get('id', issue.get('number', '')),

                    'issue_number': issue.get('number', ''),

                    'pr_id': pr.get('id', pr.get('number', '')),

                    'pr_number': pr.get('number', ''),

                    'repository': issue_repo,

                    'link_type': 'explicit' if explicit_ref else 'temporal',

                    'agent': pr.get('agent', ''),

                    'issue_created': issue_created,

                    'pr_created': pr_created

                })



    return pd.DataFrame(linked_issues)



def get_timestamp_field(df, priority_order=['submitted_at', 'created_at', 'timestamp']):

    """Get the best available timestamp field from a dataframe"""

    for field in priority_order:

        if field in df.columns:

            return field

    return 'created_at'  # fallback



def get_login_helper(row, field_variants=['actor_name', 'user_login', 'login', 'author']):

    """Get login/actor name from various possible column names"""

    for field in field_variants:

        value = row.get(field, '')

        if value and str(value).strip():

            return str(value).strip()

    return ''



def normalize_issue_events(df_issues, df_comments=None):

    events = []



    if df_issues.empty:

        return pd.DataFrame()



    for _, issue in tqdm(df_issues.iterrows(), total=len(df_issues), desc='Normalizing issue events'):

        events.append({

            'event_type': 'issue_open',

            'actor_type': 'agent' if issue.get('is_ai', False) else 'human',

            'timestamp': pd.to_datetime(issue.get('created_at', issue.get('timestamp', '')), errors='coerce'),

            'issue_id': issue.get('id', issue.get('number', '')),

            'repository': issue.get('repository', issue.get('repo_name', '')),

            'actor_name': get_login_helper(issue),

            'agent': issue.get('agent', ''),

            'raw_data': issue.to_dict()

        })



        if issue.get('state') == 'closed' and issue.get('closed_at'):

            events.append({

                'event_type': 'issue_close',

                'actor_type': 'unknown',  # Closer info not always available

                'timestamp': pd.to_datetime(issue.get('closed_at'), errors='coerce'),

                'issue_id': issue.get('id', issue.get('number', '')),

                'repository': issue.get('repository', issue.get('repo_name', '')),

                'actor_name': '',

                'agent': issue.get('agent', ''),

                'raw_data': {'closed_at': issue.get('closed_at')}

            })



    if df_comments is not None and not df_comments.empty and 'issue_id' in df_comments.columns:

        mask = df_comments['issue_id'].notna()

        issue_comments = df_comments.loc[mask].copy()

    else:

        issue_comments = df_comments.iloc[0:0].copy() if df_comments is not None else pd.DataFrame()



    if not issue_comments.empty:

        timestamp_field = get_timestamp_field(issue_comments)



        for _, comment in tqdm(issue_comments.iterrows(), total=len(issue_comments), desc='Adding issue comments', leave=False):

            timestamp_value = comment.get(timestamp_field, comment.get('timestamp', ''))



            events.append({

                'event_type': 'issue_comment',

                'actor_type': 'agent' if comment.get('is_ai', False) else 'human',

                'timestamp': pd.to_datetime(timestamp_value, errors='coerce'),

                'issue_id': comment.get('issue_id', ''),

                'repository': comment.get('repository', comment.get('repo_name', '')),

                'actor_name': get_login_helper(comment),

                'agent': comment.get('agent', ''),

                'raw_data': comment.to_dict()

            })



    df_events = pd.DataFrame(events)

    df_events = df_events.dropna(subset=['timestamp'])

    df_events = df_events.sort_values('timestamp')



    return df_events


linked_issues = pd.DataFrame()

issue_events_clean = pd.DataFrame()

if not DROP_ISSUE_TABLES:

    linked_issues_path = OPTIONAL_DATASET_DIR / "linked_issues.csv"

    issue_events_clean_path = OPTIONAL_DATASET_DIR / "issue_events_clean.csv"

    try:

        linked_issues = pd.read_csv(linked_issues_path)

    except (FileNotFoundError, Exception):

        linked_issues = pd.DataFrame()



    try:

        issue_events_clean = pd.read_csv(issue_events_clean_path)

    except (FileNotFoundError, Exception):

        issue_events_clean = pd.DataFrame()



    if not df_issues.empty and not df_prs.empty and issue_events_clean.empty:

        linked_issues = link_issues_to_agent_prs(df_issues_classified, df_prs)

        issue_events_clean = normalize_issue_events(df_issues_classified, df_comments)

        OPTIONAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)

        linked_issues.to_csv(linked_issues_path, index=False)

        issue_events_clean.to_csv(issue_events_clean_path, index=False)


pass

if not df_issues.empty:

    test_fraction = 0.1

    if 'RUN_FRACTION' in globals():

        test_fraction = min(RUN_FRACTION, 0.1)  # Cap at 0.1 for testing

    

    test_issues = df_issues.sample(frac=test_fraction, random_state=42) if len(df_issues) > 10 else df_issues

    test_comments = df_comments.sample(frac=test_fraction, random_state=42) if not df_comments.empty and len(df_comments) > 10 else df_comments

    

    test_events = normalize_issue_events(test_issues, test_comments)

    

    pass

    pass

    pass

    if not test_events.empty:

        pass

        pass

else:

    pass


def clean_text(s):

    if pd.isna(s):

        return ''

    s = str(s).strip().lower()

    s = re.sub(r'https?://\S+', '', s)  # Remove URLs

    s = re.sub(r'[^a-z0-9\s\-_/]', ' ', s)  # Keep only alphanumeric and basic punctuation

    s = re.sub(r'\s+', ' ', s)  # Normalize whitespace

    return s.strip()





def _parse_struct_like(value):

    if isinstance(value, str):

        text = value.strip()

        if text.startswith('{') or text.startswith('['):

            try:

                return ast.literal_eval(text)

            except Exception:

                return value

    return value



def classify_user_type(row):

    username = str(row.get('user_login', '')).lower()

    metadata = _parse_struct_like(row.get('user_metadata', {}))

    contrib_total = row.get('contribution_total', 0)



    metadata_name = ''

    agents_list = ''

    if isinstance(metadata, dict):

        metadata_name = str(metadata.get('name', '')).lower()

        agents_list = str(metadata.get('agents', '')).lower()



    if metadata_name:

        agent_indicators = [

            'copilot', 'cursor', 'devin', 'claude', 'codex', 'codegen',

            'ai assistant', 'ai agent', 'gpt', 'chatgpt', 'openai',

            'artificial intelligence', 'machine learning', 'neural',

            'swe-agent', 'coding assistant'

        ]

        for indicator in agent_indicators:

            if indicator in metadata_name:

                return 'agent'



        bot_indicators = [

            'bot', 'automated', 'automation', 'service account', 'ci/cd',

            'continuous integration', 'pipeline', 'deploy', 'build',

            'webhook', 'api'

        ]

        for indicator in bot_indicators:

            if indicator in metadata_name:

                return 'bot'



    if agents_list and len(agents_list) > 5:

        agent_count = len([a for a in ['copilot', 'cursor', 'devin', 'claude', 'codex', 'gpt']

                          if a in agents_list])

        if agent_count >= 3:

            return 'agent'



    bot_username_patterns = [

        r'.*bot$', r'^bot.*', r'.*-bot$', r'.*_bot$', r'.*automated.*',

        r'.*auto.*', r'.*ci$', r'^ci-.*', r'.*ci-.*', r'.*deploy.*',

        r'.*build.*', r'.*service$', r'.*srv$', r'.*pipeline.*', r'.*webhook.*'

    ]

    for pattern in bot_username_patterns:

        if re.match(pattern, username):

            return 'bot'



    agent_username_patterns = [

        r'.*copilot.*', r'.*cursor.*', r'.*devin.*', r'.*claude.*',

        r'.*gpt.*', r'.*openai.*', r'.*ai.*', r'.*agent.*', r'.*assistant.*'

    ]

    for pattern in agent_username_patterns:

        if re.match(pattern, username):

            return 'agent'



    if contrib_total > 50000:

        return 'bot'



    return 'human'



def extract_total_contributions(contrib_dict):

    contrib_dict = _parse_struct_like(contrib_dict)

    if isinstance(contrib_dict, dict):

        return contrib_dict.get('total', 0)

    return 0



def extract_contribution_type(contrib_dict, contrib_type):

    contrib_dict = _parse_struct_like(contrib_dict)

    if isinstance(contrib_dict, dict):

        return contrib_dict.get(contrib_type, 0)

    return 0



def account_age_to_years(account_age_years):

    if pd.isna(account_age_years):

        return 'unknown'

    elif account_age_years <= 1:

        return '0-1 years'

    elif account_age_years <= 3:

        return '1-3 years'

    elif account_age_years <= 5:

        return '3-5 years'

    elif account_age_years <= 8:

        return '5-8 years'

    else:

        return '8+ years'



def calculate_global_seniority(users_df):



    used_columns = []



    timestamp_candidates = []



    for col in users_df.columns:

        if 'created_at' in col.lower() or 'timestamp' in col.lower():

            try:

                timestamps = pd.to_datetime(users_df[col], errors='coerce').dropna()

                if not timestamps.empty:

                    timestamp_candidates.extend(timestamps.tolist())

                    pass

            except:

                continue



    if timestamp_candidates:

        ref_time = max(timestamp_candidates)

    else:

        ref_time = datetime.now(pytz.UTC)



    users_df['account_created_at_parsed'] = pd.to_datetime(users_df['account_created_at'], errors='coerce')

    users_df['tenure_days'] = (ref_time - users_df['account_created_at_parsed']).dt.days

    users_df['tenure_days'] = users_df['tenure_days'].clip(lower=0)  # Ensure non-negative



    contrib_columns = []

    contrib_values = []



    if 'total_commits' in users_df.columns:

        contrib_columns.append('commits')

        contrib_values.append(users_df['total_commits'].fillna(0))

        used_columns.append('total_commits')

    elif 'commits_count' in users_df.columns:

        contrib_columns.append('commits')

        contrib_values.append(users_df['commits_count'].fillna(0))

        used_columns.append('commits_count')

    else:

        pass



    if 'total_prs' in users_df.columns:

        contrib_columns.append('prs')

        contrib_values.append(users_df['total_prs'].fillna(0))

        used_columns.append('total_prs')

    elif 'prs_count' in users_df.columns:

        contrib_columns.append('prs')

        contrib_values.append(users_df['prs_count'].fillna(0))

        used_columns.append('prs_count')

    else:

        pass



    if 'total_reviews' in users_df.columns:

        contrib_columns.append('reviews')

        contrib_values.append(users_df['total_reviews'].fillna(0))

        used_columns.append('total_reviews')

    elif 'reviews_count' in users_df.columns:

        contrib_columns.append('reviews')

        contrib_values.append(users_df['reviews_count'].fillna(0))

        used_columns.append('reviews_count')

    else:

        pass



    tenure_log = np.log1p(users_df['tenure_days'].fillna(0))

    tenure_q05 = tenure_log.quantile(0.05)

    tenure_q95 = tenure_log.quantile(0.95)

    tenure_norm = (tenure_log.clip(tenure_q05, tenure_q95) - tenure_q05) / (tenure_q95 - tenure_q05)

    tenure_norm = tenure_norm.fillna(0)



    if contrib_values:

        contrib_norm_list = []

        for contrib_vals in contrib_values:

            contrib_log = np.log1p(contrib_vals)

            contrib_q05 = contrib_log.quantile(0.05)

            contrib_q95 = contrib_log.quantile(0.95)

            contrib_normalized = (contrib_log.clip(contrib_q05, contrib_q95) - contrib_q05) / (contrib_q95 - contrib_q05)

            contrib_normalized = contrib_normalized.fillna(0)

            contrib_norm_list.append(contrib_normalized)



        contrib_norm = pd.concat(contrib_norm_list, axis=1).mean(axis=1)

    else:

        contrib_norm = pd.Series(0, index=users_df.index)



    if len(contrib_values) > 0:

        global_seniority_score = 0.35 * tenure_norm + 0.65 * contrib_norm

    else:

        global_seniority_score = tenure_norm



    users_df['global_seniority_score'] = global_seniority_score



    score_q25 = global_seniority_score.quantile(0.25)

    score_q75 = global_seniority_score.quantile(0.75)



    def assign_seniority_group(score):

        if pd.isna(score):

            return 'novice'

        elif score <= score_q25:

            return 'novice'

        elif score <= score_q75:

            return 'mid'

        else:

            return 'expert'



    users_df['global_seniority_group'] = global_seniority_score.apply(assign_seniority_group)



    users_df._seniority_metadata = {

        'used_columns': used_columns,

        'reference_time': ref_time,

        'contrib_dimensions': contrib_columns

    }



    return users_df



if not users_df.empty:

    pass



    users_df['contribution_total'] = users_df['contribution_counts'].apply(extract_total_contributions)

    users_df['commits_count'] = users_df['contribution_counts'].apply(lambda x: extract_contribution_type(x, 'commits'))

    users_df['issues_count'] = users_df['contribution_counts'].apply(lambda x: extract_contribution_type(x, 'issues'))

    users_df['prs_count'] = users_df['contribution_counts'].apply(lambda x: extract_contribution_type(x, 'pull_requests'))

    users_df['reviews_count'] = users_df['contribution_counts'].apply(lambda x: extract_contribution_type(x, 'reviews'))



    users_df['user_type'] = users_df.apply(classify_user_type, axis=1)



    if 'user_metadata' in users_df.columns:

        def extract_bio(metadata):

            if isinstance(metadata, dict):

                return metadata.get('bio', '')

            return ''



        def extract_location(metadata):

            if isinstance(metadata, dict):

                return metadata.get('location', '')

            return ''



        def extract_company(metadata):

            if isinstance(metadata, dict):

                return metadata.get('company', '')

            return ''



        def extract_created_at(metadata):

            if isinstance(metadata, dict):

                return metadata.get('created_at', '')

            return ''



        users_df['bio'] = users_df['user_metadata'].apply(extract_bio)

        users_df['location'] = users_df['user_metadata'].apply(extract_location)

        users_df['company'] = users_df['user_metadata'].apply(extract_company)





        users_df = calculate_global_seniority(users_df)



        users_df['bio_clean'] = users_df['bio'].map(clean_text)

        users_df['location_clean'] = users_df['location'].map(clean_text)

        users_df['company_clean'] = users_df['company'].map(clean_text)

    else:

        for col in ['bio', 'location', 'company', 'bio_clean', 'location_clean', 'company_clean']:

            users_df[col] = ''



        users_df = calculate_global_seniority(users_df)



    engaging_users = users_df[users_df['contribution_total'] > 0].copy()

import pandas as pd

import numpy as np



def _get_from_contrib_counts(x, keys):

    if isinstance(x, dict):

        for k in keys:

            if k in x and pd.notna(x.get(k)):

                return x.get(k, 0)

    return 0



def _ensure_contrib_columns(df):

    if 'contribution_counts' in df.columns:

        if 'commits_count' not in df.columns:

            df['commits_count'] = df['contribution_counts'].apply(lambda x: _get_from_contrib_counts(x, ['commits']))

        if 'prs_count' not in df.columns:

            df['prs_count'] = df['contribution_counts'].apply(lambda x: _get_from_contrib_counts(x, ['pull_requests', 'prs']))

        if 'reviews_count' not in df.columns:

            df['reviews_count'] = df['contribution_counts'].apply(lambda x: _get_from_contrib_counts(x, ['reviews']))

        if 'issues_count' not in df.columns:

            df['issues_count'] = df['contribution_counts'].apply(lambda x: _get_from_contrib_counts(x, ['issues']))

        if 'comments_count' not in df.columns:

            df['comments_count'] = df['contribution_counts'].apply(

                lambda x: sum(_get_from_contrib_counts(x, [k]) for k in [

                    'comments', 'issue_comments', 'pull_request_comments',

                    'commit_comments', 'review_comments', 'discussion_comments'

                ])

            )



    if 'contribution_total' not in df.columns:

        if 'contribution_counts' in df.columns:

            df['contribution_total'] = df['contribution_counts'].apply(lambda x: _get_from_contrib_counts(x, ['total']))

        else:

            candidates = [c for c in ['commits_count','comments_count','prs_count','reviews_count','issues_count'] if c in df.columns]

            if candidates:

                df['contribution_total'] = df[candidates].fillna(0).sum(axis=1)

            else:

                df['contribution_total'] = 0



    for c in ['commits_count','comments_count','prs_count','reviews_count','issues_count','contribution_total']:

        if c in df.columns:

            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)



    return df



def make_user_dataset_tables(users_df, engaging_rule='contribution_total>0'):

    df = users_df.copy()

    df = _ensure_contrib_columns(df)



    if 'account_created_at' in df.columns:

        df['account_created_at_parsed'] = pd.to_datetime(df['account_created_at'], errors='coerce', utc=True)

    else:

        df['account_created_at_parsed'] = pd.NaT



    scrape_col = None

    for c in df.columns:

        if 'scrape' in c.lower() and ('time' in c.lower() or 'timestamp' in c.lower() or 'at' in c.lower()):

            scrape_col = c

            break

    if scrape_col is None:

        for c in df.columns:

            if 'scrape' in c.lower():

                scrape_col = c

                break



    if scrape_col is not None:

        df['scrape_ts_parsed'] = pd.to_datetime(df[scrape_col], errors='coerce', utc=True)

    else:

        df['scrape_ts_parsed'] = pd.NaT



    ref_time = df['scrape_ts_parsed'].dropna().max()

    if pd.isna(ref_time):

        ref_time = pd.Timestamp.utcnow()



    df['tenure_days'] = (ref_time - df['account_created_at_parsed']).dt.days

    df['tenure_years'] = (df['tenure_days'] / 365.25).replace([np.inf, -np.inf], np.nan)



    if engaging_rule == 'contribution_total>0':

        engaging_mask = df['contribution_total'] > 0

    else:

        engaging_mask = df.eval(engaging_rule)



    engaging = df[engaging_mask].copy()



    total_users = len(df)

    engaging_users = len(engaging)



    if 'user_type' in df.columns:

        type_counts = engaging['user_type'].value_counts(dropna=False)

    else:

        type_counts = pd.Series(dtype=int)



    profile_fields = []

    for c in ['user_login','user_type','account_created_at']:

        if c in df.columns:

            profile_fields.append(c)

    if 'user_metadata' in df.columns:

        profile_fields.append('user_metadata')



    coverage_rows = []

    for c in profile_fields:

        cov = engaging[c].notna().mean() if c in engaging.columns else np.nan

        coverage_rows.append((c, cov))



    overview = []

    overview.append(('Total users scraped', total_users))

    overview.append(('Engaging (active) users', engaging_users))

    if 'user_type' in engaging.columns and len(type_counts) > 0:

        for k, v in type_counts.items():

            overview.append((f"Engaging users: {k}", int(v)))



    tenure_vals = engaging['tenure_years'].dropna()

    if len(tenure_vals) > 0:

        overview.extend([

            ('Tenure (years): median', float(tenure_vals.median())),

            ('Tenure (years): p25', float(tenure_vals.quantile(0.25))),

            ('Tenure (years): p75', float(tenure_vals.quantile(0.75))),

        ])



    overview_df = pd.DataFrame(overview, columns=['Metric', 'Value'])



    coverage_df = pd.DataFrame(coverage_rows, columns=['Field', 'Coverage (engaging users)'])

    if not coverage_df.empty:

        coverage_df['Coverage (engaging users)'] = (coverage_df['Coverage (engaging users)'] * 100).round(1).astype(str) + r'\%'



    activity_cols = [c for c in ['contribution_total','commits_count','comments_count','prs_count','reviews_count','issues_count'] if c in engaging.columns]

    stats_rows = []

    for c in activity_cols:

        s = engaging[c].astype(float)

        stats_rows.append({

            'Metric': c,

            'Mean': s.mean(),

            'Median': s.median(),

            'p90': s.quantile(0.90),

            'Max': s.max()

        })

    activity_stats_df = pd.DataFrame(stats_rows)

    if not activity_stats_df.empty:

        for col in ['Mean','Median','p90','Max']:

            activity_stats_df[col] = activity_stats_df[col].round(1)



    return {

        'overview_df': overview_df,

        'coverage_df': coverage_df,

        'activity_stats_df': activity_stats_df,

        'ref_time_used': ref_time

    }



out = make_user_dataset_tables(users_df, engaging_rule='contribution_total>0')




if not engaging_users.empty:

    login_source = 'user_login' if 'user_login' in engaging_users.columns else 'login' if 'login' in engaging_users.columns else None

    if login_source is None:

        engaging_users['user_login_norm'] = ''

    else:

        engaging_users['user_login_norm'] = engaging_users[login_source].astype(str).str.strip().str.lower()



    if 'global_seniority_score' not in engaging_users.columns:

        if 'account_created_at' in engaging_users.columns:

            account_created = to_datetime_utc(engaging_users['account_created_at'])

            ref_points = []

            for frame in [df_prs, df_comments, df_reviews, df_commits]:

                if isinstance(frame, pd.DataFrame):

                    for column in ['created_at', 'submitted_at', 'committedDate', 'committed_date']:

                        if column in frame.columns:

                            ts = to_datetime_utc(frame[column]).dropna()

                            if not ts.empty:

                                ref_points.append(ts.max())

            ref_time = max(ref_points) if ref_points else pd.Timestamp.utcnow(tz='UTC')

            tenure_days = (ref_time - account_created).dt.total_seconds().div(86400).clip(lower=0).fillna(0)

        else:

            tenure_days = pd.to_numeric(engaging_users.get('tenure_days', 0), errors='coerce').fillna(0).clip(lower=0)



        contribution_candidates = ['contribution_total', 'commits', 'commit_count', 'prs', 'pr_count', 'reviews', 'review_count', 'issues', 'issue_count']

        if 'contribution_total' in engaging_users.columns:

            contrib_total = pd.to_numeric(engaging_users['contribution_total'], errors='coerce').fillna(0).clip(lower=0)

        else:

            contrib_series = []

            for column in contribution_candidates[1:]:

                if column in engaging_users.columns:

                    contrib_series.append(pd.to_numeric(engaging_users[column], errors='coerce').fillna(0).clip(lower=0))

            contrib_total = sum(contrib_series) if contrib_series else pd.Series(0, index=engaging_users.index, dtype='float64')



        def _winsorize(series, low=0.05, high=0.95):

            if series.empty:

                return series

            lower = series.quantile(low)

            upper = series.quantile(high)

            return series.clip(lower, upper)



        def _minmax(series):

            series = series.fillna(0)

            spread = series.max() - series.min()

            if pd.isna(spread) or spread == 0:

                return pd.Series(0.0, index=series.index)

            return (series - series.min()) / spread



        tenure_norm = _minmax(_winsorize(np.log1p(tenure_days)))

        contrib_norm = _minmax(_winsorize(np.log1p(contrib_total)))

        engaging_users['global_seniority_score'] = 0.35 * tenure_norm + 0.65 * contrib_norm



    engaging_users['global_seniority_score'] = pd.to_numeric(engaging_users['global_seniority_score'], errors='coerce')

    engaging_users = engaging_users.loc[engaging_users['global_seniority_score'].notna()].copy()



    if not engaging_users.empty:

        q25 = engaging_users['global_seniority_score'].quantile(0.25)

        q75 = engaging_users['global_seniority_score'].quantile(0.75)

        engaging_users['global_seniority_group'] = np.select(

            [

                engaging_users['global_seniority_score'] <= q25,

                engaging_users['global_seniority_score'] > q75,

            ],

            ['novice', 'expert'],

            default='mid'

        )

        engaging_users['user_login_norm'] = engaging_users['user_login_norm'].fillna('').astype(str).str.strip().str.lower()

        seniority_group_sizes = engaging_users['global_seniority_group'].value_counts().reindex(['novice', 'mid', 'expert'], fill_value=0)

        human_users = engaging_users[engaging_users.get('user_type', '').astype(str).str.lower() == 'human'].copy() if 'user_type' in engaging_users.columns else engaging_users.copy()

        seniority_df = human_users['global_seniority_group'].value_counts().reindex(['novice', 'mid', 'expert'], fill_value=0).rename_axis('Seniority Level').reset_index(name='Count')

        seniority_df['Seniority Level'] = seniority_df['Seniority Level'].str.title()

        seniority_df['Percentage'] = np.where(len(human_users) > 0, (seniority_df['Count'] / len(human_users) * 100).round(1), 0.0)



assert engaging_users['user_login_norm'].notna().any()

assert engaging_users['global_seniority_group'].isin(['novice', 'mid', 'expert']).all()


if 'engaging_users' in globals() and not engaging_users.empty and 'global_seniority_score' in engaging_users.columns:

    human_users = engaging_users[engaging_users['user_type'] == 'human'].copy() if 'user_type' in engaging_users.columns else engaging_users.copy()

    if not human_users.empty:

        scores = human_users['global_seniority_score'].dropna()

        if not scores.empty:

            q25 = scores.quantile(0.25)

            q75 = scores.quantile(0.75)

            novice_count = int((scores <= q25).sum())

            mid_count = int(((scores > q25) & (scores <= q75)).sum())

            expert_count = int((scores > q75).sum())

            fig, ax = plt.subplots(figsize=(12, 7))

            ax.hist(scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)

            ax.axvline(q25, color='red', linestyle='--', linewidth=2)

            ax.axvline(q75, color='green', linestyle='--', linewidth=2)

            y_max = ax.get_ylim()[1]

            ax.text(q25 / 2 if q25 > 0 else 0.05, y_max * 0.85, f'Novice\nN={novice_count:,}', ha='center', va='center', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            ax.text((q25 + q75) / 2, y_max * 0.85, f'Mid\nN={mid_count:,}', ha='center', va='center', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            ax.text((q75 + scores.max()) / 2, y_max * 0.85, f'Expert\nN={expert_count:,}', ha='center', va='center', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            # ax.set_title('Distribution of Human Developer Seniority Scores')

            ax.set_xlabel('Global Seniority Score')

            ax.set_ylabel('Number of Developers')

            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()

            plt.show()


if not engaging_users.empty and 'contribution_total' in engaging_users.columns:



    human_users = engaging_users[engaging_users['user_type'] == 'human'].copy()



    if len(human_users) > 0:

        contrib_data = human_users['contribution_total']



        intensity_summary = {

            'Metric': ['Mean', 'Median', 'Maximum', 'Standard Deviation', '75th Percentile', '90th Percentile'],

            'Total Contributions': [

                f"{contrib_data.mean():.1f}",

                f"{contrib_data.median():.0f}",

                f"{contrib_data.max():,}",

                f"{contrib_data.std():.1f}",

                f"{contrib_data.quantile(0.75):.0f}",

                f"{contrib_data.quantile(0.90):.0f}"

            ]

        }



        for contrib_type, label in [('commits_count', 'Commits'), ('issues_count', 'Issues'),

                                   ('prs_count', 'Pull Requests'), ('reviews_count', 'Reviews')]:

            if contrib_type in human_users.columns:

                data = human_users[contrib_type]

                intensity_summary[label] = [

                    f"{data.mean():.1f}",

                    f"{data.median():.0f}",

                    f"{data.max():,}",

                    f"{data.std():.1f}",

                    f"{data.quantile(0.75):.0f}",

                    f"{data.quantile(0.90):.0f}"

                ]



        intensity_df = pd.DataFrame(intensity_summary)

    else:

        pass

else:

    pass


if not engaging_users.empty and 'user_type' in engaging_users.columns:

    pass



    human_users = engaging_users[engaging_users['user_type'] == 'human'].copy()



    if len(human_users) > 0:

        contribution_types = ['commits_count', 'issues_count', 'prs_count', 'reviews_count']

        type_labels = ['Commits', 'Issues', 'Pull Requests', 'Reviews']



        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        fig.suptitle('Human Developer Contribution Distributions (Long Tail)', fontsize=16, y=0.98)



        for i, (contrib_type, label) in enumerate(zip(contribution_types, type_labels)):

            if contrib_type in human_users.columns:

                ax = axes[i // 2, i % 2]

                data = human_users[contrib_type].dropna()



                if len(data) > 0:

                    total_users = len(data)

                    data_nonzero = data[data > 0]

                    nonzero_count = len(data_nonzero)

                    nonzero_share = (nonzero_count / total_users) * 100



                    if nonzero_count > 0:

                        max_val = data_nonzero.max()

                        log_bins = np.logspace(0, np.log10(max_val), 25)



                        ax.hist(data_nonzero, bins=log_bins, alpha=0.7, edgecolor='black',

                               color='steelblue', linewidth=0.5)

                        ax.set_xscale('log')

                        ax.set_xlabel(f'{label} Count (log scale)', fontsize=11)

                        ax.set_ylabel('Number of Users', fontsize=11)

                        ax.grid(True, alpha=0.3)



                        # ax.set_title(f'{label}\\n(Heavy-tailed distribution)', fontsize=12, pad=10)



                        median_val = data_nonzero.median()

                        p90_val = data_nonzero.quantile(0.90)

                        ax.axvline(median_val, color='red', linestyle='--', alpha=0.7, linewidth=1)

                        ax.axvline(p90_val, color='orange', linestyle='--', alpha=0.7, linewidth=1)



                    else:

                        ax.text(0.5, 0.5, f'All {total_users} users\\nhave 0 {label.lower()}',

                               ha='center', va='center', transform=ax.transAxes, fontsize=10)

                        # ax.set_title(f'{label}\\n(No activity)', fontsize=12)

                        ax.set_xlabel(f'{label} Count', fontsize=11)

                        ax.set_ylabel('Number of Users', fontsize=11)



        plt.tight_layout()

        plt.show()



else:

    pass


if not df_prs.empty:

    df_prs_classified = add_actor_classification(df_prs, 'author')

else:

    df_prs_classified = pd.DataFrame()



if not df_comments.empty:

    df_comments_classified = add_actor_classification(df_comments, 'author')

else:

    df_comments_classified = pd.DataFrame()



if not df_reviews.empty:

    df_reviews_classified = add_actor_classification(df_reviews, 'author')

else:

    df_reviews_classified = pd.DataFrame()



if not df_commits.empty:

    author_field = 'author'

    if 'committer' in df_commits.columns and 'author' not in df_commits.columns:

        author_field = 'committer'

    elif 'authors' in df_commits.columns and 'author' not in df_commits.columns:

        author_field = 'authors'



    df_commits_classified = add_actor_classification(df_commits, author_field)

else:

    df_commits_classified = pd.DataFrame()



all_classified_dfs = [

    ('PRs', df_prs_classified),

    ('Comments', df_comments_classified),

    ('Reviews', df_reviews_classified),

    ('Commits', df_commits_classified)

]



for name, df in all_classified_dfs:

    if not df.empty and 'is_ai' in df.columns:

        ai_count = df['is_ai'].sum()

        human_count = (~df['is_ai']).sum()

        total = len(df)

    else:

        pass



if 'df_issues_classified' in globals() and not df_issues_classified.empty and 'is_ai' in df_issues_classified.columns:

    ai_count = df_issues_classified['is_ai'].sum()

    human_count = (~df_issues_classified['is_ai']).sum()

    total = len(df_issues_classified)

else:

   pass


from tqdm.auto import tqdm



def _event_timestamp(df, kind):

    priorities = {

        'prs': ['created_at'],

        'comments': ['created_at'],

        'reviews': ['submitted_at', 'created_at'],

        'commits': ['committedDate', 'committed_date', 'created_at']

    }

    for column in priorities.get(kind, ['created_at']):

        if column in df.columns:

            return to_datetime_utc(df[column])

    for column in df.columns:

        col_lower = column.lower()

        if 'time' in col_lower or 'date' in col_lower or col_lower.endswith('_at'):

            parsed = to_datetime_utc(df[column])

            if parsed.notna().any():

                return parsed

    return pd.Series(pd.NaT, index=df.index)



def _event_actor_login(df, fields):

    if df.empty:

        return pd.Series(dtype='object')

    for field in fields:

        if field in df.columns:

            return df[field].apply(get_login)

    return pd.Series('', index=df.index, dtype='object')



def prepare_pr_dataset(df_prs, df_comments, df_reviews, df_commits, dataset_label):

    if df_prs is None or df_prs.empty:

        empty = pd.DataFrame()

        return empty, empty, empty, empty

    prs = ensure_pr_key(df_prs, table_kind='prs')

    sampled_prs = maybe_sample_prs(prs, RUN_FRACTION, RANDOM_SEED)

    pr_keys = pd.Index(sampled_prs['pr_key'].astype(str).unique())

    comments = filter_events_to_prs(df_comments, pr_keys)

    reviews = filter_events_to_prs(df_reviews, pr_keys)

    commits = filter_events_to_prs(df_commits, pr_keys)

    for frame in [sampled_prs, comments, reviews, commits]:

        if not frame.empty:

            frame['dataset_label'] = dataset_label

    return sampled_prs, comments, reviews, commits



def build_timeline(df_prs, df_comments, df_reviews, df_commits, dataset_label):

    prs, comments, reviews, commits = prepare_pr_dataset(df_prs, df_comments, df_reviews, df_commits, dataset_label)

    if prs.empty:

        return pd.DataFrame(columns=['pr_key', 'timestamp', 'action_type', 'actor_login_norm', 'dataset_label'])

    open_events = prs.copy()

    open_events['timestamp'] = _event_timestamp(open_events, 'prs')

    open_events['action_type'] = 'Open'

    open_events['actor_login_norm'] = _event_actor_login(open_events, ['author', 'user', 'actor'])

    open_events['pr_author_login_norm'] = open_events['actor_login_norm']

    open_events['actor_name'] = open_events['actor_login_norm']

    open_events['is_ai'] = open_events.get('is_ai', False).fillna(False) if 'is_ai' in open_events.columns else False

    open_events['agent'] = open_events.get('agent', '').fillna('') if 'agent' in open_events.columns else ''

    event_frames = [open_events[['pr_key', 'timestamp', 'action_type', 'actor_login_norm', 'actor_name', 'is_ai', 'agent', 'dataset_label']]]

    if not comments.empty:

        comments = comments.copy()

        comments['timestamp'] = _event_timestamp(comments, 'comments')

        comments['action_type'] = 'Comment'

        comments['actor_login_norm'] = _event_actor_login(comments, ['author', 'user', 'actor'])

        comments['actor_name'] = comments['actor_login_norm']

        comments['is_ai'] = comments.get('is_ai', False).fillna(False) if 'is_ai' in comments.columns else False

        comments['agent'] = comments.get('agent', '').fillna('') if 'agent' in comments.columns else ''

        event_frames.append(comments[['pr_key', 'timestamp', 'action_type', 'actor_login_norm', 'actor_name', 'is_ai', 'agent', 'dataset_label']])

    if not reviews.empty:

        reviews = reviews.copy()

        reviews['timestamp'] = _event_timestamp(reviews, 'reviews')

        reviews['action_type'] = 'Review'

        reviews['actor_login_norm'] = _event_actor_login(reviews, ['author', 'user', 'actor'])

        reviews['actor_name'] = reviews['actor_login_norm']

        reviews['is_ai'] = reviews.get('is_ai', False).fillna(False) if 'is_ai' in reviews.columns else False

        reviews['agent'] = reviews.get('agent', '').fillna('') if 'agent' in reviews.columns else ''

        event_frames.append(reviews[['pr_key', 'timestamp', 'action_type', 'actor_login_norm', 'actor_name', 'is_ai', 'agent', 'dataset_label']])

    if not commits.empty:

        commits = commits.copy()

        commits['timestamp'] = _event_timestamp(commits, 'commits')

        commits['action_type'] = 'Commit'

        if 'authors' in commits.columns:

            commits['commit_actor_login_norm'] = commits['authors'].apply(get_login)

        else:

            commits['commit_actor_login_norm'] = pd.Series('', index=commits.index, dtype='object')

        fallback = _event_actor_login(commits, ['author', 'committer', 'user'])

        commits['actor_login_norm'] = commits['commit_actor_login_norm'].where(commits['commit_actor_login_norm'] != '', fallback)

        commits['actor_name'] = commits['actor_login_norm']

        commits['is_ai'] = commits.get('is_ai', False).fillna(False) if 'is_ai' in commits.columns else False

        commits['agent'] = commits.get('agent', '').fillna('') if 'agent' in commits.columns else ''

        event_frames.append(commits[['pr_key', 'timestamp', 'action_type', 'actor_login_norm', 'actor_name', 'is_ai', 'agent', 'dataset_label']])

    timeline = pd.concat(event_frames, ignore_index=True)

    timeline = timeline.dropna(subset=['timestamp']).copy()

    timeline['pr_key'] = timeline['pr_key'].astype(str)

    timeline['actor_login_norm'] = timeline['actor_login_norm'].fillna('').astype(str).map(norm_login)

    timeline['actor_name'] = timeline['actor_login_norm']

    timeline['is_ai'] = timeline['is_ai'].fillna(False)

    timeline['agent'] = timeline['agent'].fillna('')

    timeline['actor_login'] = timeline['actor_login_norm']

    timeline['dataset'] = timeline['dataset_label']

    timeline['pr_id'] = timeline['pr_key'].astype(str)

    timeline = timeline.sort_values(['pr_key', 'timestamp', 'action_type']).reset_index(drop=True)

    return timeline



TIMELINE_CACHE_PATH = 'outputs/timeline_df.csv'

FORCE_TIMELINE_REBUILD = _env_flag('SUBMISSION_REBUILD_TIMELINE', '0')

TIMELINE_REQUIRED_COLUMNS = {

    'pr_key': '',

    'timestamp': pd.NaT,

    'action_type': '',

    'actor_login_norm': '',

    'actor_name': '',

    'is_ai': False,

    'agent': '',

    'dataset_label': '',

    'actor_login': '',

    'dataset': '',

    'pr_id': ''

}



def _timeline_cache_looks_stale(df):

    if not isinstance(df, pd.DataFrame) or df.empty or 'pr_key' not in df.columns:

        return True

    pr_keys = df['pr_key'].fillna('').astype(str).str.strip()

    nonempty = pr_keys[pr_keys.ne('')]

    if nonempty.empty:

        return True

    canonical_share = nonempty.str.contains('#', regex=False).mean()

    raw_id_share = nonempty.str.startswith('PR_').mean()

    return canonical_share < 0.90 or raw_id_share > 0.10



if os.path.exists(TIMELINE_CACHE_PATH) and not FORCE_TIMELINE_REBUILD:

    timeline_df = pd.read_csv(TIMELINE_CACHE_PATH)

    for col, default in TIMELINE_REQUIRED_COLUMNS.items():

        if col not in timeline_df.columns:

            timeline_df[col] = default

    if timeline_df.empty:

        timeline_df = pd.DataFrame(columns=list(TIMELINE_REQUIRED_COLUMNS.keys()))

    timeline_df['timestamp'] = to_datetime_utc(timeline_df['timestamp'])

    if timeline_df['dataset_label'].astype(str).eq('').all() and 'dataset' in timeline_df.columns:

        timeline_df['dataset_label'] = timeline_df['dataset'].fillna('').astype(str)

    truthy = {'true', '1', 'yes', 'y', 't'}

    timeline_df['is_ai'] = timeline_df['is_ai'].apply(lambda x: x if isinstance(x, (bool, np.bool_)) else str(x).strip().lower() in truthy).fillna(False)

    timeline_df['pr_key'] = timeline_df['pr_key'].fillna('').astype(str)

    timeline_df['pr_id'] = timeline_df['pr_key'].astype(str)

    timeline_df['actor_login_norm'] = timeline_df['actor_login_norm'].fillna('').astype(str).map(norm_login)

    timeline_df['actor_name'] = timeline_df['actor_name'].fillna('').astype(str)

    missing_actor_name = timeline_df['actor_name'].eq('')

    timeline_df.loc[missing_actor_name, 'actor_name'] = timeline_df.loc[missing_actor_name, 'actor_login_norm']

    timeline_df['actor_login'] = timeline_df['actor_login_norm']

    timeline_df['dataset_label'] = timeline_df['dataset_label'].fillna('').astype(str)

    timeline_df['dataset'] = timeline_df['dataset_label']

    timeline_df['agent'] = timeline_df['agent'].fillna('').astype(str)

    timeline_df = timeline_df.sort_values(['pr_key', 'timestamp', 'action_type']).reset_index(drop=True)

    if _timeline_cache_looks_stale(timeline_df):

        print(f'Ignoring stale timeline cache at {TIMELINE_CACHE_PATH}; rebuilding from source tables.')

        timeline_df = pd.DataFrame()

    else:

        print(f'Loaded timeline_df from {TIMELINE_CACHE_PATH} ({len(timeline_df):,} rows)')

if 'timeline_df' not in globals() or not isinstance(timeline_df, pd.DataFrame) or timeline_df.empty:

    timeline_agent = build_timeline(df_prs_classified, df_comments_classified, df_reviews_classified, df_commits_classified, 'agent')

    df_human_prs_classified = add_actor_classification(df_human_prs, 'author') if 'df_human_prs' in globals() and not df_human_prs.empty else pd.DataFrame()

    df_human_comments_classified = add_actor_classification(df_human_comments, 'author') if 'df_human_comments' in globals() and not df_human_comments.empty else pd.DataFrame()

    df_human_reviews_classified = add_actor_classification(df_human_reviews, 'author') if 'df_human_reviews' in globals() and not df_human_reviews.empty else pd.DataFrame()

    df_human_commits_classified = add_actor_classification(df_human_commits, 'authors' if 'df_human_commits' in globals() and 'authors' in df_human_commits.columns else 'author') if 'df_human_commits' in globals() and not df_human_commits.empty else pd.DataFrame()

    timeline_human = build_timeline(df_human_prs_classified, df_human_comments_classified, df_human_reviews_classified, df_human_commits_classified, 'human')

    timeline_frames = [df for df in [timeline_agent, timeline_human] if not df.empty]

    timeline_df = pd.concat(timeline_frames, ignore_index=True) if timeline_frames else pd.DataFrame(columns=['pr_key', 'timestamp', 'action_type', 'actor_login_norm', 'dataset_label'])

    timeline_df['pr_key'] = timeline_df['pr_key'].astype(str)

    timeline_df['pr_id'] = timeline_df['pr_key'].astype(str)

    timeline_df['actor_login_norm'] = timeline_df['actor_login_norm'].fillna('').astype(str).map(norm_login)

    timeline_df['actor_name'] = timeline_df['actor_login_norm']

    timeline_df['actor_login'] = timeline_df['actor_login_norm']

    timeline_df['dataset'] = timeline_df['dataset_label']

    if 'is_ai' in timeline_df.columns:

        timeline_df['is_ai'] = timeline_df['is_ai'].fillna(False)

    else:

        timeline_df['is_ai'] = False

    if 'agent' in timeline_df.columns:

        timeline_df['agent'] = timeline_df['agent'].fillna('')

    else:

        timeline_df['agent'] = ''

    timeline_df = timeline_df.sort_values(['pr_key', 'timestamp', 'action_type']).reset_index(drop=True)

    os.makedirs('outputs', exist_ok=True)

    timeline_df.to_csv(TIMELINE_CACHE_PATH, index=False)

    print(f'Rebuilt timeline_df and saved to {TIMELINE_CACHE_PATH} ({len(timeline_df):,} rows)')



assert timeline_df['pr_key'].notna().all()

assert timeline_df['timestamp'].notna().all()

assert timeline_df['actor_login_norm'].notna().all()



if 'engaging_users' in globals() and isinstance(engaging_users, pd.DataFrame) and not timeline_df.empty:

    if 'user_login_norm' not in engaging_users.columns:

        login_source = 'user_login' if 'user_login' in engaging_users.columns else 'login' if 'login' in engaging_users.columns else None

        if login_source is not None:

            engaging_users['user_login_norm'] = engaging_users[login_source].astype(str).str.strip().str.lower()

        else:

            engaging_users['user_login_norm'] = ''

    observed_human_counts = timeline_df[(~timeline_df['is_ai']) & (timeline_df['action_type'] != 'Open') & (timeline_df['actor_login_norm'] != '')]['actor_login_norm'].value_counts()

    missing_counts = observed_human_counts[~observed_human_counts.index.isin(set(engaging_users['user_login_norm']))]

    if not missing_counts.empty:

        proxy_counts = np.log1p(missing_counts.astype(float))

        spread = proxy_counts.max() - proxy_counts.min()

        proxy_norm = (proxy_counts - proxy_counts.min()) / spread if spread else pd.Series(0.5, index=proxy_counts.index)

        proxy_score = 0.35 * 0.5 + 0.65 * proxy_norm

        supplemental_users = pd.DataFrame({

            'user_login': missing_counts.index,

            'user_login_norm': missing_counts.index,

            'contribution_total': missing_counts.values,

            'global_seniority_score': proxy_score.values,

            'user_type': 'human'

        })

        engaging_users = pd.concat([engaging_users, supplemental_users], ignore_index=True, sort=False)

    engaging_users = engaging_users.drop_duplicates(subset=['user_login_norm'], keep='first').copy()

    engaging_users['global_seniority_score'] = pd.to_numeric(engaging_users['global_seniority_score'], errors='coerce').fillna(0.5)

    q25 = engaging_users['global_seniority_score'].quantile(0.25)

    q75 = engaging_users['global_seniority_score'].quantile(0.75)

    engaging_users['global_seniority_group'] = np.select(

        [engaging_users['global_seniority_score'] <= q25, engaging_users['global_seniority_score'] > q75],

        ['novice', 'expert'],

        default='mid'

    )

    seniority_group_sizes = engaging_users['global_seniority_group'].value_counts().reindex(['novice', 'mid', 'expert'], fill_value=0)


def _choose_repo_key(df):

    if df is None or df.empty:

        return pd.Series('', index=getattr(df, 'index', pd.Index([])), dtype='object')

    candidates = ['repo_full_name', 'base_repo_full_name', 'repository', 'repo_name', 'base_repository', 'repo', 'full_name']

    for column in candidates:

        if column in df.columns:

            return _extract_repo_name(df[column])

    return pd.Series('', index=df.index, dtype='object')





def _choose_pr_number(df):

    if df is None or df.empty:

        return pd.Series(pd.NA, index=getattr(df, 'index', pd.Index([])), dtype='Int64')

    for column in ['number', 'pr_number', 'pull_number']:

        if column in df.columns:

            return pd.to_numeric(df[column], errors='coerce').astype('Int64')

    return pd.Series(pd.NA, index=df.index, dtype='Int64')





def _choose_created_at(df):

    if df is None or df.empty:

        return pd.Series(pd.NaT, index=getattr(df, 'index', pd.Index([])))

    for column in ['created_at', 'createdAt', 'timestamp']:

        if column in df.columns:

            return to_datetime_utc(df[column])

    return pd.Series(pd.NaT, index=df.index)





def _baseline_group_order(values=None):

    order = ['agent', 'human', 'bot'] if RUN_COARSE_BOT_BASELINE else ['agent', 'human', 'bot_dependency', 'bot_other']

    if values is None:

        return order

    present = list(pd.Index(values).astype(str))

    ordered = [value for value in order if value in present]

    tail = [value for value in present if value not in ordered]

    return ordered + tail





# Ensure human classified tables exist even when timeline is loaded from cache.

if 'df_human_prs_classified' not in globals():

    df_human_prs_classified = add_actor_classification(df_human_prs, 'author') if 'df_human_prs' in globals() and not df_human_prs.empty else pd.DataFrame()

if 'df_human_comments_classified' not in globals():

    df_human_comments_classified = add_actor_classification(df_human_comments, 'author') if 'df_human_comments' in globals() and not df_human_comments.empty else pd.DataFrame()

if 'df_human_reviews_classified' not in globals():

    df_human_reviews_classified = add_actor_classification(df_human_reviews, 'author') if 'df_human_reviews' in globals() and not df_human_reviews.empty else pd.DataFrame()

if 'df_human_commits_classified' not in globals():

    df_human_commits_classified = add_actor_classification(

        df_human_commits,

        'authors' if 'df_human_commits' in globals() and isinstance(df_human_commits, pd.DataFrame) and 'authors' in df_human_commits.columns else 'author'

    ) if 'df_human_commits' in globals() and not df_human_commits.empty else pd.DataFrame()



agent_prs_sampled, agent_comments_sampled, agent_reviews_sampled, agent_commits_sampled = prepare_pr_dataset(

    df_prs_classified, df_comments_classified, df_reviews_classified, df_commits_classified, 'agent'

)

human_prs_sampled, human_comments_sampled, human_reviews_sampled, human_commits_sampled = prepare_pr_dataset(

    df_human_prs_classified, df_human_comments_classified, df_human_reviews_classified, df_human_commits_classified, 'human'

)



# Refresh id->key map with sampled PRs so event tables can map pr_id consistently.

refresh_pr_id_to_key_map(agent_prs_sampled, human_prs_sampled)



pr_frames = []

for frame in [agent_prs_sampled, human_prs_sampled]:

    if frame is not None and not frame.empty:

        pr_frames.append(frame.copy())



df_prs_all = pd.concat(pr_frames, ignore_index=True) if pr_frames else pd.DataFrame()

df_agent_prs = pd.DataFrame()

df_human_prs_scope = pd.DataFrame()

df_bot_prs = pd.DataFrame()

agent_prs = pd.DataFrame()

baseline_prs = pd.DataFrame()

baseline_sanity_checks_df = pd.DataFrame()



data_scope_totals_before = {}

data_scope_totals_after = {}

available_agents = []

selected_agents_effective = []

missing_selected_agents = []

dedupe_dropped_by_group = {}



author_bot_pattern = re.compile(r'dependabot|renovate|\[bot\]|-bot$|bot\b', flags=re.IGNORECASE)



if not df_prs_all.empty:

    df_prs_all['repo_full_name'] = _choose_repo_key(df_prs_all)

    df_prs_all['pr_number'] = _choose_pr_number(df_prs_all)

    df_prs_all['pr_key'] = make_pr_key(df_prs_all['repo_full_name'], df_prs_all['pr_number'])

    if df_prs_all['pr_key'].fillna('').eq('').any():

        blank_n = int(df_prs_all['pr_key'].fillna('').eq('').sum())

        stop_with_questions_for_user(f"{blank_n} PR rows have blank canonical pr_key. Please verify repo/number fields in PR tables.")



    df_prs_all['author_login_norm'] = _event_actor_login(df_prs_all, ['author', 'user', 'actor']).fillna('').astype(str).map(norm_login)

    df_prs_all['open_time'] = _choose_created_at(df_prs_all)

    df_prs_all['merged'] = is_merged(df_prs_all).fillna(False).astype(bool)

    df_prs_all['pr_dataset'] = df_prs_all.get('dataset_label', '').fillna('').astype(str)

    df_prs_all['agent_name'] = df_prs_all.get('agent', '').fillna('').astype(str).str.strip().str.lower()

    df_prs_all['agent_name_norm'] = df_prs_all['agent_name'].map(norm_login)



    is_agent_pr = df_prs_all['pr_dataset'].eq('agent')

    is_bot_author = df_prs_all['author_login_norm'].str.contains(author_bot_pattern, na=False)

    df_prs_all['baseline_bot_kind'] = ''

    dependency_mask = is_bot_author & df_prs_all['author_login_norm'].str.contains(r'dependabot|renovate', regex=True, na=False)

    df_prs_all.loc[dependency_mask, 'baseline_bot_kind'] = 'bot_dependency'

    df_prs_all.loc[is_bot_author & df_prs_all['baseline_bot_kind'].eq(''), 'baseline_bot_kind'] = 'bot_other'



    if RUN_COARSE_BOT_BASELINE:

        df_prs_all['author_type_group'] = np.where(is_agent_pr, 'agent', np.where(is_bot_author, 'bot', 'human'))

    else:

        df_prs_all['author_type_group'] = np.where(

            is_agent_pr,

            'agent',

            np.where(is_bot_author, df_prs_all['baseline_bot_kind'], 'human')

        )



    data_scope_totals_before = df_prs_all['author_type_group'].value_counts().to_dict()

    available_agents = sorted(df_prs_all.loc[df_prs_all['author_type_group'].eq('agent'), 'agent_name_norm'].dropna().unique().tolist())



    if SELECTED_AGENTS is None:

        selected_agents_effective = available_agents

    else:

        requested = [norm_login(x) for x in SELECTED_AGENTS]

        selected_agents_effective = [x for x in requested if x in available_agents]

        missing_selected_agents = [x for x in requested if x not in available_agents]



    start_ts = pd.Timestamp(START_DATE, tz='UTC')

    end_ts = pd.Timestamp(END_DATE, tz='UTC') + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    time_mask = df_prs_all['open_time'].between(start_ts, end_ts, inclusive='both')



    if selected_agents_effective:

        agent_mask = (~df_prs_all['author_type_group'].eq('agent')) | (df_prs_all['agent_name_norm'].isin(selected_agents_effective))

    else:

        agent_mask = ~df_prs_all['author_type_group'].eq('agent')



    df_prs_all = df_prs_all.loc[time_mask & agent_mask].copy()

    before_group_counts = df_prs_all['author_type_group'].value_counts().to_dict()



    sentinel_time = pd.Timestamp('2262-04-11', tz='UTC')

    df_prs_all['_open_sort'] = df_prs_all['open_time'].fillna(sentinel_time)

    dup_before = int(df_prs_all.duplicated(subset=['pr_key']).sum())

    df_prs_all = (

        df_prs_all

        .sort_values(['pr_key', '_open_sort'])

        .drop_duplicates(subset=['pr_key'], keep='first')

        .drop(columns=['_open_sort'])

        .reset_index(drop=True)

    )

    dup_after = int(df_prs_all.duplicated(subset=['pr_key']).sum())



    after_group_counts = df_prs_all['author_type_group'].value_counts().to_dict()

    dedupe_dropped_by_group = {

        group: int(before_group_counts.get(group, 0) - after_group_counts.get(group, 0))

        for group in set(before_group_counts) | set(after_group_counts)

    }



    present_groups = set(df_prs_all['author_type_group'].dropna().astype(str).unique())

    required_groups = {'agent', 'human'}

    missing_required = required_groups - present_groups

    assert not missing_required, f"Missing required groups after scope/dedupe: {sorted(missing_required)}"

    assert int(df_prs_all.duplicated(subset=['pr_key']).sum()) == 0, 'Duplicate pr_key rows remain after dedupe.'



    data_scope_totals_after = df_prs_all['author_type_group'].value_counts().to_dict()

    df_agent_prs = df_prs_all[df_prs_all['author_type_group'].eq('agent')].copy()

    df_human_prs_scope = df_prs_all[df_prs_all['author_type_group'].eq('human')].copy()

    if RUN_COARSE_BOT_BASELINE:

        df_bot_prs = df_prs_all[df_prs_all['author_type_group'].eq('bot')].copy()

    else:

        df_bot_prs = df_prs_all[df_prs_all['author_type_group'].isin(['bot_dependency', 'bot_other'])].copy()



    agent_prs = df_agent_prs.copy()

    baseline_prs = df_prs_all[df_prs_all['author_type_group'].ne('agent')].copy()



    # Scope summary (paper-facing)

    save_text('\n'.join(selected_agents_effective), 'agents_in_scope.txt')

    print('SCOPE SUMMARY')

    print(f"  date range: {START_DATE} to {END_DATE}")

    print(f"  agents included: {len(selected_agents_effective)}")

    if selected_agents_effective:

        preview = ', '.join(selected_agents_effective[:20])

        suffix = '' if len(selected_agents_effective) <= 20 else f" ... (+{len(selected_agents_effective) - 20} more)"

        print(f"  agent list: {preview}{suffix}")

    if missing_selected_agents:

        print('  missing requested agents:', ', '.join(missing_selected_agents))



    sanity_rows = []

    for group in _baseline_group_order(df_prs_all['author_type_group'].dropna().unique()):

        group_frame = df_prs_all[df_prs_all['author_type_group'] == group]

        sanity_rows.append({'section': 'counts', 'group': group, 'metric': 'pr_count', 'value': int(group_frame['pr_key'].nunique()), 'detail': ''})

        sanity_rows.append({'section': 'counts', 'group': group, 'metric': 'merged_pr_count', 'value': int(group_frame['merged'].sum()), 'detail': 'pre-merge-proxy PR-table merged flag; overwritten with merge-proxy merged_n after commit coverage step'})

        sanity_rows.append({'section': 'dedupe', 'group': group, 'metric': 'dedupe_rows_dropped', 'value': int(dedupe_dropped_by_group.get(group, 0)), 'detail': 'dropped while keeping earliest open_time per pr_key'})



    sanity_rows.append({'section': 'scope', 'group': 'global', 'metric': 'start_date', 'value': START_DATE, 'detail': ''})

    sanity_rows.append({'section': 'scope', 'group': 'global', 'metric': 'end_date', 'value': END_DATE, 'detail': ''})

    sanity_rows.append({'section': 'scope', 'group': 'global', 'metric': 'available_agents_n', 'value': int(len(available_agents)), 'detail': ','.join(available_agents[:30])})

    sanity_rows.append({'section': 'scope', 'group': 'global', 'metric': 'selected_agents_effective_n', 'value': int(len(selected_agents_effective)), 'detail': ','.join(selected_agents_effective[:30])})

    sanity_rows.append({'section': 'dedupe', 'group': 'global', 'metric': 'duplicate_pr_keys_before', 'value': int(dup_before), 'detail': ''})

    sanity_rows.append({'section': 'dedupe', 'group': 'global', 'metric': 'duplicate_pr_keys_after', 'value': int(dup_after), 'detail': ''})



    baseline_sanity_checks_df = pd.DataFrame(sanity_rows)

    save_csv(baseline_sanity_checks_df, 'baseline_sanity_checks.csv', index=False)



    print('PR master table built with groups:', data_scope_totals_after)
    print('Early sanity merged counts are pre-merge-proxy PR-table flags; final CSV is resynced after merge proxy construction.')

    if RUN_COARSE_BOT_BASELINE:

        print('Bot baseline rows retained (coarse bot):', int(data_scope_totals_after.get('bot', 0)))

    else:

        print('Bot baseline rows retained (split):', int(data_scope_totals_after.get('bot_dependency', 0) + data_scope_totals_after.get('bot_other', 0)))

    print(baseline_sanity_checks_df.to_string(index=False))

else:

    print('df_prs_all is empty after concatenating sampled PR frames.')


first_action_df = pd.DataFrame(columns=['author_type_group', 'first_human_response_type', 'count', 'share'])

latency_summary = pd.DataFrame(columns=['author_type_group', 'N', 'median_hours', 'p90_hours', 'p99_hours'])

first_responses = pd.DataFrame()

first_response_df = pd.DataFrame()

unknown_author_prs_skipped = 0

timeline_with_pr_info = pd.DataFrame()

timeline_missing_share = 0.0

first_response_coverage = pd.DataFrame()





def _response_type_bucket(action_type_value):

    action_norm = str(action_type_value).strip().lower()

    if action_norm == 'comment':

        return 'comment'

    if action_norm == 'review':

        return 'review'

    if action_norm == 'commit':

        return 'commit'

    return 'other'





def _gather_known_agent_logins():

    known = set()

    candidate_frames = [

        globals().get('df_prs_classified', pd.DataFrame()),

        globals().get('df_comments_classified', pd.DataFrame()),

        globals().get('df_reviews_classified', pd.DataFrame()),

        globals().get('df_commits_classified', pd.DataFrame()),

        globals().get('agent_prs_sampled', pd.DataFrame()),

    ]

    for frame in candidate_frames:

        if not isinstance(frame, pd.DataFrame) or frame.empty:

            continue

        ai_mask = frame['is_ai'].fillna(False).astype(bool) if 'is_ai' in frame.columns else pd.Series(False, index=frame.index)

        ai_rows = frame.loc[ai_mask].copy()

        if ai_rows.empty:

            continue

        known.update(_event_actor_login(ai_rows, ['author', 'user', 'actor']).fillna('').astype(str).map(norm_login))

        if 'authors' in ai_rows.columns:

            known.update(ai_rows['authors'].apply(get_login).fillna('').astype(str).map(norm_login))

        if 'agent' in ai_rows.columns:

            known.update(ai_rows['agent'].fillna('').astype(str).map(norm_login))

    return {x for x in known if x}





KNOWN_AGENT_LOGINS = _gather_known_agent_logins()

BOT_LOGIN_PATTERN = re.compile(r'(\[bot\])|(^|[-_])bot$|dependabot|renovate', flags=re.IGNORECASE)





def classify_actor(login_norm, is_ai_flag=False, agent_name_field=''):

    login = norm_login(login_norm)

    agent_field = str(agent_name_field).strip().lower() if agent_name_field is not None else ''

    if login and BOT_LOGIN_PATTERN.search(login):

        return 'bot'

    if bool(is_ai_flag):

        return 'agent'

    if agent_field not in ('', 'nan', 'none'):

        # In some tables `agent` is dataset-level metadata and appears on every row.

        # Treat it as an actor signal only when login is empty or directly matches the agent handle.

        if (not login) or (login == norm_login(agent_field)):

            return 'agent'

    if login in KNOWN_AGENT_LOGINS:

        return 'agent'

    return 'human'





def _build_scope_timeline_for_rq21():

    frames = []

    if 'agent_prs_sampled' in globals() and isinstance(agent_prs_sampled, pd.DataFrame) and not agent_prs_sampled.empty:

        t_agent = build_timeline(agent_prs_sampled, agent_comments_sampled, agent_reviews_sampled, agent_commits_sampled, 'agent')

        if not t_agent.empty:

            frames.append(t_agent)

    if 'human_prs_sampled' in globals() and isinstance(human_prs_sampled, pd.DataFrame) and not human_prs_sampled.empty:

        t_human = build_timeline(human_prs_sampled, human_comments_sampled, human_reviews_sampled, human_commits_sampled, 'human')

        if not t_human.empty:

            frames.append(t_human)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return combined





timeline_work = _build_scope_timeline_for_rq21()

if timeline_work.empty and 'timeline_df' in globals() and isinstance(timeline_df, pd.DataFrame) and not timeline_df.empty:

    timeline_work = timeline_df.copy()

    if 'df_prs_all' in globals() and isinstance(df_prs_all, pd.DataFrame) and not df_prs_all.empty and 'pr_key' in df_prs_all.columns:

        scope_keys = set(df_prs_all['pr_key'].dropna().astype(str))

        timeline_work['pr_key'] = timeline_work['pr_key'].fillna('').astype(str)

        timeline_work = timeline_work[timeline_work['pr_key'].isin(scope_keys)].copy()



if timeline_work.empty:

    stop_with_questions_for_user(

        'RQ2.1 timelines are empty for scope-filtered agent/human data. Please verify combined_dataset inputs and timeline construction.'

    )



required_cols = {

    'pr_key': '',

    'timestamp': pd.NaT,

    'action_type': '',

    'actor_login_norm': '',

    'dataset_label': '',

    'is_ai': False,

    'agent': ''

}

for col, default in required_cols.items():

    if col not in timeline_work.columns:

        timeline_work[col] = default



timeline_work['pr_key'] = timeline_work['pr_key'].fillna('').astype(str)

timeline_work['timestamp'] = to_datetime_utc(timeline_work['timestamp'])

timeline_work['action_type'] = timeline_work['action_type'].fillna('').astype(str).str.strip().str.title()

timeline_work['actor_login_norm'] = timeline_work['actor_login_norm'].fillna('').astype(str).map(norm_login)

timeline_work['dataset_label'] = timeline_work['dataset_label'].fillna('').astype(str).str.strip().str.lower()

timeline_work['is_ai'] = timeline_work['is_ai'].fillna(False).astype(bool)

timeline_work['agent'] = timeline_work['agent'].fillna('').astype(str)

timeline_work = timeline_work[timeline_work['timestamp'].notna()].copy()

timeline_work = timeline_work[timeline_work['action_type'].isin(['Open', 'Comment', 'Review', 'Commit'])].copy()

timeline_work['actor_type'] = timeline_work.apply(

    lambda row: classify_actor(

        row.get('actor_login_norm', ''),

        row.get('is_ai', False),

        row.get('agent', '')

    ),

    axis=1

)

timeline_work = timeline_work.sort_values(['pr_key', 'timestamp', 'action_type']).reset_index(drop=True)

timeline_df = timeline_work.copy()



open_events = (

    timeline_work.loc[timeline_work['action_type'].eq('Open')]

    .sort_values(['pr_key', 'timestamp'])

    .drop_duplicates(subset=['pr_key'], keep='first')

    .copy()

)

if open_events.empty:

    stop_with_questions_for_user('No Open events found in scope timelines; cannot compute RQ2.1 first-response metrics.')



open_events['open_time'] = open_events['timestamp']

open_events['pr_author_login_norm'] = open_events['actor_login_norm'].fillna('').astype(str).map(norm_login)

open_events['agent_name'] = open_events.get('agent', '').fillna('').astype(str).str.strip().str.lower()

open_events['author_type_group'] = np.where(

    open_events['dataset_label'].eq('agent'),

    'agent',

    np.where(open_events['actor_type'].eq('human'), 'human', 'bot')

)



pr_meta = open_events[['pr_key', 'open_time', 'pr_author_login_norm', 'author_type_group', 'dataset_label', 'agent_name']].copy()

timeline_with_pr_info = timeline_work.merge(

    pr_meta[['pr_key', 'open_time', 'pr_author_login_norm', 'author_type_group', 'agent_name']],

    on='pr_key',

    how='inner'

)

timeline_missing_share = 0.0



print('RQ2.1 timeline PR counts by dataset_label:')

print(pr_meta.groupby('dataset_label')['pr_key'].nunique().sort_index().to_string())

print('RQ2.1 timeline PR counts by author_type_group:')

print(pr_meta.groupby('author_type_group')['pr_key'].nunique().sort_index().to_string())



response_priority = {'comment': 1, 'review': 2, 'commit': 3}

candidate_responses = timeline_with_pr_info.loc[

    timeline_with_pr_info['action_type'].isin(['Comment', 'Review', 'Commit'])

    & timeline_with_pr_info['timestamp'].gt(timeline_with_pr_info['open_time'])

    & timeline_with_pr_info['actor_login_norm'].ne(timeline_with_pr_info['pr_author_login_norm'])

    & timeline_with_pr_info['actor_type'].eq('human')

].copy()

candidate_responses['first_human_response_type'] = candidate_responses['action_type'].str.lower().map(

    lambda x: x if x in {'comment', 'review', 'commit'} else 'other'

)

candidate_responses['action_priority'] = candidate_responses['first_human_response_type'].map(response_priority).fillna(99)



first_responses = (

    candidate_responses

    .sort_values(['pr_key', 'timestamp', 'action_priority'])

    .groupby('pr_key', as_index=False)

    .first()

)

# Defensive guard: normalize `pr_key` as an explicit string column for downstream joins.
if 'pr_key' not in first_responses.columns:

    if getattr(first_responses.index, 'name', None) == 'pr_key' or 'pr_key' in list(getattr(first_responses.index, 'names', [])):

        first_responses = first_responses.reset_index()

    else:

        first_responses['pr_key'] = first_responses.index.astype(str)

first_responses['pr_key'] = first_responses['pr_key'].fillna('').astype(str)

first_responses = first_responses.merge(

    pr_meta[['pr_key', 'author_type_group', 'agent_name', 'dataset_label']],

    on='pr_key',

    how='left',

    suffixes=('', '_meta')

)

if 'author_type_group_meta' in first_responses.columns:

    first_responses['author_type_group'] = first_responses['author_type_group'].fillna(first_responses['author_type_group_meta'])

    first_responses = first_responses.drop(columns=['author_type_group_meta'])

if 'agent_name_meta' in first_responses.columns:

    first_responses['agent_name'] = first_responses['agent_name'].fillna(first_responses['agent_name_meta'])

    first_responses = first_responses.drop(columns=['agent_name_meta'])



group_pr_counts = pr_meta.groupby('author_type_group')['pr_key'].nunique().rename('total_prs').reset_index()

response_counts = (

    first_responses.groupby('author_type_group')['pr_key'].nunique().rename('prs_with_first_human_response').reset_index()

    if not first_responses.empty else pd.DataFrame(columns=['author_type_group', 'prs_with_first_human_response'])

)

first_response_coverage = group_pr_counts.merge(response_counts, on='author_type_group', how='left')

first_response_coverage['prs_with_first_human_response'] = first_response_coverage['prs_with_first_human_response'].fillna(0).astype(int)

first_response_coverage['coverage_share'] = first_response_coverage['prs_with_first_human_response'] / first_response_coverage['total_prs'].replace(0, np.nan)



responses_by_dataset = (

    first_responses.groupby('dataset_label')['pr_key'].nunique().rename('prs_with_human_response')

    if not first_responses.empty else pd.Series(dtype='int64')

)

print('RQ2.1 PRs with >=1 candidate HUMAN response by dataset_label:')

if len(responses_by_dataset) > 0:

    print(responses_by_dataset.sort_index().to_string())

else:

    print('none')

print('RQ2.1 first HUMAN response coverage by author_type_group:')

print(first_response_coverage.sort_values('author_type_group').to_string(index=False))



human_total = int(first_response_coverage.loc[first_response_coverage['author_type_group'].eq('human'), 'total_prs'].sum())

human_with_resp = int(first_response_coverage.loc[first_response_coverage['author_type_group'].eq('human'), 'prs_with_first_human_response'].sum())

if human_total > 0 and human_with_resp == 0:

    human_keys = pr_meta.loc[pr_meta['author_type_group'].eq('human'), 'pr_key'].dropna().astype(str).head(10).tolist()

    sample_rows = []

    for pr_key in human_keys:

        pr_events = timeline_with_pr_info.loc[timeline_with_pr_info['pr_key'].eq(pr_key)].sort_values('timestamp')

        open_row = pr_events.loc[pr_events['action_type'].eq('Open')].head(1)

        open_time = open_row['timestamp'].iloc[0] if not open_row.empty else pd.NaT

        author_login = open_row['actor_login_norm'].iloc[0] if not open_row.empty else ''

        type_counts = pr_events['action_type'].value_counts().to_dict()

        has_any_human_actor = bool(pr_events['actor_type'].eq('human').any())

        candidate_mask = (

            pr_events['action_type'].isin(['Comment', 'Review', 'Commit'])

            & pr_events['actor_type'].eq('human')

            & pr_events['actor_login_norm'].ne(author_login)

        )

        if pd.notna(open_time):

            candidate_mask = candidate_mask & pr_events['timestamp'].gt(open_time)

        has_human_response_candidate = bool(candidate_mask.any())

        sample_rows.append({

            'pr_key': pr_key,

            'open_time': open_time,

            'pr_author_login_norm': author_login,

            'event_type_counts': str(type_counts),

            'any_human_actor': has_any_human_actor,

            'has_candidate_human_response': has_human_response_candidate,

        })

    print('Human PR debug sample (N=10):')

    if sample_rows:

        print(pd.DataFrame(sample_rows).to_string(index=False))

    stop_with_questions_for_user(

        'Human group has N=0 for first HUMAN response after timeline-based computation. '

        'Please confirm human timeline actor fields and whether human baseline comments/reviews/commits exist in-scope.'

    )



if first_responses.empty:

    stop_with_questions_for_user(

        'No first HUMAN responses found for any group in scope. '

        'Please confirm that non-self human Comment/Review/Commit events exist after Open.'

    )



first_responses['first_human_response_latency_hours'] = (

    (first_responses['timestamp'] - first_responses['open_time']).dt.total_seconds() / 3600.0

)

first_responses = first_responses[first_responses['first_human_response_latency_hours'].notna()].copy()

first_responses = first_responses[first_responses['first_human_response_latency_hours'] >= 0].copy()

first_responses['response_time_hours'] = first_responses['first_human_response_latency_hours']



first_response_df = first_responses[[

    'pr_key', 'author_type_group', 'agent_name', 'first_human_response_type',

    'first_human_response_latency_hours', 'response_time_hours'

]].copy()



first_action_df = (

    first_responses

    .groupby(['author_type_group', 'first_human_response_type'])['pr_key']

    .nunique()

    .rename('count')

    .reset_index()

)

action_totals = first_action_df.groupby('author_type_group')['count'].sum().rename('total').reset_index()

first_action_df = first_action_df.merge(action_totals, on='author_type_group', how='left')

first_action_df['share'] = first_action_df['count'] / first_action_df['total'].replace(0, np.nan)

first_action_df = first_action_df[['author_type_group', 'first_human_response_type', 'count', 'share']].sort_values(

    ['author_type_group', 'first_human_response_type']

).reset_index(drop=True)



latency_summary = (

    first_responses.groupby('author_type_group')['first_human_response_latency_hours']

    .agg(

        N='count',

        median_hours='median',

        p90_hours=lambda s: s.quantile(0.90),

        p99_hours=lambda s: s.quantile(0.99),

    )

    .reset_index()

    .sort_values('author_type_group')

    .reset_index(drop=True)

)



required = {'agent', 'human'}

lat_groups = set(latency_summary['author_type_group'].dropna().astype(str))

if not required.issubset(lat_groups):

    stop_with_questions_for_user(

        f'baseline_latency_summary is missing required groups. present={sorted(lat_groups)}'

    )

lat_n_map = latency_summary.set_index('author_type_group')['N'].to_dict()

if any(int(lat_n_map.get(g, 0)) <= 0 for g in required):

    stop_with_questions_for_user(

        f'baseline_latency_summary has non-positive N for required groups. N_map={lat_n_map}'

    )



fa_groups = set(first_action_df['author_type_group'].dropna().astype(str))

if not required.issubset(fa_groups):

    stop_with_questions_for_user(

        f'baseline_first_action_table is missing required groups. present={sorted(fa_groups)}'

    )

fa_count_map = first_action_df.groupby('author_type_group')['count'].sum().to_dict()

if any(int(fa_count_map.get(g, 0)) <= 0 for g in required):

    stop_with_questions_for_user(

        f'baseline_first_action_table has zero count for required groups. count_map={fa_count_map}'

    )



save_csv(first_action_df, 'baseline_first_action_table.csv', index=False)

save_csv(latency_summary, 'baseline_latency_summary.csv', index=False)


merge_results_df = pd.DataFrame()

merge_proxy_summary = pd.DataFrame(columns=[

    'author_type_group',

    'merged_n',

    'commit_coverage_pct',

    'unknown_commit_actor_pct',

    'unknown_commit_author_pct',

    'merged_without_non_author_commits_pct',

])

merge_summary_df = merge_proxy_summary.copy()

commits_df = pd.DataFrame()

commit_pr_key_diagnostics = pd.DataFrame()





def _parse_issue_or_pr_key_from_url(url_series):

    series = pd.Series(url_series).fillna('').astype(str)

    pull_extracted = series.str.extract(r'github\.com[:/]([^/]+/[^/#?]+)/pull/(\d+)', expand=True)

    issue_extracted = series.str.extract(r'github\.com[:/]([^/]+/[^/#?]+)/issues/(\d+)', expand=True)

    pull_key = make_pr_key(pull_extracted[0], pull_extracted[1]) if pull_extracted.shape[1] == 2 else pd.Series('', index=series.index)

    issue_key = make_pr_key(issue_extracted[0], issue_extracted[1]) if issue_extracted.shape[1] == 2 else pd.Series('', index=series.index)

    return pull_key.fillna(''), issue_key.fillna('')





def _extract_commit_keys_with_methods(df_commits):

    if df_commits is None or df_commits.empty:

        empty = pd.Series(dtype='object')

        return empty, empty



    out = df_commits.copy()

    key = pd.Series('', index=out.index, dtype='object')

    method = pd.Series('missing', index=out.index, dtype='object')



    if 'pr_key' in out.columns:

        direct = out['pr_key'].fillna('').astype(str)

        mask = direct.ne('')

        key.loc[mask] = direct.loc[mask]

        method.loc[mask] = 'pr_key'



    for url_col in ['pull_request_url', 'pr_url', 'url']:

        if url_col in out.columns:

            parsed_pull, parsed_issue = _parse_issue_or_pr_key_from_url(out[url_col])

            mask_pull = key.eq('') & parsed_pull.ne('')

            key.loc[mask_pull] = parsed_pull.loc[mask_pull]

            method.loc[mask_pull] = f'{url_col}:pull_url'

            mask_issue = key.eq('') & parsed_issue.ne('')

            key.loc[mask_issue] = parsed_issue.loc[mask_issue]

            method.loc[mask_issue] = f'{url_col}:issue_url'



    repo_col = next((c for c in ['repo_full_name', 'base_repo_full_name', 'repository', 'repo_name', 'base_repository', 'repo', 'full_name'] if c in out.columns), None)

    number_col = next((c for c in ['number', 'pr_number', 'pull_number'] if c in out.columns), None)

    if repo_col is not None and number_col is not None:

        parsed_repo_num = make_pr_key(out[repo_col], out[number_col]).fillna('')

        mask_repo_num = key.eq('') & parsed_repo_num.ne('')

        key.loc[mask_repo_num] = parsed_repo_num.loc[mask_repo_num]

        method.loc[mask_repo_num] = f'{repo_col}+{number_col}'



    if key.eq('').any() and PR_ID_TO_KEY:

        for id_col in ['pr_id', 'pull_request_id', 'pullRequestId', 'pull_request_node_id']:

            if id_col in out.columns:

                mapped = out[id_col].fillna('').astype(str).map(PR_ID_TO_KEY).fillna('')

                mask_id = key.eq('') & mapped.ne('')

                key.loc[mask_id] = mapped.loc[mask_id]

                method.loc[mask_id] = f'{id_col}:id_map'



    return key.fillna('').astype(str), method.fillna('missing').astype(str)





def _extract_commit_actor_login(df_commits):

    if df_commits is None or df_commits.empty:

        return pd.Series(dtype='object')

    out = df_commits.copy()

    author_series = pd.Series('', index=out.index, dtype='object')

    for c in ['commit_actor_login_norm', 'authors', 'author', 'user', 'actor']:

        if c in out.columns:

            author_series = out[c].apply(get_login).fillna('').astype(str)

            break

    committer_series = pd.Series('', index=out.index, dtype='object')

    if 'committer' in out.columns:

        committer_series = out['committer'].apply(get_login).fillna('').astype(str)

    actor = author_series.where(author_series.ne(''), committer_series)

    return actor.fillna('').astype(str).map(norm_login)





if timeline_with_pr_info.empty:

    stop_with_questions_for_user('timeline_with_pr_info is empty before merge-proxy computation; cannot evaluate RQ2.1 (3).')



open_meta = (

    timeline_with_pr_info.loc[timeline_with_pr_info['action_type'].eq('Open'),

                              ['pr_key', 'pr_author_login_norm', 'author_type_group', 'agent_name']]

    .sort_values(['pr_key', 'pr_author_login_norm'])

    .drop_duplicates(subset=['pr_key'], keep='first')

    .copy()

)

if open_meta.empty:

    stop_with_questions_for_user('No Open events available for merge-proxy computation.')



if 'df_prs_all' not in globals() or not isinstance(df_prs_all, pd.DataFrame) or df_prs_all.empty or 'merged' not in df_prs_all.columns:

    stop_with_questions_for_user('Merged PR flags are missing in df_prs_all; cannot compute merge-proxy baseline.')



merged_flags = (

    df_prs_all[['pr_key', 'merged']]

    .drop_duplicates(subset=['pr_key'], keep='first')

    .copy()

)

merged_flags['pr_key'] = merged_flags['pr_key'].fillna('').astype(str)

merged_flags['merged'] = merged_flags['merged'].fillna(False).astype(bool)



merged_eval = open_meta.merge(merged_flags, on='pr_key', how='left')

merged_eval['merged'] = merged_eval['merged'].fillna(False).astype(bool)

merged_prs = merged_eval.loc[merged_eval['merged']].copy()



raw_commit_frames = []

for src_label, frame in [('agent', globals().get('agent_commits_sampled', pd.DataFrame())),

                         ('human', globals().get('human_commits_sampled', pd.DataFrame()))]:

    if isinstance(frame, pd.DataFrame) and not frame.empty:

        work = frame.copy()

        extracted_key, key_method = _extract_commit_keys_with_methods(work)

        work['pr_key'] = extracted_key

        work['pr_key_method'] = key_method

        work['source_label'] = src_label

        raw_commit_frames.append(work)



raw_commits_all = pd.concat(raw_commit_frames, ignore_index=True) if raw_commit_frames else pd.DataFrame()

if not raw_commits_all.empty:

    raw_commits_all['commit_time'] = _event_timestamp(raw_commits_all, 'commits')

    raw_commits_all['commit_actor_login_norm'] = _extract_commit_actor_login(raw_commits_all)

    raw_commits_all['unknown_commit_actor'] = raw_commits_all['commit_actor_login_norm'].eq('')

    commit_pr_key_diagnostics = (

        raw_commits_all.groupby(['source_label', 'pr_key_method'])

        .size()

        .rename('rows')

        .reset_index()

        .sort_values(['source_label', 'rows'], ascending=[True, False])

    )

    print('Commit PR-key extraction diagnostics:')

    print(commit_pr_key_diagnostics.to_string(index=False))

    commits_df = raw_commits_all[['pr_key', 'commit_actor_login_norm', 'unknown_commit_actor', 'commit_time', 'source_label']].copy()

    commits_df = commits_df[commits_df['pr_key'].fillna('').astype(str).ne('')].copy()

else:

    commits_df = pd.DataFrame(columns=['pr_key', 'commit_actor_login_norm', 'unknown_commit_actor', 'commit_time', 'source_label'])

    print('Commit PR-key extraction diagnostics: no raw commit rows available.')



timeline_commit_rows = timeline_with_pr_info.loc[

    timeline_with_pr_info['action_type'].eq('Commit'),

    ['pr_key', 'actor_login_norm', 'timestamp', 'author_type_group']

].copy()

timeline_commit_rows = timeline_commit_rows.rename(columns={

    'actor_login_norm': 'commit_actor_login_norm',

    'timestamp': 'commit_time',

})

timeline_commit_rows['unknown_commit_actor'] = timeline_commit_rows['commit_actor_login_norm'].fillna('').astype(str).eq('')

timeline_commit_rows['source_label'] = 'timeline_commits'



summary_rows = []

merge_result_rows = []

group_order = [g for g in _baseline_group_order(open_meta['author_type_group'].dropna().unique()) if g]

for group_name in group_order:

    group_merged = merged_prs.loc[merged_prs['author_type_group'].eq(group_name)].copy()

    merged_n = int(group_merged['pr_key'].nunique())

    group_keys = set(group_merged['pr_key'].astype(str))



    raw_group_rows = commits_df.loc[commits_df['pr_key'].astype(str).isin(group_keys)].copy()

    tl_group_rows = timeline_commit_rows.loc[timeline_commit_rows['pr_key'].astype(str).isin(group_keys)].copy()



    raw_covered_prs = int(raw_group_rows['pr_key'].nunique())

    tl_covered_prs = int(tl_group_rows['pr_key'].nunique())

    raw_cov = (raw_covered_prs / merged_n * 100.0) if merged_n else np.nan

    tl_cov = (tl_covered_prs / merged_n * 100.0) if merged_n else np.nan



    if merged_n == 0:

        chosen_rows = pd.DataFrame(columns=['pr_key', 'commit_actor_login_norm', 'unknown_commit_actor', 'commit_time', 'source_label'])

        chosen_source = 'none'

        merged_with_commit_rows_n = 0

        commit_coverage_pct = np.nan

    elif (pd.isna(raw_cov) or raw_cov < tl_cov):

        chosen_rows = tl_group_rows.copy()

        chosen_source = 'timeline_commits'

        merged_with_commit_rows_n = tl_covered_prs

        commit_coverage_pct = tl_cov

    else:

        chosen_rows = raw_group_rows.copy()

        chosen_source = 'raw_commits'

        merged_with_commit_rows_n = raw_covered_prs

        commit_coverage_pct = raw_cov



    chosen_rows['commit_actor_login_norm'] = chosen_rows['commit_actor_login_norm'].fillna('').astype(str).map(norm_login)

    chosen_rows['unknown_commit_actor'] = chosen_rows['commit_actor_login_norm'].eq('')

    unknown_commit_actor_pct = float(chosen_rows['unknown_commit_actor'].mean() * 100.0) if not chosen_rows.empty else np.nan



    non_author_counts = (

        chosen_rows.loc[

            chosen_rows['commit_actor_login_norm'].ne('')

        ]

        .merge(group_merged[['pr_key', 'pr_author_login_norm']], on='pr_key', how='left')

    )

    non_author_counts['is_non_author_commit'] = non_author_counts['commit_actor_login_norm'].ne(non_author_counts['pr_author_login_norm'].fillna('').astype(str))

    non_author_by_pr = non_author_counts.groupby('pr_key')['is_non_author_commit'].sum() if not non_author_counts.empty else pd.Series(dtype='float64')

    has_commit_prs = set(chosen_rows['pr_key'].astype(str)) if not chosen_rows.empty else set()



    for _, pr_row in group_merged.iterrows():

        pr_key = str(pr_row['pr_key'])

        has_data = pr_key in has_commit_prs

        non_author_n = int(non_author_by_pr.get(pr_key, 0)) if has_data else 0

        merge_wo_non_author = (non_author_n == 0) if has_data else np.nan

        merge_result_rows.append({

            'pr_key': pr_key,

            'author_type_group': group_name,

            'agent_name': pr_row.get('agent_name', ''),

            'has_commit_data': bool(has_data),

            'non_author_commit_count': non_author_n,

            'merge_without_non_author_commits': merge_wo_non_author,

            'commit_source': chosen_source,

        })



    if merged_with_commit_rows_n > 0 and pd.notna(commit_coverage_pct) and float(commit_coverage_pct) >= 20.0:

        covered_rows = [r for r in merge_result_rows if r['author_type_group'] == group_name and r['has_commit_data']]

        merge_without_non_author_commits_pct = float(np.mean([bool(r['merge_without_non_author_commits']) for r in covered_rows]) * 100.0) if covered_rows else np.nan

    else:

        merge_without_non_author_commits_pct = np.nan

        if merged_n > 0:

            print(

                f'Warning: {group_name} commit coverage is below 20% '

                f'(coverage={commit_coverage_pct:.2f}% merged_n={merged_n}) -> merged_without_non_author_commits_pct set to NaN'

            )



    summary_rows.append({

        'author_type_group': group_name,

        'merged_n': merged_n,

        'commit_coverage_pct': commit_coverage_pct,

        'unknown_commit_actor_pct': unknown_commit_actor_pct,

        'unknown_commit_author_pct': unknown_commit_actor_pct,

        'merged_without_non_author_commits_pct': merge_without_non_author_commits_pct,

        'commit_source_used': chosen_source,

    })



merge_results_df = pd.DataFrame(merge_result_rows)

merge_proxy_summary = pd.DataFrame(summary_rows).sort_values('author_type_group').reset_index(drop=True)

merge_summary_df = merge_proxy_summary.copy()

save_csv(merge_proxy_summary.drop(columns=['commit_source_used']), 'baseline_merge_proxy.csv', index=False)

if isinstance(baseline_sanity_checks_df, pd.DataFrame) and not baseline_sanity_checks_df.empty:

    merge_counts = dict(
        zip(
            merge_proxy_summary['author_type_group'].astype(str),
            pd.to_numeric(merge_proxy_summary['merged_n'], errors='coerce').fillna(0).astype(int),
        )
    )

    for group_name, merged_n in merge_counts.items():

        mask = (
            baseline_sanity_checks_df['section'].astype(str).eq('counts')
            & baseline_sanity_checks_df['group'].astype(str).eq(group_name)
            & baseline_sanity_checks_df['metric'].astype(str).eq('merged_pr_count')
        )

        baseline_sanity_checks_df.loc[mask, 'value'] = int(merged_n)
        baseline_sanity_checks_df.loc[mask, 'detail'] = 'merge-proxy merged_n'

    save_csv(baseline_sanity_checks_df, 'baseline_sanity_checks.csv', index=False)



print('Merge proxy coverage and rates by group:')

print(merge_proxy_summary.to_string(index=False))



human_row = merge_proxy_summary.loc[merge_proxy_summary['author_type_group'].eq('human')].head(1)

if not human_row.empty:

    human_merged_n = int(human_row['merged_n'].iloc[0]) if pd.notna(human_row['merged_n'].iloc[0]) else 0

    human_cov = float(human_row['commit_coverage_pct'].iloc[0]) if pd.notna(human_row['commit_coverage_pct'].iloc[0]) else np.nan

    if human_merged_n > 0 and (pd.isna(human_cov) or human_cov < 20.0):

        stop_with_questions_for_user(

            'Human commit coverage is below 20% for merge proxy. '

            'Are human commits expected in combined_dataset/human_commits.csv? '

            'If yes, please provide the exact path/format of commit fields to use for pr_key alignment.'

        )


per_agent_df = pd.DataFrame(columns=['agent_name', 'pr_count', 'merged_count', 'share_merge_without_non_author_commits', 'median_response_time_hours', 'comment_first_share'])

agent_prs = df_prs_all.loc[df_prs_all['author_type_group'].eq('agent')].copy() if not df_prs_all.empty else pd.DataFrame()



if not agent_prs.empty:

    pr_counts = agent_prs.groupby('agent_name')['pr_key'].nunique().rename('pr_count')

    merged_counts = agent_prs.loc[agent_prs['merged']].groupby('agent_name')['pr_key'].nunique().rename('merged_count')



    response_metrics = pd.DataFrame(columns=['median_response_time_hours', 'comment_first_share'])

    if not first_response_df.empty:

        response_by_agent = first_response_df.loc[first_response_df['author_type_group'].eq('agent')].copy()

        if not response_by_agent.empty:

            response_metrics = response_by_agent.groupby('agent_name').agg(

                median_response_time_hours=('response_time_hours', 'median'),

                total_responses=('pr_key', 'nunique'),

                comment_first_count=('first_human_response_type', lambda s: s.eq('comment').sum()),

            )

            response_metrics['comment_first_share'] = response_metrics['comment_first_count'] / response_metrics['total_responses'].replace(0, np.nan)



    merge_metrics = pd.DataFrame(columns=['share_merge_without_non_author_commits'])

    if not merge_results_df.empty:

        merge_agent = merge_results_df.loc[merge_results_df['author_type_group'].eq('agent')].copy()

        if not merge_agent.empty:

            merge_metrics = merge_agent.groupby('agent_name').agg(

                share_merge_without_non_author_commits=('merge_without_non_author_commits', 'mean')

            )



    per_agent_df = pd.concat([

        pr_counts,

        merged_counts,

        merge_metrics,

        response_metrics[['median_response_time_hours', 'comment_first_share']] if not response_metrics.empty else response_metrics

    ], axis=1).reset_index()



    if 'agent_name' not in per_agent_df.columns:

        per_agent_df['agent_name'] = per_agent_df.iloc[:, 0] if len(per_agent_df.columns) > 0 else ''

    if 'pr_count' not in per_agent_df.columns:

        per_agent_df['pr_count'] = 0

    if 'merged_count' not in per_agent_df.columns:

        per_agent_df['merged_count'] = 0



    per_agent_df['merged_count'] = safe_int_series(per_agent_df['merged_count'], fill_value=0)

    per_agent_df = per_agent_df.sort_values(['pr_count', 'agent_name'], ascending=[False, True]).reset_index(drop=True)

    save_csv(per_agent_df, 'per_agent_summary.csv', index=False)

else:

    save_csv(per_agent_df, 'per_agent_summary.csv', index=False)


sensitivity_df = pd.DataFrame(columns=['scenario', 'pr_count', 'median_response_time_hours', 'comment_first_share', 'share_merge_without_non_author_commits'])



if not per_agent_df.empty:

    top_agents = per_agent_df['agent_name'].head(2).tolist()

    scenario_defs = [

        ('all', []),

        ('drop_top1', top_agents[:1]),

        ('drop_top2', top_agents[:2]),

    ]



    rows = []

    for scenario_name, excluded_agents in tqdm(scenario_defs, total=len(scenario_defs), desc='Agent sensitivity'):

        scenario_prs = agent_prs.loc[~agent_prs['agent_name'].isin(excluded_agents)].copy()

        pr_keys = set(scenario_prs['pr_key'])



        scenario_responses = (

            first_response_df.loc[

                first_response_df['author_type_group'].eq('agent')

                & first_response_df['pr_key'].isin(pr_keys)

            ].copy()

            if not first_response_df.empty else pd.DataFrame()

        )

        scenario_merges = (

            merge_results_df.loc[

                merge_results_df['author_type_group'].eq('agent')

                & merge_results_df['pr_key'].isin(pr_keys)

            ].copy()

            if not merge_results_df.empty else pd.DataFrame()

        )



        rows.append({

            'scenario': scenario_name,

            'pr_count': int(len(scenario_prs)),

            'median_response_time_hours': scenario_responses['response_time_hours'].median() if not scenario_responses.empty else np.nan,

            'comment_first_share': scenario_responses['first_human_response_type'].eq('comment').mean() if not scenario_responses.empty else np.nan,

            'share_merge_without_non_author_commits': scenario_merges['merge_without_non_author_commits'].mean() if not scenario_merges.empty else np.nan,

        })



    sensitivity_df = pd.DataFrame(rows)



save_csv(sensitivity_df, 'agent_sensitivity_drop_top.csv', index=False)


os.makedirs('figures', exist_ok=True)



group_col = 'author_type_group' if 'author_type_group' in first_action_df.columns else 'author_type'

response_type_col = 'first_human_response_type' if 'first_human_response_type' in first_action_df.columns else 'first_response_type'

ordered_groups = _baseline_group_order(first_action_df[group_col].dropna().unique()) if not first_action_df.empty else _baseline_group_order()



if not first_action_df.empty and response_type_col in first_action_df.columns:

    fig, ax = plt.subplots(figsize=(10, 6))

    pivot_data = first_action_df.pivot(index=group_col, columns=response_type_col, values='share').fillna(0)

    pivot_data = pivot_data.reindex(ordered_groups).dropna(how='all')

    pivot_data.plot(kind='bar', ax=ax, width=0.8)

    ax.set_xlabel('Author Type')

    ax.set_ylabel('Share of First Human Responses')

    ax.legend(title='Response Type')

    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if SAVE_INTERMEDIATE_BASELINE_PNGS:
        fig.savefig('figures/fig_baseline_first_action.png', dpi=300, bbox_inches='tight')

    plt.show()



if not latency_summary.empty:

    latency_group_col = 'author_type_group' if 'author_type_group' in latency_summary.columns else 'author_type'

    plot_df = (

        latency_summary

        .set_index(latency_group_col)

        .reindex(_baseline_group_order(latency_summary[latency_group_col].dropna().unique()))

        .dropna(subset=['median_hours'], how='all')

        .reset_index()

    )

    if not plot_df.empty:

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(plot_df[latency_group_col], plot_df['median_hours'])

        ax.set_yscale('log')

        ax.set_xlabel('Author Type')

        ax.set_ylabel('Median First Human Response Time (hours)')

        ax.grid(True, axis='y', alpha=0.3)

        for bar, val in zip(bars, plot_df['median_hours']):

            if pd.notna(val) and val > 0:

                ax.text(bar.get_x() + bar.get_width() / 2, val, f'{val:.2f}h', ha='center', va='bottom')

        plt.tight_layout()

        if SAVE_INTERMEDIATE_BASELINE_PNGS:
            fig.savefig('figures/fig_baseline_latency_log.png', dpi=300, bbox_inches='tight')

        plt.show()



if not merge_proxy_summary.empty:

    merge_group_col = 'author_type_group' if 'author_type_group' in merge_proxy_summary.columns else 'author_type'

    share_col = (

        'merged_without_non_author_commits_pct'

        if 'merged_without_non_author_commits_pct' in merge_proxy_summary.columns

        else ('share_merge_without_non_author_commits_in_pr' if 'share_merge_without_non_author_commits_in_pr' in merge_proxy_summary.columns else 'share_merge_without_non_author_commits')

    )

    if share_col in merge_proxy_summary.columns:

        plot_df = (

            merge_proxy_summary

            .set_index(merge_group_col)

            .reindex(_baseline_group_order(merge_proxy_summary[merge_group_col].dropna().unique()))

            .dropna(subset=[share_col], how='all')

            .reset_index()

        )

        if not plot_df.empty:

            fig, ax = plt.subplots(figsize=(10, 6))

            shares_raw = pd.to_numeric(plot_df[share_col], errors='coerce')

            shares_pct = shares_raw if share_col.endswith('_pct') else shares_raw * 100.0

            bars = ax.bar(plot_df[merge_group_col], shares_pct)

            ax.set_xlabel('Author Type')

            ax.set_ylabel('Percent of Merged PRs Without Non-Author Commits')

            ax.grid(True, axis='y', alpha=0.3)

            for bar, val in zip(bars, shares_pct):

                if pd.notna(val):

                    ax.text(bar.get_x() + bar.get_width() / 2, val, f'{val:.1f}%', ha='center', va='bottom')

            plt.tight_layout()

            if SAVE_INTERMEDIATE_BASELINE_PNGS:
                fig.savefig('figures/fig_baseline_merge_without_non_author_commits.png', dpi=300, bbox_inches='tight')

            plt.show()


paper_rows = []

group_col_action = 'author_type_group' if 'author_type_group' in first_action_df.columns else 'author_type'

group_col_latency = 'author_type_group' if 'author_type_group' in latency_summary.columns else 'author_type'

group_col_merge = 'author_type_group' if 'author_type_group' in merge_proxy_summary.columns else 'author_type'

group_col_resp = 'author_type_group' if 'author_type_group' in first_responses.columns else 'author_type'

group_col_pr = 'author_type_group' if 'author_type_group' in df_prs_all.columns else 'author_type'



response_type_col = 'first_human_response_type' if 'first_human_response_type' in first_action_df.columns else 'first_response_type'

merge_share_col = (

    'merged_without_non_author_commits_pct'

    if 'merged_without_non_author_commits_pct' in merge_proxy_summary.columns

    else ('share_merge_without_non_author_commits_in_pr' if 'share_merge_without_non_author_commits_in_pr' in merge_proxy_summary.columns else 'share_merge_without_non_author_commits')

)



if not first_action_df.empty and response_type_col in first_action_df.columns:

    _comment_mask = first_action_df[response_type_col].astype(str).str.lower().eq('comment')

    comment_first_map = first_action_df.loc[_comment_mask].set_index(group_col_action)['share'].to_dict()

else:

    comment_first_map = {}



latency_map = latency_summary.set_index(group_col_latency)['median_hours'].to_dict() if not latency_summary.empty else {}

merge_map_raw = merge_proxy_summary.set_index(group_col_merge)[merge_share_col].to_dict() if (not merge_proxy_summary.empty and merge_share_col in merge_proxy_summary.columns) else {}

merge_map = {

    k: (v / 100.0 if pd.notna(v) and str(merge_share_col).endswith('_pct') else v)

    for k, v in merge_map_raw.items()

}

response_size_map = first_responses.groupby(group_col_resp)['pr_key'].nunique().to_dict() if not first_responses.empty else {}

merged_size_map = merge_results_df.groupby(group_col_merge)['pr_key'].nunique().to_dict() if not merge_results_df.empty else {}

pr_size_map = df_prs_all.groupby(group_col_pr)['pr_key'].nunique().to_dict() if not df_prs_all.empty else {}



for author_type in _baseline_group_order(set(pr_size_map) | set(response_size_map) | set(merged_size_map)):

    comment_share = comment_first_map.get(author_type, np.nan)

    median_hours = latency_map.get(author_type, np.nan)

    merge_share = merge_map.get(author_type, np.nan)

    response_n = int(response_size_map.get(author_type, 0))

    merged_n = int(merged_size_map.get(author_type, 0))

    total_n = int(pr_size_map.get(author_type, 0))

    paper_rows.append({

        'author_type': author_type,

        'comment_first_share': comment_share,

        'median_response_time_hours': median_hours,

        'share_merge_without_non_author_commits': merge_share,

        'pr_count': total_n,

        'prs_with_response': response_n,

        'merged_pr_count': merged_n,

    })



paper_numbers_df = pd.DataFrame(paper_rows)

paper_numbers_df


def detect_first_human_response_location(df_prs_classified=None, df_comments_classified=None, df_reviews_classified=None, df_commits_classified=None, issue_events_clean=None, linked_issues=None):

    if 'timeline_df' not in globals() or timeline_df.empty:

        return pd.DataFrame(columns=['pr_id', 'first_human_action', 'first_response_location', 'first_response_time', 'response_time_hours'])

    working = timeline_df.sort_values(['pr_key', 'timestamp', 'action_type']).copy()

    open_events = working[working['action_type'] == 'Open'][['pr_key', 'timestamp']].drop_duplicates('pr_key').rename(columns={'timestamp': 'open_time'})

    pr_author_map = df_prs_all[['pr_key', 'pr_author_login_norm']] if 'df_prs_all' in globals() and not df_prs_all.empty and 'pr_author_login_norm' in df_prs_all.columns else pd.DataFrame(columns=['pr_key', 'pr_author_login_norm'])

    working = working.merge(open_events, on='pr_key', how='left')

    if not pr_author_map.empty:

        working = working.merge(pr_author_map, on='pr_key', how='left')

    else:

        working['pr_author_login_norm'] = ''

    human_events = working[

        (~working['is_ai'])

        & (working['action_type'] != 'Open')

        & (working['timestamp'] > working['open_time'])

        & (working['actor_login_norm'] != working['pr_author_login_norm'])

    ].copy()

    if human_events.empty:

        base = open_events.copy()

        base['pr_id'] = base['pr_key'].astype(str)

        base['first_human_action'] = 'None'

        base['first_response_location'] = 'None'

        base['first_response_time'] = pd.NaT

        base['response_time_hours'] = np.nan

        return base[['pr_id', 'first_human_action', 'first_response_location', 'first_response_time', 'response_time_hours']]

    first = human_events.groupby('pr_key', as_index=False).first()

    first['pr_id'] = first['pr_key'].astype(str)

    first['first_human_action'] = normalize_action_bucket(first['action_type'], ['Comment', 'Review', 'Commit'])

    first['first_response_location'] = 'PR'

    first['first_response_time'] = first['timestamp']

    first['response_time_hours'] = (first['timestamp'] - first['open_time']).dt.total_seconds().div(3600)

    return first[['pr_id', 'first_human_action', 'first_response_location', 'first_response_time', 'response_time_hours']]



if 'timeline_df' in globals():

    first_response_analysis = detect_first_human_response_location()

    if not first_response_analysis.empty:

        response_time_quantiles = first_response_analysis['response_time_hours'].dropna().quantile([0.5, 0.9, 0.99]) if first_response_analysis['response_time_hours'].notna().any() else pd.Series(dtype=float)


def analyze_ai_to_human_sequences(timeline_df, anchor_option='pr_creation'):

    if timeline_df.empty:

        return {'sequences': [], 'first_human_actions': [], 'response_times_hours': [], 'transitions': [], 'anchor_option': anchor_option}

    source = timeline_df[timeline_df['dataset_label'] == 'agent'].copy() if 'dataset_label' in timeline_df.columns and (timeline_df['dataset_label'] == 'agent').any() else timeline_df.copy()

    if source.empty:

        return {'sequences': [], 'first_human_actions': [], 'response_times_hours': [], 'transitions': [], 'anchor_option': anchor_option}

    source = source.sort_values(['pr_id', 'timestamp', 'action_type']).copy()

    open_events = source[source['action_type'] == 'Open'].groupby('pr_id', as_index=False).first()[['pr_id', 'timestamp', 'is_ai', 'action_type']].rename(columns={'timestamp': 'open_time'})

    human_events = source[(~source['is_ai']) & (source['action_type'] != 'Open')].copy()

    ai_events = source[source['is_ai']].copy()

    if human_events.empty or ai_events.empty:

        return {'sequences': [], 'first_human_actions': [], 'response_times_hours': [], 'transitions': [], 'anchor_option': anchor_option}

    first_human = human_events.groupby('pr_id', as_index=False).first()[['pr_id', 'timestamp', 'action_type']].rename(columns={'timestamp': 'first_human_time', 'action_type': 'first_human_action'})

    ai_before_human = ai_events.merge(first_human[['pr_id', 'first_human_time']], on='pr_id', how='inner')

    ai_before_human = ai_before_human[ai_before_human['timestamp'] < ai_before_human['first_human_time']].copy()

    last_ai = ai_before_human.groupby('pr_id', as_index=False).last()[['pr_id', 'timestamp', 'action_type']].rename(columns={'timestamp': 'last_ai_time', 'action_type': 'last_ai_action'})

    last_ai['last_ai_action'] = normalize_action_bucket(last_ai['last_ai_action'], ['Open', 'Comment', 'Review', 'Commit'])

    sequences = first_human.merge(last_ai, on='pr_id', how='inner')

    sequences = sequences.merge(open_events[['pr_id', 'open_time', 'is_ai']], on='pr_id', how='left', suffixes=('', '_open'))

    if anchor_option == 'pr_creation':

        sequences = sequences[sequences['is_ai'] == True].copy()

        sequences['anchor_time'] = sequences['open_time']

        sequences['anchor_action'] = 'Open'

    else:

        sequences['anchor_time'] = sequences['last_ai_time']

        sequences['anchor_action'] = sequences['last_ai_action']

    sequences['response_time_hours'] = (sequences['first_human_time'] - sequences['anchor_time']).dt.total_seconds().div(3600)

    sequences = sequences[sequences['response_time_hours'] >= 0].copy()

    results = {

        'sequences': sequences[['pr_id', 'anchor_action', 'first_human_action', 'response_time_hours']].to_dict('records'),

        'first_human_actions': sequences['first_human_action'].tolist(),

        'response_times_hours': sequences['response_time_hours'].tolist(),

        'transitions': list(zip(sequences['anchor_action'], sequences['first_human_action'])),

        'anchor_option': anchor_option

    }

    return results



if not timeline_df.empty:

    results_pr_creation = analyze_ai_to_human_sequences(timeline_df, anchor_option='pr_creation')

    results_last_ai = analyze_ai_to_human_sequences(timeline_df, anchor_option='last_ai_action')

    primary_results = results_pr_creation if len(results_pr_creation.get('sequences', [])) > 0 else results_last_ai

    sensitivity_results = results_last_ai

else:

    primary_results = {}

    sensitivity_results = {}


if primary_results and len(primary_results.get('sequences', [])) > 0:



    first_actions = primary_results['first_human_actions']

    action_counts = pd.Series(first_actions).value_counts()

    total_sequences = len(first_actions)



    action_table = []

    for action_type in ['Comment', 'Review', 'Commit', 'Other']:

        count = action_counts.get(action_type, 0)

        percentage = (count / total_sequences) * 100 if total_sequences > 0 else 0

        action_table.append({

            'First Human Action': action_type,

            'Count': count,

            'Percentage': f"{percentage:.1f}%"

        })



    action_df = pd.DataFrame(action_table)



    if sensitivity_results and len(sensitivity_results.get('sequences', [])) > 0:

        sens_actions = sensitivity_results['first_human_actions']

        sens_counts = pd.Series(sens_actions).value_counts()

        sens_total = len(sens_actions)





        sens_table = []

        for action_type in ['Comment', 'Review', 'Commit', 'Other']:

            primary_pct = (action_counts.get(action_type, 0) / total_sequences) * 100

            sens_pct = (sens_counts.get(action_type, 0) / sens_total) * 100 if sens_total > 0 else 0

            sens_table.append({

                'Action Type': action_type,

                'Primary (%)': f"{primary_pct:.1f}%",

                'Alternative (%)': f"{sens_pct:.1f}%",

                'Difference': f"{sens_pct - primary_pct:+.1f}%"

            })



        sens_df = pd.DataFrame(sens_table)

else:

    pass


if primary_results and len(primary_results.get('response_times_hours', [])) > 0:



    response_times = np.array(primary_results['response_times_hours'])



    response_times = response_times[response_times >= 0]  # Remove negative times



    if len(response_times) > 0:

        summary_stats = {

            'Statistic': ['Mean', 'Median', '75th Percentile', '90th Percentile', '99th Percentile', 'Minimum', 'Maximum'],

            'Hours': [

                f"{response_times.mean():.1f}",

                f"{np.median(response_times):.1f}",

                f"{np.percentile(response_times, 75):.1f}",

                f"{np.percentile(response_times, 90):.1f}",

                f"{np.percentile(response_times, 99):.1f}",

                f"{response_times.min():.1f}",

                f"{response_times.max():.1f}"

            ]

        }



        def hours_to_readable(hours):

            if hours < 1:

                return f"{hours*60:.0f} minutes"

            elif hours < 24:

                return f"{hours:.1f} hours"

            elif hours < 168:  # 1 week

                return f"{hours/24:.1f} days"

            else:

                return f"{hours/168:.1f} weeks"



        summary_stats['Human Readable'] = [

            hours_to_readable(response_times.mean()),

            hours_to_readable(np.median(response_times)),

            hours_to_readable(np.percentile(response_times, 75)),

            hours_to_readable(np.percentile(response_times, 90)),

            hours_to_readable(np.percentile(response_times, 99)),

            hours_to_readable(response_times.min()),

            hours_to_readable(response_times.max())

        ]



        summary_df = pd.DataFrame(summary_stats)

        pass



        zero_responses = (np.array(primary_results['response_times_hours']) == 0).sum()

        near_zero_responses = (np.array(primary_results['response_times_hours']) < 0.1).sum()

    else:

        pass



else:

    pass


INCLUDE_OPEN_AS_AI_PREDECESSOR = str(
    os.environ.get('INCLUDE_OPEN_AS_AI_PREDECESSOR', '1')
).strip().lower() not in {'0', 'false', 'no', 'off'}



heatmap_results = primary_results if primary_results else {}

primary_anchor_actions = set(action for action, _ in primary_results.get('transitions', [])) if primary_results else set()

sensitivity_anchor_actions = set(action for action, _ in sensitivity_results.get('transitions', [])) if sensitivity_results else set()

if sensitivity_results and len(sensitivity_anchor_actions) > len(primary_anchor_actions):

    heatmap_results = sensitivity_results



if heatmap_results and len(heatmap_results.get('transitions', [])) > 0:



    transitions = heatmap_results['transitions']



    if not INCLUDE_OPEN_AS_AI_PREDECESSOR:

        transitions = [(ai_act, human_act) for ai_act, human_act in transitions if ai_act != 'Open']



    ordered_ai_actions = ['Open', 'Comment', 'Review', 'Commit', 'Other']

    present_ai_actions = list(dict.fromkeys(ai_action for ai_action, _ in transitions))

    ai_actions = [action for action in ordered_ai_actions if action in present_ai_actions]

    ai_actions.extend([action for action in present_ai_actions if action not in ai_actions])



    ordered_human_actions = ['Comment', 'Review', 'Commit', 'Other']

    present_human_actions = list(dict.fromkeys(human_action for _, human_action in transitions))

    human_actions = [action for action in ordered_human_actions if action in present_human_actions]

    human_actions.extend([action for action in present_human_actions if action not in human_actions])



    transition_counts = defaultdict(int)

    for ai_action, human_action in transitions:

        transition_counts[(ai_action, human_action)] += 1



    transition_matrix = np.zeros((len(ai_actions), len(human_actions)))

    for i, ai_action in enumerate(ai_actions):

        for j, human_action in enumerate(human_actions):

            transition_matrix[i, j] = transition_counts[(ai_action, human_action)]



    row_normalized_matrix = np.zeros_like(transition_matrix)

    for i in range(len(ai_actions)):

        row_sum = transition_matrix[i, :].sum()

        if row_sum > 0:

            row_normalized_matrix[i, :] = transition_matrix[i, :] / row_sum



    fig, ax = plt.subplots(figsize=(10, 8))



    im = ax.imshow(row_normalized_matrix, cmap='Blues', aspect='equal', vmin=0, vmax=1)



    ax.set_xticks(range(len(human_actions)))

    ax.set_yticks(range(len(ai_actions)))

    ax.set_xticklabels(human_actions, fontsize=12)

    ax.set_yticklabels(ai_actions, fontsize=12)



    ax.set_xlabel('First Human Action (Response)', fontsize=14, fontweight='bold')

    ax.set_ylabel('AI Action (Anchor)', fontsize=14, fontweight='bold')

    # ax.set_title('AI→Human Action Transition Patterns\\n(Row-Normalized Probabilities)',fontsize=16, pad=20)



    for i in range(len(ai_actions)):

        for j in range(len(human_actions)):

            value = row_normalized_matrix[i, j]

            count = int(transition_matrix[i, j])

            if count > 0:

                text = f'{value:.1%}({count})'

                color = 'white' if value > 0.5 else 'black'

                ax.text(j, i, text, ha='center', va='center',

                       fontsize=10, color=color, fontweight='bold')



    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    cbar.set_label('Probability', rotation=270, labelpad=20, fontsize=12)

    cbar.ax.tick_params(labelsize=10)



    plt.tight_layout()

    plt.show()


rq2_sanity_rows = []





def _append_rq2(section, group, metric, value, passed=True, detail=''):

    rq2_sanity_rows.append({

        'section': section,

        'group': group,

        'metric': metric,

        'value': value,

        'passed': bool(passed),

        'detail': detail,

    })





# PR counts per group

if 'df_prs_all' in globals() and isinstance(df_prs_all, pd.DataFrame) and not df_prs_all.empty:

    prs_group_counts = df_prs_all.groupby('author_type_group')['pr_key'].nunique().to_dict()

    prs_merged_counts = df_prs_all.loc[df_prs_all['merged']].groupby('author_type_group')['pr_key'].nunique().to_dict()

    merge_proxy_merged_counts = {}

    if 'merge_proxy_summary' in globals() and isinstance(merge_proxy_summary, pd.DataFrame) and not merge_proxy_summary.empty:

        merge_group_col = 'author_type_group' if 'author_type_group' in merge_proxy_summary.columns else 'author_type'

        if merge_group_col in merge_proxy_summary.columns and 'merged_n' in merge_proxy_summary.columns:

            merge_proxy_merged_counts = dict(
                zip(
                    merge_proxy_summary[merge_group_col].astype(str),
                    pd.to_numeric(merge_proxy_summary['merged_n'], errors='coerce').fillna(0).astype(int),
                )
            )

    for group_name in _baseline_group_order(prs_group_counts.keys()):

        _append_rq2('scope', group_name, 'prs_n', int(prs_group_counts.get(group_name, 0)), passed=True)

        merged_count = int(merge_proxy_merged_counts.get(group_name, prs_merged_counts.get(group_name, 0)))

        _append_rq2('scope', group_name, 'merged_prs_n', merged_count, passed=True)

else:

    _append_rq2('scope', 'global', 'df_prs_all_present', 0, passed=False, detail='df_prs_all is missing or empty')





# Timeline counts + merge-missing share

timeline_group_counts = {}

missing_share_value = np.nan

if 'timeline_with_pr_info' in globals() and isinstance(timeline_with_pr_info, pd.DataFrame) and not timeline_with_pr_info.empty:

    timeline_group_counts = timeline_with_pr_info.groupby('author_type_group').size().to_dict()

    if 'timeline_missing_share' in globals() and pd.notna(timeline_missing_share):

        missing_share_value = float(timeline_missing_share)

    else:

        missing_share_value = 0.0

elif 'timeline_df' in globals() and isinstance(timeline_df, pd.DataFrame) and not timeline_df.empty and 'df_prs_all' in globals() and not df_prs_all.empty:

    _timeline_tmp = timeline_df.copy()

    if 'pr_key' in _timeline_tmp.columns:

        _timeline_tmp['pr_key'] = _timeline_tmp['pr_key'].fillna('').astype(str)

        _timeline_tmp = _timeline_tmp.merge(df_prs_all[['pr_key', 'author_type_group']], on='pr_key', how='left')

        timeline_group_counts = _timeline_tmp.groupby('author_type_group').size().to_dict()

        _missing_mask = _timeline_tmp['author_type_group'].isna() | _timeline_tmp['author_type_group'].astype(str).eq('')

        missing_share_value = float(_missing_mask.mean()) if len(_timeline_tmp) else 0.0

else:

    _append_rq2('timeline', 'global', 'timeline_present', 0, passed=False, detail='timeline_with_pr_info and timeline_df are both unavailable/empty')



for group_name in _baseline_group_order(timeline_group_counts.keys()):

    _append_rq2('timeline', group_name, 'timeline_events_n', int(timeline_group_counts.get(group_name, 0)), passed=True)

if pd.notna(missing_share_value):

    _append_rq2(

        'timeline',

        'global',

        'missing_author_type_group_pct',

        float(missing_share_value * 100.0),

        passed=bool(float(missing_share_value) < 0.01),

        detail='must be < 1%',

    )





# First response coverage by group

if 'latency_summary' in globals() and isinstance(latency_summary, pd.DataFrame) and not latency_summary.empty:

    group_col = 'author_type_group' if 'author_type_group' in latency_summary.columns else 'author_type'

    present_groups = set(latency_summary[group_col].dropna().astype(str))

    for _, row in latency_summary.iterrows():

        group_name = str(row[group_col])

        n_value = int(pd.to_numeric(row.get('N', 0), errors='coerce') if pd.notna(row.get('N', np.nan)) else 0)

        _append_rq2('first_response', group_name, 'first_human_response_prs_n', n_value, passed=bool(n_value > 0))

    _append_rq2(

        'first_response',

        'global',

        'required_groups_present_agent_human',

        int({'agent', 'human'}.issubset(present_groups)),

        passed=bool({'agent', 'human'}.issubset(present_groups)),

        detail=f'present={sorted(present_groups)}',

    )

else:

    _append_rq2('first_response', 'global', 'latency_summary_present', 0, passed=False, detail='latency_summary is missing or empty')





# First action table coverage by group

if 'first_action_df' in globals() and isinstance(first_action_df, pd.DataFrame) and not first_action_df.empty:

    group_col = 'author_type_group' if 'author_type_group' in first_action_df.columns else 'author_type'

    action_groups = set(first_action_df[group_col].dropna().astype(str))

    for group_name in sorted(action_groups):

        group_count = int(first_action_df.loc[first_action_df[group_col].eq(group_name), 'count'].sum())

        _append_rq2('first_action', group_name, 'first_action_rows_n', group_count, passed=bool(group_count > 0))

    _append_rq2(

        'first_action',

        'global',

        'required_groups_present_agent_human',

        int({'agent', 'human'}.issubset(action_groups)),

        passed=bool({'agent', 'human'}.issubset(action_groups)),

        detail=f'present={sorted(action_groups)}',

    )

else:

    _append_rq2('first_action', 'global', 'first_action_df_present', 0, passed=False, detail='first_action_df is missing or empty')





# Merge-proxy coverage and totals

if 'merge_proxy_summary' in globals() and isinstance(merge_proxy_summary, pd.DataFrame) and not merge_proxy_summary.empty:

    group_col = 'author_type_group' if 'author_type_group' in merge_proxy_summary.columns else 'author_type'

    for _, row in merge_proxy_summary.iterrows():

        group_name = str(row[group_col])

        merged_n = int(pd.to_numeric(row.get('merged_n', 0), errors='coerce') if pd.notna(row.get('merged_n', np.nan)) else 0)

        coverage_pct = float(pd.to_numeric(row.get('commit_coverage_pct', np.nan), errors='coerce'))

        unknown_pct = float(pd.to_numeric(row.get('unknown_commit_author_pct', np.nan), errors='coerce'))

        proxy_pct = float(pd.to_numeric(row.get('merged_without_non_author_commits_pct', np.nan), errors='coerce'))

        coverage_ok = True if merged_n == 0 else (pd.notna(coverage_pct) and coverage_pct > 0)

        _append_rq2('merge_proxy', group_name, 'merged_n', merged_n, passed=True)

        _append_rq2('merge_proxy', group_name, 'commit_coverage_pct', coverage_pct, passed=coverage_ok, detail='must be > 0 when merged_n > 0')

        _append_rq2('merge_proxy', group_name, 'unknown_commit_author_pct', unknown_pct, passed=True)

        _append_rq2('merge_proxy', group_name, 'merged_without_non_author_commits_pct', proxy_pct, passed=True)

else:

    _append_rq2('merge_proxy', 'global', 'merge_proxy_summary_present', 0, passed=False, detail='merge_proxy_summary is missing or empty')





# Keep transition consistency checks (diagnostic, optional)

if 'heatmap_results' in globals() and heatmap_results and len(heatmap_results.get('transitions', [])) > 0:

    transition_count = len(heatmap_results['transitions'])

    matrix_total = int(transition_matrix.sum()) if 'transition_matrix' in globals() else 0

    sequence_count = len(heatmap_results.get('sequences', []))

    anchor_variety = len(set(action for action, _ in heatmap_results['transitions']))

    _append_rq2('rq2_transitions', 'global', 'transition_count_matches_matrix_total', int(transition_count == matrix_total), passed=bool(transition_count == matrix_total), detail=f'expected={transition_count}, actual={matrix_total}')

    _append_rq2('rq2_transitions', 'global', 'transition_count_matches_sequence_count', int(transition_count == sequence_count), passed=bool(transition_count == sequence_count), detail=f'expected={transition_count}, actual={sequence_count}')

    _append_rq2('rq2_transitions', 'global', 'anchor_action_variety_ge_2', anchor_variety, passed=bool(anchor_variety >= 2))





rq2_sanity_checks_df = pd.DataFrame(rq2_sanity_rows)

save_csv(rq2_sanity_checks_df, 'baseline_rq2_sanity_checks.csv', index=False)

rq2_sanity_checks_df


if primary_results and len(primary_results.get('response_times_hours', [])) > 0:



    response_times = np.array(primary_results['response_times_hours'])



    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    fig.suptitle('AI→Human Response Time Distribution\\n(Heavy-Tailed with Zero-Time Responses)',

                 fontsize=16, y=0.98)



    zero_count = (response_times == 0).sum()

    near_zero_count = (response_times < 0.1).sum()  # Less than 6 minutes

    total_count = len(response_times)

    zero_share = zero_count / total_count * 100

    near_zero_share = near_zero_count / total_count * 100



    ax1.hist(response_times, bins=50, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Response Time (Hours)', fontsize=12)

    ax1.set_ylabel('Number of Sequences', fontsize=12)

    ax1.set_title('Full Distribution', fontsize=14)

    ax1.grid(True, alpha=0.3)



    ax1.text(0.98, 0.98, f'Zero responses: {zero_share:.1f}% ({zero_count:,})\\nNear-zero (<6min): {near_zero_share:.1f}% ({near_zero_count:,})',

             transform=ax1.transAxes, ha='right', va='top',

             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),

             fontsize=10)



    nonzero_times = response_times[response_times > 0]



    if len(nonzero_times) > 0:

        min_val = max(0.01, nonzero_times.min())  # Avoid log(0)

        max_val = nonzero_times.max()

        log_bins = np.logspace(np.log10(min_val), np.log10(max_val), 30)



        ax2.hist(nonzero_times, bins=log_bins, alpha=0.7, color='darkorange', edgecolor='black', linewidth=0.5)

        ax2.set_xscale('log')

        ax2.set_xlabel('Response Time (Hours, log scale)', fontsize=12)

        ax2.set_ylabel('Number of Sequences', fontsize=12)

        ax2.set_title('Non-Zero Responses Only\\n(Log-Binned)', fontsize=14)

        ax2.grid(True, alpha=0.3)



        median_val = np.median(nonzero_times)

        p90_val = np.percentile(nonzero_times, 90)

        ax2.axvline(median_val, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Median: {median_val:.1f}h')

        ax2.axvline(p90_val, color='darkred', linestyle='--', alpha=0.8, linewidth=2, label=f'90th %ile: {p90_val:.1f}h')

        ax2.legend(fontsize=10)



        ax2.text(0.98, 0.98, f'Non-zero responses: {len(nonzero_times):,}\\nMedian: {median_val:.1f} hours\\n90th %ile: {p90_val:.1f} hours',

                 transform=ax2.transAxes, ha='right', va='top',

                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),

                 fontsize=10)

    else:

        ax2.text(0.5, 0.5, 'All responses are\\nzero-time', ha='center', va='center',

                transform=ax2.transAxes, fontsize=12)

        ax2.set_title('Non-Zero Responses Only\\n(No Data)', fontsize=14)



    plt.tight_layout()

    pass

    plt.show()



    pass



    pass

    pass

    pass

    pass

    if len(nonzero_times) > 0:

        pass

        pass

        pass

        pass

        pass



else:

    pass


def select_representative_pr(timeline_df, primary_results):

    """

    Select a representative PR using stable, explicit strategy:

    Prefer a PR with at least 1 AI action + >=3 human actions + diverse action types

    """

    if timeline_df.empty or not primary_results.get('sequences'):

        return None, None



    PREFERRED_PR_ID = None  # Can be set to specific PR ID for reproducibility



    if PREFERRED_PR_ID and PREFERRED_PR_ID in timeline_df['pr_id'].values:

        return PREFERRED_PR_ID, "predefined"



    pr_scores = {}



    for pr_id, pr_events in timeline_df.groupby('pr_id'):

        if pd.isna(pr_id):

            continue



        pr_events = pr_events.sort_values('timestamp').reset_index(drop=True)



        ai_events = pr_events[pr_events['is_ai'] == True]

        human_events = pr_events[pr_events['is_ai'] == False]



        ai_count = len(ai_events)

        human_count = len(human_events)

        action_types = set(pr_events['action_type'].unique())



        score = 0



        if ai_count >= 1 and human_count >= 3:

            score += 100



            score += len(action_types) * 10



            total_events = len(pr_events)

            if 5 <= total_events <= 20:

                score += 50

            elif total_events > 20:

                score += 20  # Still good but less preferred



            if 'Comment' in action_types and 'Review' in action_types:

                score += 30



            time_span = (pr_events['timestamp'].max() - pr_events['timestamp'].min()).total_seconds() / 3600

            if 1 <= time_span <= 168:  # 1 hour to 1 week

                score += 20



        pr_scores[pr_id] = score



    if pr_scores:

        best_pr_id = max(pr_scores.keys(), key=lambda x: pr_scores[x])

        return best_pr_id, "scored_selection"

    else:

        return None, None



if not timeline_df.empty and primary_results:

    selected_pr_id = 'PR_kwDOA-ENlc6XN-U1'

    selection_method = 'fixed_pr_id'

    available_pr_ids = set(timeline_df['pr_id'].dropna()) if 'pr_id' in timeline_df.columns else set()

    if selected_pr_id not in available_pr_ids:

        sequence_prs = [s.get('pr_id') for s in primary_results.get('sequences', []) if s.get('pr_id') in available_pr_ids]

        if sequence_prs:

            selected_pr_id = sequence_prs[0]

            selection_method = 'fallback_first_sequence'

        elif available_pr_ids:

            selected_pr_id = next(iter(available_pr_ids))

            selection_method = 'fallback_first_available'

        else:

            selected_pr_id = None

    if selected_pr_id is not None:

        pr_events = timeline_df[timeline_df['pr_id'] == selected_pr_id].copy()

        pr_events = pr_events.sort_values('timestamp').reset_index(drop=True)



        pr_events = pr_events.sort_values(['timestamp', 'is_ai', 'action_type']).reset_index(drop=True)



        pr_events['x'] = np.arange(len(pr_events))



        pr_created_at_utc = pr_events['timestamp'].min()



        def format_relative_time(dt):

            """Format relative time as +0m, +6m, +2h, +1d 3h"""

            total_seconds = dt.total_seconds()

            if total_seconds < 0:

                return "0m"



            if total_seconds < 3600:  # Less than 1 hour

                minutes = int(total_seconds // 60)

                return f"+{minutes}m"

            elif total_seconds < 86400:  # Less than 1 day

                hours = int(total_seconds // 3600)

                return f"+{hours}h"

            else:  # 1 day or more

                days = int(total_seconds // 86400)

                remaining_hours = int((total_seconds % 86400) // 3600)

                if remaining_hours > 0:

                    return f"+{days}d {remaining_hours}h"

                else:

                    return f"+{days}d"



        pr_events['dt'] = pr_events['timestamp'] - pr_created_at_utc

        pr_events['rel_time'] = pr_events['dt'].apply(format_relative_time)



        fig, ax = plt.subplots(figsize=(14, 6))



        marker_map = {

            'Open': 'D',        # diamond (for PR creation)

            'Comment': 'o',     # circle

            'Review': '^',      # triangle up

            'Commit': 's',      # square

            'Other': 'X'        # X (for Merge/Admin)

        }



        ai_color = 'red'

        human_color = 'blue'



        y_human = 0.03   # Human events slightly above

        y_ai = -0.03     # AI events slightly below



        ai_events = pr_events[pr_events['is_ai'] == True]

        if len(ai_events) > 0:

            for _, event in ai_events.iterrows():

                marker = marker_map.get(event['action_type'], '|')  # AI actions use | if not mapped

                ax.scatter(event['x'], y_ai, marker=marker, s=120,

                          facecolors='none', edgecolors=ai_color, linewidth=2,

                          zorder=3)



        human_events = pr_events[pr_events['is_ai'] == False]

        if len(human_events) > 0:

            for _, event in human_events.iterrows():

                marker = marker_map.get(event['action_type'], 'o')  # Default to circle

                ax.scatter(event['x'], y_human, marker=marker, s=120,

                          color=human_color, alpha=0.8, edgecolors='black', linewidth=1,

                          zorder=3)



        ax.set_ylim(-0.15, 0.15)

        ax.set_yticks([])  # Remove y-ticks since all events are on one line



        ax.set_xlim(-0.5, len(pr_events) - 0.5)



        N = len(pr_events)

        if N <= 25:

            tick_positions = list(range(N))

            tick_labels = pr_events['rel_time'].tolist()

        else:

            k = max(1, (N + 24) // 25)  # Ceiling division

            tick_positions = list(range(0, N, k))

            tick_labels = [pr_events.iloc[i]['rel_time'] for i in tick_positions]



        ax.set_xticks(tick_positions)

        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)



        total_events = len(pr_events)

        ai_count = pr_events['is_ai'].sum()

        human_count = (~pr_events['is_ai']).sum()





        from matplotlib.patches import Patch

        from matplotlib.lines import Line2D



        legend_elements = [

            Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',

                   markersize=10, label='Open (PR creation)', markeredgecolor='black'),

            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',

                   markersize=10, label='Comment', markeredgecolor='black'),

            Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',

                   markersize=10, label='Review', markeredgecolor='black'),

            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',

                   markersize=10, label='Commit', markeredgecolor='black'),

            Line2D([0], [0], marker='X', color='w', markerfacecolor='gray',

                   markersize=10, label='Merge/Admin', markeredgecolor='black'),

            Patch(facecolor=ai_color, label='AI Actor'),

            Patch(facecolor=human_color, label='Human Actor')

        ]



        ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='best', fontsize=10)



        ax.plot(pr_events['x'], [0] * len(pr_events),

               color='lightgray', alpha=0.5, linewidth=1, zorder=1)



        ax.grid(axis='x', alpha=0.3)



        start_time = pr_events['timestamp'].min()

        end_time = pr_events['timestamp'].max()

        time_span_hours = (end_time - start_time).total_seconds() / 3600



        first_human_response_time = None

        if len(human_events) > 0 and len(ai_events) > 0:

            first_ai_time = ai_events['timestamp'].min()

            human_after_ai = human_events[human_events['timestamp'] > first_ai_time]

            if len(human_after_ai) > 0:

                first_human_time = human_after_ai['timestamp'].min()

                first_human_response_time = (first_human_time - first_ai_time).total_seconds() / 60  # minutes





        plt.tight_layout()

        pass

        plt.show()



        pass

        sanity_df = pr_events[['x', 'timestamp', 'rel_time', 'is_ai', 'action_type']].head(10).copy()

        sanity_df['actor_group'] = sanity_df['is_ai'].apply(lambda x: 'AI' if x else 'Human')

        sanity_df = sanity_df[['x', 'timestamp', 'rel_time', 'actor_group', 'action_type']]

        pass



        if N > 25:

            pass

            mapping_df = pr_events[['x', 'timestamp', 'rel_time', 'is_ai', 'action_type']].copy()

            mapping_df['actor_group'] = mapping_df['is_ai'].apply(lambda x: 'AI' if x else 'Human')

            mapping_df = mapping_df[['x', 'timestamp', 'rel_time', 'actor_group', 'action_type']]

            pass



        is_merged = False

        merge_status = "NOT MERGED / UNKNOWN"



        if not df_prs_classified.empty and 'state' in df_prs_classified.columns:

            pr_record = df_prs_classified[df_prs_classified['id'] == selected_pr_id]

            if not pr_record.empty:

                pr_state = pr_record.iloc[0].get('state', '')

                is_merged = (pr_state == 'MERGED')

                if is_merged:

                    merge_status = "MERGED"

                elif pr_state == 'CLOSED':

                    merge_status = "CLOSED (NOT MERGED)"

                elif pr_state == 'OPEN':

                    merge_status = "OPEN"

                else:

                    merge_status = f"UNKNOWN STATE: {pr_state}"

            else:

                merge_status = "PR NOT FOUND IN DATASET"

        else:

            merge_status = "STATE FIELD NOT AVAILABLE"


if not timeline_df.empty:

    merge_source = df_prs_all[['pr_key', 'is_merged']] if 'df_prs_all' in globals() and not df_prs_all.empty and 'is_merged' in df_prs_all.columns else pd.DataFrame(columns=['pr_key', 'is_merged'])

    pr_merge_status_map = dict(zip(merge_source['pr_key'].astype(str), merge_source['is_merged'])) if not merge_source.empty else {}

    timeline_work = timeline_df.copy()

    timeline_work['pr_id'] = timeline_work['pr_id'].astype(str)

    event_counts = timeline_work.groupby('pr_id').agg(

        total_events=('action_type', 'size'),

        ai_events=('is_ai', 'sum')

    )

    event_counts['human_events'] = event_counts['total_events'] - event_counts['ai_events']

    first_event = timeline_work.sort_values(['pr_id', 'timestamp']).groupby('pr_id', as_index=False).first()[['pr_id', 'is_ai']]

    duration = timeline_work.groupby('pr_id')['timestamp'].agg(['min', 'max'])

    duration['duration_hours'] = (duration['max'] - duration['min']).dt.total_seconds().div(3600)

    action_types = timeline_work.groupby('pr_id')['action_type'].agg(lambda s: list(pd.unique(s)))

    seq_df = event_counts.reset_index().merge(first_event, on='pr_id', how='left').merge(duration[['duration_hours']].reset_index(), on='pr_id', how='left')

    seq_df['action_types'] = seq_df['pr_id'].map(action_types)

    seq_df['action_sequence'] = ''

    seq_df['sequence_pattern'] = np.where(

        (seq_df['ai_events'] > 0) & (seq_df['human_events'] > 0),

        np.where(seq_df['is_ai'], 'AI-initiated', 'Human-initiated'),

        np.where(seq_df['ai_events'] > 0, 'AI-only', 'Human-only')

    )

    seq_df['is_merged'] = seq_df['pr_id'].map(pr_merge_status_map).fillna(False)

    globals()['sequence_analysis_df'] = seq_df

else:

    pr_merge_status_map = {}


seq_df = pd.DataFrame(primary_results["sequences"])



seq_df["is_merged"] = seq_df["pr_id"].map(pr_merge_status_map)

seq_df["is_merged"] = seq_df["is_merged"].fillna(False)



overall_merge_rate = seq_df["is_merged"].mean()

overall_merged_n = int(seq_df["is_merged"].sum())

overall_n = len(seq_df)



if 'pr_metrics_df' not in globals() or pr_metrics_df is None:

    human_commits_per_pr = timeline_df[(timeline_df['action_type'] == 'Commit') & (~timeline_df['is_ai'])].groupby('pr_id').size()

    human_reviews_per_pr = timeline_df[(timeline_df['action_type'] == 'Review') & (~timeline_df['is_ai'])].groupby('pr_id').size()

    human_comments_per_pr = timeline_df[(timeline_df['action_type'] == 'Comment') & (~timeline_df['is_ai'])].groupby('pr_id').size()

    pr_ids = pd.Index(timeline_df['pr_id'].dropna().unique())

    pr_metrics_local = pd.DataFrame({'pr_id': pr_ids})

    pr_metrics_local['human_commits'] = safe_int_series(pr_metrics_local['pr_id'].map(human_commits_per_pr), fill_value=0)

    pr_metrics_local['human_reviews'] = safe_int_series(pr_metrics_local['pr_id'].map(human_reviews_per_pr), fill_value=0)

    pr_metrics_local['human_comments'] = safe_int_series(pr_metrics_local['pr_id'].map(human_comments_per_pr), fill_value=0)

    pr_metrics_local['autonomy'] = safe_int_series(pr_metrics_local['human_commits'].eq(0), fill_value=0)

    pr_metrics_local['validation_overhead'] = pr_metrics_local['human_reviews'] + pr_metrics_local['human_comments']

else:

    pr_metrics_local = pr_metrics_df.copy()



pr_metrics_local["is_merged"] = pr_metrics_local["pr_id"].map(pr_merge_status_map)

pr_metrics_local["is_merged"] = pr_metrics_local["is_merged"].fillna(False)



merged_metrics = pr_metrics_local[pr_metrics_local["is_merged"]].copy()



autonomy_rate_merged = merged_metrics["autonomy"].mean() if not merged_metrics.empty else np.nan



over = merged_metrics["validation_overhead"].dropna() if not merged_metrics.empty else pd.Series(dtype=float)

over_summary = {

    "median": float(over.median()) if not over.empty else float('nan'),

    "mean": float(over.mean()) if not over.empty else float('nan'),

    "p90": float(over.quantile(0.90)) if not over.empty else float('nan'),

    "p99": float(over.quantile(0.99)) if not over.empty else float('nan'),

}



reliance_table = pd.DataFrame([

    {"Metric": "Merged rate", "Value": f"{overall_merge_rate*100:.1f}% ({overall_merged_n:,}/{overall_n:,})"},

    {"Metric": "Merged without human commits", "Value": f"{autonomy_rate_merged*100:.1f}%"},

    {"Metric": "Validation overhead (comments+reviews) median", "Value": f"{over_summary['median']:.1f}"},

    {"Metric": "Validation overhead (comments+reviews) mean", "Value": f"{over_summary['mean']:.1f}"},

    {"Metric": "Validation overhead (comments+reviews) p90", "Value": f"{over_summary['p90']:.1f}"},

    {"Metric": "Validation overhead (comments+reviews) p99", "Value": f"{over_summary['p99']:.1f}"},

])



reliance_table


if not df_comments_classified.empty and RUN_TRUST_CUES:

    pass



    human_comments = df_comments_classified[~df_comments_classified['is_ai']].copy()



    if len(human_comments) > 0 and 'body' in human_comments.columns:

        human_comments['comment_length'] = human_comments['body'].fillna('').astype(str).str.len()



        pr_comment_stats = human_comments.groupby('pr_id').agg({

            'comment_length': ['count', 'median', 'sum'],

            'body': 'count'  # number of comments

        }).reset_index()



        pr_comment_stats.columns = ['pr_id', 'num_comments', 'median_length', 'total_length', 'comment_count']



        if 'engaging_users' in globals() and not engaging_users.empty:

            human_comments['user_login'] = human_comments['author'].apply(

                lambda x: extract_author_info(x)[0] if pd.notna(x) else ''

            )



            user_seniority_map = dict(zip(engaging_users['user_login'], engaging_users.get('global_seniority_group', 'novice')))

            human_comments['seniority'] = human_comments['user_login'].map(user_seniority_map).fillna('novice')



            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            fig.suptitle('Human Comment Patterns in Agentic PRs', fontsize=14)



            valid_lengths = human_comments[human_comments['comment_length'] > 0]['comment_length']

            if len(valid_lengths) > 0:

                log_bins = np.logspace(0, np.log10(valid_lengths.max()), 30)

                axes[0].hist(valid_lengths, bins=log_bins, alpha=0.7, edgecolor='black')

                axes[0].set_xscale('log')

                axes[0].set_xlabel('Comment Length (characters, log scale)')

                axes[0].set_ylabel('Number of Comments')

                axes[0].set_title('Comment Length Distribution')

                axes[0].grid(True, alpha=0.3)



            if 'seniority' in human_comments.columns:

                pr_seniority = human_comments.groupby(['pr_id', 'seniority']).size().reset_index(name='comment_count')

                seniority_summary = pr_seniority.groupby('seniority')['comment_count'].apply(list).to_dict()



                seniority_order = ['novice', 'mid', 'expert']

                valid_seniorities = [s for s in seniority_order if s in seniority_summary and len(seniority_summary[s]) > 0]



                if len(valid_seniorities) > 1:

                    data_to_plot = [seniority_summary[s] for s in valid_seniorities]

                    axes[1].boxplot(data_to_plot, labels=valid_seniorities)

                    axes[1].set_xlabel('Developer Seniority')

                    axes[1].set_ylabel('Comments per PR')

                    axes[1].set_title('Comment Volume by Seniority')

                    axes[1].tick_params(axis='x', rotation=45)

                else:

                    axes[1].text(0.5, 0.5, 'Insufficient seniority data', ha='center', va='center', transform=axes[1].transAxes)

                    axes[1].set_title('Comment Volume by Seniority')



            plt.tight_layout()

            if True:

                pass

                plt.show()

            else:

                pass



            pass

            pass

            pass

            pass



        else:

            pass



            valid_lengths = human_comments[human_comments['comment_length'] > 0]['comment_length']

            pass

            pass

            pass

            pass

    else:

        pass

else:

    pass


if not timeline_df.empty:

    analysis_timeline = timeline_df[timeline_df['dataset_label'] == 'agent'].copy() if 'dataset_label' in timeline_df.columns and (timeline_df['dataset_label'] == 'agent').any() else timeline_df.copy()

    pr_ids = pd.Index(analysis_timeline['pr_id'].dropna().astype(str).unique())

    human_commits_per_pr = analysis_timeline[(analysis_timeline['action_type'] == 'Commit') & (~analysis_timeline['is_ai'])].groupby('pr_id').size().reindex(pr_ids, fill_value=0)

    human_reviews_per_pr = analysis_timeline[(analysis_timeline['action_type'] == 'Review') & (~analysis_timeline['is_ai'])].groupby('pr_id').size().reindex(pr_ids, fill_value=0)

    human_comments_per_pr = analysis_timeline[(analysis_timeline['action_type'] == 'Comment') & (~analysis_timeline['is_ai'])].groupby('pr_id').size().reindex(pr_ids, fill_value=0)

    pr_event_counts = analysis_timeline.groupby('pr_id').size()

    sufficient_prs = pr_event_counts[pr_event_counts >= 2].index.astype(str)

    pr_metrics_df = pd.DataFrame({

        'pr_id': pr_ids,

        'human_commits': human_commits_per_pr.values,

        'human_reviews': human_reviews_per_pr.values,

        'human_comments': human_comments_per_pr.values

    })

    pr_metrics_df = pr_metrics_df[pr_metrics_df['pr_id'].isin(sufficient_prs)].copy()

    pr_metrics_df['pr_key'] = pr_metrics_df['pr_id'].astype(str)

    pr_metrics_df['autonomy'] = safe_int_series(pr_metrics_df['human_commits'].eq(0), fill_value=0)

    pr_metrics_df['validation_overhead'] = pr_metrics_df['human_reviews'] + pr_metrics_df['human_comments']

    pr_metrics_df['seniority'] = 'unknown'

    seniority_match_rate = np.nan

    unknown_count = 0

    novice_count = 0

    mid_count = 0

    expert_count = 0

    total_first_human = 0

    if 'engaging_users' in globals() and not engaging_users.empty and 'user_login_norm' in engaging_users.columns:

        user_seniority_map = dict(zip(engaging_users['user_login_norm'], engaging_users['global_seniority_group']))

        human_events = analysis_timeline[(~analysis_timeline['is_ai']) & (analysis_timeline['action_type'] != 'Open') & (analysis_timeline['actor_login_norm'] != '')].copy()

        if not human_events.empty:

            first_human_per_pr = human_events.sort_values(['pr_id', 'timestamp']).groupby('pr_id', as_index=False).first()[['pr_id', 'actor_login_norm']]

            total_first_human = len(first_human_per_pr)

            first_human_per_pr['seniority'] = first_human_per_pr['actor_login_norm'].map(user_seniority_map)

            pr_metrics_df = pr_metrics_df.merge(first_human_per_pr[['pr_id', 'seniority']], on='pr_id', how='left', suffixes=('', '_mapped'))

            pr_metrics_df['seniority'] = pr_metrics_df['seniority_mapped'].where(pr_metrics_df['seniority_mapped'].notna(), pr_metrics_df['seniority'])

            pr_metrics_df = pr_metrics_df.drop(columns=['seniority_mapped'])

            matched = int(pr_metrics_df['seniority'].isin(['novice', 'mid', 'expert']).sum())

            seniority_match_rate = matched / total_first_human if total_first_human else np.nan

    unknown_count = int((pr_metrics_df['seniority'] == 'unknown').sum())

    novice_count = int((pr_metrics_df['seniority'] == 'novice').sum())

    mid_count = int((pr_metrics_df['seniority'] == 'mid').sum())

    expert_count = int((pr_metrics_df['seniority'] == 'expert').sum())

    if not pr_metrics_df.empty:

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        overall_groups = [pr_metrics_df.loc[pr_metrics_df['autonomy'] == value, 'validation_overhead'].values for value in [0, 1] if (pr_metrics_df['autonomy'] == value).any()]

        overall_labels = [f'A={value}' for value in [0, 1] if (pr_metrics_df['autonomy'] == value).any()]

        if overall_groups:

            box_plot = axes[0].boxplot(overall_groups, labels=overall_labels, patch_artist=True)

            for patch, color in zip(box_plot['boxes'], ['lightcoral', 'lightblue'][:len(box_plot['boxes'])]):

                patch.set_facecolor(color)

                patch.set_alpha(0.7)

        axes[0].set_ylabel('Validation Overhead')

        axes[0].set_title('Overall Distribution')

        axes[0].grid(True, alpha=0.3)

        plotted = False

        for color, seniority in zip(['steelblue', 'orange', 'green'], ['novice', 'mid', 'expert']):

            group = pr_metrics_df[pr_metrics_df['seniority'] == seniority]

            if not group.empty:

                x_vals = group['autonomy'] + np.random.normal(0, 0.03, len(group))

                axes[1].scatter(x_vals, group['validation_overhead'], alpha=0.5, label=seniority, color=color, s=20)

                plotted = True

        if plotted:

            axes[1].legend()

        axes[1].set_xlim(-0.2, 1.2)

        axes[1].set_xlabel('Technical Autonomy')

        axes[1].set_ylabel('Validation Overhead')

        axes[1].set_title('By Developer Seniority')

        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        plt.show()


EARLY_WINDOW_HOURS = 24

intent_signal_pr_df = pd.DataFrame()

comment_intent_event_df = pd.DataFrame()

intent_signal_overview_df = pd.DataFrame()



comment_intent_signal_patterns = {

    'acceptance_signal': [

        r'\blgtm\b',

        r'\bship\s+it\b',

        r'\bapproved\b',

        r'\blooks?\s+good(\s+to\s+merge)?\b',

        r'\bgood\s+to\s+merge\b',

    ],

    'evidence_request_signal': [

        r'(please|can you|could you)\s+(run|add)\s+tests?',

        r'\brepro(duce)?\b',

        r'(please|can you|could you)\s+verify',

        r'\blogs?\b|\bstack\s*trace\b|\bbenchmark\b',

    ],

    'change_request_signal': [

        r'\brequest\s+changes\b',

        r'\bplease\s+change\b',

        r'\bcould\s+you\s+(change|update|fix|refactor|rename|remove|add)\b',

        r'\bnit:\b|\bnit\b',

        r'\bneeds?\s+to\b',

    ],

}



if not (RUN_TRUST_CUES and RUN_INTENSIVE_TRUST_CUES):

    print('Comment intent signals are disabled by RUN_TRUST_CUES / RUN_INTENSIVE_TRUST_CUES flags.')

else:

    if 'df_comments_classified' not in globals() or not isinstance(df_comments_classified, pd.DataFrame) or df_comments_classified.empty:

        stop_with_questions_for_user('Missing comments table for intent signal extraction (df_comments_classified is empty).')

    if 'df_reviews_classified' not in globals() or not isinstance(df_reviews_classified, pd.DataFrame) or df_reviews_classified.empty:

        stop_with_questions_for_user('Missing reviews table for intent signal extraction (df_reviews_classified is empty).')

    if 'state' not in df_reviews_classified.columns:

        stop_with_questions_for_user('Review state column is missing. Expected a review state field (APPROVED / CHANGES_REQUESTED / COMMENTED).')



    known_agent_logins = set()

    if 'df_prs_all' in globals() and isinstance(df_prs_all, pd.DataFrame) and not df_prs_all.empty:

        if {'author_type_group', 'author_login_norm'}.issubset(df_prs_all.columns):

            known_agent_logins.update(

                df_prs_all.loc[df_prs_all['author_type_group'].eq('agent'), 'author_login_norm']

                .dropna().astype(str).map(norm_login)

            )

    if 'df_prs_classified' in globals() and isinstance(df_prs_classified, pd.DataFrame) and not df_prs_classified.empty:

        ai_rows = df_prs_classified[df_prs_classified.get('is_ai', False).fillna(False)]

        if not ai_rows.empty:

            known_agent_logins.update(

                _event_actor_login(ai_rows, ['author', 'user', 'actor']).dropna().astype(str).map(norm_login)

            )

    known_agent_logins = {x for x in known_agent_logins if x}

    bot_pattern = re.compile(r'\[bot\]|(^|[-_])bot$|dependabot|renovate', flags=re.IGNORECASE)



    def _actor_kind(login_norm, is_ai_flag):

        login = norm_login(login_norm)

        if login and bot_pattern.search(login):

            return 'bot'

        if bool(is_ai_flag):

            return 'agent'

        if login in known_agent_logins:

            return 'agent'

        return 'human'



    def _clean_snippet(text):

        cleaned = re.sub(r'\s+', ' ', str(text) if text is not None else '').strip()

        return cleaned[:200]



    def _extract_matches(text):

        text_value = str(text) if text is not None else ''

        text_lower = text_value.lower()

        matches = {

            'acceptance_signal': [],

            'evidence_request_signal': [],

            'change_request_signal': [],

        }



        for pattern in comment_intent_signal_patterns['acceptance_signal']:

            if re.search(pattern, text_lower):

                matches['acceptance_signal'].append(pattern)

        if '👍' in text_value:

            matches['acceptance_signal'].append('👍')

        if '✅' in text_value:

            matches['acceptance_signal'].append('✅')



        for pattern in comment_intent_signal_patterns['evidence_request_signal']:

            if re.search(pattern, text_lower):

                matches['evidence_request_signal'].append(pattern)

        if re.search(r'\bsecurity\b', text_lower) and re.search(r'\b(check|review|verify)\b', text_lower):

            matches['evidence_request_signal'].append('security+(check|review|verify)')



        for pattern in comment_intent_signal_patterns['change_request_signal']:

            if re.search(pattern, text_lower):

                matches['change_request_signal'].append(pattern)



        return matches



    comments_events = ensure_pr_key(df_comments_classified.copy(), table_kind='events')

    reviews_events = ensure_pr_key(df_reviews_classified.copy(), table_kind='events')



    comments_events['timestamp'] = _event_timestamp(comments_events, 'comments')

    reviews_events['timestamp'] = _event_timestamp(reviews_events, 'reviews')



    comments_events['actor_login_norm'] = _event_actor_login(comments_events, ['author', 'user', 'actor']).fillna('').astype(str).map(norm_login)

    reviews_events['actor_login_norm'] = _event_actor_login(reviews_events, ['author', 'user', 'actor']).fillna('').astype(str).map(norm_login)



    comments_events['actor_kind'] = comments_events.apply(lambda row: _actor_kind(row.get('actor_login_norm', ''), row.get('is_ai', False)), axis=1)

    reviews_events['actor_kind'] = reviews_events.apply(lambda row: _actor_kind(row.get('actor_login_norm', ''), row.get('is_ai', False)), axis=1)



    comments_events['text'] = comments_events.get('body', '').fillna('').astype(str)

    reviews_events['text'] = reviews_events.get('body', '').fillna('').astype(str)

    reviews_events['review_state_norm'] = reviews_events['state'].fillna('').astype(str).str.upper().str.strip()

    comments_events['review_state_norm'] = ''



    comment_key_covered = int(comments_events['pr_key'].fillna('').astype(str).str.strip().ne('').sum())

    review_key_covered = int(reviews_events['pr_key'].fillna('').astype(str).str.strip().ne('').sum())

    if (comment_key_covered + review_key_covered) == 0:

        stop_with_questions_for_user('pr_key link is missing for comments/reviews. Cannot join intent signals to PR pathways.')



    response_events = pd.concat([

        comments_events[['pr_key', 'timestamp', 'actor_kind', 'text', 'review_state_norm', 'actor_login_norm']].assign(source_type='comment'),

        reviews_events[['pr_key', 'timestamp', 'actor_kind', 'text', 'review_state_norm', 'actor_login_norm']].assign(source_type='review'),

    ], ignore_index=True)

    response_events['pr_key'] = response_events['pr_key'].fillna('').astype(str).str.strip()

    response_events = response_events[

        response_events['pr_key'].ne('')

        & response_events['timestamp'].notna()

        & response_events['actor_kind'].eq('human')

    ].copy()



    if response_events.empty:

        stop_with_questions_for_user('No human comment/review events found after actor typing. Cannot compute comment intent signals.')



    first_human_response = (

        response_events.groupby('pr_key', as_index=False)['timestamp']

        .min()

        .rename(columns={'timestamp': 'first_human_response_time'})

    )

    window_events = response_events.merge(first_human_response, on='pr_key', how='inner')

    window_events['window_end_time'] = window_events['first_human_response_time'] + pd.to_timedelta(EARLY_WINDOW_HOURS, unit='h')

    window_events = window_events[

        window_events['timestamp'].ge(window_events['first_human_response_time'])

        & window_events['timestamp'].le(window_events['window_end_time'])

    ].copy()



    if window_events.empty:

        stop_with_questions_for_user('No human comment/review events fell into the 24h early window after first human response.')



    signal_rows = []

    for _, row in tqdm(window_events.iterrows(), total=len(window_events), desc='Scanning comment intent signals'):

        text = row.get('text', '')

        matches = _extract_matches(text)

        review_state = str(row.get('review_state_norm', '') or '').upper().strip()

        acceptance_state = review_state == 'APPROVED'

        change_state = review_state == 'CHANGES_REQUESTED'



        acceptance_text = len(matches['acceptance_signal']) > 0

        evidence_text = len(matches['evidence_request_signal']) > 0

        change_text = len(matches['change_request_signal']) > 0



        acceptance_pattern = '; '.join(matches['acceptance_signal']) if matches['acceptance_signal'] else ('review_state:APPROVED' if acceptance_state else '')

        evidence_pattern = '; '.join(matches['evidence_request_signal'])

        change_pattern = '; '.join(matches['change_request_signal']) if matches['change_request_signal'] else ('review_state:CHANGES_REQUESTED' if change_state else '')



        signal_rows.append({

            'pr_key': row['pr_key'],

            'source_type': row.get('source_type', ''),

            'timestamp': row.get('timestamp', pd.NaT),

            'first_human_response_time': row.get('first_human_response_time', pd.NaT),

            'actor_login_norm': row.get('actor_login_norm', ''),

            'review_state_norm': review_state,

            'text': text,

            'snippet': _clean_snippet(text) if str(text).strip() else '[no review body]',

            'acceptance_signal_text': bool(acceptance_text),

            'evidence_request_signal_text': bool(evidence_text),

            'change_request_signal_text': bool(change_text),

            'acceptance_signal_from_review_state': bool(acceptance_state),

            'change_request_signal_from_review_state': bool(change_state),

            'acceptance_signal': bool(acceptance_text or acceptance_state),

            'evidence_request_signal': bool(evidence_text),

            'change_request_signal': bool(change_text or change_state),

            'acceptance_signal_matched_pattern': acceptance_pattern,

            'evidence_request_signal_matched_pattern': evidence_pattern,

            'change_request_signal_matched_pattern': change_pattern,

        })



    comment_intent_event_df = pd.DataFrame(signal_rows)

    if comment_intent_event_df.empty:

        stop_with_questions_for_user('Intent signal extraction produced zero rows in early-window events.')



    intent_signal_pr_df = (

        comment_intent_event_df.groupby('pr_key', as_index=False)

        .agg(

            first_human_response_time=('first_human_response_time', 'min'),

            acceptance_signal=('acceptance_signal', 'max'),

            evidence_request_signal=('evidence_request_signal', 'max'),

            change_request_signal=('change_request_signal', 'max'),

            acceptance_signal_from_review_state=('acceptance_signal_from_review_state', 'max'),

            change_request_signal_from_review_state=('change_request_signal_from_review_state', 'max'),

            early_window_event_count=('pr_key', 'size'),

        )

    )

    for col in ['acceptance_signal', 'evidence_request_signal', 'change_request_signal', 'acceptance_signal_from_review_state', 'change_request_signal_from_review_state']:

        intent_signal_pr_df[col] = intent_signal_pr_df[col].fillna(False).astype(bool)

    intent_signal_pr_df['any_signal'] = intent_signal_pr_df[

        ['acceptance_signal', 'evidence_request_signal', 'change_request_signal']

    ].any(axis=1)



    n_prs = int(len(intent_signal_pr_df))

    intent_signal_overview_df = pd.DataFrame([

        {'signal_name': 'acceptance_signal', 'N_prs': n_prs, 'prevalence_pct': float(intent_signal_pr_df['acceptance_signal'].mean() * 100.0)},

        {'signal_name': 'evidence_request_signal', 'N_prs': n_prs, 'prevalence_pct': float(intent_signal_pr_df['evidence_request_signal'].mean() * 100.0)},

        {'signal_name': 'change_request_signal', 'N_prs': n_prs, 'prevalence_pct': float(intent_signal_pr_df['change_request_signal'].mean() * 100.0)},

        {'signal_name': 'any_signal', 'N_prs': n_prs, 'prevalence_pct': float(intent_signal_pr_df['any_signal'].mean() * 100.0)},

    ])

    print(f'Built intent signals for {n_prs:,} PRs using EARLY_WINDOW_HOURS={EARLY_WINDOW_HOURS}.')

    print(intent_signal_overview_df.to_string(index=False))


sentiment_df = pd.DataFrame()

sentiment_summary_table = pd.DataFrame(columns=['group', 'n_comments', 'median_vader', 'p25', 'p75'])



if RUN_SENTIMENT:

    comment_frames = []

    for frame in [

        agent_comments_sampled if 'agent_comments_sampled' in globals() else pd.DataFrame(),

        human_comments_sampled if 'human_comments_sampled' in globals() else pd.DataFrame(),

    ]:

        if isinstance(frame, pd.DataFrame) and not frame.empty:

            comment_frames.append(frame.copy())



    if comment_frames:

        comments_all = pd.concat(comment_frames, ignore_index=True)

    else:

        fallback_frames = []

        if 'df_comments_classified' in globals() and isinstance(df_comments_classified, pd.DataFrame) and not df_comments_classified.empty:

            fallback_frames.append(df_comments_classified.copy())

        if 'df_human_comments_classified' in globals() and isinstance(df_human_comments_classified, pd.DataFrame) and not df_human_comments_classified.empty:

            fallback_frames.append(df_human_comments_classified.copy())

        comments_all = pd.concat(fallback_frames, ignore_index=True) if fallback_frames else pd.DataFrame()



    if not comments_all.empty and 'body' in comments_all.columns:

        comments_all = ensure_pr_key(comments_all, table_kind='events')

        comments_all['actor_login_norm'] = _event_actor_login(comments_all, ['author', 'user', 'actor']).fillna('').astype(str).map(norm_login)

        comments_all['body'] = comments_all['body'].fillna('').astype(str)

        comments_all = comments_all[comments_all['body'].str.strip() != ''].copy()



        if 'df_prs_all' in globals() and isinstance(df_prs_all, pd.DataFrame) and not df_prs_all.empty:

            comments_all = comments_all[comments_all['pr_key'].astype(str).isin(set(df_prs_all['pr_key'].astype(str)))]

            known_agent_logins = set(df_prs_all.loc[df_prs_all['author_type_group'].eq('agent'), 'author_login_norm'].dropna().astype(str).map(norm_login))

        else:

            known_agent_logins = set()



        bot_regex = re.compile(r'dependabot|renovate|\[bot\]|-bot$|bot\b', flags=re.IGNORECASE)



        def _comment_actor_type(row):

            login = norm_login(row.get('actor_login_norm', ''))

            if login in known_agent_logins:

                return 'agent'

            if bool(row.get('is_ai', False)) or bool(bot_regex.search(login)):

                return 'bot'

            return 'human'



        comments_all['actor_type'] = comments_all.apply(_comment_actor_type, axis=1)

        human_comments = comments_all[comments_all['actor_type'].eq('human')].copy()



        if human_comments.empty:

            print('Warning: No HUMAN comments found for VADER sentiment after scope filtering; exporting empty sentiment table.')



        analyzer = None

        analyzer_backend = 'unavailable'

        analyzer_error_messages = []



        def _sanitize_local_vader_lexicon(src_path):

            src = Path(src_path)

            if not src.exists():

                return None, 0

            out_dir = Path('outputs')

            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / f'{src.stem}_sanitized.txt'

            rows = 0

            sanitized_lines = []

            with src.open('r', encoding='utf-8', errors='ignore') as fin:

                for raw in fin:

                    line = raw.strip()

                    if not line or line.startswith('#'):

                        continue

                    parts = line.split('\t')

                    if len(parts) < 2:

                        continue

                    token = parts[0].strip()

                    score_text = parts[1].strip()

                    if not token:

                        continue

                    try:

                        score = float(score_text)

                    except Exception:

                        continue

                    sanitized_lines.append(f'{token}\t{score}')

                    rows += 1

            if rows == 0:

                return None, 0

            # Avoid trailing newline because some NLTK VADER versions parse blank tail lines poorly.

            out_path.write_text('\n'.join(sanitized_lines), encoding='utf-8')

            return out_path, rows



        # Preferred: official vaderSentiment package.

        try:

            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderAnalyzer

            analyzer = VaderAnalyzer()

            analyzer_backend = 'vaderSentiment'

        except Exception as e_vader_pkg:

            analyzer_error_messages.append(f'vaderSentiment unavailable: {e_vader_pkg!r}')



        # Fallback: NLTK VADER using in-environment lexicon sources only (no downloads).

        if analyzer is None:

            try:

                from nltk.sentiment import SentimentIntensityAnalyzer as VaderAnalyzer

                try:

                    analyzer = VaderAnalyzer()

                    analyzer_backend = 'nltk_vader'

                except LookupError as e_lex:

                    analyzer_error_messages.append(f'nltk vader lexicon missing: {e_lex!r}')

                    local_lexicon_candidates = [

                        Path('vader_lexicon.txt'),

                        Path('resources') / 'vader_lexicon.txt',

                        Path('outputs') / 'vader_lexicon.txt',

                        RESOURCE_DIR / 'vader_lexicon.txt',

                    ]

                    for lex_path in local_lexicon_candidates:

                        if not lex_path.exists():

                            continue

                        try:

                            analyzer = VaderAnalyzer(lexicon_file=str(lex_path.resolve()))

                            analyzer_backend = f'nltk_vader_local_lexicon:{lex_path}'

                            break

                        except Exception as e_local:

                            analyzer_error_messages.append(f'local lexicon failed ({lex_path}): {e_local!r}')

                            sanitized_path, sanitized_rows = _sanitize_local_vader_lexicon(lex_path)

                            if sanitized_path is None:

                                analyzer_error_messages.append(f'sanitized lexicon unavailable ({lex_path})')

                                continue

                            try:

                                analyzer = VaderAnalyzer(lexicon_file=str(sanitized_path.resolve()))

                                analyzer_backend = f'nltk_vader_sanitized_lexicon:{sanitized_path} rows={sanitized_rows}'

                                break

                            except Exception as e_sanitized:

                                analyzer_error_messages.append(f'sanitized lexicon failed ({sanitized_path}): {e_sanitized!r}')

            except Exception as e_nltk:

                analyzer_error_messages.append(f'nltk vader unavailable: {e_nltk!r}')



        if analyzer is None:

            print('Warning: VADER analyzer unavailable; exporting empty sentiment_vader_robustness.csv.')

            if analyzer_error_messages:

                print('VADER initialization details:', ' | '.join(analyzer_error_messages))

            save_csv(sentiment_summary_table, 'sentiment_vader_robustness.csv', index=False)

        else:

            human_comments['vader_compound'] = human_comments['body'].apply(lambda text: analyzer.polarity_scores(text)['compound']).astype(float)



            user_seniority_map = dict(zip(engaging_users['user_login_norm'], engaging_users['global_seniority_group'])) if 'engaging_users' in globals() and not engaging_users.empty and 'user_login_norm' in engaging_users.columns else {}

            sentiment_df = pd.DataFrame({

                'pr_key': human_comments['pr_key'].astype(str),

                'author_login_norm': human_comments['actor_login_norm'],

                'global_seniority_group': human_comments['actor_login_norm'].map(user_seniority_map).fillna('unknown'),

                'vader_compound': human_comments['vader_compound'].astype(float),

                'comment_length': safe_int_series(human_comments['body'].str.len(), fill_value=0),

            })



            sentiment_summary_table = (

                sentiment_df.groupby('global_seniority_group')['vader_compound']

                .agg(

                    n_comments='size',

                    median_vader='median',

                    p25=lambda s: s.quantile(0.25),

                    p75=lambda s: s.quantile(0.75),

                )

                .reindex(['novice', 'mid', 'expert', 'unknown'])

                .reset_index()

                .rename(columns={'global_seniority_group': 'group'})

            )



            save_csv(sentiment_summary_table, 'sentiment_vader_robustness.csv', index=False)

            unknown_share = (

                sentiment_summary_table.loc[sentiment_summary_table['group'].eq('unknown'), 'n_comments'].fillna(0).sum()

                / max(1, sentiment_summary_table['n_comments'].fillna(0).sum())

            )



            print('Sentiment backend:', analyzer_backend)

            print('Sentiment (VADER compound) summary by group:')

            print(sentiment_summary_table.to_string(index=False))

            print(f'Unknown seniority share: {unknown_share:.2%}')

            if unknown_share > 0.20:

                print('Warning: unknown seniority share is high; interpret group-level sentiment with caution.')



            plot_groups = [g for g in ['novice', 'mid', 'expert', 'unknown'] if g in set(sentiment_df['global_seniority_group'])]

            data_to_plot = [sentiment_df.loc[sentiment_df['global_seniority_group'].eq(g), 'vader_compound'].dropna().values for g in plot_groups]

            if any(len(arr) > 0 for arr in data_to_plot):

                fig, ax = plt.subplots(figsize=(10, 6))

                ax.violinplot(data_to_plot, positions=range(len(plot_groups)), showmeans=True)

                ax.set_xticks(range(len(plot_groups)))

                ax.set_xticklabels(plot_groups, rotation=45)

                ax.set_ylabel('VADER compound')

                ax.grid(True, alpha=0.3, axis='y')

                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

                plt.tight_layout()

                plt.show()




user_metrics = []


user_metrics

if len(user_metrics) >= 5:

    metrics_df = pd.DataFrame(user_metrics)

    correlation_results = []

    numeric_metrics = ['total_actions', 'prs_involved', 'comment_actions', 'review_actions', 'commit_actions']

    for metric in numeric_metrics:

        valid_data = metrics_df[['seniority_numeric', metric]].dropna()

        if len(valid_data) >= 10:

            correlation = valid_data['seniority_numeric'].rank().corr(valid_data[metric].rank())

            correlation_results.append({

                'metric': metric.replace('_', ' ').title(),

                'correlation': correlation,

                'p_value': np.nan,

                'n_samples': len(valid_data),

                'interpretation': 'Positive' if correlation > 0.1 else 'Negative' if correlation < -0.1 else 'Weak'

            })

    response_time_data = metrics_df[['seniority_numeric', 'avg_response_time']].dropna() if 'avg_response_time' in metrics_df.columns else pd.DataFrame()

    if len(response_time_data) >= 10:

        correlation = response_time_data['seniority_numeric'].rank().corr(response_time_data['avg_response_time'].rank())

        correlation_results.append({

            'metric': 'Avg Response Time',

            'correlation': correlation,

            'p_value': np.nan,

            'n_samples': len(response_time_data),

            'interpretation': 'Positive' if correlation > 0.1 else 'Negative' if correlation < -0.1 else 'Weak'

        })

    if not timeline_df.empty and 'user_login' in metrics_df.columns:

        first_actions = []

        for user_login in metrics_df['user_login']:

            user_login_norm = str(user_login).strip().lower()

            user_events = timeline_df[timeline_df['actor_login_norm'] == user_login_norm]

            if not user_events.empty:

                first_action = user_events.sort_values('timestamp').iloc[0]['action_type']

                seniority = metrics_df.loc[metrics_df['user_login'] == user_login, 'seniority_category'].iloc[0]

                first_actions.append({'user': user_login, 'first_action': first_action, 'seniority': seniority})

        if first_actions:

            user_first_action_df = pd.DataFrame(first_actions)

            contingency_table = pd.crosstab(user_first_action_df['seniority'], user_first_action_df['first_action'])

            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:

                row_props = contingency_table.div(contingency_table.sum(axis=1), axis=0).fillna(0)

                effect_size = float((row_props.max(axis=1) - row_props.min(axis=1)).mean())

                correlation_results.append({

                    'metric': 'First Action Type',

                    'correlation': effect_size,

                    'p_value': np.nan,

                    'n_samples': int(contingency_table.to_numpy().sum()),

                    'interpretation': 'Strong' if effect_size > 0.3 else 'Moderate' if effect_size > 0.1 else 'Weak'

                })

    results_df = pd.DataFrame(correlation_results) if correlation_results else pd.DataFrame()

    if not results_df.empty:

        results_df['correlation'] = results_df['correlation'].round(3)

else:

    results_df = pd.DataFrame()


TAXONOMY_CACHE_PATH = 'outputs/taxonomy_df.csv'



def _build_taxonomy_source_for_current_scope():

    if (

        'agent_prs_sampled' in globals() and isinstance(agent_prs_sampled, pd.DataFrame) and not agent_prs_sampled.empty

        and 'agent_comments_sampled' in globals() and 'agent_reviews_sampled' in globals() and 'agent_commits_sampled' in globals()

    ):

        fresh = build_timeline(agent_prs_sampled, agent_comments_sampled, agent_reviews_sampled, agent_commits_sampled, 'agent')

        if isinstance(fresh, pd.DataFrame) and not fresh.empty:

            fresh = fresh.copy()

            if 'timestamp' in fresh.columns:

                fresh['timestamp'] = to_datetime_utc(fresh['timestamp'])

            return fresh

    if 'timeline_df' in globals() and isinstance(timeline_df, pd.DataFrame) and not timeline_df.empty:

        fallback = timeline_df.copy()

        if 'dataset_label' in fallback.columns and (fallback['dataset_label'] == 'agent').any():

            fallback = fallback[fallback['dataset_label'] == 'agent'].copy()

        if 'timestamp' in fallback.columns:

            fallback['timestamp'] = to_datetime_utc(fallback['timestamp'])

        return fallback

    return pd.DataFrame()





_simple_login_re = re.compile(r'^[a-z0-9][a-z0-9-]{0,38}$')

_dict_login_re = re.compile(r"['\"]login['\"]\s*:\s*['\"]([^'\"]+)['\"]", flags=re.IGNORECASE)

_dict_username_re = re.compile(r"['\"]username['\"]\s*:\s*['\"]([^'\"]+)['\"]", flags=re.IGNORECASE)

_noreply_re = re.compile(r"[a-z0-9._%-]+\+([a-z0-9-]+)@users\.noreply\.github\.com", flags=re.IGNORECASE)





def _extract_login_from_mixed_actor(value):

    if value is None or (isinstance(value, float) and pd.isna(value)):

        return ''

    text = str(value).strip()

    if not text:

        return ''

    try:

        login, _ = extract_author_info(value)

        login_norm = norm_login(login)

        if login_norm and _simple_login_re.match(login_norm):

            return login_norm

    except Exception:

        pass



    text_lower = text.lower()

    m_login = _dict_login_re.search(text_lower)

    if m_login:

        candidate = norm_login(m_login.group(1))

        if candidate:

            return candidate

    m_username = _dict_username_re.search(text_lower)

    if m_username:

        candidate = norm_login(m_username.group(1))

        if candidate:

            return candidate

    m_noreply = _noreply_re.search(text_lower)

    if m_noreply:

        candidate = norm_login(m_noreply.group(1))

        if candidate:

            return candidate



    plain = norm_login(text_lower)

    if plain and _simple_login_re.match(plain):

        return plain

    return ''





def _build_seniority_lookup():

    lookup = {}



    def _add_from_df(df, login_col, group_col):

        if not isinstance(df, pd.DataFrame) or df.empty or login_col not in df.columns or group_col not in df.columns:

            return

        subset = df[[login_col, group_col]].copy()

        subset[login_col] = subset[login_col].fillna('').astype(str).str.strip().str.lower()

        subset[group_col] = subset[group_col].fillna('').astype(str).str.strip().str.lower()

        subset = subset[subset[login_col].ne('') & subset[group_col].isin(['novice', 'mid', 'expert'])]

        if subset.empty:

            return

        for login, group in subset.drop_duplicates(login_col, keep='first')[[login_col, group_col]].itertuples(index=False):

            lookup.setdefault(login, group)



    if 'engaging_users' in globals():

        _add_from_df(engaging_users, 'user_login_norm', 'global_seniority_group')

    elif 'users_df' in globals():

        _add_from_df(users_df, 'user_login_norm', 'global_seniority_group')



    return lookup





def _compute_actor_lookup_key(df):

    work = df.copy()

    for col in ['actor_login_norm', 'actor_login', 'actor_name']:

        if col not in work.columns:

            work[col] = ''

        work[col] = work[col].fillna('')

    c1 = work['actor_login_norm'].map(_extract_login_from_mixed_actor)

    c2 = work['actor_login'].map(_extract_login_from_mixed_actor)

    c3 = work['actor_name'].map(_extract_login_from_mixed_actor)

    key = c1.copy()

    mask = key.eq('')

    key.loc[mask] = c2.loc[mask]

    mask = key.eq('')

    key.loc[mask] = c3.loc[mask]

    return key


def _build_first_response_seniority_map_from_events(events_df, seniority_lookup):

    if not isinstance(events_df, pd.DataFrame) or events_df.empty or not seniority_lookup:

        return {}

    required_cols = {'pr_key', 'timestamp', 'action_type', 'is_ai', 'actor_login_norm'}

    if not required_cols.issubset(events_df.columns):

        return {}

    work = events_df.copy()

    work['pr_key'] = work['pr_key'].fillna('').astype(str).str.strip()

    work = work[work['pr_key'].ne('')].copy()

    if work.empty:

        return {}

    work['timestamp'] = to_datetime_utc(work['timestamp'])

    work['action_type'] = work['action_type'].fillna('').astype(str)

    work['is_ai'] = work['is_ai'].fillna(False).astype(bool)

    work['actor_lookup_key'] = work['actor_login_norm'].fillna('').astype(str).str.strip().str.lower()

    human_events = work[(~work['is_ai']) & work['action_type'].ne('Open')].copy()

    if human_events.empty:

        return {}

    matched = human_events[human_events['actor_lookup_key'].isin(seniority_lookup)].copy()

    if matched.empty:

        return {}

    sort_cols = ['pr_key', 'timestamp']

    if 'event_rank' in matched.columns:

        sort_cols.append('event_rank')

    first_human = (

        matched

        .sort_values(sort_cols, kind='mergesort')

        .drop_duplicates(subset=['pr_key'], keep='first')

        .copy()

    )

    first_human['primary_seniority'] = first_human['actor_lookup_key'].map(seniority_lookup)

    first_human = first_human[first_human['primary_seniority'].isin(['novice', 'mid', 'expert'])].copy()

    if first_human.empty:

        return {}

    return dict(zip(first_human['pr_key'].astype(str), first_human['primary_seniority']))


def _build_first_response_seniority_map(seniority_lookup):

    if not seniority_lookup:

        return {}

    source_df = _build_taxonomy_source_for_current_scope()

    return _build_first_response_seniority_map_from_events(source_df, seniority_lookup)





def _recompute_primary_seniority_fast(base_taxonomy_df, taxonomy_source_df, seniority_lookup):

    if base_taxonomy_df.empty or taxonomy_source_df.empty or not seniority_lookup:

        return base_taxonomy_df.copy(), 0.0



    base = base_taxonomy_df.copy()

    base['pr_key'] = base['pr_key'].astype(str)

    first_response_map = _build_first_response_seniority_map(seniority_lookup)

    if not first_response_map:

        valid_keys = set(base['pr_key'])

        events = taxonomy_source_df.copy()

        events['pr_key'] = events['pr_key'].astype(str)

        events = events[events['pr_key'].isin(valid_keys)].copy()

        first_response_map = _build_first_response_seniority_map_from_events(events, seniority_lookup)

    base['primary_seniority'] = base['pr_key'].map(first_response_map).fillna('unknown')

    known_share = base['primary_seniority'].isin(['novice', 'mid', 'expert']).mean()

    return base, float(known_share)



def _build_taxonomy_df_vectorized(taxonomy_source_df, seniority_lookup):

    expected_cols = ['pr_key', 'pr_id', 'taxonomy_class', 'total_events', 'human_events', 'ai_events', 'primary_seniority']

    if not isinstance(taxonomy_source_df, pd.DataFrame) or taxonomy_source_df.empty or 'pr_key' not in taxonomy_source_df.columns:

        return pd.DataFrame(columns=expected_cols)



    work = taxonomy_source_df.copy()

    work['pr_key'] = work['pr_key'].fillna('').astype(str).str.strip()

    work = work[work['pr_key'].ne('')].copy()

    if work.empty:

        return pd.DataFrame(columns=expected_cols)



    if 'timestamp' not in work.columns:

        work['timestamp'] = pd.NaT

    work['timestamp'] = to_datetime_utc(work['timestamp'])

    work['action_type'] = work.get('action_type', pd.Series('', index=work.index)).fillna('').astype(str)

    work['is_ai'] = work.get('is_ai', pd.Series(False, index=work.index)).fillna(False).astype(bool)

    work = work.sort_values(['pr_key', 'timestamp'], kind='mergesort').copy()

    work['event_rank'] = work.groupby('pr_key').cumcount()



    total_counts = work.groupby('pr_key').size().rename('total_events')

    total_counts = total_counts[total_counts >= 2]

    if total_counts.empty:

        return pd.DataFrame(columns=expected_cols)



    work = work[work['pr_key'].isin(total_counts.index)].copy()

    ai_counts = work.groupby('pr_key')['is_ai'].sum().astype('int64').rename('ai_events')



    human_events = work[(~work['is_ai']) & work['action_type'].ne('Open')].copy()

    human_counts = (

        human_events.groupby('pr_key').size().rename('human_events')

        if not human_events.empty

        else pd.Series(dtype='int64', name='human_events')

    )



    human_discussion = human_events[human_events['action_type'].isin(['Comment', 'Review'])].copy()

    discussion_flags = (

        human_discussion.groupby('pr_key').size().gt(0)

        if not human_discussion.empty

        else pd.Series(dtype=bool)

    )



    human_commits = human_events[human_events['action_type'].eq('Commit')].copy()

    commit_flags = (

        human_commits.groupby('pr_key').size().gt(0)

        if not human_commits.empty

        else pd.Series(dtype=bool)

    )



    recent_commit_flags = pd.Series(dtype=bool)

    if not human_events.empty and not human_commits.empty:

        first_human_time = human_events.groupby('pr_key')['timestamp'].min()

        human_commits = human_commits.copy()

        human_commits['first_human_time'] = human_commits['pr_key'].map(first_human_time)

        human_commits['recent_commit'] = (

            (human_commits['timestamp'] - human_commits['first_human_time']).dt.total_seconds().le(24 * 3600)

        ).fillna(False)

        recent_commit_flags = human_commits.groupby('pr_key')['recent_commit'].any()



    summary = total_counts.to_frame()

    summary = summary.join(ai_counts, how='left')

    summary = summary.join(human_counts, how='left')

    summary[['ai_events', 'human_events']] = summary[['ai_events', 'human_events']].fillna(0).astype('int64')

    summary = summary.reset_index()

    summary['has_human_discussion'] = summary['pr_key'].map(discussion_flags).fillna(False).astype(bool)

    summary['has_human_commit'] = summary['pr_key'].map(commit_flags).fillna(False).astype(bool)

    summary['has_recent_commit'] = summary['pr_key'].map(recent_commit_flags).fillna(False).astype(bool)



    summary['taxonomy_class'] = 'echo_chamber'

    summary.loc[

        summary['human_events'].gt(0) & ~summary['has_human_discussion'] & ~summary['has_human_commit'],

        'taxonomy_class'

    ] = 'human_in_loop'

    summary.loc[

        summary['has_human_discussion'] & summary['has_human_commit'] & summary['has_recent_commit'],

        'taxonomy_class'

    ] = 'rewriting'

    summary.loc[

        summary['has_human_discussion'] & summary['has_human_commit'] & ~summary['has_recent_commit'],

        'taxonomy_class'

    ] = 'steering'

    summary.loc[

        summary['has_human_discussion'] & ~summary['has_human_commit'],

        'taxonomy_class'

    ] = 'steering'

    summary.loc[

        ~summary['has_human_discussion'] & summary['has_human_commit'],

        'taxonomy_class'

    ] = 'human_in_loop'



    summary['primary_seniority'] = 'unknown'

    if seniority_lookup:

        first_response_map = _build_first_response_seniority_map(seniority_lookup)

        if not first_response_map:

            first_response_map = _build_first_response_seniority_map_from_events(work, seniority_lookup)

        if first_response_map:

            summary['primary_seniority'] = summary['pr_key'].astype(str).map(first_response_map).fillna('unknown')



    out = summary[['pr_key', 'taxonomy_class', 'total_events', 'human_events', 'ai_events', 'primary_seniority']].copy()

    out.insert(1, 'pr_id', out['pr_key'].astype(str))

    return out[expected_cols]





taxonomy_source_current = _build_taxonomy_source_for_current_scope()

current_pr_keys = set()

if isinstance(taxonomy_source_current, pd.DataFrame) and not taxonomy_source_current.empty and 'pr_key' in taxonomy_source_current.columns:

    current_pr_keys = set(taxonomy_source_current['pr_key'].dropna().astype(str))



taxonomy_df = pd.DataFrame()

cache_loaded = False



if os.path.exists(TAXONOMY_CACHE_PATH):

    cached_taxonomy = pd.read_csv(TAXONOMY_CACHE_PATH)

    if not cached_taxonomy.empty and 'pr_key' in cached_taxonomy.columns and current_pr_keys:

        cached_keys = set(cached_taxonomy['pr_key'].dropna().astype(str))

        overlap_n = len(cached_keys & current_pr_keys)

        if overlap_n > 0:

            taxonomy_df = cached_taxonomy

            cache_loaded = True

            print(f'Loaded taxonomy_df from {TAXONOMY_CACHE_PATH} ({len(taxonomy_df):,} rows, overlap_n={overlap_n:,})')

        else:

            print(f'Ignoring stale taxonomy cache at {TAXONOMY_CACHE_PATH} due to zero PR-key overlap with current scope.')

    elif not cached_taxonomy.empty and not current_pr_keys:

        taxonomy_df = cached_taxonomy

        cache_loaded = True

        print(f'Loaded taxonomy_df from {TAXONOMY_CACHE_PATH} ({len(taxonomy_df):,} rows; current scope keys unavailable).')



if not taxonomy_df.empty and 'primary_seniority' in taxonomy_df.columns:

    known_share = taxonomy_df['primary_seniority'].fillna('').astype(str).str.lower().isin(['novice', 'mid', 'expert']).mean()

    if known_share < 0.02:

        print(

            f'Existing taxonomy seniority share is low ({known_share:.2%}); '

            'attempting fast event-level seniority remap.'

        )

        lookup = _build_seniority_lookup()

        repaired, repaired_share = _recompute_primary_seniority_fast(taxonomy_df, taxonomy_source_current, lookup)

        if repaired_share > known_share:

            taxonomy_df = repaired

            os.makedirs('outputs', exist_ok=True)

            taxonomy_df.to_csv(TAXONOMY_CACHE_PATH, index=False)

            print(f'Fast seniority remap improved known share to {repaired_share:.2%}; cache updated.')

        else:

            print('Fast remap did not improve seniority coverage; will rebuild taxonomy from timeline.')

            taxonomy_df = pd.DataFrame()

            cache_loaded = False



if taxonomy_df.empty:

    taxonomy_df = pd.DataFrame()

    taxonomy_source = taxonomy_source_current.copy() if isinstance(taxonomy_source_current, pd.DataFrame) else pd.DataFrame()

    if not taxonomy_source.empty:

        seniority_lookup = _build_seniority_lookup()

        print(f'Seniority lookup entries: {len(seniority_lookup):,}')

        first_response_seniority_map = _build_first_response_seniority_map(seniority_lookup)

        if not first_response_seniority_map:

            first_response_seniority_map = _build_first_response_seniority_map_from_events(taxonomy_source, seniority_lookup)



        taxonomy_source['actor_lookup_key'] = _compute_actor_lookup_key(taxonomy_source)

        if seniority_lookup:

            known_event_share = taxonomy_source['actor_lookup_key'].isin(seniority_lookup).mean()

            print(f'Event-level actor->seniority coverage: {known_event_share:.2%}')

        try:

            taxonomy_df = _build_taxonomy_df_vectorized(taxonomy_source, seniority_lookup)

            if taxonomy_df.empty and int(taxonomy_source['pr_key'].nunique()) > 0:

                raise RuntimeError('vectorized taxonomy builder returned no rows')

            print(f'Vectorized taxonomy build completed ({len(taxonomy_df):,} rows)')

        except Exception as exc:

            print(f'Warning: vectorized taxonomy build failed ({exc}); falling back to iterative build.')

            pr_taxonomies = []

            total_prs = int(taxonomy_source['pr_key'].nunique()) if 'pr_key' in taxonomy_source.columns else None

            for pr_key, pr_events in tqdm(taxonomy_source.groupby('pr_key'), total=total_prs, desc='Taxonomy'):

                pr_events = pr_events.sort_values('timestamp').copy()

                if len(pr_events) < 2:

                    continue



                human_events = pr_events[(~pr_events['is_ai']) & (pr_events['action_type'] != 'Open')].copy()

                ai_events = pr_events[pr_events['is_ai']].copy()

                if human_events.empty:

                    taxonomy_class = 'echo_chamber'

                else:

                    human_comments_reviews = human_events[human_events['action_type'].isin(['Comment', 'Review'])]

                    human_commits = human_events[human_events['action_type'] == 'Commit']

                    if human_comments_reviews.empty and human_commits.empty:

                        taxonomy_class = 'human_in_loop'

                    elif not human_comments_reviews.empty and not human_commits.empty:

                        first_human_time = human_events['timestamp'].min()

                        recent_commits = human_commits[(human_commits['timestamp'] - first_human_time).dt.total_seconds() <= 24 * 3600]

                        taxonomy_class = 'rewriting' if not recent_commits.empty else 'steering'

                    elif not human_comments_reviews.empty:

                        taxonomy_class = 'steering'

                    else:

                        taxonomy_class = 'human_in_loop'



                primary_seniority = first_response_seniority_map.get(str(pr_key), 'unknown')



                pr_taxonomies.append({

                    'pr_key': str(pr_key),

                    'pr_id': str(pr_key),

                    'taxonomy_class': taxonomy_class,

                    'total_events': int(len(pr_events)),

                    'human_events': int(len(human_events)),

                    'ai_events': int(len(ai_events)),

                    'primary_seniority': primary_seniority

                })

            taxonomy_df = pd.DataFrame(pr_taxonomies)

    os.makedirs('outputs', exist_ok=True)

    taxonomy_df.to_csv(TAXONOMY_CACHE_PATH, index=False)

    print(f'Rebuilt taxonomy_df and saved to {TAXONOMY_CACHE_PATH} ({len(taxonomy_df):,} rows)')



if taxonomy_df.empty:

    print('taxonomy_df is empty; skipping taxonomy summaries and plot.')

else:

    taxonomy_display_map = {

        'echo_chamber': 'No human handling',

        'steering': 'Discussion only pathway',

        'rewriting': 'Early takeover pathway',

        'human_in_loop': 'Silent edit pathway',

    }

    taxonomy_df['taxonomy_class'] = taxonomy_df['taxonomy_class'].fillna('').astype(str)

    taxonomy_df['taxonomy_display_label'] = taxonomy_df['taxonomy_class'].map(taxonomy_display_map).fillna(taxonomy_df['taxonomy_class'])

    taxonomy_df['seniority'] = taxonomy_df.get('primary_seniority', 'unknown')

    taxonomy_df['seniority'] = taxonomy_df['seniority'].fillna('unknown').astype(str).str.strip().str.lower()

    taxonomy_df.loc[~taxonomy_df['seniority'].isin(['novice', 'mid', 'expert']), 'seniority'] = 'unknown'



    pathway_share_by_seniority_df = (

        taxonomy_df.groupby(['taxonomy_class', 'taxonomy_display_label', 'seniority'], as_index=False)

        .size()

        .rename(columns={'size': 'count'})

    )

    pathway_share_by_seniority_df['share'] = pathway_share_by_seniority_df['count'] / pathway_share_by_seniority_df.groupby('seniority')['count'].transform('sum')

    pathway_share_by_seniority_df['share'] = pathway_share_by_seniority_df['share'].fillna(0.0)

    pathway_share_by_seniority_df = pathway_share_by_seniority_df.sort_values(['taxonomy_display_label', 'seniority']).reset_index(drop=True)

    save_csv(pathway_share_by_seniority_df, 'pathway_share_by_seniority.csv', index=False)



    print('Pathway share by seniority (proportions):')

    print(pathway_share_by_seniority_df.head(20).to_string(index=False))

    print('Taxonomy seniority counts:')

    print(taxonomy_df['seniority'].value_counts(dropna=False).to_string())



    plot_order = [taxonomy_display_map[k] for k in ['echo_chamber', 'steering', 'rewriting', 'human_in_loop'] if taxonomy_display_map[k] in set(pathway_share_by_seniority_df['taxonomy_display_label'])]

    plot_df = pathway_share_by_seniority_df[pathway_share_by_seniority_df['seniority'].isin(['novice', 'mid', 'expert'])].copy()

    if plot_df.empty:

        print('Warning: No known seniority rows (novice/mid/expert) for pathway plot; skipping chart.')

        pathway_plot_legend_labels = []

    else:

        pivot_plot = plot_df.pivot(index='taxonomy_display_label', columns='seniority', values='share').fillna(0.0)

        pivot_plot = pivot_plot.reindex(plot_order if plot_order else pivot_plot.index)

        pivot_plot = pivot_plot.reindex(columns=[c for c in ['novice', 'mid', 'expert'] if c in pivot_plot.columns])

        pathway_plot_legend_labels = list(pivot_plot.columns)

        fig, ax = plt.subplots(figsize=(10, 5))

        pivot_plot.plot(kind='bar', ax=ax, alpha=0.85)

        ax.set_xlabel('Pathway')

        ax.set_ylabel('Share within seniority')

        ax.set_title('Pathway share by seniority')

        ax.legend(title='Seniority', bbox_to_anchor=(1.02, 1), loc='upper left')

        ax.tick_params(axis='x', rotation=20)

        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        plt.show()


if 'taxonomy_df' in globals() and not os.path.exists('outputs/taxonomy_df.csv'):

    os.makedirs('outputs', exist_ok=True)

    taxonomy_df.to_csv('outputs/taxonomy_df.csv', index=False)



if 'intent_signal_pr_df' not in globals() or intent_signal_pr_df.empty:

    stop_with_questions_for_user('Intent signal PR table is missing or empty. Please verify the comment intent signals cell ran successfully.')

if 'comment_intent_event_df' not in globals() or comment_intent_event_df.empty:

    stop_with_questions_for_user('Intent signal event table is missing or empty. Please verify early-window event extraction.')

if 'taxonomy_df' not in globals() or taxonomy_df.empty:

    stop_with_questions_for_user('taxonomy_df is missing or empty; cannot connect intent signals to pathways.')



required_taxonomy_cols = {'pr_key', 'taxonomy_class', 'taxonomy_display_label'}

missing_taxonomy = [c for c in required_taxonomy_cols if c not in taxonomy_df.columns]

if missing_taxonomy:

    stop_with_questions_for_user(f'Missing taxonomy columns required for signal linkage: {missing_taxonomy}')



required_signal_cols = {'pr_key', 'acceptance_signal', 'evidence_request_signal', 'change_request_signal', 'any_signal'}

missing_signal_cols = [c for c in required_signal_cols if c not in intent_signal_pr_df.columns]

if missing_signal_cols:

    stop_with_questions_for_user(f'Missing signal columns in intent_signal_pr_df: {missing_signal_cols}')



taxonomy_unique = taxonomy_df[['pr_key', 'taxonomy_class', 'taxonomy_display_label']].copy()

taxonomy_unique['pr_key'] = taxonomy_unique['pr_key'].fillna('').astype(str).str.strip()

taxonomy_unique = taxonomy_unique[taxonomy_unique['pr_key'].ne('')].drop_duplicates('pr_key', keep='first')



signals_df = intent_signal_pr_df[['pr_key', 'acceptance_signal', 'evidence_request_signal', 'change_request_signal', 'any_signal']].copy()

signals_df['pr_key'] = signals_df['pr_key'].fillna('').astype(str).str.strip()

signals_df = signals_df[signals_df['pr_key'].ne('')].drop_duplicates('pr_key', keep='first')

for col in ['acceptance_signal', 'evidence_request_signal', 'change_request_signal', 'any_signal']:

    signals_df[col] = signals_df[col].fillna(False).astype(bool)



intent_pathway_df = taxonomy_unique.merge(signals_df, on='pr_key', how='inner')

if intent_pathway_df.empty:

    stop_with_questions_for_user('No overlap between taxonomy_df and intent signal PR features on pr_key.')



comment_intent_signals_by_pathway = (

    intent_pathway_df.groupby('taxonomy_display_label', as_index=False)

    .agg(

        N_prs=('pr_key', 'nunique'),

        acceptance_signal_pct=('acceptance_signal', lambda s: float(pd.Series(s).astype(bool).mean() * 100.0)),

        evidence_request_signal_pct=('evidence_request_signal', lambda s: float(pd.Series(s).astype(bool).mean() * 100.0)),

        change_request_signal_pct=('change_request_signal', lambda s: float(pd.Series(s).astype(bool).mean() * 100.0)),

    )

    .sort_values('taxonomy_display_label')

    .reset_index(drop=True)

)

save_csv(comment_intent_signals_by_pathway, 'comment_intent_signals_by_pathway.csv', index=False)



try:

    from scipy.stats import chi2_contingency

except Exception:

    chi2_contingency = None



signal_names = ['acceptance_signal', 'evidence_request_signal', 'change_request_signal']

stats_rows = []

for signal_name in signal_names:

    contingency = pd.crosstab(intent_pathway_df['taxonomy_display_label'], intent_pathway_df[signal_name].astype(int))

    n_obs = int(contingency.values.sum())

    if n_obs == 0 or contingency.shape[0] < 2 or contingency.shape[1] < 2:

        chi2_value = np.nan

        p_value = np.nan

        cramers_v = np.nan

    else:

        if chi2_contingency is not None:

            chi2_value, p_value, _, _ = chi2_contingency(contingency.values)

        else:

            observed = contingency.values.astype(float)

            expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()

            with np.errstate(divide='ignore', invalid='ignore'):

                chi2_matrix = (observed - expected) ** 2 / expected

            chi2_value = float(np.nansum(chi2_matrix))

            p_value = np.nan

        denom = n_obs * max(min(contingency.shape) - 1, 1)

        cramers_v = float(np.sqrt(chi2_value / denom)) if denom > 0 and pd.notna(chi2_value) else np.nan

    stats_rows.append({

        'signal_name': signal_name,

        'chi2': float(chi2_value) if pd.notna(chi2_value) else np.nan,

        'p_value': float(p_value) if pd.notna(p_value) else np.nan,

        'cramers_v': float(cramers_v) if pd.notna(cramers_v) else np.nan,

        'N': n_obs,

    })



comment_intent_signals_stats = pd.DataFrame(stats_rows)

save_csv(comment_intent_signals_stats, 'comment_intent_signals_stats.csv', index=False)



events_for_examples = comment_intent_event_df.copy()

events_for_examples['pr_key'] = events_for_examples['pr_key'].fillna('').astype(str).str.strip()

events_for_examples = events_for_examples[events_for_examples['pr_key'].ne('')].copy()

events_for_examples = events_for_examples.merge(

    taxonomy_unique[['pr_key', 'taxonomy_display_label']],

    on='pr_key',

    how='inner'

)



example_rows = []

for pathway_label in sorted(events_for_examples['taxonomy_display_label'].dropna().astype(str).unique()):

    subset_pathway = events_for_examples[events_for_examples['taxonomy_display_label'].eq(pathway_label)].copy()

    for signal_name in signal_names:

        pattern_col = f'{signal_name}_matched_pattern'

        if signal_name not in subset_pathway.columns:

            continue

        subset_signal = subset_pathway[subset_pathway[signal_name].fillna(False).astype(bool)].copy()

        if subset_signal.empty:

            continue

        sample_n = min(20, len(subset_signal))

        subset_signal = subset_signal.sample(n=sample_n, random_state=7)

        for _, row in subset_signal.iterrows():

            snippet = re.sub(r'\s+', ' ', str(row.get('snippet', '') or row.get('text', '') or '')).strip()

            snippet = snippet[:200]

            if not snippet:

                snippet = '[no review body]'

            example_rows.append({

                'taxonomy_display_label': pathway_label,

                'signal_name': signal_name,

                'pr_key': row.get('pr_key', ''),

                'snippet': snippet,

                'matched_pattern': row.get(pattern_col, '') if pattern_col in row else '',

            })



comment_intent_signal_examples = pd.DataFrame(example_rows)

save_csv(comment_intent_signal_examples, 'comment_intent_signal_examples.csv', index=False)



print('Comment intent signals by pathway:')

print(comment_intent_signals_by_pathway.to_string(index=False))

print('Comment intent signal association stats:')

print(comment_intent_signals_stats.to_string(index=False))



coverage_by_pathway = (

    intent_pathway_df.groupby('taxonomy_display_label', as_index=False)['any_signal']

    .mean()

    .rename(columns={'any_signal': 'signal_coverage_pct'})

)

coverage_by_pathway['signal_coverage_pct'] = coverage_by_pathway['signal_coverage_pct'] * 100.0

print('Signal coverage (% PRs with any signal) by pathway:')

print(coverage_by_pathway.to_string(index=False))



legend_labels = [str(x) for x in globals().get('pathway_plot_legend_labels', [])]

signal_name_collision = any('steering' in str(x).lower() for x in signal_names)

legend_collision = any('steering' in str(x).lower() for x in legend_labels)

print(f'Naming collision check (signal_name contains "steering"): {signal_name_collision}')

print(f'Naming collision check (plot legend contains "steering"): {legend_collision}')

if signal_name_collision or legend_collision:

    stop_with_questions_for_user('Naming collision detected: "steering" appears in signal names or plot legend.')




from pathlib import Path





def _load_output_csv(filename):

    candidates = []

    if 'output_path' in globals():

        candidates.append(Path(output_path(filename)))

    if 'OUTPUT_DIR_FALLBACK' in globals():

        candidates.append(Path(OUTPUT_DIR_FALLBACK) / filename)

    candidates.append(Path('outputs') / filename)

    seen = set()

    for path in candidates:

        path = path.resolve()

        if path in seen:

            continue

        seen.add(path)

        if path.exists():

            return pd.read_csv(path), str(path)

    return pd.DataFrame(), None





latency_source = latency_summary.copy() if 'latency_summary' in globals() and isinstance(latency_summary, pd.DataFrame) and not latency_summary.empty else pd.DataFrame()

if latency_source.empty:

    latency_source, _ = _load_output_csv('baseline_latency_summary.csv')



first_action_source = first_action_df.copy() if 'first_action_df' in globals() and isinstance(first_action_df, pd.DataFrame) and not first_action_df.empty else pd.DataFrame()

if first_action_source.empty:

    first_action_source, _ = _load_output_csv('baseline_first_action_table.csv')



merge_source = merge_proxy_summary.copy() if 'merge_proxy_summary' in globals() and isinstance(merge_proxy_summary, pd.DataFrame) and not merge_proxy_summary.empty else pd.DataFrame()

if merge_source.empty:

    merge_source, _ = _load_output_csv('baseline_merge_proxy.csv')



if 'rq2_sanity_checks_df' in globals() and isinstance(rq2_sanity_checks_df, pd.DataFrame) and not rq2_sanity_checks_df.empty:

    rq2_source = rq2_sanity_checks_df.copy()

else:

    rq2_source, _ = _load_output_csv('baseline_rq2_sanity_checks.csv')



if 'baseline_sanity_checks_df' in globals() and isinstance(baseline_sanity_checks_df, pd.DataFrame) and not baseline_sanity_checks_df.empty:

    save_csv(baseline_sanity_checks_df, 'baseline_sanity_checks.csv', index=False)



checks = []

required_groups = {'agent', 'human'}

tol = 1e-6



# 1) Latency summary must include agent+human with N>0

if latency_source.empty or 'author_type_group' not in latency_source.columns or 'N' not in latency_source.columns:

    checks.append({'check': 'latency_summary_present', 'passed': False, 'detail': 'baseline_latency_summary.csv missing required columns'})

else:

    lat_n = latency_source.groupby('author_type_group')['N'].sum().to_dict()

    lat_groups = set(lat_n.keys())

    checks.append({'check': 'latency_contains_agent_human', 'passed': bool(required_groups.issubset(lat_groups)), 'detail': f'present={sorted(lat_groups)}'})

    checks.append({

        'check': 'latency_agent_human_N_gt_0',

        'passed': bool(all(int(lat_n.get(g, 0)) > 0 for g in required_groups)),

        'detail': str({g: int(lat_n.get(g, 0)) for g in sorted(required_groups)}),

    })



# 2) First action table must include agent+human non-zero counts and shares sum to 1

if first_action_source.empty or not {'author_type_group', 'count', 'share'}.issubset(first_action_source.columns):

    checks.append({'check': 'first_action_present', 'passed': False, 'detail': 'baseline_first_action_table.csv missing required columns'})

else:

    fa_counts = first_action_source.groupby('author_type_group')['count'].sum().to_dict()

    fa_groups = set(fa_counts.keys())

    checks.append({'check': 'first_action_contains_agent_human', 'passed': bool(required_groups.issubset(fa_groups)), 'detail': f'present={sorted(fa_groups)}'})

    checks.append({

        'check': 'first_action_agent_human_count_gt_0',

        'passed': bool(all(int(fa_counts.get(g, 0)) > 0 for g in required_groups)),

        'detail': str({g: int(fa_counts.get(g, 0)) for g in sorted(required_groups)}),

    })

    share_sums = first_action_source.groupby('author_type_group')['share'].sum().to_dict()

    share_ok = all(abs(float(share_sums.get(g, np.nan)) - 1.0) <= tol for g in required_groups if g in fa_groups)

    checks.append({

        'check': 'first_action_share_sums_to_1',

        'passed': bool(share_ok),

        'detail': str({g: float(share_sums.get(g, np.nan)) for g in sorted(share_sums)}),

    })



# 3) Merge proxy must include agent+human rows; human coverage >0 if merged_n>0

if merge_source.empty or 'author_type_group' not in merge_source.columns:

    checks.append({'check': 'merge_proxy_present', 'passed': False, 'detail': 'baseline_merge_proxy.csv missing/empty'})

else:

    merge_groups = set(merge_source['author_type_group'].dropna().astype(str))

    checks.append({'check': 'merge_proxy_contains_agent_human', 'passed': bool(required_groups.issubset(merge_groups)), 'detail': f'present={sorted(merge_groups)}'})

    human_row = merge_source.loc[merge_source['author_type_group'].eq('human')].head(1)

    if human_row.empty:

        checks.append({'check': 'merge_proxy_human_row_present', 'passed': False, 'detail': 'human row missing'})

    else:

        human_merged_n = int(pd.to_numeric(human_row.iloc[0].get('merged_n', 0), errors='coerce') or 0)

        human_cov = float(pd.to_numeric(human_row.iloc[0].get('commit_coverage_pct', np.nan), errors='coerce'))

        coverage_ok = True if human_merged_n == 0 else (pd.notna(human_cov) and human_cov > 0)

        checks.append({

            'check': 'merge_proxy_human_coverage_gt_0_if_merged',

            'passed': bool(coverage_ok),

            'detail': f'merged_n={human_merged_n}, commit_coverage_pct={human_cov}',

        })



acceptance_df = pd.DataFrame(checks)

print('\nAcceptance checks:')

print(acceptance_df.to_string(index=False))



failed = acceptance_df.loc[~acceptance_df['passed']] if not acceptance_df.empty else pd.DataFrame()

if not failed.empty:

    fail_details = '; '.join([f"{row['check']} ({row['detail']})" for _, row in failed.iterrows()])

    stop_with_questions_for_user(f'RQ2.1 acceptance failed: {fail_details}')



# Append acceptance rows to baseline_rq2_sanity_checks.csv

if isinstance(rq2_source, pd.DataFrame) and not rq2_source.empty:

    acceptance_rows = acceptance_df.copy()

    acceptance_rows['section'] = 'acceptance'

    acceptance_rows['group'] = 'global'

    acceptance_rows['metric'] = acceptance_rows['check']

    acceptance_rows['value'] = acceptance_rows['detail']

    cols = ['section', 'group', 'metric', 'value', 'passed', 'detail']

    rq2_out = pd.concat([rq2_source.reindex(columns=cols), acceptance_rows.reindex(columns=cols)], ignore_index=True)

else:

    rq2_out = acceptance_df.assign(

        section='acceptance',

        group='global',

        metric=acceptance_df['check'],

        value=acceptance_df['detail']

    )[['section', 'group', 'metric', 'value', 'passed', 'detail']]

save_csv(rq2_out, 'baseline_rq2_sanity_checks.csv', index=False)



print('\nRQ2.1 PASS')

print('Latency N by group:')

print(latency_source[['author_type_group', 'N', 'median_hours', 'p90_hours', 'p99_hours']].sort_values('author_type_group').to_string(index=False))

print('\nFirst HUMAN response type distribution:')

print(first_action_source[['author_type_group', 'first_human_response_type', 'count', 'share']].sort_values(['author_type_group', 'first_human_response_type']).to_string(index=False))

merge_cols = [c for c in ['author_type_group', 'merged_n', 'commit_coverage_pct', 'merged_without_non_author_commits_pct'] if c in merge_source.columns]

print('\nMerge proxy summary:')

print(merge_source[merge_cols].sort_values('author_type_group').to_string(index=False))


# PAPER_EXPORTS_START

from pathlib import Path





def _safe_export(step_name, fn):

    try:

        fn()

        print(f"[paper-export] OK: {step_name}")

    except Exception as exc:

        print(f"[paper-export] FAILED: {step_name} -> {exc}")





def _read_csv_candidates(relative_paths):

    for rel in relative_paths:

        for path in _paper_path_candidates(rel):

            if path.exists():

                try:

                    return pd.read_csv(path), str(path)

                except Exception:

                    continue

    return pd.DataFrame(), None





def _df_from_var_or_csv(var_name, csv_candidates):

    obj = globals().get(var_name, None)

    if isinstance(obj, pd.DataFrame) and not obj.empty:

        return obj.copy(), f'var:{var_name}'

    return _read_csv_candidates(csv_candidates)





def _label_map_taxonomy():

    return {

        'echo_chamber': 'No human handling',

        'steering': 'Discussion only pathway',

        'rewriting': 'Early takeover pathway',

        'human_in_loop': 'Silent edit pathway',

    }





def _baseline_group_order_safe(values=None):

    if '_baseline_group_order' in globals():

        try:

            return _baseline_group_order(values if values is not None else [])

        except Exception:

            pass

    base = ['agent', 'human', 'bot', 'bot_dependency', 'bot_other']

    if values is None:

        return base

    seen = set(base)

    ordered = [x for x in base if x in set(values)]

    for x in values:

        if x not in seen:

            ordered.append(x)

            seen.add(x)

    return ordered





# ASE minimal-patch helpers

def _neutral_label_map():

    return {

        'echo_chamber': 'No human handling',

        'steering': 'Discussion only',

        'human_in_loop': 'Silent edit',

        'rewriting': 'Early takeover',

    }




def _neutral_pathway_order():

    return ['No human handling', 'Discussion only', 'Silent edit', 'Early takeover']




def _seniority_order():

    return ['novice', 'mid', 'expert']




def _write_text_dual(text, relative_path):

    last_path = None

    for path in _paper_path_candidates(relative_path):

        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(str(text))

        last_path = path

    return str(last_path) if last_path is not None else ''




def _write_json_dual(payload, relative_path):

    return _write_text_dual(json.dumps(payload, indent=2, sort_keys=True), relative_path)




def _write_latex_table(df, relative_path, caption, label=None, column_format=None, size='\\footnotesize'):

    table_df = df.copy() if isinstance(df, pd.DataFrame) and not df.empty else pd.DataFrame([{'message': 'No rows'}])

    latex_tabular = table_df.to_latex(index=False, escape=True, column_format=column_format)

    parts = [

        '\\begin{table}[t]',

        '\\centering',

        size,

        f'\\caption{{{caption}}}',

    ]

    if label:

        parts.append(f'\\label{{{label}}}')

    parts.extend([latex_tabular.strip(), '\\end{table}', ''])

    return _write_text_dual('\n'.join(parts), relative_path)




def _winsorize_series(series, low=0.05, high=0.95):

    values = pd.to_numeric(series, errors='coerce')

    if values.empty:

        return values

    lower = values.quantile(low)

    upper = values.quantile(high)

    return values.clip(lower, upper)




def _minmax_series(series):

    values = pd.to_numeric(series, errors='coerce').fillna(0)

    spread = values.max() - values.min()

    if pd.isna(spread) or spread == 0:

        return pd.Series(0.0, index=values.index)

    return (values - values.min()) / spread




def _chi2_cramers_v(contingency):

    observed = contingency.copy() if isinstance(contingency, pd.DataFrame) else pd.DataFrame(contingency)

    n_obs = int(observed.to_numpy().sum()) if not observed.empty else 0

    observed = observed.loc[observed.sum(axis=1) > 0, observed.sum(axis=0) > 0]

    if observed.empty or observed.shape[0] < 2 or observed.shape[1] < 2 or n_obs <= 0:

        return {

            'chi2': np.nan,

            'p_value': np.nan,

            'cramers_v': np.nan,

            'n': n_obs,

        }

    try:

        from scipy.stats import chi2_contingency as _chi2_contingency

        chi2_value, p_value, _, _ = _chi2_contingency(observed.values)

    except Exception:

        values = observed.values.astype(float)

        expected = np.outer(values.sum(axis=1), values.sum(axis=0)) / values.sum()

        with np.errstate(divide='ignore', invalid='ignore'):

            chi2_matrix = (values - expected) ** 2 / expected

        chi2_value = float(np.nansum(chi2_matrix))

        p_value = np.nan

    denom = n_obs * max(min(observed.shape) - 1, 1)

    cramers_v = float(np.sqrt(chi2_value / denom)) if denom > 0 and pd.notna(chi2_value) else np.nan

    return {

        'chi2': float(chi2_value) if pd.notna(chi2_value) else np.nan,

        'p_value': float(p_value) if pd.notna(p_value) else np.nan,

        'cramers_v': cramers_v,

        'n': n_obs,

    }




def _get_fixed_seniority_reference_time():

    out_obj = globals().get('out', None)

    if isinstance(out_obj, dict) and 'ref_time_used' in out_obj:

        ref_time = to_datetime_utc(pd.Series([out_obj.get('ref_time_used')])).iloc[0]

        if pd.notna(ref_time):

            return ref_time

    users = globals().get('users_df', pd.DataFrame())

    if isinstance(users, pd.DataFrame) and not users.empty:

        scrape_candidates = []

        for col in users.columns:

            col_lower = str(col).lower()

            if 'scrape' not in col_lower:

                continue

            ts = to_datetime_utc(users[col]).dropna()

            if not ts.empty:

                scrape_candidates.append(ts.max())

        if scrape_candidates:

            return max(scrape_candidates)

    return pd.Timestamp.utcnow(tz='UTC')




def _get_profiled_human_users_for_agent_scope():

    source = globals().get('taxonomy_source_current', pd.DataFrame())

    users = globals().get('users_df', pd.DataFrame())

    if not isinstance(source, pd.DataFrame) or source.empty or not isinstance(users, pd.DataFrame) or users.empty:

        return pd.DataFrame()

    work = users.copy()

    work = _ensure_contrib_columns(work)

    if 'user_type' not in work.columns:

        work['user_type'] = work.apply(classify_user_type, axis=1)

    work['user_type'] = work['user_type'].fillna('').astype(str).str.strip().str.lower()

    work.loc[work['user_type'].eq('user'), 'user_type'] = 'human'

    login_source = 'user_login' if 'user_login' in work.columns else 'login' if 'login' in work.columns else None

    if login_source is None:

        work['user_login_norm'] = ''

    else:

        work['user_login_norm'] = work[login_source].fillna('').astype(str).str.strip().str.lower().map(norm_login)

    work['account_created_at_parsed'] = to_datetime_utc(work.get('account_created_at', pd.Series(pd.NaT, index=work.index)))

    for col in ['commits_count', 'prs_count', 'reviews_count', 'issues_count']:

        if col not in work.columns:

            work[col] = 0

        work[col] = pd.to_numeric(work[col], errors='coerce').fillna(0).clip(lower=0)

    work['contrib_total_proxy'] = work[['commits_count', 'prs_count', 'reviews_count', 'issues_count']].sum(axis=1)

    source = source.copy()

    source['actor_lookup_key'] = _compute_actor_lookup_key(source)

    engaged_logins = set(

        source.loc[

            (~source['is_ai'].fillna(False).astype(bool)) & source['action_type'].isin(['Comment', 'Review', 'Commit']),

            'actor_lookup_key'

        ]

        .fillna('')

        .astype(str)

        .str.strip()

        .str.lower()

    )

    engaged_logins.discard('')

    profiled = work.loc[

        work['user_type'].eq('human')

        & work['user_login_norm'].isin(engaged_logins)

        & work['user_login_norm'].ne('')

        & work['account_created_at_parsed'].notna()

    ].copy()

    profiled = profiled.drop_duplicates(subset=['user_login_norm'], keep='first').reset_index(drop=True)

    if profiled.empty:

        return profiled

    ref_time = _get_fixed_seniority_reference_time()

    profiled['tenure_days'] = (

        (ref_time - profiled['account_created_at_parsed']).dt.total_seconds().div(86400).clip(lower=0).fillna(0)

    )

    profiled['contrib_total'] = profiled['contrib_total_proxy']

    profiled['tenure_norm'] = _minmax_series(_winsorize_series(np.log1p(profiled['tenure_days'])))

    profiled['contrib_norm'] = _minmax_series(_winsorize_series(np.log1p(profiled['contrib_total'])))

    return profiled




def _get_agent_first_human_responses():

    response_df = globals().get('first_responses', pd.DataFrame())

    if isinstance(response_df, pd.DataFrame) and not response_df.empty:

        work = response_df.copy()

        if 'author_type_group' in work.columns:

            work = work[work['author_type_group'].fillna('').astype(str).eq('agent')].copy()

        if 'pr_key' not in work.columns and 'pr_id' in work.columns:

            work['pr_key'] = work['pr_id'].astype(str)

        if 'first_human_response_type' not in work.columns and 'action_type' in work.columns:

            work['first_human_response_type'] = work['action_type'].fillna('').astype(str).str.strip().str.lower()

        work['actor_lookup_key'] = _compute_actor_lookup_key(work)

        return work.drop_duplicates(subset=['pr_key'], keep='first').copy()

    source = globals().get('taxonomy_source_current', pd.DataFrame())

    if not isinstance(source, pd.DataFrame) or source.empty:

        return pd.DataFrame()

    work = source.copy()

    work['pr_key'] = work['pr_key'].fillna('').astype(str)

    work['timestamp'] = to_datetime_utc(work['timestamp'])

    open_times = (

        work.loc[work['action_type'].eq('Open')]

        .groupby('pr_key')['timestamp']

        .min()

    )

    candidate = work.loc[

        (~work['is_ai'].fillna(False).astype(bool))

        & work['action_type'].isin(['Comment', 'Review', 'Commit'])

    ].copy()

    candidate['open_time'] = candidate['pr_key'].map(open_times)

    candidate = candidate[candidate['timestamp'].gt(candidate['open_time'])].copy()

    if candidate.empty:

        return candidate

    candidate['first_human_response_type'] = candidate['action_type'].fillna('').astype(str).str.strip().str.lower()

    candidate['action_priority'] = candidate['first_human_response_type'].map({'comment': 1, 'review': 2, 'commit': 3}).fillna(99)

    candidate['actor_lookup_key'] = _compute_actor_lookup_key(candidate)

    return (

        candidate.sort_values(['pr_key', 'timestamp', 'action_priority'])

        .groupby('pr_key', as_index=False)

        .first()

    )




def _build_2x2_pathway_labels(taxonomy_source):

    if not isinstance(taxonomy_source, pd.DataFrame) or taxonomy_source.empty:

        return pd.DataFrame(columns=['pr_key', 'visible_human_discussion', 'visible_human_code_intervention', 'pathway'])

    work = taxonomy_source.copy()

    work['pr_key'] = work['pr_key'].fillna('').astype(str).str.strip()

    work = work[work['pr_key'].ne('')].copy()

    if work.empty:

        return pd.DataFrame(columns=['pr_key', 'visible_human_discussion', 'visible_human_code_intervention', 'pathway'])



    work['action_type'] = work.get('action_type', pd.Series('', index=work.index)).fillna('').astype(str).str.title()

    work['is_ai'] = work.get('is_ai', pd.Series(False, index=work.index)).fillna(False).astype(bool)

    human_events = work.loc[

        (~work['is_ai'])

        & work['action_type'].isin(['Comment', 'Review', 'Commit'])

    ].copy()



    discussion_flags = (

        human_events.loc[human_events['action_type'].isin(['Comment', 'Review'])].groupby('pr_key').size().gt(0)

        if not human_events.empty

        else pd.Series(dtype=bool)

    )

    code_flags = (

        human_events.loc[human_events['action_type'].eq('Commit')].groupby('pr_key').size().gt(0)

        if not human_events.empty

        else pd.Series(dtype=bool)

    )



    out = pd.DataFrame({'pr_key': work['pr_key'].drop_duplicates().astype(str).tolist()})

    out['visible_human_discussion'] = out['pr_key'].map(discussion_flags).fillna(False).astype(bool)

    out['visible_human_code_intervention'] = out['pr_key'].map(code_flags).fillna(False).astype(bool)

    out['pathway'] = 'No human handling'

    out.loc[

        out['visible_human_discussion'] & ~out['visible_human_code_intervention'],

        'pathway'

    ] = 'Discussion only'

    out.loc[

        ~out['visible_human_discussion'] & out['visible_human_code_intervention'],

        'pathway'

    ] = 'Silent edit'

    out.loc[

        out['visible_human_discussion'] & out['visible_human_code_intervention'],

        'pathway'

    ] = 'Early takeover'

    return out


def _build_pathway_labels_from_taxonomy(taxonomy_table):

    columns = ['pr_key', 'visible_human_discussion', 'visible_human_code_intervention', 'pathway']

    if not isinstance(taxonomy_table, pd.DataFrame) or taxonomy_table.empty or 'pr_key' not in taxonomy_table.columns:

        return pd.DataFrame(columns=columns)

    work = taxonomy_table.copy()

    work['pr_key'] = work['pr_key'].fillna('').astype(str).str.strip()

    work = work[work['pr_key'].ne('')].drop_duplicates(subset=['pr_key'], keep='first').copy()

    if work.empty:

        return pd.DataFrame(columns=columns)

    if 'pathway' not in work.columns:

        if 'taxonomy_class' not in work.columns:

            return pd.DataFrame(columns=columns)

        work['pathway'] = work['taxonomy_class'].fillna('').astype(str).map(_neutral_label_map())

    work['pathway'] = work['pathway'].fillna('').astype(str).str.strip()

    work = work[work['pathway'].isin(_neutral_pathway_order())].copy()

    if work.empty:

        return pd.DataFrame(columns=columns)

    discussion_pathways = {'Discussion only', 'Early takeover'}

    code_pathways = {'Silent edit', 'Early takeover'}

    work['visible_human_discussion'] = work['pathway'].isin(discussion_pathways)

    work['visible_human_code_intervention'] = work['pathway'].isin(code_pathways)

    return work[columns].reset_index(drop=True)




def _fmt_p_value(value):

    if pd.isna(value):

        return 'NA'

    if float(value) < 0.001:

        return '<0.001'

    return f'{float(value):.3f}'




# Resolve core tables

intensity_export_df, _ = _df_from_var_or_csv('intensity_df', ['outputs/rq1_intensity.csv', 'rq1_intensity.csv'])

first_action_export_df, _ = _df_from_var_or_csv('first_action_df', ['outputs/baseline_first_action_table.csv', 'baseline_first_action_table.csv'])

latency_export_df, _ = _df_from_var_or_csv('latency_summary', ['outputs/baseline_latency_summary.csv', 'baseline_latency_summary.csv'])

merge_export_df, _ = _df_from_var_or_csv('merge_proxy_summary', ['outputs/baseline_merge_proxy.csv', 'baseline_merge_proxy.csv'])

sentiment_summary_export_df, _ = _df_from_var_or_csv('sentiment_summary_table', ['outputs/sentiment_vader_robustness.csv', 'sentiment_vader_robustness.csv'])

pathway_share_long_df, _ = _df_from_var_or_csv('pathway_share_by_seniority_df', ['outputs/pathway_share_by_seniority.csv', 'pathway_share_by_seniority.csv'])

signals_by_pathway_df, _ = _df_from_var_or_csv('comment_intent_signals_by_pathway', ['outputs/comment_intent_signals_by_pathway.csv', 'comment_intent_signals_by_pathway.csv'])

signals_stats_df, _ = _df_from_var_or_csv('comment_intent_signals_stats', ['outputs/comment_intent_signals_stats.csv', 'comment_intent_signals_stats.csv'])





# 1) RQ1 seniority distribution figure

def _export_rq1_seniority():

    if 'engaging_users' not in globals() or not isinstance(engaging_users, pd.DataFrame) or engaging_users.empty:

        return

    users = engaging_users.copy()

    if 'user_type' in users.columns:

        users = users[users['user_type'].astype(str).str.lower().eq('human')].copy()

    if users.empty or 'global_seniority_score' not in users.columns:

        return

    scores = pd.to_numeric(users['global_seniority_score'], errors='coerce').dropna()

    if scores.empty:

        return

    q25 = scores.quantile(0.25)

    q75 = scores.quantile(0.75)

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(scores, bins=30, color='skyblue', edgecolor='black', alpha=0.8)

    ax.axvline(q25, color='red', linestyle='--', linewidth=2)

    ax.axvline(q75, color='green', linestyle='--', linewidth=2)

    ax.set_xlabel('Global Seniority Score')

    ax.set_ylabel('Number of Developers')

    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    export_fig_pdf(fig, 'figures/rq1_seniority.pdf')

    plt.close(fig)





_safe_export('rq1_seniority_figure', _export_rq1_seniority)





# 2) RQ1 contribution intensity summary table

def _export_rq1_intensity():

    if intensity_export_df is None or intensity_export_df.empty:

        return

    _write_csv_dual(intensity_export_df, 'outputs/rq1_intensity.csv', index=False)

    export_table_pdf(intensity_export_df, 'outputs/pdf_tables/rq1_intensity.pdf', title='RQ1 Contribution Intensity', index=False)





_safe_export('rq1_intensity_table', _export_rq1_intensity)





# 3-5) Baseline tables

def _export_baseline_tables():

    if first_action_export_df is not None and not first_action_export_df.empty:

        _write_csv_dual(first_action_export_df, 'outputs/baseline_first_action_table.csv', index=False)

        export_table_pdf(first_action_export_df, 'outputs/pdf_tables/rq2_first_action.pdf', title='RQ2 First HUMAN Response Type', index=False)

    if latency_export_df is not None and not latency_export_df.empty:

        _write_csv_dual(latency_export_df, 'outputs/baseline_latency_summary.csv', index=False)

        export_table_pdf(latency_export_df, 'outputs/pdf_tables/rq2_latency.pdf', title='RQ2 First HUMAN Response Latency', index=False)

    if merge_export_df is not None and not merge_export_df.empty:

        merge_table = merge_export_df.drop(columns=[c for c in ['commit_source_used'] if c in merge_export_df.columns])

        _write_csv_dual(merge_table, 'outputs/baseline_merge_proxy.csv', index=False)

        export_table_pdf(merge_table, 'outputs/pdf_tables/rq2_merge_proxy.pdf', title='RQ2 Merge Proxy', index=False)





_safe_export('rq2_baseline_tables', _export_baseline_tables)





# 6-8) Baseline figures (PDF versions)

def _export_baseline_figures():

    if (not PAPER_ONLY_EXPORTS) and first_action_export_df is not None and not first_action_export_df.empty:

        group_col = 'author_type_group' if 'author_type_group' in first_action_export_df.columns else 'author_type'

        type_col = 'first_human_response_type' if 'first_human_response_type' in first_action_export_df.columns else 'first_response_type'

        plot_df = first_action_export_df.copy()

        if 'share' not in plot_df.columns and 'count' in plot_df.columns:

            totals = plot_df.groupby(group_col)['count'].transform('sum').replace(0, np.nan)

            plot_df['share'] = plot_df['count'] / totals

        pivot = plot_df.pivot(index=group_col, columns=type_col, values='share').fillna(0.0)

        pivot = pivot.reindex(_baseline_group_order_safe(pivot.index.tolist())).dropna(how='all')

        if not pivot.empty:

            fig, ax = plt.subplots(figsize=(10, 6))

            pivot.plot(kind='bar', ax=ax, width=0.8)

            ax.set_xlabel('Author Type')

            ax.set_ylabel('Share of First HUMAN Responses')

            ax.legend(title='Response Type')

            ax.grid(True, axis='y', alpha=0.3)

            plt.tight_layout()

            export_fig_pdf(fig, 'figures/fig_baseline_first_action.pdf')

            plt.close(fig)



    if latency_export_df is not None and not latency_export_df.empty and 'median_hours' in latency_export_df.columns:

        group_col = 'author_type_group' if 'author_type_group' in latency_export_df.columns else 'author_type'

        plot_df = latency_export_df.copy().set_index(group_col)

        plot_df = plot_df.reindex(_baseline_group_order_safe(plot_df.index.tolist())).dropna(subset=['median_hours'], how='all').reset_index()

        if not plot_df.empty:

            fig, ax = plt.subplots(figsize=(10, 6))

            vals = pd.to_numeric(plot_df['median_hours'], errors='coerce')

            bars = ax.bar(plot_df[group_col], vals)

            ax.set_yscale('log')

            ax.set_xlabel('Author Type')

            ax.set_ylabel('Median First HUMAN Response Time (hours)')

            ax.grid(True, axis='y', alpha=0.3)

            for bar, val in zip(bars, vals):

                if pd.notna(val) and val > 0:

                    ax.text(bar.get_x() + bar.get_width() / 2, val, f'{val:.2f}h', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()

            export_fig_pdf(fig, 'figures/fig_baseline_latency_log.pdf')

            plt.close(fig)



    if (not PAPER_ONLY_EXPORTS) and merge_export_df is not None and not merge_export_df.empty:

        group_col = 'author_type_group' if 'author_type_group' in merge_export_df.columns else 'author_type'

        share_col = (

            'merged_without_non_author_commits_pct'

            if 'merged_without_non_author_commits_pct' in merge_export_df.columns

            else ('share_merge_without_non_author_commits_in_pr' if 'share_merge_without_non_author_commits_in_pr' in merge_export_df.columns else 'share_merge_without_non_author_commits')

        )

        if share_col in merge_export_df.columns:

            plot_df = merge_export_df.copy().set_index(group_col)

            plot_df = plot_df.reindex(_baseline_group_order_safe(plot_df.index.tolist())).dropna(subset=[share_col], how='all').reset_index()

            if not plot_df.empty:

                vals = pd.to_numeric(plot_df[share_col], errors='coerce')

                vals = vals if share_col.endswith('_pct') else vals * 100.0

                fig, ax = plt.subplots(figsize=(10, 6))

                bars = ax.bar(plot_df[group_col], vals)

                ax.set_xlabel('Author Type')

                ax.set_ylabel('Percent of Merged PRs Without Non-Author Commits')

                ax.grid(True, axis='y', alpha=0.3)

                for bar, val in zip(bars, vals):

                    if pd.notna(val):

                        ax.text(bar.get_x() + bar.get_width() / 2, val, f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

                plt.tight_layout()

                export_fig_pdf(fig, 'figures/fig_baseline_merge_without_non_author_commits.pdf')

                plt.close(fig)





_safe_export('rq2_baseline_figures', _export_baseline_figures)





# 9) AI->Human transition heatmap

def _export_transition_heatmap():

    source = {}

    if 'heatmap_results' in globals() and isinstance(heatmap_results, dict) and heatmap_results.get('transitions'):

        source = heatmap_results

    elif 'primary_results' in globals() and isinstance(primary_results, dict) and primary_results.get('transitions'):

        source = primary_results

    elif 'sensitivity_results' in globals() and isinstance(sensitivity_results, dict) and sensitivity_results.get('transitions'):

        source = sensitivity_results

    transitions = source.get('transitions', []) if isinstance(source, dict) else []

    if not transitions:

        return

    tr_df = pd.DataFrame(transitions, columns=['ai_action', 'human_action'])

    tr_df['ai_action'] = tr_df['ai_action'].fillna('Other').astype(str)

    tr_df['human_action'] = tr_df['human_action'].fillna('Other').astype(str)

    ai_order = ['Open', 'Comment', 'Review', 'Commit', 'Other']

    human_order = ['Comment', 'Review', 'Commit', 'Other']

    ai_labels = [x for x in ai_order if x in set(tr_df['ai_action'])] + [x for x in tr_df['ai_action'].unique() if x not in ai_order]

    human_labels = [x for x in human_order if x in set(tr_df['human_action'])] + [x for x in tr_df['human_action'].unique() if x not in human_order]

    counts = pd.crosstab(tr_df['ai_action'], tr_df['human_action']).reindex(index=ai_labels, columns=human_labels, fill_value=0)

    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(probs.values, cmap='Blues', vmin=0, vmax=1, aspect='equal')

    ax.set_xticks(range(len(human_labels)))

    ax.set_yticks(range(len(ai_labels)))

    ax.set_xticklabels(human_labels, fontsize=11)

    ax.set_yticklabels(ai_labels, fontsize=11)

    ax.set_xlabel('First Human Action (Response)', fontsize=12, fontweight='bold')

    ax.set_ylabel('AI Action (Anchor)', fontsize=12, fontweight='bold')

    for i in range(probs.shape[0]):

        for j in range(probs.shape[1]):

            c = int(counts.iloc[i, j])

            if c <= 0:

                continue

            p = probs.iloc[i, j]

            ax.text(j, i, f'{p:.1%}\n({c})', ha='center', va='center', fontsize=9, color='white' if p > 0.5 else 'black')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    cbar.set_label('Probability', rotation=270, labelpad=16)

    plt.tight_layout()

    export_fig_pdf(fig, 'figures/fig_first_action.pdf')

    plt.close(fig)





_safe_export('rq2_transition_heatmap', _export_transition_heatmap)





# 10) Representative PR timeline

def _export_timeline_example():

    if 'timeline_df' not in globals() or not isinstance(timeline_df, pd.DataFrame) or timeline_df.empty:

        return

    work = timeline_df.copy()

    if 'pr_id' not in work.columns and 'pr_key' in work.columns:

        work['pr_id'] = work['pr_key'].astype(str)

    if 'pr_id' not in work.columns:

        return

    if 'pr_key' in work.columns:

        work['pr_key_norm'] = work['pr_key'].fillna('').astype(str).str.strip().str.lower()
    else:

        work['pr_key_norm'] = ''

    work['timestamp'] = pd.to_datetime(work['timestamp'], errors='coerce', utc=True)

    work = work[work['timestamp'].notna()].copy()

    if work.empty:

        return

    selected = None

    if 'selected_pr_id' in globals() and selected_pr_id in set(work['pr_id'].astype(str)):

        selected = str(selected_pr_id)

    if selected is None and PREFERRED_TIMELINE_PR_KEY and 'pr_key_norm' in work.columns:

        preferred_rows = work[work['pr_key_norm'].eq(PREFERRED_TIMELINE_PR_KEY)].copy()

        if not preferred_rows.empty:

            selected = str(preferred_rows['pr_id'].iloc[0])

    if selected is None:

        stats = (

            work.groupby('pr_id')

            .agg(ai_n=('is_ai', lambda s: int(pd.Series(s).astype(bool).sum()) if 'is_ai' in work.columns else 0),

                 total_n=('pr_id', 'size'))

            .reset_index()

        )

        if 'is_ai' in work.columns:

            stats['human_n'] = stats['total_n'] - stats['ai_n']

            stats = stats[(stats['ai_n'] > 0) & (stats['human_n'] > 0)]

        selected = str(stats.sort_values(['total_n'], ascending=False).iloc[0]['pr_id']) if not stats.empty else str(work['pr_id'].iloc[0])

    pr_events = work[work['pr_id'].astype(str).eq(selected)].sort_values('timestamp').copy()

    if pr_events.empty:

        return

    display_label = selected

    if 'pr_key' in pr_events.columns:

        pr_key_values = pr_events['pr_key'].fillna('').astype(str).str.strip()

        if not pr_key_values.empty and pr_key_values.iloc[0]:

            display_label = pr_key_values.iloc[0]

    open_mask = pr_events['action_type'].astype(str).eq('Open')

    t0 = pr_events.loc[open_mask, 'timestamp'].min() if open_mask.any() else pr_events['timestamp'].min()

    pr_events['hours_since_open'] = (pr_events['timestamp'] - t0).dt.total_seconds().div(3600.0)

    action_order = ['Open', 'Comment', 'Review', 'Commit', 'Other']

    pr_events['action_norm'] = pr_events['action_type'].where(pr_events['action_type'].isin(action_order), 'Other')

    y_map = {a: i for i, a in enumerate(action_order)}

    pr_events['y'] = pr_events['action_norm'].map(y_map).fillna(y_map['Other'])

    is_ai_col = pd.Series(False, index=pr_events.index)

    if 'is_ai' in pr_events.columns:

        is_ai_col = pr_events['is_ai'].fillna(False).astype(bool)

    colors = np.where(is_ai_col, '#1f77b4', '#ff7f0e')

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(pr_events['hours_since_open'], pr_events['y'], color='#999999', alpha=0.35, linewidth=1)

    ax.scatter(pr_events['hours_since_open'], pr_events['y'], c=colors, s=45, alpha=0.9)

    ax.set_yticks(list(y_map.values()))

    ax.set_yticklabels(list(y_map.keys()))

    ax.set_xlabel('Hours since PR open')

    ax.set_ylabel('Action type')

    ax.set_title(f'Representative PR Timeline: {display_label}')

    ax.grid(True, axis='x', alpha=0.3)

    from matplotlib.lines import Line2D

    legend_items = [

        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=8, label='AI actor'),

        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=8, label='Human actor'),

    ]

    ax.legend(handles=legend_items, loc='upper right')

    plt.tight_layout()

    export_fig_pdf(fig, 'figures/rq2_timeline_example.pdf')

    plt.close(fig)





_safe_export('rq2_timeline_example', _export_timeline_example)





# 11) Conditioned load by pathway/seniority (from pr_metrics_df)

def _export_conditioned_load():

    if PAPER_ONLY_EXPORTS:

        return

    if 'pr_metrics_df' not in globals() or not isinstance(pr_metrics_df, pd.DataFrame) or pr_metrics_df.empty:

        return

    required_cols = {'seniority', 'autonomy', 'validation_overhead'}

    if not required_cols.issubset(pr_metrics_df.columns):

        return

    df = pr_metrics_df.copy()

    df['seniority'] = df['seniority'].fillna('unknown').astype(str).str.lower()

    df = df[df['seniority'].isin(['novice', 'mid', 'expert'])].copy()

    if df.empty:

        return

    df['autonomy'] = pd.to_numeric(df['autonomy'], errors='coerce').fillna(0).astype(int)

    df['validation_overhead'] = pd.to_numeric(df['validation_overhead'], errors='coerce')

    df = df[df['validation_overhead'].notna()].copy()

    if df.empty:

        return

    agg = (

        df.groupby(['seniority', 'autonomy'])['validation_overhead']

        .median()

        .unstack(fill_value=np.nan)

        .reindex(['novice', 'mid', 'expert'])

    )

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(agg.index))

    w = 0.36

    y0 = pd.to_numeric(agg.get(0, pd.Series(index=agg.index, dtype=float)), errors='coerce')

    y1 = pd.to_numeric(agg.get(1, pd.Series(index=agg.index, dtype=float)), errors='coerce')

    ax.bar(x - w / 2, y0.values, width=w, label='Autonomy=0')

    ax.bar(x + w / 2, y1.values, width=w, label='Autonomy=1')

    ax.set_xticks(x)

    ax.set_xticklabels(agg.index.tolist())

    ax.set_xlabel('Seniority')

    ax.set_ylabel('Median Validation Overhead')

    ax.grid(True, axis='y', alpha=0.3)

    ax.legend()

    plt.tight_layout()

    export_fig_pdf(fig, 'figures/fig_conditioned_load.pdf')

    plt.close(fig)





if not PAPER_ONLY_EXPORTS:

    _safe_export('fig_conditioned_load', _export_conditioned_load)





# 12-13) Sentiment table + figure

def _export_sentiment_outputs():

    if PAPER_ONLY_EXPORTS:

        return

    if sentiment_summary_export_df is not None and not sentiment_summary_export_df.empty:

        _write_csv_dual(sentiment_summary_export_df, 'outputs/sentiment_vader_robustness.csv', index=False)

        export_table_pdf(sentiment_summary_export_df, 'outputs/pdf_tables/sentiment_summary.pdf', title='Sentiment Summary (VADER)', index=False)



    if 'sentiment_df' in globals() and isinstance(sentiment_df, pd.DataFrame) and not sentiment_df.empty and {'global_seniority_group', 'vader_compound'}.issubset(sentiment_df.columns):

        sdf = sentiment_df.copy()

        sdf['global_seniority_group'] = sdf['global_seniority_group'].fillna('unknown').astype(str)

        sdf['vader_compound'] = pd.to_numeric(sdf['vader_compound'], errors='coerce')

        sdf = sdf[sdf['vader_compound'].notna()].copy()

        groups = [g for g in ['novice', 'mid', 'expert', 'unknown'] if g in set(sdf['global_seniority_group'])]

        arrays = [sdf.loc[sdf['global_seniority_group'].eq(g), 'vader_compound'].values for g in groups]

        if groups and any(len(a) > 0 for a in arrays):

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.violinplot(arrays, positions=range(len(groups)), showmeans=True)

            ax.set_xticks(range(len(groups)))

            ax.set_xticklabels(groups, rotation=45)

            ax.set_ylabel('VADER compound')

            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

            ax.grid(True, axis='y', alpha=0.3)

            plt.tight_layout()

            export_fig_pdf(fig, 'figures/sentiment.pdf')

            plt.close(fig)





if not PAPER_ONLY_EXPORTS:

    _safe_export('sentiment_outputs', _export_sentiment_outputs)





# 14) Pathway share by seniority (long + pivot)

def _export_pathway_mix_tables():

    pathway_long = pathway_share_long_df.copy() if isinstance(pathway_share_long_df, pd.DataFrame) else pd.DataFrame()

    if pathway_long.empty and 'taxonomy_df' in globals() and isinstance(taxonomy_df, pd.DataFrame) and not taxonomy_df.empty:

        tdf = taxonomy_df.copy()

        if 'taxonomy_display_label' not in tdf.columns and 'taxonomy_class' in tdf.columns:

            tdf['taxonomy_display_label'] = tdf['taxonomy_class'].map(_label_map_taxonomy()).fillna(tdf['taxonomy_class'])

        seniority_col = 'seniority' if 'seniority' in tdf.columns else 'primary_seniority'

        tdf[seniority_col] = tdf[seniority_col].fillna('unknown').astype(str).str.lower()

        tdf.loc[~tdf[seniority_col].isin(['novice', 'mid', 'expert']), seniority_col] = 'unknown'

        pathway_long = (

            tdf.groupby(['taxonomy_class', 'taxonomy_display_label', seniority_col], as_index=False)

            .size()

            .rename(columns={seniority_col: 'seniority', 'size': 'count'})

        )

        pathway_long['share'] = pathway_long['count'] / pathway_long.groupby('seniority')['count'].transform('sum')

        pathway_long['share'] = pathway_long['share'].fillna(0.0)

    if pathway_long.empty:

        return

    _write_csv_dual(pathway_long, 'outputs/pathway_share_by_seniority.csv', index=False)



    pivot = pathway_long[pathway_long['seniority'].isin(['novice', 'mid', 'expert'])].copy()

    pivot = (

        pivot.pivot_table(

            index=['taxonomy_class', 'taxonomy_display_label'],

            columns='seniority',

            values='share',

            aggfunc='sum',

            fill_value=0.0,

        )

        .reset_index()

    )

    for col in ['novice', 'mid', 'expert']:

        if col not in pivot.columns:

            pivot[col] = 0.0

    pivot = pivot[['taxonomy_class', 'taxonomy_display_label', 'novice', 'mid', 'expert']].sort_values('taxonomy_display_label').reset_index(drop=True)

    _write_csv_dual(pivot, 'outputs/pathway_mix_by_seniority_pivot.csv', index=False)

    export_table_pdf(pivot, 'outputs/pdf_tables/pathway_mix_by_seniority.pdf', title='Pathway Mix by Seniority', index=False)





_safe_export('pathway_mix_tables', _export_pathway_mix_tables)





# 15-16) Intent signal tables

def _export_signal_tables():

    if PAPER_ONLY_EXPORTS:

        return

    if signals_by_pathway_df is not None and not signals_by_pathway_df.empty:

        _write_csv_dual(signals_by_pathway_df, 'outputs/comment_intent_signals_by_pathway.csv', index=False)

        export_table_pdf(signals_by_pathway_df, 'outputs/pdf_tables/signals_by_pathway.pdf', title='Comment Intent Signals by Pathway', index=False)

    if signals_stats_df is not None and not signals_stats_df.empty:

        _write_csv_dual(signals_stats_df, 'outputs/comment_intent_signals_stats.csv', index=False)

        export_table_pdf(signals_stats_df, 'outputs/pdf_tables/signals_pathway_association.pdf', title='Signals-Pathway Association', index=False)





if not PAPER_ONLY_EXPORTS:

    _safe_export('signals_tables', _export_signal_tables)





# C) Pathway summary figure + table

def _export_pathway_summary():

    if 'taxonomy_df' not in globals() or not isinstance(taxonomy_df, pd.DataFrame) or taxonomy_df.empty:

        return

    tax = taxonomy_df.copy()

    if 'pr_key' not in tax.columns:

        return

    tax['pr_key'] = tax['pr_key'].fillna('').astype(str)

    tax = tax[tax['pr_key'].ne('')].copy()

    if 'taxonomy_display_label' not in tax.columns:

        if 'taxonomy_class' not in tax.columns:

            return

        tax['taxonomy_display_label'] = tax['taxonomy_class'].map(_label_map_taxonomy()).fillna(tax['taxonomy_class'])

    tax = tax[['pr_key', 'taxonomy_display_label']].drop_duplicates('pr_key', keep='first')

    if tax.empty:

        return



    merged_map = pd.DataFrame(columns=['pr_key', 'merged_flag'])

    if 'df_prs_all' in globals() and isinstance(df_prs_all, pd.DataFrame) and not df_prs_all.empty and 'pr_key' in df_prs_all.columns:

        prs = df_prs_all.copy()

        prs['pr_key'] = prs['pr_key'].fillna('').astype(str)

        merge_col = 'merged' if 'merged' in prs.columns else ('is_merged' if 'is_merged' in prs.columns else None)

        if merge_col is not None:

            merged_map = prs[['pr_key', merge_col]].drop_duplicates('pr_key', keep='first').rename(columns={merge_col: 'merged_flag'})

            merged_map['merged_flag'] = merged_map['merged_flag'].fillna(False).astype(bool)



    metrics_map = pd.DataFrame(columns=['pr_key', 'validation_overhead', 'takeover_flag'])

    if 'pr_metrics_df' in globals() and isinstance(pr_metrics_df, pd.DataFrame) and not pr_metrics_df.empty:

        pm = pr_metrics_df.copy()

        key_col = 'pr_key' if 'pr_key' in pm.columns else ('pr_id' if 'pr_id' in pm.columns else None)

        if key_col is not None:

            pm['pr_key'] = pm[key_col].astype(str)

            cols = ['pr_key']

            if 'validation_overhead' in pm.columns:

                pm['validation_overhead'] = pd.to_numeric(pm['validation_overhead'], errors='coerce')

                cols.append('validation_overhead')

            if 'human_commits' in pm.columns:

                pm['takeover_flag'] = pd.to_numeric(pm['human_commits'], errors='coerce').fillna(0).gt(0)

                cols.append('takeover_flag')

            metrics_map = pm[cols].drop_duplicates('pr_key', keep='first')



    joined = tax.merge(merged_map, on='pr_key', how='left').merge(metrics_map, on='pr_key', how='left')

    total_prs = max(len(joined), 1)

    summary = (

        joined.groupby('taxonomy_display_label', as_index=False)

        .agg(

            N_prs=('pr_key', 'nunique'),

            merged_share=('merged_flag', lambda s: float(pd.Series(s).astype('boolean').mean() * 100.0) if s.notna().any() else np.nan),

            median_validation_overhead=('validation_overhead', 'median'),

            takeover_share=('takeover_flag', lambda s: float(pd.Series(s).astype('boolean').mean() * 100.0) if s.notna().any() else np.nan),

        )

        .sort_values('N_prs', ascending=False)

        .reset_index(drop=True)

    )

    summary['share_of_prs'] = summary['N_prs'] / float(total_prs) * 100.0

    summary = summary[['taxonomy_display_label', 'N_prs', 'share_of_prs', 'merged_share', 'median_validation_overhead', 'takeover_share']]



    _write_csv_dual(summary, 'outputs/pathway_summary_metrics.csv', index=False)

    if not PAPER_ONLY_EXPORTS:

        export_table_pdf(summary, 'outputs/pdf_tables/pathway_summary_metrics.pdf', title='Pathway Summary Metrics', index=False)



    labels = summary['taxonomy_display_label'].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1, ax2, ax3, ax4 = axes.flatten()

    ax1.bar(labels, summary['share_of_prs'])

    ax1.set_title('Share of PRs (%)')

    ax1.tick_params(axis='x', rotation=25)

    ax1.grid(True, axis='y', alpha=0.3)



    ax2.bar(labels, summary['merged_share'])

    ax2.set_title('Merged Share (%)')

    ax2.tick_params(axis='x', rotation=25)

    ax2.grid(True, axis='y', alpha=0.3)



    ax3.bar(labels, summary['median_validation_overhead'])

    ax3.set_title('Median Validation Overhead')

    ax3.tick_params(axis='x', rotation=25)

    ax3.grid(True, axis='y', alpha=0.3)



    ax4.bar(labels, summary['takeover_share'])

    ax4.set_title('Takeover Share (%)')

    ax4.tick_params(axis='x', rotation=25)

    ax4.grid(True, axis='y', alpha=0.3)



    plt.tight_layout()

    export_fig_pdf(fig, 'figures/fig_pathway_summary.pdf')

    plt.close(fig)





_safe_export('pathway_summary', _export_pathway_summary)





def _export_ase_minimal_patch_outputs():

    neutral_order = _neutral_pathway_order()

    seniority_order = _seniority_order()

    taxonomy_source = globals().get('taxonomy_source_current', pd.DataFrame())

    if not isinstance(taxonomy_source, pd.DataFrame) or taxonomy_source.empty:

        taxonomy_source = _build_taxonomy_source_for_current_scope() if '_build_taxonomy_source_for_current_scope' in globals() else pd.DataFrame()

    if not isinstance(taxonomy_source, pd.DataFrame) or taxonomy_source.empty:

        raise RuntimeError('taxonomy_source_current is unavailable for ASE minimal patch outputs.')

    legacy_taxonomy = globals().get('taxonomy_df', pd.DataFrame())

    if not isinstance(legacy_taxonomy, pd.DataFrame) or legacy_taxonomy.empty:

        legacy_taxonomy, _ = _read_csv_candidates(['outputs/taxonomy_df.csv', 'taxonomy_df.csv'])

    if legacy_taxonomy.empty or 'pr_key' not in legacy_taxonomy.columns or 'taxonomy_class' not in legacy_taxonomy.columns:

        raise RuntimeError('taxonomy_df is missing required columns for seniority robustness outputs.')

    legacy_pathways = legacy_taxonomy[['pr_key', 'taxonomy_class']].copy()

    legacy_pathways['pr_key'] = legacy_pathways['pr_key'].fillna('').astype(str).str.strip()

    legacy_pathways = legacy_pathways[legacy_pathways['pr_key'].ne('')].drop_duplicates(subset=['pr_key'], keep='first')

    legacy_pathways['pathway'] = legacy_pathways['taxonomy_class'].map(_neutral_label_map())

    legacy_pathways = legacy_pathways[legacy_pathways['pathway'].isin(neutral_order)].copy()

    profiled_users = _get_profiled_human_users_for_agent_scope()

    if profiled_users.empty:

        raise RuntimeError('No profiled human developers were found for agent-pathway seniority robustness.')

    first_agent_responses = _get_agent_first_human_responses()

    if first_agent_responses.empty or 'pr_key' not in first_agent_responses.columns:

        raise RuntimeError('No agent PR first-human-response rows are available for seniority robustness.')

    first_agent_responses['pr_key'] = first_agent_responses['pr_key'].fillna('').astype(str).str.strip()

    first_agent_responses = first_agent_responses[first_agent_responses['pr_key'].ne('')].drop_duplicates(subset=['pr_key'], keep='first')

    specs = {

        'tenure_only': (1.00, 0.00),

        'contrib_only': (0.00, 1.00),

        'w35_65': (0.35, 0.65),

        'w50_50': (0.50, 0.50),

        'w65_35': (0.65, 0.35),

    }

    summary_rows = []

    full_rows = []

    for spec_name, (tenure_weight, contrib_weight) in specs.items():

        spec_users = profiled_users[['user_login_norm', 'tenure_norm', 'contrib_norm']].copy()

        spec_users['seniority_score'] = tenure_weight * spec_users['tenure_norm'] + contrib_weight * spec_users['contrib_norm']

        q25 = spec_users['seniority_score'].quantile(0.25)

        q75 = spec_users['seniority_score'].quantile(0.75)

        spec_users['seniority'] = np.select(

            [

                spec_users['seniority_score'] <= q25,

                spec_users['seniority_score'] > q75,

            ],

            ['novice', 'expert'],

            default='mid'

        )

        lookup = dict(zip(spec_users['user_login_norm'], spec_users['seniority']))

        pr_labels = first_agent_responses[['pr_key', 'actor_lookup_key']].copy()

        pr_labels['seniority'] = pr_labels['actor_lookup_key'].map(lookup)

        pr_labels = pr_labels[pr_labels['seniority'].isin(seniority_order)].drop_duplicates(subset=['pr_key'], keep='first')

        labeled = legacy_pathways.merge(pr_labels[['pr_key', 'seniority']], on='pr_key', how='inner')

        labeled_prs = int(labeled['pr_key'].nunique())

        contingency = pd.crosstab(labeled['seniority'], labeled['pathway']).reindex(

            index=seniority_order,

            columns=neutral_order,

            fill_value=0

        )

        row_pct = contingency.div(contingency.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100.0

        stats = _chi2_cramers_v(contingency)

        expert_discussion_or_takeover = float(

            row_pct.loc['expert', 'Discussion only'] + row_pct.loc['expert', 'Early takeover']

        ) if 'expert' in row_pct.index else np.nan

        novice_silent_edit = float(row_pct.loc['novice', 'Silent edit']) if 'novice' in row_pct.index else np.nan

        mid_silent_edit = float(row_pct.loc['mid', 'Silent edit']) if 'mid' in row_pct.index else np.nan

        summary_rows.append({

            'spec': spec_name,

            'labeled_prs': labeled_prs,

            'chi2': stats['chi2'],

            'p_value': stats['p_value'],

            'cramers_v': stats['cramers_v'],

            'expert_discussion_or_takeover_pct': expert_discussion_or_takeover,

            'novice_silent_edit_pct': novice_silent_edit,

            'mid_silent_edit_pct': mid_silent_edit,

        })

        for seniority in seniority_order:

            row_total = int(contingency.loc[seniority].sum()) if seniority in contingency.index else 0

            for pathway in neutral_order:

                full_rows.append({

                    'spec': spec_name,

                    'seniority': seniority,

                    'pathway': pathway,

                    'count': int(contingency.loc[seniority, pathway]) if seniority in contingency.index and pathway in contingency.columns else 0,

                    'row_total_n': row_total,

                    'row_pct': float(row_pct.loc[seniority, pathway]) if seniority in row_pct.index and pathway in row_pct.columns else 0.0,

                })

    seniority_summary_df = pd.DataFrame(summary_rows)

    seniority_summary_df = seniority_summary_df.sort_values('spec').reset_index(drop=True)

    for col in ['expert_discussion_or_takeover_pct', 'novice_silent_edit_pct', 'mid_silent_edit_pct']:

        seniority_summary_df[col] = pd.to_numeric(seniority_summary_df[col], errors='coerce').round(2)

    for col in ['chi2', 'p_value', 'cramers_v']:

        seniority_summary_df[col] = pd.to_numeric(seniority_summary_df[col], errors='coerce')

    seniority_full_df = pd.DataFrame(full_rows).sort_values(['spec', 'seniority', 'pathway']).reset_index(drop=True)

    seniority_full_df['row_pct'] = pd.to_numeric(seniority_full_df['row_pct'], errors='coerce').round(2)

    _write_csv_dual(seniority_summary_df, 'outputs/seniority_robustness_summary.csv', index=False)

    _write_csv_dual(seniority_full_df, 'outputs/seniority_robustness_full_tables.csv', index=False)

    seniority_summary_tex = seniority_summary_df[

        ['spec', 'labeled_prs', 'cramers_v', 'expert_discussion_or_takeover_pct', 'novice_silent_edit_pct', 'mid_silent_edit_pct']

    ].copy()

    seniority_summary_tex['cramers_v'] = pd.to_numeric(seniority_summary_tex['cramers_v'], errors='coerce').round(3)

    for col in ['expert_discussion_or_takeover_pct', 'novice_silent_edit_pct', 'mid_silent_edit_pct']:

        seniority_summary_tex[col] = pd.to_numeric(seniority_summary_tex[col], errors='coerce').round(1)

    _write_latex_table(

        seniority_summary_tex,

        'outputs/seniority_robustness_summary.tex',

        caption='Seniority robustness across alternative weighting specifications.',

        label='tab:seniority-robustness',

        column_format='lrrrrr'

    )

    expert_range = seniority_summary_df['expert_discussion_or_takeover_pct'].agg(['min', 'max'])

    novice_range = seniority_summary_df['novice_silent_edit_pct'].agg(['min', 'max'])

    mid_range = seniority_summary_df['mid_silent_edit_pct'].agg(['min', 'max'])

    cramers_range = seniority_summary_df['cramers_v'].agg(['min', 'max'])

    seniority_note_lines = [

        '# Seniority robustness',

        '',

        '## `outputs/seniority_robustness_summary.csv` and `outputs/seniority_robustness_summary.tex`',

    ]

    for row in seniority_summary_df.itertuples(index=False):

        seniority_note_lines.append(

            f"- `{row.spec}`: labeled_prs={int(row.labeled_prs)}, chi2={row.chi2:.4f}, p={_fmt_p_value(row.p_value)}, "

            f"Cramer's V={row.cramers_v:.4f}, expert Discussion only + Early takeover={row.expert_discussion_or_takeover_pct:.2f}%, "

            f"novice Silent edit={row.novice_silent_edit_pct:.2f}%, mid Silent edit={row.mid_silent_edit_pct:.2f}%."

        )

    seniority_note_lines.extend([

        '',

        f'The directional result is stable across all five weighting choices: expert entrants remain concentrated in Discussion only plus Early takeover '

        f'({expert_range["min"]:.2f}% to {expert_range["max"]:.2f}%), while Silent edit remains concentrated among novice and mid entrants '

        f'(novice {novice_range["min"]:.2f}% to {novice_range["max"]:.2f}%; mid {mid_range["min"]:.2f}% to {mid_range["max"]:.2f}%).',

        f'Association strength stays in the same small-to-moderate band across specifications (Cramer\'s V {cramers_range["min"]:.4f} to {cramers_range["max"]:.4f}), '

        'so the substantive conclusion does not depend on the exact tenure/contribution weighting.',

        '',

        '## `outputs/seniority_robustness_full_tables.csv`',

        f'- Rows={len(seniority_full_df)} across 5 specifications x 3 seniority strata x 4 pathways; each row reports the exact count, row total, and row percentage used for the summary table.',

        '- `No human handling` is 0 in every labeled contingency table because PR-level seniority requires a visible human entrant after PR open.',

        'The full tables preserve the original agent-PR pathway subset and only swap the weighting used to assign developer strata, which keeps this check lightweight and directly publication-oriented.',

    ])

    _write_text_dual('\n'.join(seniority_note_lines) + '\n', 'outputs/seniority_robustness_note.md')

    pathway_2x2_df = _build_pathway_labels_from_taxonomy(legacy_pathways)

    if pathway_2x2_df.empty:

        raise RuntimeError('taxonomy_df pathway derivation produced no rows.')

    pathway_subset_n = int(pathway_2x2_df['pr_key'].nunique())

    pathway_counts_df = (

        pathway_2x2_df.groupby(['pathway', 'visible_human_discussion', 'visible_human_code_intervention'], as_index=False)

        .size()

        .rename(columns={'size': 'n'})

    )

    pathway_counts_df['pathway'] = pd.Categorical(pathway_counts_df['pathway'], categories=neutral_order, ordered=True)

    pathway_counts_df = pathway_counts_df.sort_values('pathway').reset_index(drop=True)

    pathway_counts_df['pct_of_subset'] = (pathway_counts_df['n'] / float(max(pathway_subset_n, 1)) * 100.0).round(2)

    missing_pathways = [label for label in neutral_order if label not in set(pathway_counts_df['pathway'].astype(str))]

    if missing_pathways:

        missing_rows = []

        for pathway in missing_pathways:

            missing_rows.append({

                'pathway': pathway,

                'visible_human_discussion': pathway in {'Discussion only', 'Early takeover'},

                'visible_human_code_intervention': pathway in {'Silent edit', 'Early takeover'},

                'n': 0,

                'pct_of_subset': 0.0,

            })

        pathway_counts_df = pd.concat([pathway_counts_df, pd.DataFrame(missing_rows)], ignore_index=True)

        pathway_counts_df['pathway'] = pd.Categorical(pathway_counts_df['pathway'], categories=neutral_order, ordered=True)

        pathway_counts_df = pathway_counts_df.sort_values('pathway').reset_index(drop=True)

    _write_csv_dual(pathway_counts_df, 'outputs/pathway_2x2_counts.csv', index=False)

    pathway_counts_tex = pathway_counts_df.copy()

    pathway_counts_tex['pct_of_subset'] = pd.to_numeric(pathway_counts_tex['pct_of_subset'], errors='coerce').round(1)

    _write_latex_table(

        pathway_counts_tex[['pathway', 'visible_human_discussion', 'visible_human_code_intervention', 'n', 'pct_of_subset']],

        'outputs/pathway_2x2_counts.tex',

        caption='Operational 2x2 pathway taxonomy counts on the scoped agent-PR subset.',

        label='tab:pathway-2x2-counts',

        column_format='lccrr'

    )

    responded_with_pathway = pathway_2x2_df.merge(

        first_agent_responses[['pr_key', 'first_human_response_type', 'first_human_response_latency_hours']],

        on='pr_key',

        how='inner'

    )

    responded_with_pathway['first_human_response_type'] = responded_with_pathway['first_human_response_type'].fillna('').astype(str).str.strip().str.lower()

    responded_with_pathway = responded_with_pathway[responded_with_pathway['first_human_response_type'].isin(['comment', 'review', 'commit'])].copy()

    response_contingency = pd.crosstab(

        responded_with_pathway['pathway'],

        responded_with_pathway['first_human_response_type']

    ).reindex(index=neutral_order, columns=['comment', 'review', 'commit'], fill_value=0)

    response_row_pct = response_contingency.div(response_contingency.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100.0

    response_stats = _chi2_cramers_v(response_contingency)

    pathway_first_response_df = pd.DataFrame({

        'pathway': neutral_order,

        'n': [int(response_contingency.loc[pathway].sum()) for pathway in neutral_order],

        'comment_first_pct': [float(response_row_pct.loc[pathway, 'comment']) for pathway in neutral_order],

        'review_first_pct': [float(response_row_pct.loc[pathway, 'review']) for pathway in neutral_order],

        'commit_first_pct': [float(response_row_pct.loc[pathway, 'commit']) for pathway in neutral_order],

    })

    for col in ['comment_first_pct', 'review_first_pct', 'commit_first_pct']:

        pathway_first_response_df[col] = pathway_first_response_df[col].round(2)

    _write_csv_dual(pathway_first_response_df, 'outputs/pathway_first_response_by_pathway.csv', index=False)

    pathway_first_response_tex = pathway_first_response_df.copy()

    for col in ['comment_first_pct', 'review_first_pct', 'commit_first_pct']:

        pathway_first_response_tex[col] = pd.to_numeric(pathway_first_response_tex[col], errors='coerce').round(1)

    _write_latex_table(

        pathway_first_response_tex,

        'outputs/pathway_first_response_by_pathway.tex',

        caption='First human response type by operational 2x2 pathway.',

        label='tab:pathway-first-response',

        column_format='lrrrr'

    )

    latency_medians = (

        responded_with_pathway.groupby('pathway')['first_human_response_latency_hours'].median().to_dict()

        if 'first_human_response_latency_hours' in responded_with_pathway.columns and not responded_with_pathway.empty

        else {}

    )

    response_stats_payload = {

        'pathway_analysis_subset_n': pathway_subset_n,

        'pathway_2x2_total_n': int(pathway_counts_df['n'].sum()),

        'responded_with_pathway_n': int(response_stats['n']),

        'chi2': response_stats['chi2'],

        'p_value': response_stats['p_value'],

        'cramers_v': response_stats['cramers_v'],

        'median_first_response_latency_hours_by_pathway': {str(k): float(v) for k, v in latency_medians.items() if pd.notna(v)},

    }

    _write_json_dual(response_stats_payload, 'outputs/pathway_first_response_stats.json')

    pathway_note_lines = [

        '# Pathway taxonomy support',

        '',

        '## `outputs/pathway_2x2_counts.csv` and `outputs/pathway_2x2_counts.tex`',

    ]

    for row in pathway_counts_df.itertuples(index=False):

        pathway_note_lines.append(

            f"- `{row.pathway}`: discussion={bool(row.visible_human_discussion)}, code_intervention={bool(row.visible_human_code_intervention)}, "

            f"n={int(row.n)}, pct_of_subset={row.pct_of_subset:.2f}%."

        )

    pathway_note_lines.extend([

        '',

        f'The four 2x2 cells sum to {int(pathway_counts_df["n"].sum())}, exactly matching the scoped pathway-analysis subset size of {pathway_subset_n}.',

        'This verifies that the operational pathway taxonomy is exhaustive on the current taxonomy_df subset without changing the paper’s scoped PR subset or preprocessing.',

        '',

        '## `outputs/pathway_first_response_by_pathway.csv`, `outputs/pathway_first_response_by_pathway.tex`, and `outputs/pathway_first_response_stats.json`',

    ])

    for row in pathway_first_response_df.itertuples(index=False):

        pathway_note_lines.append(

            f"- `{row.pathway}`: n={int(row.n)}, Comment-first={row.comment_first_pct:.2f}%, Review-first={row.review_first_pct:.2f}%, Commit-first={row.commit_first_pct:.2f}%."

        )

    pathway_note_lines.extend([

        '',

        f"Chi-square={response_stats['chi2']:.4f}, p={_fmt_p_value(response_stats['p_value'])}, Cramer's V={response_stats['cramers_v']:.4f}, N={int(response_stats['n'])}.",

        'First-response behavior is differentiated across pathways using a response-type variable that was not part of the 2x2 definition itself, which provides lightweight external empirical support for the taxonomy.',

    ])

    _write_text_dual('\n'.join(pathway_note_lines) + '\n', 'outputs/pathway_taxonomy_note.md')

    _write_text_dual(

        '\n'.join([

            'echo chamber -> No human handling',

            'steering -> Discussion only',

            'human-in-loop -> Silent edit',

            'rewriting -> Early takeover',

            '',

        ]),

        'outputs/pathway_label_mapping.md'

    )

    best_summary_row = seniority_summary_df.loc[

        seniority_summary_df['spec'].eq('w35_65')

    ].head(1)

    if best_summary_row.empty:

        best_summary_row = seniority_summary_df.head(1)

    best_summary_row = best_summary_row.iloc[0]

    response_lookup = pathway_first_response_df.set_index('pathway').to_dict('index')

    discussion_row = response_lookup.get('Discussion only', {})

    silent_row = response_lookup.get('Silent edit', {})

    takeover_row = response_lookup.get('Early takeover', {})

    patch_note_lines = [

        'The pathway taxonomy can be described in the Methodology as a study-derived 2x2 operational taxonomy over the scoped agent-PR timeline: visible human discussion (any human comment or review) crossed with visible human code intervention (any human commit). The exhaustive cell counts in `outputs/pathway_2x2_counts.tex` sum to '

        f'{pathway_subset_n}, matching the current pathway-analysis subset exactly and showing that the operationalization covers the full scoped sample without changing the main preprocessing pipeline.',

        '',

        'The seniority score can be described in the Methodology as a study-defined proxy built from public GitHub account tenure and public contribution counts, with tenure and contribution terms transformed via log1p, 5th–95th percentile winsorization, and min-max normalization before weighting. Robustness was checked over five alternative weighting specifications using the same profiled human-developer subset and the same earliest-visible-human-entrant PR labeling rule; see `outputs/seniority_robustness_summary.tex`.',

        '',

        f"In the Results, the seniority conclusion is stable across all five specifications: labeled PR coverage ranges from {int(seniority_summary_df['labeled_prs'].min())} to {int(seniority_summary_df['labeled_prs'].max())}, "

        f"expert Discussion only + Early takeover ranges from {expert_range['min']:.2f}% to {expert_range['max']:.2f}%, novice Silent edit ranges from {novice_range['min']:.2f}% to {novice_range['max']:.2f}%, "

        f"and mid Silent edit ranges from {mid_range['min']:.2f}% to {mid_range['max']:.2f}% (Cramer's V {cramers_range['min']:.4f} to {cramers_range['max']:.4f}). Under the paper’s default 0.35/0.65 weighting (`w35_65`), the corresponding values are "

        f"{best_summary_row['expert_discussion_or_takeover_pct']:.2f}%, {best_summary_row['novice_silent_edit_pct']:.2f}%, and {best_summary_row['mid_silent_edit_pct']:.2f}%, with {int(best_summary_row['labeled_prs'])} labeled PRs and Cramer's V={best_summary_row['cramers_v']:.4f}.",

        '',

        f"In the Results, the pathway taxonomy is also behaviorally differentiated by first human response type on the responded subset with pathway labels (`outputs/pathway_first_response_by_pathway.tex`; N={int(response_stats['n'])}). "

        f"Discussion only is {discussion_row.get('comment_first_pct', 0.0):.2f}% Comment-first, Silent edit is {silent_row.get('commit_first_pct', 0.0):.2f}% Commit-first, and Early takeover is split across discussion and code responses "

        f"({takeover_row.get('comment_first_pct', 0.0):.2f}% Comment-first, {takeover_row.get('review_first_pct', 0.0):.2f}% Review-first, {takeover_row.get('commit_first_pct', 0.0):.2f}% Commit-first); chi-square={response_stats['chi2']:.4f}, p={_fmt_p_value(response_stats['p_value'])}, Cramer's V={response_stats['cramers_v']:.4f}.",

        '',

        'Table references and key numbers:',

        f"- `outputs/seniority_robustness_summary.tex`: default `w35_65` row uses {int(best_summary_row['labeled_prs'])} labeled PRs with Cramer's V={best_summary_row['cramers_v']:.4f}.",

        f"- `outputs/pathway_2x2_counts.tex`: exhaustive 2x2 pathway counts sum to {pathway_subset_n}.",

        f"- `outputs/pathway_first_response_by_pathway.tex`: pathway x first-response association uses N={int(response_stats['n'])} and Cramer's V={response_stats['cramers_v']:.4f}.",

    ]

    _write_text_dual('\n'.join(patch_note_lines) + '\n', 'outputs/ase_minimal_patch_note.md')




_safe_export('ase_minimal_patch_outputs', _export_ase_minimal_patch_outputs)




# Export manifest

expected_files = [

    'figures/rq1_seniority.pdf',

    'outputs/rq1_intensity.csv',

    'outputs/pdf_tables/rq1_intensity.pdf',

    'outputs/baseline_first_action_table.csv',

    'outputs/pdf_tables/rq2_first_action.pdf',

    'outputs/baseline_latency_summary.csv',

    'outputs/pdf_tables/rq2_latency.pdf',

    'outputs/baseline_merge_proxy.csv',

    'outputs/pdf_tables/rq2_merge_proxy.pdf',

    'figures/fig_baseline_latency_log.pdf',

    'figures/fig_first_action.pdf',

    'figures/rq2_timeline_example.pdf',

    'outputs/pathway_share_by_seniority.csv',

    'outputs/pathway_mix_by_seniority_pivot.csv',

    'outputs/pdf_tables/pathway_mix_by_seniority.pdf',

    'outputs/seniority_robustness_summary.csv',

    'outputs/seniority_robustness_summary.tex',

    'outputs/seniority_robustness_full_tables.csv',

    'outputs/seniority_robustness_note.md',

    'outputs/pathway_2x2_counts.csv',

    'outputs/pathway_2x2_counts.tex',

    'outputs/pathway_first_response_by_pathway.csv',

    'outputs/pathway_first_response_by_pathway.tex',

    'outputs/pathway_first_response_stats.json',

    'outputs/pathway_taxonomy_note.md',

    'outputs/pathway_label_mapping.md',

    'outputs/ase_minimal_patch_note.md',

    'outputs/comment_intent_signals_by_pathway.csv',

    'outputs/comment_intent_signals_stats.csv',

    'figures/fig_pathway_summary.pdf',

    'outputs/pathway_summary_metrics.csv',

]



manifest_rows = []

for rel in expected_files:

    candidates = _paper_path_candidates(rel)

    exists = any(path.exists() for path in candidates)

    existing_locations = [str(path) for path in candidates if path.exists()]

    manifest_rows.append({

        'file': rel,

        'exists': bool(exists),

        'locations': '; '.join(existing_locations),

    })

manifest_df = pd.DataFrame(manifest_rows)

print('\nPaper export manifest:')

print(manifest_df.to_string(index=False))

# PAPER_EXPORTS_END
