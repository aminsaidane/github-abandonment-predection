"""From a GitHub repository link to the model's inputs, and from a score to an explanation.

Live path (needs GITHUB_TOKEN): link -> GitHub API -> the same collector code that built the
training data (`scritpts/collect_early_activity.py`) -> the 14 columns the model reads.
The model is a logistic regression, so the explanation is exact: each feature contributes
coefficient x (its standardized value - the training average), and the contributions add up to
the repo's log-odds minus the average repo's log-odds.
"""
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scritpts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import collect_early_activity as collector  # noqa: E402
from src.predict import predict_risk  # noqa: E402

WINDOW_DAYS = collector.WINDOWS["6m"]
TRAIN_YEARS = (2009, 2023)                       # creation years in the study
KNOWN_LANGS = {"python", "javascript", "java", "typescript", "c++", "c#", "c", "go", "php", "ruby",
               "rust", "swift", "shell", "html", "jupyter notebook", "unknown"}

LABELS = {
    "commits_6m": "Commits in the first 6 months",
    "active_weeks_6m": "Active weeks (out of 26)",
    "commits_last_30d_6m": "Commits in the last 30 days of the window",
    "days_since_last_commit_6m": "Days since the last commit (at month 6)",
    "contributors_6m": "Contributors",
    "nonowner_contributors_6m": "Contributors other than the owner",
    "forks_6m": "Forks",
    "issues_opened_6m": "Issues opened",
    "prs_opened_6m": "Pull requests opened",
    "issues_prs_closed_6m": "Issues and PRs closed",
    "external_issue_authors_6m": "People other than the owner who opened issues or PRs",
    "precreation_commits": "Commits before creation (imported history)",
    "is_org": "Owned by an organization",
    "language_group": "Main language",
}


class RepoError(Exception):
    """A problem the user can fix or must know about (bad link, repo too young, not found)."""
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------------ link handling
def parse_repo(text: str):
    s = (text or "").strip()
    s = re.sub(r"^git@github\.com:", "", s)
    s = re.sub(r"^(https?://)?(www\.)?github\.com/", "", s)
    parts = [p for p in s.split("/") if p]
    if len(parts) < 2:
        raise RepoError("Enter a repository link such as https://github.com/owner/repo")
    owner, repo = parts[0], re.sub(r"\.git$", "", parts[1])
    if not (re.fullmatch(r"[A-Za-z0-9_.-]+", owner) and re.fullmatch(r"[A-Za-z0-9_.-]+", repo)):
        raise RepoError("That does not look like a GitHub repository link.")
    return owner, repo


def lang_group(language):
    """Same rule as Hop pipeline 2: 15 common languages and 'unknown', everything else 'other'."""
    s = str(language).strip().lower() if language else "unknown"
    return s if s in KNOWN_LANGS else "other"


# ------------------------------------------------------------------ live collection
def fetch_live(owner, repo, gh, input_columns):
    """Collect the first 6 months of a repo and return (features, meta, warnings)."""
    status, meta, _ = gh.get(f"{collector.API}/repos/{owner}/{repo}")
    if status in (404, 451):
        raise RepoError("Repository not found. It may be private, renamed or deleted.", 404)
    if status != 200 or not isinstance(meta, dict):
        raise RepoError(f"GitHub answered HTTP {status}. Try again in a moment.", 502)

    now = datetime.now(timezone.utc)
    created = collector.parse_ts(meta["created_at"])
    age_days = (now - created).days
    if age_days < WINDOW_DAYS:
        raise RepoError(f"This repository is only {age_days} days old. The model needs its first "
                        f"{WINDOW_DAYS} days (6 months) of activity.", 422)

    rec = collector.collect_repo(gh, meta["full_name"], meta["created_at"])
    row = collector.build_row(rec)
    if row["early_status"] != "ok":
        raise RepoError(f"Could not collect this repository's history ({row['early_status']}).", 502)

    features = {c: row[c] for c in input_columns if c in row}
    features["is_org"] = int(meta["owner"]["type"] == "Organization")
    features["language_group"] = lang_group(meta.get("language"))
    missing = [c for c in input_columns if c not in features]
    if missing:
        raise RepoError(f"Internal error, missing inputs: {missing}", 500)

    warnings = []
    if created.year > TRAIN_YEARS[1]:
        warnings.append(f"Created in {created.year}: newer than the repositories the model learned from "
                        f"({TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}), so treat the score with extra caution.")
    if meta.get("stargazers_count", 0) < 5:
        warnings.append("Fewer than 5 stars. The model was trained on repositories with 5 or more stars.")
    if rec.get("capped"):
        warnings.append("Very active repository: some counts hit the 1,000-item API limit, as in the training data.")

    pushed = collector.parse_ts(meta["pushed_at"]) if meta.get("pushed_at") else None
    days_quiet = (now - pushed).days if pushed else None
    info = {
        "full_name": meta["full_name"], "url": meta["html_url"], "description": meta.get("description"),
        "created_at": meta["created_at"][:10], "age_days": age_days, "stars": meta.get("stargazers_count"),
        "language": meta.get("language") or "unknown", "owner_type": meta["owner"]["type"],
        "today": ({"last_push": pushed.date().isoformat(), "days_since_last_push": days_quiet,
                   "abandoned_by_definition": days_quiet is not None and days_quiet > 365}
                  if pushed else None),
    }
    return features, info, warnings


# ------------------------------------------------------------------ scoring and explanation
def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _raw_display(col, value):
    if col == "is_org":
        return "Yes" if int(value) else "No"
    return value if col == "language_group" else (int(value) if float(value).is_integer() else round(float(value), 1))


def score_features(features: dict, bundle: dict, reference: dict) -> dict:
    cols = bundle["input_columns"]
    df = pd.DataFrame([{c: features[c] for c in cols}])
    scored = predict_risk(df, bundle).iloc[0]

    pipe = bundle["model"]
    prep, lr = pipe.named_steps["prep"], pipe.named_steps["model"]
    names = list(prep.get_feature_names_out())
    z = prep.transform(df)[0]
    mean = np.array([reference["transformed_means"][n] for n in names])
    contrib = lr.coef_[0] * (z - mean)
    logit = float(lr.intercept_[0] + lr.coef_[0] @ z)
    avg_logit = float(lr.intercept_[0] + lr.coef_[0] @ mean)
    p_raw = float(_sigmoid(logit))

    grouped = {}
    for n, c in zip(names, contrib):
        key = "language_group" if n.startswith("language_group_") else n     # one-hot columns -> one feature
        grouped[key] = grouped.get(key, 0.0) + float(c)
    rows = []
    for col in cols:
        c = grouped[col]
        rows.append({
            "feature": col, "label": LABELS.get(col, col), "value": _raw_display(col, features[col]),
            "typical": reference["typical"].get(col), "contribution": c,
            "points": float(100 * (p_raw - _sigmoid(logit - c))),       # share of the score due to this feature
            "effect": "raises" if c > 0 else "lowers",
        })
    rows.sort(key=lambda r: -abs(r["contribution"]))

    return {
        "p_abandoned": float(scored["p_abandoned"]), "p_active": float(scored["p_active"]),
        "flagged": bool(scored["flagged"]), "threshold": float(bundle["threshold"]),
        "average_repo_p_abandoned": float(_sigmoid(avg_logit)),
        "contributions": rows,
        "inputs": {c: _raw_display(c, features[c]) for c in cols},
        "calibration": bundle["calibration"],
    }
