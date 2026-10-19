"""Central configuration: paths, feature lists, and modeling constants."""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATA_PATH = DATA_DIR / "pipeline2_output.csv"      # final output of the Hop ETL
REJECTED_PATH = DATA_DIR / "rejected_rows.csv"     # Hop's error stream (not used for training)
EARLY_DATA_PATH = DATA_DIR / "pipeline3_output.csv"        # Hop pipeline 3: labeled data + first-6-month activity
REJECTED_EARLY_PATH = DATA_DIR / "rejected_early.csv"      # repos with no early data (deleted / blocked)
MODEL_PATH = MODELS_DIR / "best_model.pkl"
CV_RESULTS_PATH = REPORTS_DIR / "cv_results.csv"
EVAL_REPORT_PATH = REPORTS_DIR / "evaluation_report.txt"

# ---------------------------------------------------------------- target
TARGET = "abandoned"            # 1 = no push for > 365 days at snapshot time
POSITIVE_LABEL = "abandoned"
NEGATIVE_LABEL = "active"

# ---------------------------------------------------------------- features
# Raw count features. Heavily right-skewed, so preprocess applies log1p.
NUMERIC_FEATURES = [
    "assignableUserCount",
    "defaultBranchCommitCount",
    "diskUsageKb",
    "forks",
    "issues",
    "languageCount",
    "pullRequests",
    "repo_age_days",
    "stars",
    "topicCount",
    "watchers",
    # ratios computed in Hop Pipeline 2
    "commits_per_day",
    "stars_per_day",
    "issues_per_pr",
]

# 0/1 flags and description length, computed in Hop Pipeline 2 ("derived flags")
DERIVED_FEATURES = [
    "has_license",
    "has_code_of_conduct",
    "has_description",
    "description_length",
]

# Grouped in Hop Pipeline 2 with fixed lookup lists ("group language" / "group license")
CATEGORICAL_FEATURES = [
    "language_group",   # top 15 languages + "unknown"; everything else -> "other"
    "license_group",    # 6 common licenses + "none"; everything else -> "other"
]

# Real-but-misleading extremes, capped at the 99th percentile of the TRAINING split
# (EDA notebook, section 2.5): kernel copies inherit 400k-700k upstream commits,
# and org-owned repos report every org member as an assignable user (~2,000).
WINSORIZE_FEATURES = ["defaultBranchCommitCount", "commits_per_day", "assignableUserCount"]
WINSORIZE_QUANTILE = 0.99

# Ratios that are always > 0 and mostly < 1: log1p(x) ~ x there and barely reduces
# skew (11.2 -> 0.5 with a plain log for commits_per_day). Every other numeric
# feature has zeros, so it keeps log1p.
PLAIN_LOG_FEATURES = ["commits_per_day", "stars_per_day"]

# Columns deliberately NOT used as features (kept here so the choice is explicit).
# They stay in the CSV: Phase 3 (survival analysis) needs pushedAt / createdAt.
#   - days_since_last_push, pushedAt : the label is defined from these -> direct leakage
#   - isArchived                     : archived ~= abandoned by definition -> near leakage
#   - createdAt                      : redundant with repo_age_days
#   - name, nameWithOwner, owner     : identifiers
# (isFork, forkingAllowed, parent were constant/empty and are dropped in Hop.)
LEAKAGE_COLUMNS = ["days_since_last_push", "pushedAt", "isArchived"]
ID_COLUMNS = ["name", "nameWithOwner", "owner"]

# ---------------------------------------------------------------- modeling
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
PRIMARY_METRIC = "roc_auc"       # threshold-free; robust to the ~84/16 class imbalance
DECISION_THRESHOLD = 0.5         # probability above which a repo is flagged "abandoned"
