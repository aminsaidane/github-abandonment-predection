"""Experiments on the chosen model (logistic regression, 6-month window): capping, correlated features, trend.

Each variant changes ONE thing in the reference model and is scored with the same nested CV as the
benchmark (same 5 outer folds, same 20-configuration inner search), on the TRAINING split only.
A variant is compared with the reference fold by fold: it counts as a real change only if its mean
gain or loss is larger than the standard deviation of the 5 paired differences.

Experiments
  1. Capping: every early count capped at the training 95th or 99th percentile before the log.
  2. Correlated features: drop the weaker member of each pair with Spearman |rho| > 0.8.
  3. Trend: activity in months 4-6 (6-month total minus 3-month total), i.e. is the repo slowing down.

Run:
    python -m src.experiments          # about 3 minutes
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression

from src import config
from src.preprocess import build_early_preprocessor, early_feature_columns, early_window_columns, get_early_train_test
from src.train_early import nested_cv

warnings.filterwarnings("ignore")
N_ITER = 20
MODEL = LogisticRegression(max_iter=2000)
SPACE = {"model__C": loguniform(1e-3, 1e2)}

# Spearman |rho| > 0.8 on the training split (main() prints every such pair). Commits, active weeks and
# days since last commit are also correlated with each other but are kept: they are the three strongest
# single signals (src/baselines.py). From the other pairs, the more indirect member is dropped.
CORRELATED_DROPS = {
    "drop contributors_6m (rho 0.83 with non-owner contributors)": ["contributors_6m"],
    "drop issues_prs_closed_6m (rho 0.85 with PRs opened)": ["issues_prs_closed_6m"],
    "drop external_issue_authors_6m (rho 0.81 with closed)": ["external_issue_authors_6m"],
    "drop all three": ["contributors_6m", "issues_prs_closed_6m", "external_issue_authors_6m"],
}
TREND_COUNTS = ["commits", "issues_opened", "prs_opened", "forks", "nonowner_contributors"]


def add_trend(X: pd.DataFrame) -> pd.DataFrame:
    """Months 4-6 activity = 6-month total - 3-month total (never negative: totals only grow)."""
    X = X.copy()
    for f in TREND_COUNTS:
        X[f"{f}_m4to6"] = (X[f"{f}_6m"] - X[f"{f}_3m"]).clip(lower=0)
    X["active_weeks_m4to6"] = (X["active_weeks_6m"] - X["active_weeks_3m"]).clip(lower=0)
    return X


def variants():
    counts, bounded = early_window_columns("6m")
    base_cols = early_feature_columns("6m")
    yield "reference", "logistic regression, 6 months", base_cols, build_early_preprocessor("6m")
    for q in (0.99, 0.95):
        yield "capping", f"cap counts at the {int(q * 100)}th percentile", base_cols, \
            build_early_preprocessor("6m", cap_quantile=q)
    for name, drop in CORRELATED_DROPS.items():
        yield "correlated", name, [c for c in base_cols if c not in drop], \
            build_early_preprocessor("6m", counts=[c for c in counts if c not in drop])
    trend_counts = [f"{f}_m4to6" for f in TREND_COUNTS]
    yield "trend", "+ months 4-6 activity (6 columns)", base_cols + trend_counts + ["active_weeks_m4to6"], \
        build_early_preprocessor("6m", counts=counts + trend_counts, bounded=bounded + ["active_weeks_m4to6"])


def main():
    X, _, y, _ = get_early_train_test()
    X = add_trend(X)

    num = [c for c in early_feature_columns("6m") if c != "language_group"]
    rho = X[num].corr(method="spearman")
    print("Pairs with Spearman |rho| > 0.8:")
    for i, a in enumerate(num):
        for b in num[i + 1:]:
            if abs(rho.loc[a, b]) > 0.8:
                print(f"  {a} / {b}: {rho.loc[a, b]:.2f}")

    rows, ref = [], None
    for exp, name, cols, prep in variants():
        auc, auc_sd, ap, ap_sd, params, auc_f, ap_f = nested_cv(
            X, y, cols, MODEL, True, SPACE, return_folds=True, n_iter=N_ITER, prep=prep)
        if ref is None:
            ref = (np.array(auc_f), np.array(ap_f))
        d_auc, d_ap = np.array(auc_f) - ref[0], np.array(ap_f) - ref[1]
        sd = d_auc.std(ddof=1) if exp != "reference" else 0.0
        verdict = "reference" if exp == "reference" else (
            "better" if d_auc.mean() > sd else "worse" if -d_auc.mean() > sd else "no real change")
        rows.append({"experiment": exp, "variant": name, "n_inputs": len(cols), "roc_auc": auc, "roc_auc_sd": auc_sd,
                     "ap_active": ap, "ap_active_sd": ap_sd, "delta_roc_auc": d_auc.mean(), "delta_sd": sd,
                     "delta_ap_active": d_ap.mean(), "verdict": verdict})
        print(f"  {exp:<11} {name:<62} ROC-AUC {auc:.4f} ({d_auc.mean():+.4f} ± {sd:.4f})  "
              f"AP {ap:.4f} ({d_ap.mean():+.4f})  -> {verdict}", flush=True)

    pd.DataFrame(rows).round(4).to_csv(config.EXPERIMENTS_PATH, index=False)
    print(f"\nwritten to {config.EXPERIMENTS_PATH.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main()
