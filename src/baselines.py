"""Baselines the forecasting models must beat, scored by 5-fold CV on the training split only.

Each baseline turns ONE early signal (or nothing) into a score, with no fitting, so it shows what
a simple rule already achieves before any model. The test split is not touched here; it is kept
for the final evaluation of the chosen model.

Metrics (positive class = abandoned):
  - ROC-AUC: chance that a random abandoned repo scores higher than a random active one (0.5 = no skill).
  - AP (active): average precision for finding the ACTIVE repos, the 16 % minority. Its no-skill
    level is the active share (~0.16), so it shows skill that ROC-AUC can hide under imbalance.

Run:
    python -m src.baselines
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src import config
from src.preprocess import get_early_train_test

# name -> (column, sign). Score = sign * column, oriented so that higher = more likely abandoned.
RULES = {
    "constant (always 'abandoned')": (None, 0),
    "no outside contributor": ("nonowner_contributors", "none"),
    "days since last commit": ("days_since_last_commit", +1),
    "few active weeks": ("active_weeks", -1),
    "few commits": ("commits", -1),
}


def rule_score(X: pd.DataFrame, rule, window: str) -> np.ndarray:
    col, sign = rule
    if col is None:
        return np.zeros(len(X))
    values = X[f"{col}_{window}"].to_numpy(dtype=float)
    if sign == "none":                      # binary rule: 1 if the repo has no outside contributor
        return (values == 0).astype(float)
    return sign * values


def evaluate(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    rows = []
    for window in ("3m", "6m"):
        for name, rule in RULES.items():
            auc, ap = [], []
            for _, idx in cv.split(X, y):            # no fitting: each fold is only a held-out sample
                s, t = rule_score(X.iloc[idx], rule, window), y.iloc[idx]
                auc.append(roc_auc_score(t, s))
                ap.append(average_precision_score(1 - t, -s))
            rows.append({"window": window, "baseline": name,
                         "roc_auc": np.mean(auc), "roc_auc_sd": np.std(auc),
                         "ap_active": np.mean(ap), "ap_active_sd": np.std(ap)})
    return pd.DataFrame(rows)


def main():
    X_train, _, y_train, _ = get_early_train_test()
    res = evaluate(X_train, y_train)
    res.round(3).to_csv(config.BASELINES_PATH, index=False)
    print(f"Training split: {len(X_train)} repos, active share {1 - y_train.mean():.1%} (no-skill AP)\n")
    print(res.round(3).to_string(index=False))
    print(f"\nwritten to {config.BASELINES_PATH.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main()
