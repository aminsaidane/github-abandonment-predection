"""Temporal validation: train on older repositories, test on newer ones.

A random split mixes repos from every period, so it can flatter a model that will later be used on
repos it has never seen. Here the split follows the creation date (`createdAt`): the model never
sees a repo newer than the ones it is scored on. Only the TRAINING split is used (the 985 held-out
repos stay untouched for the final evaluation). Model: the reference logistic regression on the
6-month features, with C tuned on 3 inner folds inside the training period.

Three views:
  main      train on repos created up to MAIN_CUTOFF, test on the later ones
  rolling   three expanding windows: train on everything before a block, test on that block
  control   the same train/test sizes as `main` but split at random, repeated 20 times. If the
            temporal score falls inside this range, time itself costs the model nothing.

Besides ROC-AUC and AP(active) it reports the mean predicted P(abandoned) next to the observed
rate. The abandonment rate falls from ~0.91 for old repos to ~0.74 for 2023 ones (newer repos had
less time to go quiet before the snapshot), so a model trained on old repos may over-predict
abandonment on new ones even when its ranking is fine.

Run:
    python -m src.temporal_validation
    python -m src.temporal_validation --data path/to/pipeline3_output.csv
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

from src import config
from src.preprocess import build_early_preprocessor, get_early_train_test

warnings.filterwarnings("ignore", category=UserWarning)
SEED = config.RANDOM_STATE
MAIN_CUTOFF = 2020                                  # main split: created <= 2020 -> train, later -> test
ROLLING_BLOCKS = [(2018, 2019), (2020, 2021), (2022, 2023)]   # test blocks (inclusive years)
N_CONTROL = 20
N_BOOT = 1000
RESULTS_PATH = config.REPORTS_DIR / "temporal_validation.csv"
BASELINE_COL = "days_since_last_commit_6m"          # best single rule in src/baselines.py


def fit_lr(X, y):
    """Reference logistic regression, C tuned by a small randomized search on 3 inner folds."""
    pipe = Pipeline([("prep", build_early_preprocessor("6m", scale=True)),
                     ("model", LogisticRegression(max_iter=2000))])
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(pipe, {"model__C": loguniform(1e-3, 1e2)}, n_iter=12, cv=inner,
                                scoring="roc_auc", random_state=SEED, n_jobs=1, refit=True)
    return search.fit(X, y)


def bootstrap_auc_ci(y, p, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    y, p, aucs = np.asarray(y), np.asarray(p), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            aucs.append(roc_auc_score(y[i], p[i]))
    return np.percentile(aucs, [2.5, 97.5])


def evaluate(name, Xtr, ytr, Xte, yte):
    model = fit_lr(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    lo, hi = bootstrap_auc_ci(yte, p)
    return {
        "scheme": name, "n_train": len(ytr), "n_test": len(yte),
        "abandon_rate_train": ytr.mean(), "abandon_rate_test": yte.mean(),
        "roc_auc": roc_auc_score(yte, p), "auc_ci_low": lo, "auc_ci_high": hi,
        "ap_active": average_precision_score(1 - yte, -p), "ap_active_no_skill": 1 - yte.mean(),
        "baseline_auc": roc_auc_score(yte, Xte[BASELINE_COL]),
        "mean_pred_abandoned": p.mean(), "C": model.best_params_["model__C"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=config.EARLY_DATA_PATH)
    args = ap.parse_args()

    X, _, y, _ = get_early_train_test(args.data)          # training split only
    year = pd.to_datetime(X["createdAt"], utc=True).dt.year
    rows = []

    old = (year <= MAIN_CUTOFF).to_numpy()
    rows.append(evaluate(f"main: train <= {MAIN_CUTOFF}, test later", X[old], y[old], X[~old], y[~old]))

    for lo, hi in ROLLING_BLOCKS:
        tr, te = (year < lo).to_numpy(), ((year >= lo) & (year <= hi)).to_numpy()
        rows.append(evaluate(f"rolling: train < {lo}, test {lo}-{hi}", X[tr], y[tr], X[te], y[te]))

    # control: same sizes as the main split, random assignment
    n_te = int((~old).sum())
    aucs = []
    for tr, te in StratifiedShuffleSplit(N_CONTROL, test_size=n_te, random_state=SEED).split(X, y):
        m = fit_lr(X.iloc[tr], y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    control = (np.mean(aucs), np.std(aucs))

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.round(3).drop(columns=["C"]).to_string(index=False))
    print(f"\ncontrol (random split, same sizes as main, {N_CONTROL} repeats): "
          f"ROC-AUC {control[0]:.3f} ± {control[1]:.3f}")
    main_auc = out.loc[0, "roc_auc"]
    print(f"main temporal ROC-AUC {main_auc:.3f} is {(main_auc - control[0]) / control[1]:+.1f} SD from the control mean")

    out["control_auc_mean"], out["control_auc_sd"] = control
    out.round(4).to_csv(RESULTS_PATH, index=False)
    print(f"written to {RESULTS_PATH.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main()
