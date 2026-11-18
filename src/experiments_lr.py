"""Experiments on the logistic regression, scored on the TRAINING split only (test split untouched).

Same protocol as src/train_early.py and src/benchmark.py: the same 5 outer folds, and inside each
training fold a randomized search of C on 3 inner folds. Every variant is compared with the
reference model on the SAME folds, so the gain is the mean of the paired fold differences, and a
change counts as real only when it exceeds one SD of those differences (the rule used in the
benchmark). A tie goes to the simpler variant.

Three experiments (month-6 window by default):
  capping   cap the early counts at none / 95th / 99th percentile (learned on the training fold)
  corr      remove one member of each strongly correlated pair of features; L1 zeros shown for reference
  trend     activity in months 4-6 vs months 1-3, built as (6m - 3m) vs 3m. No raw commit dates are
            needed: the 3-month and 6-month windows both start at creation and have lengths of 91
            and 182 days, so the second window minus the first is exactly months 4-6.

Run:
    python -m src.experiments_lr                       # all three, 6m window
    python -m src.experiments_lr --only trend
    python -m src.experiments_lr --data path/to/pipeline3_output.csv
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config
from src.preprocess import Winsorizer, early_window_columns, get_early_train_test

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)   # penalty= is renamed in newest scikit-learn
SEED = config.RANDOM_STATE
N_ITER, INNER_FOLDS = 12, 3
CORR_THRESHOLD = 0.80
RESULTS_PATH = config.REPORTS_DIR / "lr_experiments.csv"
CORR_PATH = config.REPORTS_DIR / "tables" / "lr_correlated_pairs.csv"

# Counts that simply add up over time, so months 4-6 = value(6m) - value(3m).
ADDITIVE = ["commits", "active_weeks", "forks", "issues_opened", "prs_opened"]


# ------------------------------------------------------------------ features
def add_trend_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Add <f>_m1_3 and <f>_m4_6 for the additive features, plus log-ratio trends."""
    X = X.copy()
    for f in ADDITIVE:
        X[f"{f}_m1_3"] = X[f"{f}_3m"]
        X[f"{f}_m4_6"] = (X[f"{f}_6m"] - X[f"{f}_3m"]).clip(lower=0)
    # positive = busier in months 4-6 than in months 1-3, negative = fading
    for f in ("commits", "active_weeks"):
        X[f"trend_{f}"] = np.log1p(X[f"{f}_m4_6"]) - np.log1p(X[f"{f}_m1_3"])
    return X


def build_prep(window="6m", cap_q=None, drop=(), extra_counts=(), extra_plain=()) -> ColumnTransformer:
    """Same layout as preprocess.build_early_preprocessor, with options for the experiments.

    cap_q        cap the logged counts at this training-fold quantile (None = no capping)
    drop         columns removed from the model
    extra_counts added count columns (log1p + scale, like the others)
    extra_plain  added signed columns, scaled only (e.g. the log-ratio trends)
    """
    counts, bounded = early_window_columns(window)
    counts = [c for c in counts + config.CREATION_NUMERIC + list(extra_counts) if c not in drop]
    bounded = [c for c in bounded + list(extra_plain) if c not in drop]
    flags = [c for c in config.CREATION_FLAGS if c not in drop]

    steps = [("cap", Winsorizer(counts, cap_q))] if cap_q is not None else []
    steps += [("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
              ("scale", StandardScaler())]
    return ColumnTransformer(
        [("counts", Pipeline(steps), counts),
         ("bounded", StandardScaler(), bounded),
         ("flags", StandardScaler(), flags),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.CREATION_CATEGORICAL)],
        verbose_feature_names_out=False,
    )


# ------------------------------------------------------------------ scoring
def nested_cv(X, y, prep):
    """Outer 5-fold; tune C on 3 inner folds in each training fold; return per-fold ROC-AUC and AP(active)."""
    outer = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=SEED)
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    pipe = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000))])
    auc, ap = [], []
    for tr, te in outer.split(X, y):
        search = RandomizedSearchCV(clone(pipe), {"model__C": loguniform(1e-3, 1e2)}, n_iter=N_ITER,
                                    cv=inner, scoring="roc_auc", random_state=SEED, n_jobs=1, refit=True)
        search.fit(X.iloc[tr], y.iloc[tr])
        p = search.predict_proba(X.iloc[te])[:, 1]
        auc.append(roc_auc_score(y.iloc[te], p))
        ap.append(average_precision_score(1 - y.iloc[te], -p))
    return np.array(auc), np.array(ap)


def run_variants(X, y, variants, experiment, ref_name):
    """Score every variant; compare with `ref_name` fold by fold."""
    folds = {}
    for name, kw in variants.items():
        folds[name] = nested_cv(X, y, build_prep(**kw))
    ref_auc = folds[ref_name][0]
    rows = []
    for name, (auc, ap) in folds.items():
        diff = auc - ref_auc
        delta, sd = diff.mean(), diff.std()
        if name == ref_name:
            verdict = "reference"
        elif abs(delta) <= sd:
            verdict = "tie (keep the simpler one)"
        else:
            verdict = "better" if delta > 0 else "worse"
        rows.append({"experiment": experiment, "variant": name,
                     "roc_auc": auc.mean(), "roc_auc_sd": auc.std(),
                     "ap_active": ap.mean(), "ap_active_sd": ap.std(),
                     "delta_auc": delta, "delta_sd": sd, "verdict": verdict})
        print(f"  {experiment:<8} {name:<46} ROC-AUC {auc.mean():.3f} ± {auc.std():.3f}  "
              f"AP {ap.mean():.3f}  Δ {delta:+.4f} (SD {sd:.4f})  {verdict}", flush=True)
    return rows


# ------------------------------------------------------------------ experiments
def exp_capping(X, y, window):
    variants = {"no capping": dict(window=window),
                "cap at 95th percentile": dict(window=window, cap_q=0.95),
                "cap at 99th percentile": dict(window=window, cap_q=0.99)}
    return run_variants(X, y, variants, "capping", "no capping")


def correlated_pairs(X, window):
    """Pairs of model inputs with |Spearman| above the threshold (computed on the training split)."""
    counts, bounded = early_window_columns(window)
    cols = counts + bounded + config.CREATION_NUMERIC + config.CREATION_FLAGS
    corr = X[cols].astype(float).corr(method="spearman").abs()
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(cols) for b in cols[i + 1:] if corr.loc[a, b] >= CORR_THRESHOLD]
    return pd.DataFrame(sorted(pairs, key=lambda t: -t[2]), columns=["feature_a", "feature_b", "abs_spearman"])


def l1_zeros(X, y, window, C=1.0):
    """Features whose coefficient the L1 model sets to zero (a hint of which are redundant)."""
    prep = build_prep(window)
    pipe = Pipeline([("prep", prep), ("model", LogisticRegression(penalty="l1", solver="liblinear", C=C))]).fit(X, y)
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = pipe.named_steps["model"].coef_[0]
    return [n for n, c in zip(names, coef) if c == 0]


def exp_correlated(X, y, window):
    pairs = correlated_pairs(X, window)
    CORR_PATH.parent.mkdir(parents=True, exist_ok=True)
    pairs.round(3).to_csv(CORR_PATH, index=False)
    print(f"\n  {len(pairs)} pairs with |Spearman| >= {CORR_THRESHOLD}; table -> "
          f"{CORR_PATH.relative_to(config.ROOT_DIR)}")
    print(pairs.round(3).to_string(index=False) if len(pairs) else "  none")
    print(f"  L1 (C=1) sets to zero: {l1_zeros(X, y, window) or 'nothing'}\n", flush=True)

    variants = {"all features": dict(window=window)}
    for a, b, _ in pairs.head(6).itertuples(index=False):       # drop each member of the strongest pairs
        variants[f"drop {a}"] = dict(window=window, drop=[a])
        variants[f"drop {b}"] = dict(window=window, drop=[b])
    dropped = []                                                 # greedy: one member of every pair, all at once
    for a, b, _ in pairs.itertuples(index=False):
        if a not in dropped and b not in dropped:
            dropped.append(b)
    if dropped:
        variants[f"drop one of each pair ({len(dropped)} features)"] = dict(window=window, drop=dropped)
        print(f"  'one of each pair' removes: {dropped}")
    return run_variants(X, y, variants, "corr", "all features")


def exp_trend(X, y, window):
    if window != "6m":
        raise SystemExit("trend features compare months 4-6 with months 1-3, so they need the 6m window")
    X = add_trend_columns(X)
    halves = [f"{f}_m1_3" for f in ADDITIVE] + [f"{f}_m4_6" for f in ADDITIVE]
    trends = ["trend_commits", "trend_active_weeks"]
    variants = {
        "no trend (reference)": dict(window=window),
        "+ trend (log-ratio of commits and active weeks)": dict(window=window, extra_plain=trends),
        "+ trend + months 1-3 and 4-6 counts": dict(window=window, extra_counts=halves, extra_plain=trends),
    }
    return X, run_variants(X, y, variants, "trend", "no trend (reference)")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=["capping", "corr", "trend"])
    ap.add_argument("--window", default="6m", choices=["3m", "6m"])
    ap.add_argument("--data", type=lambda p: __import__("pathlib").Path(p), default=config.EARLY_DATA_PATH)
    args = ap.parse_args()

    X, _, y, _ = get_early_train_test(args.data)       # training split only
    rows = []
    if args.only in (None, "capping"):
        rows += exp_capping(X, y, args.window)
    if args.only in (None, "corr"):
        rows += exp_correlated(X, y, args.window)
    if args.only in (None, "trend"):
        rows += exp_trend(X, y, args.window)[1]

    out = pd.DataFrame(rows)
    if args.only is None:
        out.round(4).to_csv(RESULTS_PATH, index=False)
        print(f"\nwritten to {RESULTS_PATH.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main()
