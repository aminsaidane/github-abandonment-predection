"""Forecasting models on early-life features, compared with the baselines by nested cross-validation.

Everything here uses the TRAINING split only (the test split is kept for the final evaluation):
  - outer loop: the same 5 stratified folds as src/baselines.py, so scores are directly comparable;
  - inner loop: a small randomized hyperparameter search (3 folds) inside each outer training fold,
    so the reported scores are not inflated by the tuning.

For each window (3m, 6m) it scores logistic regression, random forest and XGBoost, each with and
without class weighting. It then runs the ablation: the best 6-month model again, with the three
snapshot-only fields (license, description, code of conduct) added.

Run:
    python -m src.train_early            # about 10-20 minutes on a laptop
"""
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config
from src.preprocess import build_early_preprocessor, early_feature_columns, get_early_train_test

warnings.filterwarnings("ignore", category=UserWarning)
SEED = config.RANDOM_STATE
N_ITER = 12          # random configurations tried per inner search
INNER_FOLDS = 3


def candidates(neg_over_pos: float):
    """name -> (estimator, needs scaling, search space, {weighted: extra params})."""
    return {
        "logistic regression": (
            LogisticRegression(max_iter=2000),
            True,
            {"model__C": loguniform(1e-3, 1e2)},
            {False: {}, True: {"class_weight": "balanced"}},
        ),
        "random forest": (
            RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
            False,
            {"model__min_samples_leaf": randint(1, 30), "model__max_features": uniform(0.2, 0.6),
             "model__max_depth": [None, 6, 10, 16]},
            {False: {}, True: {"class_weight": "balanced"}},
        ),
        "xgboost": (
            XGBClassifier(n_estimators=300, eval_metric="logloss", n_jobs=-1, random_state=SEED, verbosity=0),
            False,
            {"model__max_depth": randint(2, 7), "model__learning_rate": loguniform(0.01, 0.2),
             "model__subsample": uniform(0.6, 0.4), "model__colsample_bytree": uniform(0.5, 0.5),
             "model__min_child_weight": randint(1, 10)},
            # abandoned (1) is the majority: weighting it by n_active / n_abandoned balances the classes
            {False: {}, True: {"scale_pos_weight": neg_over_pos}},
        ),
    }


def nested_cv(X, y, cols, estimator, scale, space, extra_flags=()):
    """Outer 5-fold CV; in each training fold, tune on 3 inner folds, then score the held-out fold."""
    outer = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=SEED)
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    window = "3m" if cols[0].endswith("_3m") else "6m"
    pipe = Pipeline([("prep", build_early_preprocessor(window, scale=scale, extra_flags=extra_flags)),
                     ("model", estimator)])
    auc, ap, params = [], [], []
    for tr, te in outer.split(X, y):
        search = RandomizedSearchCV(clone(pipe), space, n_iter=N_ITER, cv=inner, scoring="roc_auc",
                                    random_state=SEED, n_jobs=1, refit=True)
        search.fit(X.iloc[tr][cols], y.iloc[tr])
        p = search.predict_proba(X.iloc[te][cols])[:, 1]          # probability of "abandoned"
        auc.append(roc_auc_score(y.iloc[te], p))
        ap.append(average_precision_score(1 - y.iloc[te], 1 - p))  # finding the active repos
        params.append({k.replace("model__", ""): (round(v, 4) if isinstance(v, float) else v)
                       for k, v in search.best_params_.items()})
    return np.mean(auc), np.std(auc), np.mean(ap), np.std(ap), params


def main():
    X, _, y, _ = get_early_train_test()
    neg_over_pos = (1 - y.mean()) / y.mean()
    base = pd.read_csv(config.BASELINES_PATH)
    rows = []

    def record(window, name, weighted, features, res):
        auc, auc_sd, ap, ap_sd, params = res
        rows.append({"window": window, "model": name, "class_weight": weighted, "features": features,
                     "roc_auc": auc, "roc_auc_sd": auc_sd, "ap_active": ap, "ap_active_sd": ap_sd,
                     "params_fold1": str(params[0])})
        print(f"  {window} {name:<20} weighted={weighted!s:<5} {features:<22} "
              f"ROC-AUC {auc:.3f} ± {auc_sd:.3f}   AP(active) {ap:.3f} ± {ap_sd:.3f}", flush=True)

    for window in ("3m", "6m"):
        cols = early_feature_columns(window)
        for name, (est, scale, space, weights) in candidates(neg_over_pos).items():
            for weighted, extra in weights.items():
                t = time.time()
                res = nested_cv(X, y, cols, clone(est).set_params(**extra), scale, space)
                record(window, name, weighted, "early + creation", res)
                print(f"    ({time.time() - t:.0f} s)", flush=True)

    # ablation: best 6-month configuration, plus the snapshot-only fields
    res = pd.DataFrame(rows)
    best = res[res.window == "6m"].sort_values("roc_auc", ascending=False).iloc[0]
    est, scale, space, weights = candidates(neg_over_pos)[best.model]
    cols = early_feature_columns("6m") + config.SNAPSHOT_ONLY_FIELDS
    print(f"\nAblation on the best 6m model ({best.model}, weighted={best.class_weight}):", flush=True)
    res_ab = nested_cv(X, y, cols, clone(est).set_params(**weights[best.class_weight]), scale, space,
                       extra_flags=config.SNAPSHOT_ONLY_FIELDS)
    record("6m", best.model, best.class_weight, "+ snapshot-only fields", res_ab)

    out = pd.DataFrame(rows)
    for window in ("3m", "6m"):                      # best single-rule baseline for reference
        b = base[base.window == window].sort_values("roc_auc", ascending=False).iloc[0]
        out.loc[len(out)] = {"window": window, "model": f"baseline: {b.baseline}", "class_weight": False,
                             "features": "one signal", "roc_auc": b.roc_auc, "roc_auc_sd": b.roc_auc_sd,
                             "ap_active": b.ap_active, "ap_active_sd": b.ap_active_sd, "params_fold1": ""}
    out = out.sort_values(["window", "roc_auc"], ascending=[True, False])
    out.round(3).to_csv(config.EARLY_CV_RESULTS_PATH, index=False)
    print(f"\nwritten to {config.EARLY_CV_RESULTS_PATH.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main()
