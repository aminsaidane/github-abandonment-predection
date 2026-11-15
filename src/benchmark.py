"""Benchmark of 14 classifiers on the 6-month early-life features, then the top 3 at 3 months.

Same protocol as src/train_early.py: nested cross-validation on the TRAINING split only (outer = the
5 baseline folds, inner = 3-fold randomized search, 20 configurations per model), so every model is tuned and scored the same way.

Choosing a model: models are compared on their fold-by-fold ROC-AUC against the best one. A model
whose mean difference from the best is within one standard deviation of that difference is
"tied with the best"; among tied models we prefer the simplest to explain.

Run:
    python -m src.benchmark          # about 30-45 minutes
"""
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src import config
from src.preprocess import early_feature_columns, get_early_train_test
from src.train_early import nested_cv

warnings.filterwarnings("ignore")
S = config.RANDOM_STATE
N_ITER = 20          # configurations tried per inner search, the same budget for every model

# name -> (family, estimator, needs scaling, search space). Simplicity rank: lower = easier to explain.
MODELS = {
    "Logistic regression": ("linear", LogisticRegression(max_iter=2000), True,
                            {"model__C": loguniform(1e-3, 1e2)}),
    "Logistic regression (L1)": ("linear", LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000), True,
                                 {"model__C": loguniform(1e-3, 1e2)}),
    "Linear discriminant analysis": ("linear", LinearDiscriminantAnalysis(solver="lsqr"), True,
                                     {"model__shrinkage": uniform(0, 1)}),
    "Gaussian naive Bayes": ("probabilistic", GaussianNB(), True,
                             {"model__var_smoothing": loguniform(1e-10, 1e-1)}),
    "k-nearest neighbours": ("distance", KNeighborsClassifier(), True,
                             {"model__n_neighbors": randint(5, 200), "model__weights": ["uniform", "distance"]}),
    "Support vector machine (RBF)": ("kernel", SVC(kernel="rbf"), True,
                                     {"model__C": loguniform(1e-2, 1e2), "model__gamma": loguniform(1e-3, 1)}),
    "Decision tree": ("tree", DecisionTreeClassifier(random_state=S), False,
                      {"model__max_depth": randint(2, 12), "model__min_samples_leaf": randint(5, 150)}),
    "Random forest": ("ensemble", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=S), False,
                      {"model__min_samples_leaf": randint(1, 30), "model__max_features": uniform(0.2, 0.6),
                       "model__max_depth": [None, 6, 10, 16]}),
    "Extra trees": ("ensemble", ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=S), False,
                    {"model__min_samples_leaf": randint(1, 30), "model__max_features": uniform(0.2, 0.7),
                     "model__max_depth": [None, 6, 10, 16]}),
    "AdaBoost": ("ensemble", AdaBoostClassifier(random_state=S), False,
                 {"model__n_estimators": randint(50, 400), "model__learning_rate": loguniform(0.01, 1)}),
    "Gradient boosting (hist.)": ("ensemble", HistGradientBoostingClassifier(random_state=S), False,
                                  {"model__learning_rate": loguniform(0.01, 0.3), "model__max_leaf_nodes": randint(4, 40),
                                   "model__min_samples_leaf": randint(10, 100), "model__l2_regularization": loguniform(1e-4, 10)}),
    "XGBoost": ("ensemble", XGBClassifier(n_estimators=300, eval_metric="logloss", n_jobs=-1, random_state=S, verbosity=0), False,
                {"model__max_depth": randint(2, 7), "model__learning_rate": loguniform(0.01, 0.2),
                 "model__subsample": uniform(0.6, 0.4), "model__colsample_bytree": uniform(0.5, 0.5),
                 "model__min_child_weight": randint(1, 10)}),
    "LightGBM": ("ensemble", LGBMClassifier(n_estimators=300, random_state=S, verbose=-1, n_jobs=-1), False,
                 {"model__num_leaves": randint(4, 40), "model__learning_rate": loguniform(0.01, 0.2),
                  "model__min_child_samples": randint(10, 100), "model__subsample": uniform(0.6, 0.4),
                  "model__subsample_freq": [1], "model__colsample_bytree": uniform(0.5, 0.5)}),
    "Neural network (MLP)": ("neural", MLPClassifier(max_iter=600, early_stopping=True, random_state=S), True,
                             {"model__hidden_layer_sizes": [(16,), (32,), (64,), (32, 16), (64, 32)],
                              "model__alpha": loguniform(1e-5, 1e-1), "model__learning_rate_init": loguniform(1e-4, 1e-2)}),
}
SIMPLICITY = {"linear": 1, "probabilistic": 1, "tree": 2, "distance": 3, "kernel": 4, "ensemble": 4, "neural": 5}


def run(X, y, window, names):
    cols = early_feature_columns(window)
    rows = []
    for name in names:
        family, est, scale, space = MODELS[name]
        t = time.time()
        auc, auc_sd, ap, ap_sd, params, auc_f, ap_f = nested_cv(X, y, cols, est, scale, space, return_folds=True, n_iter=N_ITER)
        rows.append({"window": window, "model": name, "family": family, "roc_auc": auc, "roc_auc_sd": auc_sd,
                     "ap_active": ap, "ap_active_sd": ap_sd, "seconds": round(time.time() - t),
                     "folds_auc": auc_f, "params_fold1": str(params[0])})
        print(f"  {window} {name:<30} ROC-AUC {auc:.3f} ± {auc_sd:.3f}   AP(active) {ap:.3f} ± {ap_sd:.3f}"
              f"   ({time.time() - t:.0f} s)", flush=True)
    return pd.DataFrame(rows)


def mark_ties(df):
    """Paired comparison with the best model on the same 5 folds."""
    best = df.loc[df.roc_auc.idxmax()]
    diffs = [np.array(best.folds_auc) - np.array(f) for f in df.folds_auc]
    df["gap_to_best"] = [d.mean() for d in diffs]
    df["gap_sd"] = [d.std(ddof=1) for d in diffs]
    df["tied_with_best"] = [(m <= s) or (m == 0) for m, s in zip(df.gap_to_best, df.gap_sd)]
    return df


def plot(df, base, path):
    d = df.sort_values("roc_auc")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True)
    for ax, (m, sd, label, b) in zip(axes, [("roc_auc", "roc_auc_sd", "ROC-AUC", base.roc_auc),
                                            ("ap_active", "ap_active_sd", "AP (active)", base.ap_active)]):
        colors = ["#2E7D8C" if t else "#9BB7BD" for t in d.tied_with_best]
        ax.errorbar(d[m], range(len(d)), xerr=d[sd], fmt="none", ecolor="#9BB7BD", capsize=3)
        ax.scatter(d[m], range(len(d)), c=colors, zorder=3, s=40)
        ax.axvline(b, color="#C0504D", ls="--", lw=1)
        ax.text(b, len(d) - 0.4, " best single rule", color="#C0504D", fontsize=8, va="bottom")
        ax.set_xlabel(f"{label}, mean of 5 outer folds ± SD")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(range(len(d)), d.model)
    fig.suptitle("Model benchmark, first 6 months (dark = tied with the best)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    X, _, y, _ = get_early_train_test()
    base = pd.read_csv(config.BASELINES_PATH)
    print("6-month window, all models:", flush=True)
    res6 = mark_ties(run(X, y, "6m", list(MODELS)))
    top3 = res6.sort_values("roc_auc", ascending=False).model.head(3).tolist()
    print("\n3-month window, top 3:", flush=True)
    res3 = mark_ties(run(X, y, "3m", top3))

    out = pd.concat([res6, res3])
    out["simplicity"] = out.family.map(SIMPLICITY)
    out = out.sort_values(["window", "roc_auc"], ascending=[False, False]).drop(columns="folds_auc")
    out.round(4).to_csv(config.BENCHMARK_PATH, index=False)

    b6 = base[base.window == "6m"].sort_values("roc_auc", ascending=False).iloc[0]
    plot(res6, b6, config.FIGURES_DIR / "model_benchmark.png")
    tied = out[(out.window == "6m") & out.tied_with_best].sort_values(["simplicity", "roc_auc"], ascending=[True, False])
    print(f"\nTied with the best at 6 months: {', '.join(tied.model)}")
    print(f"Simplest of them: {tied.iloc[0].model}")
    print(f"written to {config.BENCHMARK_PATH.relative_to(config.ROOT_DIR)} and reports/figures/model_benchmark.png")


if __name__ == "__main__":
    main()
