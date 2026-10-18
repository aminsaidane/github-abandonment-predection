"""Evaluate the saved model on the held-out test split.

Outputs (in reports/):
  evaluation_report.txt       metrics + baselines + classification report
  figures/confusion_matrix.png
  figures/roc_curve.png
  figures/shap_summary.png    beeswarm: direction + magnitude per feature
  figures/shap_importance.png mean |SHAP| bar chart

    python -m src.evaluate
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
)

from src import config
from src.preprocess import get_train_test

LABELS = [config.NEGATIVE_LABEL, config.POSITIVE_LABEL]


# ------------------------------------------------------------------ metrics
def score(y_true, y_pred, proba_abandoned=None) -> dict:
    out = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": (y_true == y_pred).mean(),
    }
    if proba_abandoned is not None:
        out["roc_auc"] = roc_auc_score(y_true, proba_abandoned)
        out["pr_auc_active"] = average_precision_score(1 - y_true, 1 - proba_abandoned)
    return out


def heuristic_baseline(X_train, y_train, X_test, feature="commits_per_day"):
    """One-feature rule: 'abandoned if <feature> < t', with t chosen on the
    TRAINING split to maximise balanced accuracy. The bar any model must clear."""
    candidates = np.unique(np.quantile(X_train[feature], np.linspace(0.01, 0.99, 99)))
    best_t = max(
        candidates,
        key=lambda t: balanced_accuracy_score(y_train, (X_train[feature] < t).astype(int)),
    )
    return (X_test[feature] < best_t).astype(int), best_t


# ------------------------------------------------------------------ SHAP
def shap_values_for(bundle, X_train, X_test):
    prep = bundle["model"].named_steps["prep"]
    clf = bundle["model"].named_steps["clf"]
    names = prep.get_feature_names_out()
    Xt_test = pd.DataFrame(prep.transform(X_test), columns=names, index=X_test.index)

    if bundle["model_name"] == "logistic_regression":
        Xt_train = pd.DataFrame(prep.transform(X_train), columns=names)
        explainer = shap.LinearExplainer(clf, Xt_train)
    else:
        explainer = shap.TreeExplainer(clf)

    sv = explainer(Xt_test)
    if sv.values.ndim == 3:            # RandomForest returns one slice per class
        sv = sv[:, :, 1]
    return sv


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------ main
def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(config.MODEL_PATH)
    model, threshold = bundle["model"], bundle["threshold"]

    X_train, X_test, y_train, y_test = get_train_test()
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    model_scores = score(y_test, y_pred, proba)
    majority_scores = score(y_test, np.ones_like(y_test))
    heur_pred, heur_t = heuristic_baseline(X_train, y_train, X_test)
    # Ranking by -commits_per_day gives the heuristic a ROC-AUC, so the
    # threshold-free comparison is apples to apples.
    heur_rank = -X_test["commits_per_day"]
    heur_scores = score(y_test, heur_pred)
    heur_scores["roc_auc"] = roc_auc_score(y_test, heur_rank)
    heur_scores["pr_auc_active"] = average_precision_score(1 - y_test, -heur_rank)

    comparison = pd.DataFrame({
        f"model ({bundle['model_name']})": model_scores,
        "majority class (always 'abandoned')": majority_scores,
        f"heuristic: commits_per_day < {heur_t:.4f}": heur_scores,
    }).T

    lines = [
        f"Model: {bundle['model_name']}  |  threshold {threshold}",
        f"Test set: {len(y_test)} repos, abandon rate {y_test.mean():.1%}",
        "",
        "Cross-validation (training split):",
        bundle["cv_results"].round(3).to_string(index=False),
        "",
        "Held-out test set vs baselines:",
        comparison.round(3).to_string(),
        "",
        "Classification report (test):",
        classification_report(y_test, y_pred, target_names=LABELS, digits=3),
    ]
    report = "\n".join(lines)
    config.EVAL_REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)

    # confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=LABELS, cmap="Blues", colorbar=False)
    plt.title(f"Confusion matrix — {bundle['model_name']} (test)")
    save_fig(config.FIGURES_DIR / "confusion_matrix.png")

    # ROC
    RocCurveDisplay.from_predictions(y_test, proba, name=bundle["model_name"])
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    plt.title("ROC curve (test)")
    save_fig(config.FIGURES_DIR / "roc_curve.png")

    # SHAP
    sv = shap_values_for(bundle, X_train, X_test)
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title("SHAP: push toward 'abandoned' (+) vs 'active' (−)")
    save_fig(config.FIGURES_DIR / "shap_summary.png")

    shap.plots.bar(sv, max_display=15, show=False)
    plt.title("Mean |SHAP| feature importance")
    save_fig(config.FIGURES_DIR / "shap_importance.png")

    print(f"Report  -> {config.EVAL_REPORT_PATH.relative_to(config.ROOT_DIR)}")
    print(f"Figures -> {config.FIGURES_DIR.relative_to(config.ROOT_DIR)}/")


if __name__ == "__main__":
    main()
