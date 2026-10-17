"""Compare Logistic Regression, Random Forest and XGBoost with stratified CV
on the training split, refit the best one, and save it.

    python -m src.train
"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config
from src.preprocess import build_preprocessor, get_train_test

SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc_active": None,          # filled below (average precision for the minority class)
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
}


def _pr_auc_active(estimator, X, y):
    """Average precision for detecting ACTIVE repos (the 16% minority).
    PR-AUC on the 84% abandoned class would look great no matter what."""
    from sklearn.metrics import average_precision_score
    proba_active = estimator.predict_proba(X)[:, 0]
    return average_precision_score(1 - y, proba_active)


SCORING["pr_auc_active"] = _pr_auc_active


def build_models(y_train: pd.Series) -> dict:
    # Abandoned (1) is the majority class, so XGBoost's weight on it is < 1.
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    rs = config.RANDOM_STATE
    return {
        "logistic_regression": Pipeline([
            ("prep", build_preprocessor(scale=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=rs)),
        ]),
        "random_forest": Pipeline([
            ("prep", build_preprocessor(scale=False)),
            ("clf", RandomForestClassifier(
                n_estimators=400, min_samples_leaf=5, class_weight="balanced",
                n_jobs=-1, random_state=rs)),
        ]),
        "xgboost": Pipeline([
            ("prep", build_preprocessor(scale=False)),
            ("clf", XGBClassifier(
                n_estimators=400, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=neg / pos, eval_metric="logloss",
                n_jobs=-1, random_state=rs)),
        ]),
    }


def main():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = get_train_test()
    print(f"Train {len(X_train)} rows | abandon rate {y_train.mean():.1%} | "
          f"{config.CV_FOLDS}-fold stratified CV\n")

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    rows = []
    models = build_models(y_train)
    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1)
        row = {"model": name}
        for metric in SCORING:
            vals = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_std"] = vals.std()
        rows.append(row)
        print(f"{name:<20} ROC-AUC {row['roc_auc_mean']:.3f} ± {row['roc_auc_std']:.3f} | "
              f"PR-AUC(active) {row['pr_auc_active_mean']:.3f} | "
              f"bal-acc {row['balanced_accuracy_mean']:.3f} | "
              f"F1-macro {row['f1_macro_mean']:.3f}")

    results = pd.DataFrame(rows).sort_values(f"{config.PRIMARY_METRIC}_mean", ascending=False)
    results.to_csv(config.CV_RESULTS_PATH, index=False)

    best_name = results.iloc[0]["model"]
    print(f"\nBest by {config.PRIMARY_METRIC}: {best_name} -> refitting on full training split")
    best = models[best_name].fit(X_train, y_train)

    joblib.dump({
        "model": best,
        "model_name": best_name,
        "features": list(X_train.columns),
        "threshold": config.DECISION_THRESHOLD,
        "cv_results": results,
    }, config.MODEL_PATH)
    print(f"Saved -> {config.MODEL_PATH.relative_to(config.ROOT_DIR)}")
    print(f"CV table -> {config.CV_RESULTS_PATH.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main()
