"""Load the final model and score repositories (used by the dashboard).

    from src.predict import load_final_model, predict_risk
    bundle = load_final_model()
    scored = predict_risk(df, bundle)      # df: one row per repo, with bundle["input_columns"]

`p_abandoned` is the calibrated probability that the repo will be abandoned (no push for more than
365 days at snapshot time) given its first 6 months. `flagged` is True when it reaches the cut-off
chosen on the training data. Probabilities describe repos like those in the study; they also
depend on how long a repo has existed (older repos had longer to go quiet), see
reports/temporal_validation.csv.
"""
import joblib
import pandas as pd

from src import config


def load_final_model(path=config.FINAL_MODEL_PATH):
    return joblib.load(path)


def predict_risk(df: pd.DataFrame, bundle=None) -> pd.DataFrame:
    bundle = bundle or load_final_model()
    missing = [c for c in bundle["input_columns"] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    raw = bundle["model"].predict_proba(df[bundle["input_columns"]])[:, 1]
    p = bundle["calibrator"].predict(raw)
    out = pd.DataFrame({"p_abandoned": p, "flagged": p >= bundle["threshold"]}, index=df.index)
    out["p_active"] = 1 - out["p_abandoned"]
    return out
