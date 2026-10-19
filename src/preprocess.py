"""Load the Hop ETL output, split it, and build the scikit-learn
preprocessing transformer.

Feature engineering (ratios, 0/1 flags, language/license grouping) is done
upstream in Hop Pipeline 2. This module only does the steps that must be
LEARNED from the training split (imputation, log/scale, one-hot), so they
cannot leak information from the test set.

Run directly for a quick data summary:
    python -m src.preprocess
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config


# ------------------------------------------------------------------ loading
def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    """Read the CSV produced by Hop Pipeline 2 and check the expected columns exist."""
    df = pd.read_csv(path)
    missing = set(feature_columns() + [config.TARGET]) - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing columns {sorted(missing)}. "
            "Re-run the Hop feature engineering pipeline."
        )
    return df


# ------------------------------------------------------------------ features
def feature_columns():
    return config.NUMERIC_FEATURES + config.DERIVED_FEATURES + config.CATEGORICAL_FEATURES


def get_X_y(df: pd.DataFrame):
    X = df[feature_columns()]
    y = df[config.TARGET].astype(int)
    return X, y


def get_train_test(path=config.DATA_PATH):
    """Deterministic stratified split, shared by train.py and evaluate.py."""
    df = load_data(path)
    X, y = get_X_y(df)
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )


# ------------------------------------------------------------------ transformer
def safe_log(x):
    """Natural log for strictly positive ratios; clips at 1e-6 so a zero can never give -inf."""
    return np.log(np.clip(x, 1e-6, None))


class Winsorizer(BaseEstimator, TransformerMixin):
    """Cap selected columns at an upper quantile learned from the TRAINING data.

    Used for variables whose extreme values are real but misleading (see the
    EDA notebook, section 2.5): kernel copies that inherit hundreds of
    thousands of upstream commits, and organisation-wide collaborator counts.
    Other columns pass through unchanged.
    """

    def __init__(self, columns=(), quantile=0.99):
        self.columns = columns
        self.quantile = quantile

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.caps_ = {c: float(X[c].quantile(self.quantile)) for c in self.columns if c in X.columns}
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        for c, cap in self.caps_.items():
            X[c] = X[c].clip(upper=cap)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Numeric: 99th-pct cap (3 vars) -> median impute -> log -> optional scaling.
    Counts (which contain zeros) use log1p; the strictly positive ratios in
    config.PLAIN_LOG_FEATURES use a plain log. Categoricals are one-hot encoded.

    Every step is fitted on the training split only. Trees don't need scaling,
    but the log is harmless for them and keeps SHAP plots readable, so the same
    transformer is used for every model.
    """
    def numeric_pipeline(log_fn, name):
        steps = [
            ("cap", Winsorizer(config.WINSORIZE_FEATURES, config.WINSORIZE_QUANTILE)),
            ("impute", SimpleImputer(strategy="median")),
            (name, FunctionTransformer(log_fn, feature_names_out="one-to-one")),
        ]
        if scale:
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps)

    plain_log = config.PLAIN_LOG_FEATURES
    log1p_cols = [c for c in config.NUMERIC_FEATURES if c not in plain_log]

    derived_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        derived_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline(np.log1p, "log1p"), log1p_cols),     # counts with zeros
            ("ratio", numeric_pipeline(safe_log, "log"), plain_log),        # strictly positive ratios
            ("der", Pipeline(derived_steps), config.DERIVED_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             config.CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


# ------------------------------------------------------------------ summary
def main():
    df = load_data()
    print(f"Loaded {len(df)} rows x {df.shape[1]} cols from {config.DATA_PATH.name}")
    print(f"Abandon rate: {df[config.TARGET].mean():.1%}")
    if config.REJECTED_PATH.exists():
        rej = pd.read_csv(config.REJECTED_PATH)
        print(f"Rejected by Hop: {len(rej)} rows (abandon rate {rej[config.TARGET].mean():.0%})")

    X_train, X_test, y_train, y_test = get_train_test()
    print(f"Train: {len(X_train)}  Test: {len(X_test)}  Features: {X_train.shape[1]}")
    print("\nArchived vs abandoned (why isArchived is excluded):")
    print(pd.crosstab(df["isArchived"], df[config.TARGET]))


if __name__ == "__main__":
    main()
