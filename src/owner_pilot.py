"""Owner-history pilot: does the owner's track record add information the early-activity model lacks?

Why a test and not an AUC difference: on ~300 repos the AUC is too noisy (+/- 0.04) to see a gain of
0.01. A likelihood-ratio test of the owner features, given the score the current model already
gives, is far more powerful. Rough power: 150 active + 150 abandoned repos detect an owner effect
worth about +0.01 AUC with ~80 % power.

Design (fixed BEFORE looking at any owner data):
  sample    150 active + 150 abandoned repos drawn at random from the TRAINING split only
            (`--make-sample`). Oversampling the minority class is what gives the test its power;
            it shifts the intercept, not the owner coefficients.
  baseline  the reference logistic regression (6-month features) trained on the REST of the
            training split, so its score on the pilot repos is out-of-sample.
  test      logit(abandoned) ~ score            (reduced)
            logit(abandoned) ~ score + 3 owner features   (full)
            joint likelihood-ratio test, 3 degrees of freedom.
  features  log1p(account age), log1p(prior repos), dormant share of old prior repos
            (scritpts/collect_owner_history.py; all measured as of the repo's creation).
  gate      joint p < 0.05  -> collect the owner features for all repos, retrain, and compare by
                               nested CV like every other experiment
            otherwise       -> drop owner history and report the test.

The test split is never touched.

Run:
    python -m src.owner_pilot --make-sample      # writes data/owner_pilot_sample.csv
    python scritpts/collect_owner_history.py --repos data/owner_pilot_sample.csv
    python -m src.owner_pilot                    # analysis -> reports/owner_pilot.csv
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.preprocess import build_early_preprocessor, get_early_train_test

warnings.filterwarnings("ignore", category=UserWarning)
SEED = config.RANDOM_STATE
N_PER_CLASS = 150
ALPHA = 0.05
SAMPLE_PATH = config.DATA_DIR / "owner_pilot_sample.csv"
OWNER_PATH = config.DATA_DIR / "owner_history.csv"
RESULTS_PATH = config.REPORTS_DIR / "owner_pilot.csv"
OWNER_FEATURES = ["log_account_age", "log_prior_repos", "dormant_share"]


def make_sample(data_path):
    """150 active + 150 abandoned repos from the training split."""
    X, _, y, _ = get_early_train_test(data_path)
    pick = pd.concat([y[y == c].sample(N_PER_CLASS, random_state=SEED) for c in (0, 1)]).index
    out = pd.DataFrame({"nameWithOwner": X.loc[pick, "nameWithOwner"], "abandoned": y.loc[pick]})
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SAMPLE_PATH, index=False)
    print(f"{len(out)} repos ({N_PER_CLASS} per class) -> {SAMPLE_PATH}")


def owner_design(df):
    return pd.DataFrame({
        "log_account_age": np.log1p(df["owner_account_age_days"].astype(float)),
        "log_prior_repos": np.log1p(df["owner_prior_repos"].astype(float)),
        "dormant_share": df["owner_prior_dormant_share"].astype(float).fillna(0.0),   # no old repo -> 0
    })


def fit_ll(A, y):
    """Near-unpenalized logistic regression and its log-likelihood."""
    m = LogisticRegression(C=1e8, max_iter=5000).fit(A, y)
    return m, -log_loss(y, m.predict_proba(A)[:, 1], normalize=False)


def lrt(ll_full, ll_reduced, df):
    stat = 2 * (ll_full - ll_reduced)
    return stat, chi2.sf(max(stat, 0.0), df)


def reference_score(X_rest, y_rest, X_pilot):
    """Logit score of the reference model trained on the training repos outside the pilot."""
    pipe = Pipeline([("prep", build_early_preprocessor("6m", scale=True)),
                     ("model", LogisticRegression(max_iter=2000))])
    search = RandomizedSearchCV(pipe, {"model__C": loguniform(1e-3, 1e2)}, n_iter=12,
                                cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                                scoring="roc_auc", random_state=SEED, refit=True)
    search.fit(X_rest, y_rest)
    return search.decision_function(X_pilot)


def analyze(data_path):
    X, _, y, _ = get_early_train_test(data_path)                 # training split only
    sample = pd.read_csv(SAMPLE_PATH)
    owner = pd.read_csv(OWNER_PATH)
    in_pilot = X["nameWithOwner"].isin(sample["nameWithOwner"]).to_numpy()
    if in_pilot.sum() != len(sample):
        raise SystemExit("some sample repos are not in the training split: regenerate the sample with --make-sample")

    s_all = pd.Series(reference_score(X[~in_pilot], y[~in_pilot], X[in_pilot]), index=X.index[in_pilot], name="score")
    pilot = (pd.concat([X.loc[in_pilot, ["nameWithOwner"]], y[in_pilot].rename("y"), s_all], axis=1)
             .merge(owner, on="nameWithOwner", how="left"))
    print(f"pilot repos: {len(pilot)}; owner status: {pilot['owner_status'].value_counts(dropna=False).to_dict()}")
    pilot = pilot[pilot["owner_status"] == "ok"].reset_index(drop=True)
    if pilot.empty:
        raise SystemExit("no pilot repo has owner_status 'ok': the collection failed (see the status counts above). "
                         "Fix the cause (for http_401: the GitHub token), rerun the collector, then this analysis.")
    n, n_act = len(pilot), int((pilot["y"] == 0).sum())
    print(f"analysed: {n} repos ({n_act} active, {n - n_act} abandoned)")
    if min(n_act, n - n_act) < 100:
        print("WARNING: fewer than 100 repos in a class, the test is weaker than designed")

    yv, s = pilot["y"].to_numpy(), pilot["score"].to_numpy()
    Z = pd.DataFrame(StandardScaler().fit_transform(owner_design(pilot)), columns=OWNER_FEATURES)
    print(f"\nreference model on the pilot repos: ROC-AUC {roc_auc_score(yv, s):.3f}")

    # univariate look
    print("\nowner features alone (ROC-AUC for abandoned, 0.5 = no signal):")
    for c in OWNER_FEATURES:
        print(f"  {c:<18} {roc_auc_score(yv, Z[c]):.3f}")

    # joint test
    A_red, A_full = s[:, None], np.column_stack([s, Z.to_numpy()])
    _, ll_red = fit_ll(A_red, yv)
    m_full, ll_full = fit_ll(A_full, yv)
    stat, p = lrt(ll_full, ll_red, len(OWNER_FEATURES))
    rows = [{"test": "joint (3 owner features)", "chi2": stat, "df": 3, "p_value": p, "odds_ratio_per_sd": np.nan}]
    print(f"\njoint likelihood-ratio test: chi2 = {stat:.2f}, df = 3, p = {p:.4f}")

    print("\neach feature given the others (odds ratio per SD of the feature, 1 = no effect):")
    for i, c in enumerate(OWNER_FEATURES):
        keep = [0] + [j + 1 for j in range(len(OWNER_FEATURES)) if j != i]
        _, ll_drop = fit_ll(A_full[:, keep], yv)
        st, pv = lrt(ll_full, ll_drop, 1)
        orr = float(np.exp(m_full.coef_[0][i + 1]))
        rows.append({"test": c, "chi2": st, "df": 1, "p_value": pv, "odds_ratio_per_sd": orr})
        print(f"  {c:<18} OR {orr:.2f}   p = {pv:.4f}")

    # informational only: out-of-fold AUC gain on the pilot (too noisy to gate on)
    diffs = []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED).split(A_full, yv):
        a = LogisticRegression(C=1e8, max_iter=5000).fit(A_red[tr], yv[tr])
        b = LogisticRegression(C=1e8, max_iter=5000).fit(A_full[tr], yv[tr])
        diffs.append(roc_auc_score(yv[te], b.decision_function(A_full[te])) - roc_auc_score(yv[te], a.decision_function(A_red[te])))
    print(f"\ninformational, not the gate: out-of-fold AUC gain {np.mean(diffs):+.4f} (SD across folds {np.std(diffs):.4f})")

    out = pd.DataFrame(rows)
    out["n_repos"], out["n_active"] = n, n_act
    out.round(4).to_csv(RESULTS_PATH, index=False)
    print(f"written to {RESULTS_PATH.relative_to(config.ROOT_DIR)}")
    print("\nGATE: " + ("p < 0.05 -> collect owner features for all repos and compare by nested CV."
                        if p < ALPHA else "p >= 0.05 -> drop owner history and report this test."))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--make-sample", action="store_true")
    ap.add_argument("--data", type=Path, default=config.EARLY_DATA_PATH)
    args = ap.parse_args()
    make_sample(args.data) if args.make_sample else analyze(args.data)


if __name__ == "__main__":
    main()
