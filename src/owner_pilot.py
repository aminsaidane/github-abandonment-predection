"""Owner-history pilot: does the owner's track record improve the forecast?

Reads data/owner_history.csv (scritpts/collect_owner_history.py, 1,000 training repos) and compares,
on those repos only, logistic regression on the 6-month features with and without the owner
features. Same nested CV as the benchmark; the comparison is paired fold by fold.

Decision rule agreed beforehand: run the full collection only if the gain in ROC-AUC is above
+0.01 AND larger than the standard deviation of the paired fold differences.

Run:
    python -m src.owner_pilot
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import loguniform, mannwhitneyu
from sklearn.linear_model import LogisticRegression

from src import config
from src.preprocess import build_early_preprocessor, early_feature_columns, early_window_columns, get_early_train_test
from src.train_early import nested_cv

warnings.filterwarnings("ignore")
OWNER_PATH = config.DATA_DIR / "owner_history.csv"
OWNER_COUNTS = ["owner_account_age_days", "owner_prior_repos", "owner_repos_prev_year"]
SPACE = {"model__C": loguniform(1e-3, 1e2)}


def load():
    X, _, y, _ = get_early_train_test()
    own = pd.read_csv(OWNER_PATH)
    own = own[own.owner_status == "ok"]
    d = X.join(y).merge(own, on="nameWithOwner", how="inner")
    d["owner_has_track"] = (d.owner_track_checked > 0).astype(int)
    d["owner_track_active_share"] = d.owner_track_active_share.fillna(0.0)   # 0 + flag when nothing to check
    return d.drop(columns=[config.TARGET]), d[config.TARGET].astype(int)


def main():
    X, y = load()
    print(f"Pilot: {len(X)} repos with owner data, abandoned {y.mean():.1%}\n")

    print("Single-feature check (Mann-Whitney, r > 0 = higher in abandoned repos):")
    for c in OWNER_COUNTS + ["owner_track_active_share"]:
        sub = X[c].notna() if c != "owner_track_active_share" else X.owner_has_track == 1
        a, b = X.loc[sub & (y == 1), c], X.loc[sub & (y == 0), c]
        u, p = mannwhitneyu(a, b)
        print(f"  {c:<26} median abandoned {a.median():>7.2f}  active {b.median():>7.2f}  "
              f"r = {2 * u / (len(a) * len(b)) - 1:+.2f}  p = {p:.1e}")

    counts, bounded = early_window_columns("6m")
    base = early_feature_columns("6m")
    variants = {
        "reference (6-month features)": (base, build_early_preprocessor("6m")),
        "+ owner history": (base + OWNER_COUNTS + ["owner_track_active_share", "owner_has_track"],
                            build_early_preprocessor("6m", counts=counts + OWNER_COUNTS,
                                                     bounded=bounded + ["owner_track_active_share"],
                                                     extra_flags=["owner_has_track"])),
    }
    rows, ref = [], None
    print()
    for name, (cols, prep) in variants.items():
        auc, auc_sd, ap, ap_sd, _, auc_f, ap_f = nested_cv(X, y, cols, LogisticRegression(max_iter=2000), True,
                                                           SPACE, return_folds=True, n_iter=20, prep=prep)
        ref = ref or (np.array(auc_f), np.array(ap_f))
        d = np.array(auc_f) - ref[0]
        rows.append({"variant": name, "n_repos": len(X), "roc_auc": auc, "roc_auc_sd": auc_sd, "ap_active": ap,
                     "ap_active_sd": ap_sd, "delta_roc_auc": d.mean(), "delta_sd": d.std(ddof=1) if d.any() else 0.0,
                     "delta_ap_active": (np.array(ap_f) - ref[1]).mean()})
        print(f"  {name:<30} ROC-AUC {auc:.3f} ± {auc_sd:.3f}  AP(active) {ap:.3f} ± {ap_sd:.3f}  "
              f"gain {d.mean():+.3f} ± {rows[-1]['delta_sd']:.3f}")

    gain, sd = rows[-1]["delta_roc_auc"], rows[-1]["delta_sd"]
    go = gain > 0.01 and gain > sd
    print(f"\nDecision: {'RUN the full collection' if go else 'STOP, the gain is too small'} "
          f"(gain {gain:+.3f}, needs > +0.010 and > its SD {sd:.3f})")
    pd.DataFrame(rows).round(4).to_csv(config.REPORTS_DIR / "owner_pilot.csv", index=False)


if __name__ == "__main__":
    main()
