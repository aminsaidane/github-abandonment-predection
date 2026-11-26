"""Step 5: what does the final model say? Coefficients with bootstrap intervals, plus SHAP.

Uses the TRAINING split only (the test split was spent in src/final_evaluate.py), and the saved
final model (models/final_model.pkl). Run from the repo root:

    python -m src.interpret                 # ~1-2 min (500 bootstrap refits)
    python -m src.interpret --n-boot 50     # quick check

Outputs
    reports/tables/interpretation_coefficients.csv   one row per input feature (language grouped)
    reports/tables/interpretation_language.csv       the 16 language effects
    reports/tables/interpretation_bands.csv          observed abandonment rate by band of the top features
    reports/figures/interp_coefficients.png          odds ratio per +1 SD, 95% bootstrap interval
    reports/figures/interp_shap.png                  SHAP dot plot (training split)

How to read the numbers
    Every input is standardised (counts go through log1p first), so a coefficient is the change in the
    log-odds of "abandoned" for a repo that is 1 standard deviation higher on that feature, with the other
    features held fixed. exp(coef) is the odds ratio. The model is strongly regularised (C = 0.029), so
    coefficients are shrunk towards zero on purpose.
    For a linear model SHAP is exact: shap_i = coef_i * (z_i - mean z_i). We compute it with the shap
    package and check it against that formula (the dashboard uses the same formula).
"""
import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from dashboard.features import LABELS
from src import config
from src.predict import load_final_model
from src.preprocess import get_early_train_test

warnings.filterwarnings("ignore")
SEED = config.RANDOM_STATE
LANG_PREFIX = "language_group_"


def label(name):
    return LABELS.get(name, name)


def tidy_names(names):
    """Feature names of the transformed matrix -> (numeric names, language names)."""
    lang = [n for n in names if n.startswith(LANG_PREFIX)]
    return [n for n in names if not n.startswith(LANG_PREFIX)], lang


def transform(pipe, X):
    Z = pipe.named_steps["prep"].transform(X)
    return pd.DataFrame(Z, columns=pipe.named_steps["prep"].get_feature_names_out(), index=X.index)


def bootstrap_coefs(pipe, X, y, n_boot):
    """Refit the whole pipeline (scaling included, same C) on bootstrap resamples of the training split."""
    rng = np.random.default_rng(SEED)
    rows, n = [], len(X)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        p = clone(pipe).fit(X.iloc[idx], y.iloc[idx])
        names = p.named_steps["prep"].get_feature_names_out()
        rows.append(pd.Series(p.named_steps["model"].coef_[0], index=names))
        if (b + 1) % 100 == 0:
            print(f"  bootstrap {b + 1}/{n_boot}")
    return pd.DataFrame(rows).reindex(columns=pipe.named_steps["prep"].get_feature_names_out())


def coefficient_table(pipe, X, y, boot):
    Z = transform(pipe, X)
    coef = pd.Series(pipe.named_steps["model"].coef_[0], index=Z.columns)
    num, lang = tidy_names(Z.columns)
    rows = []
    for f in num:
        lo, hi = boot[f].quantile([0.025, 0.975])
        auc = roc_auc_score(y, Z[f])              # >0.5: higher value goes with MORE abandonment (on its own)
        rows.append(dict(feature=f, label=label(f), coef=coef[f], odds_ratio=np.exp(coef[f]),
                         ci_low=lo, ci_high=hi, or_low=np.exp(lo), or_high=np.exp(hi),
                         excludes_zero=bool(lo > 0 or hi < 0), alone_auc=auc,
                         alone_spearman=spearmanr(Z[f], y)[0],
                         share_boot_same_sign=float((np.sign(boot[f]) == np.sign(coef[f])).mean())))
    t = pd.DataFrame(rows)
    # sign disagreement between the model coefficient and the feature on its own = collinearity flag
    t["sign_flips_vs_alone"] = np.sign(t["coef"]) != np.sign(t["alone_auc"] - 0.5)
    return t, coef, Z


def language_table(coef, boot, Z, y):
    _, lang = tidy_names(coef.index)
    rows = []
    for f in lang:
        lo, hi = boot[f].quantile([0.025, 0.975])
        rows.append(dict(language=f[len(LANG_PREFIX):], n=int(Z[f].sum()), abandon_rate=float(y[Z[f] == 1].mean()),
                         coef=coef[f], ci_low=lo, ci_high=hi, excludes_zero=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(rows).sort_values("coef", ascending=False)


def band_table(X, y):
    """Observed abandonment rate by band of the top features (training split). Feeds the dashboard."""
    spec = [("days_since_last_commit_6m", [-1, 14, 60, 120, 160, 183], ["0-14 days", "15-60", "61-120", "121-160", "161-182"]),
            ("active_weeks_6m", [-1, 1, 3, 6, 12, 26], ["0-1 weeks", "2-3", "4-6", "7-12", "13-26"])]
    rows = []
    for col, bins, labels in spec:
        g = pd.cut(X[col], bins, labels=labels, include_lowest=True)
        for b, d in pd.DataFrame({"g": g, "y": y.values}).groupby("g", observed=True):
            rows.append(dict(feature=col, band=b, n=len(d), abandon_rate=d["y"].mean()))
    binary = [("is_org", X["is_org"] > 0, "Organisation", "Personal"),
              ("external_issue_authors_6m", X["external_issue_authors_6m"] > 0, "Someone else opened an issue or PR", "Nobody else did"),
              ("prs_opened_6m", X["prs_opened_6m"] > 0, "Pull requests opened", "None opened")]
    for col, mask, yes, no in binary:
        for name, m in ((yes, mask), (no, ~mask)):
            rows.append(dict(feature=col, band=name, n=int(m.sum()), abandon_rate=float(y[m.values].mean())))
    return pd.DataFrame(rows)


def shap_values(pipe, Z):
    """SHAP via the shap package, checked against the closed form for a linear model."""
    lr = pipe.named_steps["model"]
    masker = shap.maskers.Independent(Z.values, max_samples=len(Z))     # full training split as background
    sv = shap.LinearExplainer(lr, masker).shap_values(Z.values)
    closed = (Z.values - Z.values.mean(axis=0)) * lr.coef_[0]
    err = float(np.abs(sv - closed).max())
    assert err < 1e-8, f"SHAP differs from coef*(z-mean) by {err}"
    S = pd.DataFrame(sv, columns=Z.columns, index=Z.index)
    num, lang = tidy_names(Z.columns)
    S_g = S[num].copy()
    S_g["language_group"] = S[lang].sum(axis=1)       # one-hot columns summed into one feature
    return S_g, err


def fig_coefficients(t, path):
    t = t.sort_values("coef")
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    y = np.arange(len(t))
    colors = np.where(t["coef"] > 0, "#c8412f", "#2f6db5")
    ax.hlines(y, t["or_low"], t["or_high"], color=colors, lw=2, alpha=0.55)
    ax.scatter(t["odds_ratio"], y, c=colors, s=45, zorder=3)
    ax.axvline(1, color="grey", lw=1, ls="--")
    names = [f"{l}{' *' if f else ''}{'' if e else '  (interval includes 1)'}"
             for l, e, f in zip(t["label"], t["excludes_zero"], t["sign_flips_vs_alone"])]
    ax.set_yticks(y, names)
    ax.set_xlabel("Odds ratio of 'abandoned' per +1 standard deviation, other features held fixed")
    ax.set_title("What moves the odds of abandonment\n(training split, 95% bootstrap interval)", fontsize=11)
    ax.text(0.99, 0.01, "left: lowers the odds   |   right: raises the odds", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="grey")
    fig.text(0.01, 0.005, "* On its own this feature goes with LESS abandonment; the sign flips only once the other features are held fixed (they overlap).",
             fontsize=8, color="dimgrey")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_shap(S, Z, path, seed=SEED):
    order = S.abs().mean().sort_values().index
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    for i, f in enumerate(order):
        s = S[f].values
        jitter = rng.uniform(-0.28, 0.28, len(s))
        if f == "language_group":
            ax.scatter(s, i + jitter, s=4, c="#9aa3b2", alpha=0.5, linewidths=0)
        else:
            v = np.clip(Z[f].values, -2, 2)           # colour = how high the feature is for that repo
            ax.scatter(s, i + jitter, s=4, c=v, cmap="coolwarm", vmin=-2, vmax=2, alpha=0.6, linewidths=0)
    ax.axvline(0, color="grey", lw=1)
    ax.set_yticks(range(len(order)), [label(f) if f != "language_group" else "Main language (grey: no scale)" for f in order])
    ax.set_xlabel("SHAP value: effect on the log-odds of 'abandoned' (right raises the risk)")
    ax.set_title("Which features move individual predictions (training split)", fontsize=11)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(-2, 2))
    cb = fig.colorbar(sm, ax=ax, pad=0.01)
    cb.set_label("feature value (blue low, red high)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()

    bundle = load_final_model()
    pipe, cols = bundle["model"], bundle["input_columns"]
    X_tr, _, y_tr, _ = get_early_train_test()          # the test split is not used here
    X_tr = X_tr[cols]
    print(f"Training split: {len(X_tr)} repos, {y_tr.mean():.1%} abandoned; C = {pipe.named_steps['model'].C:.4f}")

    print(f"Bootstrap ({args.n_boot} refits) ...")
    boot = bootstrap_coefs(pipe, X_tr, y_tr, args.n_boot)
    t, coef, Z = coefficient_table(pipe, X_tr, y_tr, boot)
    lang = language_table(coef, boot, Z, y_tr)
    S, err = shap_values(pipe, Z)
    print(f"SHAP check passed (max |shap - coef*(z-mean)| = {err:.1e})")

    imp = S.abs().mean().rename("mean_abs_shap")
    t = t.merge(imp.rename_axis("feature").reset_index(), on="feature", how="left")
    lang_imp = float(imp["language_group"])
    t = t.sort_values("mean_abs_shap", ascending=False)

    tables, figs = config.REPORTS_DIR / "tables", config.FIGURES_DIR
    tables.mkdir(parents=True, exist_ok=True)
    t.round(5).to_csv(tables / "interpretation_coefficients.csv", index=False)
    lang.round(5).to_csv(tables / "interpretation_language.csv", index=False)
    band_table(X_tr, y_tr).round(5).to_csv(tables / "interpretation_bands.csv", index=False)
    fig_coefficients(t, figs / "interp_coefficients.png")
    fig_shap(S, Z, figs / "interp_shap.png")

    pd.set_option("display.width", 200, "display.max_columns", 20)
    show = t[["label", "coef", "odds_ratio", "or_low", "or_high", "excludes_zero", "alone_auc", "sign_flips_vs_alone", "mean_abs_shap"]]
    print("\nNumeric features, by mean |SHAP|:\n", show.round(3).to_string(index=False))
    print(f"\nLanguage as one feature: mean |SHAP| = {lang_imp:.3f}")
    print("\nLanguage effects (log-odds, vs the average language):\n", lang.round(3).to_string(index=False))
    print("\nWrote tables and figures to", config.REPORTS_DIR)


if __name__ == "__main__":
    main()
