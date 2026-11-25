"""Final model: freeze it on the TRAINING split, then evaluate it ONCE on the held-out test split.

Two stages, kept apart so that every decision is written down before the test labels are read:

  --fit        training split only. Tunes C, tests whether a recalibration helps, chooses the cut-off,
               and writes reports/final_decisions.json and reports/tables/final_threshold_tradeoff.csv.
  --evaluate   reads the decisions, refits on the whole training split, scores the test split once
               and writes the metrics, figures and models/final_model.pkl. Refuses to run twice.
  --dry-run    runs both stages on a pseudo-test carved out of the training split, outputs in a
               temporary folder. Use it to check the code; the real test split is never read.

Choices made in advance (they are the rules, not results):
  model        logistic regression on the 6-month features, no class weights. Weights and a tuned
               cut-off do the same job (they move the line, not the ranking), so only the cut-off is tuned.
  C            randomized search, 30 draws, 5 folds, ROC-AUC, on the training split.
  calibration  out-of-fold probabilities (5 folds) are compared with and without a sigmoid or isotonic
               recalibration by cross-validated Brier score. A recalibration is kept only if it improves
               Brier by more than 0.001 AND by more than one SD of the paired fold differences.
  cut-off      the threshold on P(abandoned) that maximises balanced accuracy (mean of the recall on
               abandoned and active repos) on the calibrated out-of-fold probabilities. "Flagged" means
               predicted abandoned.

    python -m src.final_evaluate --fit
    python -m src.final_evaluate --evaluate
    python -m src.final_evaluate --dry-run
"""
import argparse
import json
import platform
import tempfile
import warnings
from datetime import date
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, log_loss,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from src import config
from src.calibration import CALIBRATORS, make_calibrator
from src.preprocess import build_early_preprocessor, early_window_columns, get_early_train_test

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
SEED = config.RANDOM_STATE
WINDOW = "6m"
N_ITER, N_FOLDS, N_BOOT = 30, 5, 1000
MIN_BRIER_GAIN = 0.001
THRESHOLDS = np.round(np.arange(0.20, 0.995, 0.005), 3)
BASELINE_COL = "days_since_last_commit_6m"          # best single rule in src/baselines.py
PERIODS = ([-np.inf, 2017, 2019, 2021, np.inf], ["<= 2017", "2018-2019", "2020-2021", "2022-2023"])


class Paths:
    """Where the outputs go: the repo for a real run, a temporary folder for --dry-run."""
    def __init__(self, root=None):
        if root is None:
            self.reports, self.model = config.REPORTS_DIR, config.FINAL_MODEL_PATH
        else:
            root = Path(root)
            self.reports, self.model = root / "reports", root / "models" / "final_model.pkl"
        self.figures, self.tables = self.reports / "figures", self.reports / "tables"
        self.decisions = self.reports / "final_decisions.json"
        self.metrics = self.reports / "final_test_metrics.json"
        self.report = self.reports / "final_evaluation_report.txt"
        self.tradeoff = self.tables / "final_threshold_tradeoff.csv"
        for d in (self.figures, self.tables, self.model.parent):
            d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ helpers
def input_columns():
    """The columns the model reads (everything else in the data is ignored)."""
    counts, bounded = early_window_columns(WINDOW)
    return counts + bounded + config.CREATION_NUMERIC + config.CREATION_FLAGS + config.CREATION_CATEGORICAL


def make_pipeline(C):
    return Pipeline([("prep", build_early_preprocessor(WINDOW, scale=True)),
                     ("model", LogisticRegression(C=C, max_iter=2000))])


def ece(y, p, bins=10):
    """Expected calibration error with equal-count bins."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges[1:-1], p, side="right"), 0, bins - 1)
    return float(sum((idx == b).sum() / len(p) * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def reliability(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges[1:-1], p, side="right"), 0, bins - 1)
    pts = [(p[idx == b].mean(), y[idx == b].mean()) for b in range(bins) if (idx == b).any()]
    return np.array(pts)


def threshold_table(y, p):
    """Recall, precision and balanced accuracy for every cut-off. Positive class = abandoned."""
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for t in THRESHOLDS:
        flag = p >= t
        tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
        fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
        rec_ab, rec_ac = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
        rows.append({"threshold": t, "recall_abandoned": rec_ab, "recall_active": rec_ac,
                     "precision_abandoned": tp / max(tp + fp, 1), "precision_active": tn / max(tn + fn, 1),
                     "balanced_accuracy": (rec_ab + rec_ac) / 2, "share_flagged": flag.mean()})
    return pd.DataFrame(rows)


def best_threshold(y, p):
    """Cut-off maximising balanced accuracy (middle of the plateau when several tie)."""
    tab = threshold_table(y, p)
    top = tab[tab["balanced_accuracy"] >= tab["balanced_accuracy"].max() - 1e-12]["threshold"]
    return float(top.median()), tab


def boot_ci(fn, y, p, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    y, p, vals = np.asarray(y), np.asarray(p), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(fn(y[i], p[i]))
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


def by_period(years, y, p):
    cat = pd.cut(years, bins=PERIODS[0], labels=PERIODS[1])
    rows = []
    for lab in PERIODS[1]:
        m = np.asarray(cat == lab)
        if m.sum() < 30:
            continue
        rows.append({"created": lab, "n": int(m.sum()), "observed_abandoned": float(y[m].mean()),
                     "mean_predicted": float(p[m].mean()),
                     "roc_auc": float(roc_auc_score(y[m], p[m])) if y[m].min() != y[m].max() else None})
    return rows


# ------------------------------------------------------------------ stage 1: fit
def fit_stage(X, y, paths):
    """Everything here sees the training split only."""
    years = pd.to_datetime(X["createdAt"], utc=True).dt.year.to_numpy()
    X = X[input_columns()]
    print(f"training repos: {len(y)} ({y.mean():.1%} abandoned)")

    # 1. C
    search = RandomizedSearchCV(make_pipeline(1.0), {"model__C": loguniform(1e-3, 1e2)}, n_iter=N_ITER,
                                cv=StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED),
                                scoring="roc_auc", random_state=SEED, n_jobs=1, refit=False).fit(X, y)
    C = float(search.best_params_["model__C"])
    print(f"C = {C:.4f}  (CV ROC-AUC {search.best_score_:.4f})")

    # 2. out-of-fold probabilities
    p_oof = cross_val_predict(make_pipeline(C), X, y, cv=StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + 1),
                              method="predict_proba")[:, 1]

    # 3. does a recalibration help? (cross-validated on the out-of-fold probabilities)
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + 2).split(p_oof, y))
    briers = {name: [] for name in CALIBRATORS}
    for tr, va in folds:
        for name in CALIBRATORS:
            cal = make_calibrator(name).fit(p_oof[tr], y.iloc[tr])
            briers[name].append(brier_score_loss(y.iloc[va], cal.predict(p_oof[va])))
    briers = {k: np.array(v) for k, v in briers.items()}
    cal_rows, chosen = [], "none"
    for name, b in briers.items():
        gain = briers["none"] - b
        cal_rows.append({"calibration": name, "brier": float(b.mean()), "gain_vs_none": float(gain.mean()),
                         "gain_sd": float(gain.std())})
    best = max((r for r in cal_rows if r["calibration"] != "none"), key=lambda r: r["gain_vs_none"])
    if best["gain_vs_none"] > MIN_BRIER_GAIN and best["gain_vs_none"] > best["gain_sd"]:
        chosen = best["calibration"]
    print("calibration candidates:", {r["calibration"]: round(r["brier"], 5) for r in cal_rows}, "-> chosen:", chosen)

    # 4. cut-off on cross-fitted calibrated out-of-fold probabilities
    p_cal = np.empty_like(p_oof)
    for tr, va in folds:
        p_cal[va] = make_calibrator(chosen).fit(p_oof[tr], y.iloc[tr]).predict(p_oof[va])
    t_star, tab = best_threshold(y, p_cal)
    rng = np.random.default_rng(SEED)
    boot_t = [best_threshold(y.iloc[i], p_cal[i])[0] for i in (rng.integers(0, len(y), len(y)) for _ in range(200))]
    row = tab[tab["threshold"] == t_star].iloc[0]
    tab.round(4).to_csv(paths.tradeoff, index=False)
    print(f"cut-off {t_star:.3f}  (bootstrap SD {np.std(boot_t):.3f}); recall abandoned {row.recall_abandoned:.3f}, "
          f"recall active {row.recall_active:.3f}, balanced accuracy {row.balanced_accuracy:.3f}")

    decisions = {
        "written": str(date.today()), "n_train": int(len(y)), "abandon_rate_train": float(y.mean()),
        "window": WINDOW, "class_weight": None, "C": C, "cv_roc_auc_for_C": float(search.best_score_),
        "calibration": chosen, "calibration_candidates": cal_rows, "min_brier_gain_rule": MIN_BRIER_GAIN,
        "threshold": t_star, "threshold_bootstrap_sd": float(np.std(boot_t)),
        "criterion": "maximise balanced accuracy on calibrated out-of-fold probabilities; flagged = predicted abandoned",
        "oof": {"roc_auc": float(roc_auc_score(y, p_cal)), "ap_active": float(average_precision_score(1 - y, 1 - p_cal)),
                "brier": float(brier_score_loss(y, p_cal)), "ece": ece(y, p_cal),
                "mean_predicted": float(p_cal.mean()), "balanced_accuracy_at_threshold": float(row.balanced_accuracy),
                "recall_abandoned": float(row.recall_abandoned), "recall_active": float(row.recall_active)},
        "oof_by_creation_period": by_period(years, y.to_numpy(), p_cal),
        "versions": {"sklearn": sklearn.__version__, "python": platform.python_version()},
    }
    return decisions, p_oof


# ------------------------------------------------------------------ stage 2: evaluate
def evaluate_stage(X_tr, y_tr, X_te, y_te, decisions, paths):
    cols = input_columns()
    pipe = make_pipeline(decisions["C"]).fit(X_tr[cols], y_tr)
    # calibrator: fitted on the training out-of-fold probabilities, exactly as decided in --fit
    p_oof = cross_val_predict(make_pipeline(decisions["C"]), X_tr[cols], y_tr,
                              cv=StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + 1), method="predict_proba")[:, 1]
    cal = make_calibrator(decisions["calibration"]).fit(p_oof, y_tr)
    t = decisions["threshold"]

    p = cal.predict(pipe.predict_proba(X_te[cols])[:, 1])
    y = y_te.to_numpy()
    flag = p >= t
    tn, fp, fn, tp = confusion_matrix(y, flag).ravel()
    ba = lambda yy, pp: (((pp >= t) & (yy == 1)).sum() / max((yy == 1).sum(), 1) + ((pp < t) & (yy == 0)).sum() / max((yy == 0).sum(), 1)) / 2
    m = {
        "n_test": int(len(y)), "abandon_rate_test": float(y.mean()), "threshold": t, "calibration": decisions["calibration"],
        "roc_auc": float(roc_auc_score(y, p)), "roc_auc_ci": boot_ci(roc_auc_score, y, p),
        "ap_abandoned": float(average_precision_score(y, p)), "ap_abandoned_no_skill": float(y.mean()),
        "ap_active": float(average_precision_score(1 - y, 1 - p)), "ap_active_ci": boot_ci(lambda yy, pp: average_precision_score(1 - yy, 1 - pp), y, p),
        "ap_active_no_skill": float(1 - y.mean()),
        "baseline_roc_auc_single_rule": float(roc_auc_score(y, X_te[BASELINE_COL])),
        "brier": float(brier_score_loss(y, p)), "brier_ci": boot_ci(brier_score_loss, y, p),
        "log_loss": float(log_loss(y, p)), "ece": ece(y, p),
        "mean_predicted": float(p.mean()),
        "confusion": {"true_active_pred_active": int(tn), "true_active_flagged": int(fp),
                      "true_abandoned_missed": int(fn), "true_abandoned_flagged": int(tp)},
        "balanced_accuracy": float(ba(y, p)), "balanced_accuracy_ci": boot_ci(ba, y, p),
        "recall_abandoned": float(tp / (tp + fn)), "recall_active": float(tn / (tn + fp)),
        "precision_abandoned": float(tp / max(tp + fp, 1)), "precision_active": float(tn / max(tn + fn, 1)),
        "share_flagged": float(flag.mean()),
        "cv_estimate_roc_auc": decisions["oof"]["roc_auc"],
        "by_creation_period": by_period(pd.to_datetime(X_te["createdAt"], utc=True).dt.year.to_numpy(), y, p),
    }
    paths.metrics.write_text(json.dumps(m, indent=2))

    # ---- model file for the dashboard
    bundle = {"model": pipe, "calibrator": cal, "calibration": decisions["calibration"], "threshold": t,
              "input_columns": cols, "window": WINDOW, "positive_label": config.POSITIVE_LABEL,
              "trained_on": f"training split, n={len(y_tr)}", "decisions": decisions,
              "test_summary": {k: m[k] for k in ("roc_auc", "roc_auc_ci", "ap_active", "brier", "ece")},
              "versions": {"sklearn": sklearn.__version__, "python": platform.python_version(), "numpy": np.__version__}}
    joblib.dump(bundle, paths.model)

    make_figures(y, p, p_oof_cal=cal.predict(p_oof), y_tr=y_tr.to_numpy(), t=t, paths=paths, m=m)
    write_report(m, decisions, paths)
    return m


def make_figures(y, p, p_oof_cal, y_tr, t, paths, m):
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(figsize=(5, 4.5))
    plt.plot(fpr, tpr, label=f"final model (AUC {m['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="chance")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend(); plt.title("ROC, held-out test split")
    save(paths.figures / "final_roc_curve.png")

    plt.figure(figsize=(5, 4.5))
    pts, pts_oof = reliability(y, p), reliability(y_tr, p_oof_cal)
    plt.plot([0.5, 1], [0.5, 1], "--", color="grey", label="perfect")
    plt.plot(pts_oof[:, 0], pts_oof[:, 1], "o-", alpha=0.5, label="training (out-of-fold)")
    plt.plot(pts[:, 0], pts[:, 1], "o-", label=f"test (ECE {m['ece']:.3f})")
    plt.xlabel("Predicted P(abandoned)"); plt.ylabel("Observed share abandoned"); plt.legend(); plt.title("Calibration")
    save(paths.figures / "final_calibration.png")

    c = m["confusion"]
    cm = np.array([[c["true_active_pred_active"], c["true_active_flagged"]], [c["true_abandoned_missed"], c["true_abandoned_flagged"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}\n({v / cm[i].sum():.0%} of row)", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["not flagged", "flagged at risk"]); ax.set_yticks([0, 1], ["active", "abandoned"])
    ax.set_title(f"Confusion matrix at cut-off {t:.3f}")
    save(paths.figures / "final_confusion_matrix.png")

    tab = pd.read_csv(paths.tradeoff)
    plt.figure(figsize=(6, 4.2))
    for col, lab in [("recall_abandoned", "recall abandoned"), ("recall_active", "recall active"), ("balanced_accuracy", "balanced accuracy")]:
        plt.plot(tab["threshold"], tab[col], label=lab)
    plt.axvline(t, color="k", ls=":", label="chosen cut-off")
    plt.xlabel("Cut-off on P(abandoned)"); plt.legend(); plt.title("Cut-off trade-off (training, out-of-fold)")
    save(paths.figures / "final_threshold_tradeoff.png")


def save(path):
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def write_report(m, d, paths):
    c = m["confusion"]
    lines = [
        "FINAL EVALUATION, held-out test split (read once)",
        f"model: logistic regression, {d['window']} features, C = {d['C']:.4f}, calibration = {d['calibration']}, cut-off = {m['threshold']:.3f}",
        f"test repos: {m['n_test']} ({m['abandon_rate_test']:.1%} abandoned)",
        "",
        f"ROC-AUC            {m['roc_auc']:.3f}  (95% CI {m['roc_auc_ci'][0]:.3f}-{m['roc_auc_ci'][1]:.3f})   cross-validation estimate {m['cv_estimate_roc_auc']:.3f}",
        f"single-rule baseline ({BASELINE_COL}) ROC-AUC {m['baseline_roc_auc_single_rule']:.3f}",
        f"AP, active class   {m['ap_active']:.3f}  (95% CI {m['ap_active_ci'][0]:.3f}-{m['ap_active_ci'][1]:.3f}; no-skill = share active {m['ap_active_no_skill']:.3f})",
        f"AP, abandoned      {m['ap_abandoned']:.3f}  (no-skill {m['ap_abandoned_no_skill']:.3f})",
        f"Brier score        {m['brier']:.4f}  (95% CI {m['brier_ci'][0]:.4f}-{m['brier_ci'][1]:.4f});  ECE {m['ece']:.3f};  mean predicted {m['mean_predicted']:.3f} vs observed {m['abandon_rate_test']:.3f}",
        "",
        f"At the cut-off {m['threshold']:.3f} ({m['share_flagged']:.1%} of repos flagged at risk):",
        f"                    not flagged   flagged",
        f"  true active       {c['true_active_pred_active']:>11}   {c['true_active_flagged']:>7}",
        f"  true abandoned    {c['true_abandoned_missed']:>11}   {c['true_abandoned_flagged']:>7}",
        f"  recall abandoned {m['recall_abandoned']:.3f}, recall active {m['recall_active']:.3f}, balanced accuracy {m['balanced_accuracy']:.3f} (95% CI {m['balanced_accuracy_ci'][0]:.3f}-{m['balanced_accuracy_ci'][1]:.3f})",
        f"  precision abandoned {m['precision_abandoned']:.3f}, precision active {m['precision_active']:.3f}",
        "",
        "By creation period (older repos had longer to be abandoned):",
    ]
    lines += [f"  {r['created']:<10} n={r['n']:<4} observed {r['observed_abandoned']:.3f}  predicted {r['mean_predicted']:.3f}  AUC {r['roc_auc'] if r['roc_auc'] is None else round(r['roc_auc'], 3)}"
              for r in m["by_creation_period"]]
    paths.report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fit", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    ap.add_argument("--data", type=Path, default=config.EARLY_DATA_PATH)
    ap.add_argument("--allow-rerun", action="store_true", help="evaluate again although the test split was already used")
    args = ap.parse_args()

    X_tr, X_te, y_tr, y_te = get_early_train_test(args.data)

    if args.dry_run:
        out = Path(tempfile.mkdtemp(prefix="final_dry_run_"))
        paths = Paths(out)
        A, B, ya, yb = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=SEED)
        print(f"DRY RUN: {len(A)} train / {len(B)} pseudo-test, all from the training split. Output: {out}\n")
        decisions, _ = fit_stage(A, ya, paths)
        paths.decisions.write_text(json.dumps(decisions, indent=2))
        evaluate_stage(A, ya, B, yb, decisions, paths)
        print(f"\nfiles: {sorted(p.name for p in out.rglob('*') if p.is_file())}")
        return

    paths = Paths()
    if args.fit:
        decisions, _ = fit_stage(X_tr, y_tr, paths)
        paths.decisions.write_text(json.dumps(decisions, indent=2))
        print(f"\nDecisions written to {paths.decisions.relative_to(config.ROOT_DIR)}. The test split has not been scored or looked at.")
    else:
        if not paths.decisions.exists():
            raise SystemExit("run --fit first: the decisions must be written down before the test split is scored")
        if paths.metrics.exists() and not args.allow_rerun:
            raise SystemExit(f"{paths.metrics.name} exists: the test split was already used. Use --allow-rerun only if you accept that.")
        decisions = json.loads(paths.decisions.read_text())
        evaluate_stage(X_tr, y_tr, X_te, y_te, decisions, paths)
        print(f"\nModel saved to {paths.model.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main()
