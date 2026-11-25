"""Prepare the dashboard's offline data from the study data (run once; the outputs are committed).

  dashboard/reference.json  training-split averages used to explain a score (transformed feature
                            means, raw medians, abandonment rate). Training split only.
  dashboard/examples.json   a dozen real repositories from the held-out test split, with their
                            features and what actually happened, for a demo that needs no internet.
                            They are chosen to show the model honestly: right and wrong calls.

    python -m dashboard.make_examples
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard.features import KNOWN_LANGS, LABELS  # noqa: F401  (import checks the module loads)
from src import config
from src.predict import load_final_model, predict_risk
from src.preprocess import get_early_train_test

OUT = Path(__file__).resolve().parent
PER_BUCKET = {"flagged_and_abandoned": 3, "flagged_but_active": 3, "not_flagged_and_active": 3, "missed": 2}


def main():
    bundle = load_final_model()
    cols = bundle["input_columns"]
    X_tr, X_te, y_tr, y_te = get_early_train_test()

    # ---- reference values from the TRAINING split
    prep = bundle["model"].named_steps["prep"]
    Z = prep.transform(X_tr[cols])
    means = dict(zip(prep.get_feature_names_out(), Z.mean(axis=0).tolist()))
    numeric = [c for c in cols if c != "language_group"]
    typical = {c: float(X_tr[c].median()) for c in numeric}
    typical["language_group"] = X_tr["language_group"].mode().iloc[0]
    ref = {"transformed_means": means, "typical": typical, "abandon_rate_train": float(y_tr.mean()),
           "n_train": int(len(y_tr))}
    (OUT / "reference.json").write_text(json.dumps(ref, indent=1))

    # ---- examples from the TEST split
    s = predict_risk(X_te[cols], bundle)
    d = X_te.assign(p=s["p_abandoned"], flagged=s["flagged"], abandoned=y_te)
    buckets = {
        "flagged_and_abandoned": d[d.flagged & (d.abandoned == 1)],
        "flagged_but_active": d[d.flagged & (d.abandoned == 0)],
        "not_flagged_and_active": d[~d.flagged & (d.abandoned == 0)],
        "missed": d[~d.flagged & (d.abandoned == 1)],
    }
    rng = np.random.default_rng(config.RANDOM_STATE)
    examples = []
    for name, n in PER_BUCKET.items():
        pool = buckets[name].sort_values("stars", ascending=False).head(25)      # well-known repos first
        for _, r in pool.iloc[rng.permutation(len(pool))[:n]].iterrows():
            examples.append({
                "name": r["nameWithOwner"], "case": name, "stars": int(r["stars"]),
                "created_at": str(r["createdAt"])[:10], "language": r["primaryLanguage"],
                "observed": "abandoned" if r["abandoned"] else "active",
                "features": {c: (r[c].item() if hasattr(r[c], "item") else r[c]) for c in cols},
            })
    (OUT / "examples.json").write_text(json.dumps(examples, indent=1))
    print(f"reference.json ({len(means)} transformed features) and examples.json ({len(examples)} repos) written")
    print(pd.Series([e["case"] for e in examples]).value_counts().to_dict())


if __name__ == "__main__":
    main()
