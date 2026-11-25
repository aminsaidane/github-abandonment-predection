"""Check that the live link path reproduces the training features.

Fetches a few repositories from the GitHub API with the dashboard's own code and compares the
14 model inputs with the same repositories in data/pipeline3_output.csv (collected earlier with
the same collector). Run it once with your own GITHUB_TOKEN before trusting the live path:

    python -m dashboard.verify_live                       # 3 repos from the test split examples
    python -m dashboard.verify_live owner/name owner/name

Small differences are normal for very active repositories (the collector reads live API pages).
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

from dashboard import features as F
from src import config
from src.predict import load_final_model

HERE = Path(__file__).resolve().parent


def main(names):
    F.collector.load_dotenv()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("Set GITHUB_TOKEN first (environment variable or .env).")
    bundle = load_final_model()
    cols = bundle["input_columns"]
    ref = json.loads((HERE / "reference.json").read_text())
    df = pd.read_csv(config.DATA_DIR / "pipeline3_output.csv")
    df["_key"] = df["nameWithOwner"].str.lower()
    gh = F.collector.GitHub(token)
    if not names:
        names = [e["name"] for e in json.loads((HERE / "examples.json").read_text())][:3]
    bad = 0
    for name in names:
        owner, repo = F.parse_repo(name)
        hit = df[df["_key"] == f"{owner}/{repo}".lower()]
        if hit.empty:
            print(f"{name}: not in pipeline3_output.csv, skipped")
            continue
        csv_row = hit.iloc[0]
        feats, info, warns = F.fetch_live(owner, repo, gh, cols)
        diffs = []
        for c in cols:
            a, b = feats[c], csv_row[c]
            same = (str(a) == str(b)) if isinstance(a, str) else abs(float(a) - float(b)) <= 0.5
            if not same:
                diffs.append((c, b, a))
        out = F.score_features(feats, bundle, ref)
        print(f"{name}: live p_abandoned={out['p_abandoned']:.3f} | "
              + ("all 14 inputs match the CSV" if not diffs else f"{len(diffs)} input(s) differ"))
        for c, b, a in diffs:
            print(f"    {c}: csv={b}  live={a}")
        bad += bool(diffs)
    print("OK" if not bad else f"{bad} repo(s) with differences, look at the lines above")


if __name__ == "__main__":
    main(sys.argv[1:])
