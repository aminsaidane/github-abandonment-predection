"""Dashboard API + the built React app, in one process.

    uvicorn dashboard.server:app --port 8000        # then open http://localhost:8000

Needs GITHUB_TOKEN (in the environment or in .env) for the live link lookup. Without it the
dashboard still works with the example repositories.
"""
import json
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dashboard import features as F
from src import config
from src.predict import load_final_model

HERE = Path(__file__).resolve().parent
DIST = HERE / "frontend" / "dist"
CACHE_SECONDS = 3600

app = FastAPI(title="GitHub abandonment forecast")
BUNDLE = load_final_model()
REFERENCE = json.loads((HERE / "reference.json").read_text())
EXAMPLES = {e["name"]: e for e in json.loads((HERE / "examples.json").read_text())}

F.collector.load_dotenv()
_gh, _lock, _cache = None, threading.Lock(), {}


def github_client():
    global _gh
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise HTTPException(503, "Live lookup is off: no GITHUB_TOKEN. The example repositories still work.")
    if _gh is None:
        _gh = F.collector.GitHub(token)
    return _gh


class UrlIn(BaseModel):
    url: str


class ExampleIn(BaseModel):
    name: str


@app.get("/api/health")
def health():
    return {"live_lookup": bool(os.environ.get("GITHUB_TOKEN")), "threshold": BUNDLE["threshold"],
            "calibration": BUNDLE["calibration"], "trained_on": BUNDLE["trained_on"],
            "training_years": list(F.TRAIN_YEARS)}


@app.get("/api/examples")
def examples():
    return [{k: e[k] for k in ("name", "case", "stars", "language", "created_at")} for e in EXAMPLES.values()]


@app.post("/api/score_example")
def score_example(body: ExampleIn):
    e = EXAMPLES.get(body.name)
    if not e:
        raise HTTPException(404, "Unknown example")
    out = F.score_features(e["features"], BUNDLE, REFERENCE)
    out["repo"] = {"full_name": e["name"], "url": f"https://github.com/{e['name']}", "created_at": e["created_at"],
                   "stars": e["stars"], "language": e["language"], "today": None,
                   "observed": e["observed"]}
    out["warnings"] = []
    out["source"] = "example (held-out test split)"
    return out


@app.post("/api/predict")
def predict(body: UrlIn):
    try:
        owner, repo = F.parse_repo(body.url)
        key = f"{owner}/{repo}".lower()
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_SECONDS:
            return hit[1]
        gh = github_client()
        with _lock:                                   # one collection at a time (rate limits)
            feats, info, warnings = F.fetch_live(owner, repo, gh, BUNDLE["input_columns"])
        out = F.score_features(feats, BUNDLE, REFERENCE)
        out.update(repo=info, warnings=warnings, source="live from the GitHub API")
        _cache[key] = (time.time(), out)
        return out
    except F.RepoError as e:
        raise HTTPException(e.status, str(e))


@app.get("/api/performance")
def performance():
    m = json.loads((config.REPORTS_DIR / "final_test_metrics.json").read_text())
    d = json.loads((config.REPORTS_DIR / "final_decisions.json").read_text())
    return {"metrics": m, "decisions": {k: d[k] for k in ("C", "calibration", "threshold", "criterion", "n_train", "oof")},
            "figures": ["final_roc_curve.png", "final_calibration.png", "final_confusion_matrix.png",
                        "final_threshold_tradeoff.png"]}


app.mount("/figures", StaticFiles(directory=config.FIGURES_DIR), name="figures")

if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        return FileResponse(f if path and f.is_file() else DIST / "index.html")
else:
    @app.get("/")
    def no_build():
        return {"message": "Frontend not built. Run: cd dashboard/frontend && npm install && npm run build"}
