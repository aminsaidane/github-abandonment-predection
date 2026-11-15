"""Collect each repo OWNER's track record, as it stood at month 6 of the repo (pilot).

For a target repo created at C, the forecast is made at T = C + 182 days. Everything here is
measured as of T, so the features cannot see the future:
  owner_account_age_days     age of the owner's account when the repo was created
  owner_prior_repos          non-fork repos the owner created before this one
  owner_repos_prev_year      of those, created in the 365 days before this one
  owner_track_checked        older repos checked for activity (the 10 most recent created >= 1 year before T)
  owner_track_active_share   share of those with at least one commit in the year before T
NOT used, because they leak the future: current follower / repo counts and the current last-push date
of the owner's other repos (an owner active after T is likely to keep the target alive too).

Limitation: the API lists only repos that still exist, so deleted (often abandoned) projects are missing.

Two steps, both run by default:
  collect   GitHub API -> data/owner_history_raw.jsonl   (one line per target repo, resumable)
  features  raw JSONL  -> data/owner_history.csv          (one row per target repo)

Usage, from the repo root (needs GITHUB_TOKEN in .env, like collect_early_activity.py):
  python scritpts/collect_owner_history.py --limit 20       # trial run
  python scritpts/collect_owner_history.py                  # pilot: 1,000 training repos (~2-3 h, resumable)
  python scritpts/collect_owner_history.py --features-only
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_early_activity import API, GitHub, collect_list, iso, load_dotenv, parse_ts  # noqa: E402

RAW_PATH = ROOT / "data" / "owner_history_raw.jsonl"
OUT_CSV = ROOT / "data" / "owner_history.csv"
FORECAST_DAYS = 182          # T = creation + 6 months, the forecasting date
TRACK_MAX = 10               # older repos checked per owner
PILOT_SIZE = 1000


def pilot_targets(size=PILOT_SIZE):
    """Stratified sample of the TRAINING split only (the test split stays untouched)."""
    from sklearn.model_selection import train_test_split
    from src.preprocess import get_early_train_test
    X, _, y, _ = get_early_train_test()
    if size >= len(X):
        sel = X
    else:
        sel, _ = train_test_split(X, train_size=size, stratify=y, random_state=42)
    return [(r.nameWithOwner, r.owner, r.createdAt) for r in sel.itertuples()]


def owner_profile(gh, cache, login):
    """Account creation date + all repos the owner still has (forks excluded). Cached per owner."""
    if login in cache:
        return cache[login]
    status, user, _ = gh.get(f"{API}/users/{login}")
    if status != 200:
        cache[login] = {"status": f"http_{status}"}
        return cache[login]
    repos, st = collect_list(gh, f"/users/{login}/repos", {"type": "owner", "sort": "created", "direction": "asc"})
    cache[login] = {
        "status": "ok" if st in (200, "capped") else f"repos_http_{st}",
        "type": user.get("type"),
        "account_created": user.get("created_at"),
        "repos_capped": st == "capped",
        "repos": [{"full_name": r["full_name"], "created": r["created_at"]} for r in repos if not r.get("fork")],
    }
    return cache[login]


def active_in_year_before(gh, full_name, t):
    """True if the repo has at least one commit in the 365 days before t; None if it can't be read."""
    status, body, _ = gh.get(f"{API}/repos/{full_name}/commits",
                             params={"since": iso(t - timedelta(days=365)), "until": iso(t), "per_page": 1})
    if status == 200 and isinstance(body, list):
        return len(body) > 0
    if status == 409:                     # empty repository
        return False
    return None


def collect_target(gh, cache, name, owner, created_at):
    created = parse_ts(created_at)
    t = created + timedelta(days=FORECAST_DAYS)
    rec = {"nameWithOwner": name, "owner": owner, "createdAt": created_at}
    prof = owner_profile(gh, cache, owner)
    rec["status"] = prof["status"]
    if prof["status"] != "ok":
        return rec
    prior = [r for r in prof["repos"] if parse_ts(r["created"]) < created and r["full_name"].lower() != name.lower()]
    old = [r for r in prior if parse_ts(r["created"]) <= t - timedelta(days=365)]
    old = sorted(old, key=lambda r: r["created"], reverse=True)[:TRACK_MAX]
    rec.update({
        "owner_type_api": prof["type"],
        "account_created": prof["account_created"],
        "repos_capped": prof["repos_capped"],
        "prior": [r["created"] for r in prior],
        "track": [{"full_name": r["full_name"], "active": active_in_year_before(gh, r["full_name"], t)} for r in old],
    })
    return rec


def collect(limit=None):
    targets = pilot_targets()
    if limit:
        targets = targets[:limit]
    done = set()
    if RAW_PATH.exists():
        with open(RAW_PATH, encoding="utf-8") as f:
            done = {json.loads(line)["nameWithOwner"] for line in f if line.strip()}
    todo = [t for t in targets if t[0] not in done]
    print(f"{len(targets)} target repos, {len(done)} already collected, {len(todo)} to go")
    load_dotenv()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("GITHUB_TOKEN is not set (put it in .env at the repo root)")
    gh, cache, t0 = GitHub(token), {}, time.time()
    with open(RAW_PATH, "a", encoding="utf-8") as out:
        for i, (name, owner, created) in enumerate(todo, 1):
            try:
                rec = collect_target(gh, cache, name, owner, created)
            except Exception as e:              # keep going; the repo is retried on the next run
                print(f"  {name}: {e.__class__.__name__}, skipped")
                continue
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if i % 25 == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                print(f"  {i}/{len(todo)} done, {gh.calls} API calls, ~{rate * (len(todo) - i) / 60:.0f} min left")


def features():
    rows = []
    with open(RAW_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            row = {"nameWithOwner": rec["nameWithOwner"], "owner_status": rec["status"]}
            if rec["status"] == "ok":
                created = parse_ts(rec["createdAt"])
                checked = [x for x in rec["track"] if x["active"] is not None]
                row.update({
                    "owner_account_age_days": max(0, (created - parse_ts(rec["account_created"])).days),
                    "owner_prior_repos": len(rec["prior"]),
                    "owner_repos_prev_year": sum(created - parse_ts(c) <= timedelta(days=365) for c in rec["prior"]),
                    "owner_track_checked": len(checked),
                    "owner_track_active_share": (sum(x["active"] for x in checked) / len(checked)) if checked else "",
                    "owner_repos_capped": int(rec["repos_capped"]),
                })
            rows.append(row)
    cols = ["nameWithOwner", "owner_status", "owner_account_age_days", "owner_prior_repos", "owner_repos_prev_year",
            "owner_track_checked", "owner_track_active_share", "owner_repos_capped"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    ok = sum(r["owner_status"] == "ok" for r in rows)
    print(f"{len(rows)} rows -> {OUT_CSV.relative_to(ROOT)} ({ok} ok, {len(rows) - ok} owner not found)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, help="only the first N pilot repos (trial run)")
    ap.add_argument("--features-only", action="store_true", help="skip the API, rebuild the CSV")
    args = ap.parse_args()
    if not args.features_only:
        collect(args.limit)
    features()


if __name__ == "__main__":
    main()
