"""Collect each repo's EARLY-LIFE activity from the GitHub REST API.

For every repo in data/pipeline2_output.csv, this script records what happened
in its first 6 months after creation (commits, contributors, stars, forks,
issues, pull requests), then turns it into per-repo features for a 3-month and
a 6-month window. Only information visible at the end of the window is used,
so a model trained on these features genuinely forecasts abandonment.

Two steps, both run by default:
  collect   GitHub API -> data/early_activity_raw.jsonl  (one line per repo, resumable)
  features  raw JSONL  -> data/early_activity.csv         (one row per repo)

Setup (once): create a GitHub personal access token with read-only access to
public repositories, then put it in an environment variable:
  PowerShell : $env:GITHUB_TOKEN = "ghp_..."
  Mac/Linux  : export GITHUB_TOKEN=ghp_...

Usage, from the repo root:
  python scritpts/collect_early_activity.py --limit 20     # trial run on 20 repos
  python scritpts/collect_early_activity.py                # full run (~5 hours, resumable)
  python scritpts/collect_early_activity.py --features-only
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "data" / "pipeline2_output.csv"
RAW_PATH = ROOT / "data" / "early_activity_raw.jsonl"
OUT_CSV = ROOT / "data" / "early_activity.csv"

API = "https://api.github.com"
WINDOWS = {"3m": 91, "6m": 182}          # window lengths in days
COLLECT_DAYS = max(WINDOWS.values())      # collect events up to the longest window
MAX_PAGES = 10                            # 100 items/page -> at most 1,000 items per endpoint
RATE_LIMIT_FLOOR = 50                     # pause when fewer calls than this remain


# ====================================================================== HTTP
class GitHub:
    """Small REST client: auth, pagination, rate-limit pauses, retries."""

    def __init__(self, token: str):
        self.s = requests.Session()
        self.s.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "github-abandonment-prediction",
        })
        self.calls = 0

    def get(self, url, params=None, accept=None):
        """GET with retries. Returns (status_code, json_or_None, response)."""
        headers = {"Accept": accept} if accept else None
        for attempt in range(6):
            try:
                r = self.s.get(url, params=params, headers=headers, timeout=30)
            except requests.RequestException as e:
                wait = 2 ** attempt * 5
                print(f"    network error ({e.__class__.__name__}), retry in {wait}s")
                time.sleep(wait)
                continue
            self.calls += 1
            self._respect_rate_limit(r)

            # secondary rate limit / abuse detection
            if r.status_code in (403, 429) and (
                "rate limit" in r.text.lower() or "retry-after" in r.headers
            ):
                wait = int(r.headers.get("Retry-After", 60))
                print(f"    rate limited, waiting {wait}s")
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                wait = 2 ** attempt * 5
                print(f"    server error {r.status_code}, retry in {wait}s")
                time.sleep(wait)
                continue
            try:
                body = r.json()
            except ValueError:
                body = None
            return r.status_code, body, r
        raise RuntimeError(f"giving up on {url}")

    def _respect_rate_limit(self, r):
        remaining = r.headers.get("X-RateLimit-Remaining")
        reset = r.headers.get("X-RateLimit-Reset")
        if remaining is not None and int(remaining) < RATE_LIMIT_FLOOR and reset:
            wait = max(0, int(reset) - int(time.time())) + 5
            print(f"    {remaining} API calls left, pausing {wait // 60} min until the limit resets")
            time.sleep(wait)

    def pages(self, path, params, accept=None, stop=None):
        """Yield items page by page. `stop(item)` -> True ends paging early.
        Returns via StopIteration value whether MAX_PAGES was hit."""
        params = dict(params, per_page=100)
        url = f"{API}{path}"
        for page in range(1, MAX_PAGES + 1):
            status, body, r = self.get(url, params={**params, "page": page}, accept=accept)
            if status != 200 or not isinstance(body, list):
                return status
            for item in body:
                if stop and stop(item):
                    return 200
                yield item
            if len(body) < 100 or 'rel="next"' not in r.headers.get("Link", ""):
                return 200
        return "capped"


def collect_list(gh, path, params, accept=None, stop=None):
    """Run gh.pages() and return (items, status) where status is 200/'capped'/HTTP code."""
    items, gen = [], gh.pages(path, params, accept=accept, stop=stop)
    while True:
        try:
            items.append(next(gen))
        except StopIteration as done:
            return items, done.value


# ====================================================================== collect
def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def iso(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def collect_repo(gh, full_name, created_at):
    """Fetch early-life events for one repo. Only dates and authors are kept."""
    created = parse_ts(created_at)
    end = created + timedelta(days=COLLECT_DAYS)
    rec = {"nameWithOwner": full_name, "createdAt": created_at, "collected_at": iso(datetime.now(timezone.utc))}

    status, repo, _ = gh.get(f"{API}/repos/{full_name}")
    if status != 200:
        rec["status"] = "not_found" if status in (404, 451) else f"http_{status}"
        return rec
    rec["owner_login"] = repo["owner"]["login"]
    rec["owner_type"] = repo["owner"]["type"]           # User / Organization
    rec["current_full_name"] = repo["full_name"]        # differs if the repo was renamed
    path = f"/repos/{repo['full_name']}"
    caps = {}

    # commits up to the window end (newest first); includes pre-creation (imported) history
    commits, st = collect_list(gh, f"{path}/commits", {"until": iso(end)})
    if st == 409:                                        # empty repository
        commits, st = [], 200
    caps["commits"] = st
    rec["commits"] = [
        {
            "date": c["commit"]["committer"]["date"],
            "author": (c.get("author") or {}).get("login")
                      or (c["commit"]["author"] or {}).get("email", "").lower(),
        }
        for c in commits
    ]

    # stars, oldest first, stop past the window
    stars, st = collect_list(
        gh, f"{path}/stargazers", {},
        accept="application/vnd.github.star+json",
        stop=lambda s: parse_ts(s["starred_at"]) > end,
    )
    caps["stars"] = st
    rec["stars"] = [s["starred_at"] for s in stars]

    # forks, oldest first, stop past the window
    forks, st = collect_list(
        gh, f"{path}/forks", {"sort": "oldest"},
        stop=lambda f: parse_ts(f["created_at"]) > end,
    )
    caps["forks"] = st
    rec["forks"] = [f["created_at"] for f in forks]

    # issues + pull requests, oldest first, stop past the window
    issues, st = collect_list(
        gh, f"{path}/issues", {"state": "all", "sort": "created", "direction": "asc"},
        stop=lambda i: parse_ts(i["created_at"]) > end,
    )
    if st == 410:                                        # issues disabled
        issues, st = [], 200
    caps["issues"] = st
    rec["issues"] = [
        {
            "created": i["created_at"],
            "closed": i.get("closed_at"),
            "author": (i.get("user") or {}).get("login"),
            "is_pr": "pull_request" in i,
        }
        for i in issues
    ]

    rec["capped"] = sorted(k for k, v in caps.items() if v == "capped")
    errors = {k: v for k, v in caps.items() if v not in (200, "capped")}
    rec["status"] = "ok" if not errors else "partial:" + ",".join(f"{k}={v}" for k, v in errors.items())
    return rec


def load_targets(limit=None):
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        rows = [(r["nameWithOwner"].strip(), r["createdAt"].strip()) for r in csv.DictReader(f)]
    return rows[:limit] if limit else rows


def done_names():
    """Repos already collected for good. Partial or failed ones are retried on the next run."""
    if not RAW_PATH.exists():
        return set()
    done = set()
    with open(RAW_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("status") in ("ok", "not_found"):
                    done.add(rec["nameWithOwner"])
    return done


def run_collect(limit=None):
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("GITHUB_TOKEN is not set. See the instructions at the top of this file.")
    gh = GitHub(token)

    targets = load_targets(limit)
    done = done_names()
    todo = [t for t in targets if t[0] not in done]
    print(f"{len(targets)} repos, {len(done & {t[0] for t in targets})} already collected, {len(todo)} to go")

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(RAW_PATH, "a", encoding="utf-8") as out:
        for n, (name, created) in enumerate(todo, 1):
            try:
                rec = collect_repo(gh, name, created)
            except Exception as e:                      # keep going; record the failure
                rec = {"nameWithOwner": name, "createdAt": created, "status": f"error:{e.__class__.__name__}"}
            out.write(json.dumps(rec) + "\n")
            out.flush()                                  # every repo is saved immediately
            if n % 25 == 0 or n == len(todo):
                per = (time.time() - start) / n
                left = per * (len(todo) - n) / 60
                print(f"  {n}/{len(todo)}  ({gh.calls} API calls, ~{left:.0f} min left)")
    print(f"Raw data -> {RAW_PATH.relative_to(ROOT)}")


# ====================================================================== features
def day_offset(ts, created):
    return (parse_ts(ts) - created).total_seconds() / 86400


def window_features(rec, days):
    """Features using ONLY events between creation and creation + `days`."""
    created = parse_ts(rec["createdAt"])
    owner = (rec.get("owner_login") or "").lower()
    inside = lambda off: 0 <= off <= days

    c_off = [(day_offset(c["date"], created), c["author"]) for c in rec.get("commits", [])]
    c_in = [(o, a) for o, a in c_off if inside(o)]
    authors = {a.lower() for _, a in c_in if a}
    issues = [i for i in rec.get("issues", []) if inside(day_offset(i["created"], created))]
    opened_issues = [i for i in issues if not i["is_pr"]]
    opened_prs = [i for i in issues if i["is_pr"]]
    closed = [i for i in issues if i["closed"] and day_offset(i["closed"], created) <= days]
    last_commit = max((o for o, _ in c_in), default=None)

    return {
        "commits": len(c_in),
        "active_weeks": len({int(o // 7) for o, _ in c_in}),
        "commits_last_30d": sum(1 for o, _ in c_in if o > days - 30),
        "days_since_last_commit": round(days - last_commit, 1) if last_commit is not None else days,
        "contributors": len(authors),
        "nonowner_contributors": len(authors - {owner}),
        "stars": sum(1 for s in rec.get("stars", []) if inside(day_offset(s, created))),
        "forks": sum(1 for f in rec.get("forks", []) if inside(day_offset(f, created))),
        "issues_opened": len(opened_issues),
        "prs_opened": len(opened_prs),
        "issues_prs_closed": len(closed),
        "external_issue_authors": len({i["author"].lower() for i in issues if i["author"]} - {owner}),
    }


def build_row(rec):
    row = {
        "nameWithOwner": rec["nameWithOwner"],
        "early_status": rec.get("status", "missing"),
        "owner_type": rec.get("owner_type", ""),
        "renamed": int(bool(rec.get("current_full_name"))
                       and rec["current_full_name"].lower() != rec["nameWithOwner"].lower()),
        "capped": "|".join(rec.get("capped", [])),
    }
    if row["early_status"] != "ok":
        return row
    created = parse_ts(rec["createdAt"])
    row["precreation_commits"] = sum(
        1 for c in rec.get("commits", []) if day_offset(c["date"], created) < 0
    )
    for label, days in WINDOWS.items():
        for k, v in window_features(rec, days).items():
            row[f"{k}_{label}"] = v
    return row


FEATURE_COLUMNS = (
    ["nameWithOwner", "early_status", "owner_type", "renamed", "capped", "precreation_commits"]
    + [f"{k}_{label}" for label in WINDOWS for k in (
        "commits", "active_weeks", "commits_last_30d", "days_since_last_commit",
        "contributors", "nonowner_contributors", "stars", "forks",
        "issues_opened", "prs_opened", "issues_prs_closed", "external_issue_authors")]
)


def run_features():
    if not RAW_PATH.exists():
        sys.exit(f"{RAW_PATH.relative_to(ROOT)} not found. Run the collect step first.")
    latest = {}
    with open(RAW_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                latest[rec["nameWithOwner"]] = rec      # a re-collected repo overrides older lines
    rows = [build_row(r) for r in latest.values()]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    statuses = {}
    for r in rows:
        key = r["early_status"].split(":")[0]
        statuses[key] = statuses.get(key, 0) + 1
    capped = sum(1 for r in rows if r["capped"])
    print(f"{len(rows)} repos -> {OUT_CSV.relative_to(ROOT)}  status counts: {statuses}"
          + (f"  ({capped} hit the {MAX_PAGES * 100}-item cap on some endpoint)" if capped else ""))


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, help="only the first N repos (trial run)")
    ap.add_argument("--features-only", action="store_true", help="skip the API, rebuild the CSV")
    args = ap.parse_args()
    if not args.features_only:
        run_collect(args.limit)
    run_features()


if __name__ == "__main__":
    main()
