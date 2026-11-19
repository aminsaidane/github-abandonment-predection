"""Collect each repo OWNER's history, as it stood when the repo was created.

Question behind it: does what an owner did with their earlier repositories help forecast whether a
new one is abandoned? To stay a genuine forecast, every feature is measured as of the repo's
creation date, using only facts that cannot change afterwards:

  owner_account_age_days      days between the owner's account creation and this repo's creation
  owner_prior_repos           the owner's other (non-fork) repos created before this one
  owner_prior_old_repos       of those, how many were already at least 365 days old at that time
  owner_prior_dormant_share   of those old repos, the share whose LAST push (ever) came more than
                              365 days before this repo was created. Such a repo had already been
                              silent for a year at that date, so this is observable at that date.
                              Empty when the owner had no old repo.

Left out on purpose because they describe the owner TODAY and would leak the future: follower
count, stars, archived flags, anything about pushes after this repo's creation.

Known limits: deleted repos are invisible (survivorship), and a repo transferred between accounts
carries its new owner. Owners with more than 1,000 public repos are cut at the 1,000 oldest and
flagged (`owner_capped`).

Two steps, both run by default:
  collect   GitHub API -> data/owner_history_raw.jsonl  (one line per owner, resumable)
  features  raw JSONL  -> data/owner_history.csv        (one row per repo)

Cost: about 2-3 API calls per owner, so 300 repos take ~15 minutes and the full set of ~4,900 repos
about 3 hours, well inside the 5,000 calls/hour limit. Same GITHUB_TOKEN setup as
collect_early_activity.py (token in the environment or in .env).

Usage, from the repo root:
  python scritpts/collect_owner_history.py --repos data/owner_pilot_sample.csv   # pilot
  python scritpts/collect_owner_history.py                                       # every repo
  python scritpts/collect_owner_history.py --features-only
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_early_activity import API, GitHub, ROOT, collect_list, iso, load_dotenv, parse_ts  # noqa: E402

DEFAULT_INPUT = ROOT / "data" / "pipeline3_output.csv"
DEFAULT_RAW = ROOT / "data" / "owner_history_raw.jsonl"
DEFAULT_OUT = ROOT / "data" / "owner_history.csv"
OLD_DAYS = 365
OUT_COLUMNS = ["nameWithOwner", "owner_status", "owner_account_age_days", "owner_prior_repos",
               "owner_prior_old_repos", "owner_prior_dormant_share", "owner_capped"]


# ====================================================================== collect
def collect_owner(gh, login):
    """One owner: account creation date and the (created, pushed, fork) of every public repo."""
    rec = {"owner": login, "collected_at": iso(datetime.now(timezone.utc))}
    status, user, _ = gh.get(f"{API}/users/{login}")
    if status != 200 or not isinstance(user, dict):
        rec["status"] = "not_found" if status in (404, 451) else f"http_{status}"
        return rec
    rec["account_created"] = user["created_at"]
    rec["account_type"] = user["type"]
    repos, st = collect_list(gh, f"/users/{login}/repos",
                             {"type": "owner", "sort": "created", "direction": "asc"})
    rec["repos"] = [{"name": r["name"], "created": r["created_at"],
                     "pushed": r.get("pushed_at"), "fork": bool(r["fork"])} for r in repos]
    rec["capped"] = st == "capped"
    rec["status"] = "ok" if st in (200, "capped") else f"partial:repos={st}"
    return rec


def load_targets(input_csv, repos_csv=None, limit=None):
    """(nameWithOwner, createdAt) of the repos to process, optionally restricted to a list."""
    with open(input_csv, newline="", encoding="utf-8") as f:
        rows = [(r["nameWithOwner"].strip(), r["createdAt"].strip()) for r in csv.DictReader(f)]
    if repos_csv:
        with open(repos_csv, newline="", encoding="utf-8") as f:
            keep = {r["nameWithOwner"].strip() for r in csv.DictReader(f)}
        rows = [r for r in rows if r[0] in keep]
    return rows[:limit] if limit else rows


def done_owners(raw_path):
    """Owners collected for good. Partial or failed ones are retried on the next run."""
    done = set()
    if raw_path.exists():
        with open(raw_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("status") in ("ok", "not_found"):
                        done.add(rec["owner"].lower())
    return done


def run_collect(targets, raw_path):
    load_dotenv()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("GITHUB_TOKEN is not set. See the instructions in collect_early_activity.py.")
    gh = GitHub(token)
    owners = list(dict.fromkeys(name.split("/")[0] for name, _ in targets))
    done = done_owners(raw_path)
    todo = [o for o in owners if o.lower() not in done]
    print(f"{len(owners)} owners, {len(owners) - len(todo)} already collected, {len(todo)} to go")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(raw_path, "a", encoding="utf-8") as out:
        for n, login in enumerate(todo, 1):
            try:
                rec = collect_owner(gh, login)
            except Exception as e:                       # keep going; record the failure
                rec = {"owner": login, "status": f"error:{e.__class__.__name__}"}
            out.write(json.dumps(rec) + "\n")
            out.flush()                                   # every owner is saved immediately
            if n % 25 == 0 or n == len(todo):
                left = (time.time() - start) / n * (len(todo) - n) / 60
                print(f"  {n}/{len(todo)}  ({gh.calls} API calls, ~{left:.0f} min left)")
    print(f"Raw data -> {raw_path}")


# ====================================================================== features
def owner_features(rec, repo_full_name, created_at):
    """Features of one repo, from its owner's record. Only facts observable at `created_at`."""
    t0 = parse_ts(created_at)
    own_name = repo_full_name.split("/")[1].lower()
    cutoff = t0 - timedelta(days=OLD_DAYS)
    prior = [r for r in rec["repos"]
             if not r["fork"] and r["name"].lower() != own_name and parse_ts(r["created"]) < t0]
    old = [r for r in prior if parse_ts(r["created"]) < cutoff]
    # last push ever before the cutoff => no push in the year before t0, known at t0
    dormant = [r for r in old if parse_ts(r["pushed"] or r["created"]) < cutoff]
    return {
        "owner_account_age_days": max(0, (t0 - parse_ts(rec["account_created"])).days),
        "owner_prior_repos": len(prior),
        "owner_prior_old_repos": len(old),
        "owner_prior_dormant_share": round(len(dormant) / len(old), 4) if old else "",
        "owner_capped": int(rec.get("capped", False)),
    }


def run_features(targets, raw_path, out_path):
    if not raw_path.exists():
        sys.exit(f"{raw_path} not found. Run the collect step first.")
    latest = {}
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                latest[rec["owner"].lower()] = rec        # a re-collected owner overrides older lines
    rows, counts = [], {}
    for name, created in targets:
        rec = latest.get(name.split("/")[0].lower())
        if rec is None:
            row = {"nameWithOwner": name, "owner_status": "missing"}
        elif rec.get("status") != "ok":
            row = {"nameWithOwner": name, "owner_status": rec.get("status", "missing")}
        else:
            row = {"nameWithOwner": name, "owner_status": "ok", **owner_features(rec, name, created)}
        counts[row["owner_status"].split(":")[0]] = counts.get(row["owner_status"].split(":")[0], 0) + 1
        rows.append(row)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} repos -> {out_path}  status counts: {counts}")


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repos", type=Path, help="CSV with a nameWithOwner column: only these repos (e.g. the pilot sample)")
    ap.add_argument("--limit", type=int, help="only the first N repos (trial run)")
    ap.add_argument("--features-only", action="store_true", help="skip the API, rebuild the CSV")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    targets = load_targets(args.input, args.repos, args.limit)
    if not args.features_only:
        run_collect(targets, args.raw)
    run_features(targets, args.raw, args.out)


if __name__ == "__main__":
    main()
