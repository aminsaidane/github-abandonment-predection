"""
Filter, label, and stratified-sample the GitHub Public Repository Metadata
dataset down to 5,000 rows, preserving the natural distribution of the
abandoned/active label and primary language.

IMPORTANT: this dataset is a static snapshot (collected once, not updated
live). We use the MOST RECENT `pushedAt` timestamp found anywhere in the
dataset as our reference "now" for computing staleness -- NOT the real
current date. Using the real current date would make almost every repo
look "abandoned" simply because years have passed since the snapshot was
taken, regardless of how active the repo actually was at that time.

Usage:
    python sample_dataset.py --input data/repos.json --output data/repos_sample_5000.json
"""
import argparse
import json

import pandas as pd

MIN_AGE_MONTHS = 24          # only repos old enough (at snapshot time) to label reliably
ABANDON_THRESHOLD_DAYS = 365  # no push in this many days (relative to snapshot) -> abandoned
SAMPLE_SIZE = 5000
RANDOM_STATE = 42


def load_records(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.json_normalize(data)


def add_age_and_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["createdAt"] = pd.to_datetime(df["createdAt"], utc=True, errors="coerce")
    df["pushedAt"] = pd.to_datetime(df["pushedAt"], utc=True, errors="coerce")

    # Reference point = the dataset's own snapshot time, approximated as the
    # single most recent pushedAt value found in the whole dataset.
    reference_date = df["pushedAt"].max()
    print(f"Using {reference_date.date()} as the dataset's reference 'now' "
          f"(most recent pushedAt found in the data)")

    df["repo_age_days"] = (reference_date - df["createdAt"]).dt.days
    df["days_since_last_push"] = (reference_date - df["pushedAt"]).dt.days
    df["abandoned"] = (df["days_since_last_push"] > ABANDON_THRESHOLD_DAYS).astype(int)

    return df


def filter_eligible(df: pd.DataFrame) -> pd.DataFrame:
    min_age_days = MIN_AGE_MONTHS * 30
    eligible = df[
        df["repo_age_days"].notna()
        & (df["repo_age_days"] >= min_age_days)
        & df["days_since_last_push"].notna()
    ].copy()
    return eligible


def stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Sample n rows, preserving proportions of (abandoned x primaryLanguage)."""
    df = df.copy()
    top_languages = df["primaryLanguage"].value_counts().nlargest(15).index
    df["lang_group"] = df["primaryLanguage"].where(
        df["primaryLanguage"].isin(top_languages), "Other"
    )

    strata_col = ["abandoned", "lang_group"]
    group_sizes = df.groupby(strata_col, dropna=False).size()
    proportions = group_sizes / group_sizes.sum()
    target_counts = (proportions * n).round().astype(int)

    sampled_parts = []
    for key, target_n in target_counts.items():
        if target_n <= 0:
            continue
        subset = df[
            (df["abandoned"] == key[0]) & (df["lang_group"] == key[1])
        ]
        take_n = min(target_n, len(subset))
        sampled_parts.append(subset.sample(n=take_n, random_state=RANDOM_STATE))

    sample = pd.concat(sampled_parts, ignore_index=True)

    if len(sample) < n:
        remaining = df.drop(sample.index, errors="ignore")
        top_up = remaining.sample(
            n=min(n - len(sample), len(remaining)), random_state=RANDOM_STATE
        )
        sample = pd.concat([sample, top_up], ignore_index=True)

    return sample.drop(columns=["lang_group"]).sample(
        frac=1, random_state=RANDOM_STATE
    ).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw dataset JSON")
    parser.add_argument("--output", required=True, help="Path to write sampled JSON")
    parser.add_argument("--n", type=int, default=SAMPLE_SIZE)
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    df = load_records(args.input)
    print(f"Loaded {len(df)} raw records")

    df = add_age_and_label(df)

    # Sanity check: report the abandon rate on the FULL dataset too, so you
    # can see whether the skew is inherent to the data or specific to the
    # 24-month-eligible subset.
    print(f"Abandon rate across ALL records (any age): {df['abandoned'].mean():.1%}")

    eligible = filter_eligible(df)
    print(
        f"{len(eligible)} records are 24+ months old (at snapshot time) and "
        f"eligible for labeling (abandon rate: {eligible['abandoned'].mean():.1%})"
    )

    if len(eligible) < args.n:
        raise ValueError(
            f"Only {len(eligible)} eligible records found, fewer than "
            f"requested sample size {args.n}. Lower --n or relax MIN_AGE_MONTHS."
        )

    sample = stratified_sample(eligible, args.n)
    print(
        f"Sampled {len(sample)} records "
        f"(abandon rate in sample: {sample['abandoned'].mean():.1%})"
    )

    sample.to_json(args.output, orient="records", date_format="iso")
    print(f"Wrote sample -> {args.output}")


if __name__ == "__main__":
    main()