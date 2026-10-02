# Getting and Sampling the Dataset

This project uses the **GitHub Public Repository Metadata** dataset from Kaggle
(all public repos with 5+ stars): https://www.kaggle.com/datasets/pelmers/github-repository-metadata-with-5-stars

> **Note:** this dataset is a static snapshot (collected at a fixed point in
> time, not updated live). Our sampling script accounts for this when
> labeling abandonment — see the explanation at the bottom of this doc.

## 1. Set up the Kaggle API

1. Install the CLI: `pip install kaggle`
2. Go to kaggle.com → your profile → Settings → API → "Create New Token".
   This downloads `kaggle.json`.
3. Move it to `~/.kaggle/kaggle.json` (Mac/Linux) or
   `C:\Users\<YourName>\.kaggle\kaggle.json` (Windows).
4. On Mac/Linux, lock down its permissions: `chmod 600 ~/.kaggle/kaggle.json`

## 2. Download the dataset

```bash
kaggle datasets download -d pelmers/github-repository-metadata-with-5-stars -p ./data --unzip
```

This downloads the full dataset (a large JSON file — all public repos with
5+ stars) into the `data/` folder.

## 3. Verify the download

```bash
python -c "import json; json.load(open('data/<filename>.json'))"
```

If this runs without error, the file is valid JSON.

**Important:** never pretty-print or reformat this file before using it —
reformatting (indenting) a JSON array this size turns it from one compact
line into tens of thousands of lines, which can make tools like Apache Hop
try to render it line-by-line and run out of memory. Keep it compact.

## 4. Sample the dataset down to 5,000 rows

Run the sampling script:

```bash
python scripts/sample_dataset.py --input data/<your-downloaded-file>.json --output data/repos_sample_5000.json
```

### What this script does

1. **Labels abandonment.** For each repo, computes:
   - `repo_age_days` — how old the repo is
   - `days_since_last_push` — how long since its last commit
   - `abandoned` — `1` if `days_since_last_push > 365`, else `0`

   Since the dataset is a static snapshot, these are computed **relative to
   the most recent `pushedAt` timestamp found anywhere in the dataset** —
   not today's real date. Using today's real date would make almost every
   repo look abandoned simply because years have passed since the snapshot
   was taken, regardless of how active the repo actually was.

2. **Filters to eligible repos.** Only keeps repos that were already 24+
   months old at snapshot time — recent repos can't be labeled reliably
   (not enough time has passed to know if they'll be abandoned).

3. **Stratified samples 5,000 rows.** Instead of taking the first 5,000 rows
   (which would introduce ordering bias — the dataset isn't randomly
   sorted), it samples while preserving the true proportions of
   `abandoned` status × primary language, so the sample reflects the real
   population instead of some arbitrary slice.

4. **Writes the result** to `data/repos_sample_5000.json`, shuffled, ready
   to load into Apache Hop or pandas.

### Reading the console output

```
Using 2023-01-28 as the dataset's reference 'now' ...
Abandon rate across ALL records (any age): X%
N records are 24+ months old ... (abandon rate: Y%)
Sampled 5000 records (abandon rate in sample: Y%)
```

Check that the final sampled abandon rate (Y%) is a reasonable mix — not
0% or 100%. If it's heavily skewed toward one class, that's worth
investigating before moving on to the ETL/modeling steps.

## 5. Adjusting the sample

Edit these constants at the top of `scripts/sample_dataset.py` if needed:

| Constant | Default | Meaning |
|---|---|---|
| `MIN_AGE_MONTHS` | 24 | Minimum repo age (at snapshot time) to be eligible for labeling |
| `ABANDON_THRESHOLD_DAYS` | 365 | No push in this many days → labeled abandoned |
| `SAMPLE_SIZE` | 5000 | Rows in the final sample |