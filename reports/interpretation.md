# What the final model learned (step 5)

Computed on the **training split only** (3,936 repositories, 83.9% abandoned) with the saved final model
(logistic regression, 6-month window, C = 0.029). The test split is not used here. Reproduce with
`python -m src.interpret`; tables are in `reports/tables/interpretation_*.csv`, figures in
`reports/figures/interp_coefficients.png` and `interp_shap.png`.

## How to read the numbers

Every input is standardised (counts go through log1p first). An **odds ratio** says how the odds of
"abandoned" change for a repository that is 1 standard deviation higher on that feature, with the other
features held fixed. Above 1 raises the risk, below 1 lowers it. The interval is a 95% bootstrap interval
(500 refits of the whole pipeline). The model is strongly regularised, so effects are shrunk on purpose.
**SHAP** values are exact for a linear model (`coef × (value − mean)`); we computed them with the `shap`
package and checked that they match this formula. The dashboard's "why this score" chart uses the same numbers.

| Feature | Odds ratio per +1 SD (95% interval) | Interval excludes 1 | Mean abs. SHAP |
|---|---|---|---|
| Days since the last commit (at month 6) | 1.60 (1.44–1.79) | yes | 0.433 |
| Owned by an organization | 0.78 (0.71–0.87) | yes | 0.211 |
| Active weeks (out of 26) | 0.78 (0.69–0.87) | yes | 0.185 |
| Forks * | 1.20 (1.08–1.33) | yes | 0.138 |
| People other than the owner who opened issues or PRs | 0.84 (0.73–0.96) | yes | 0.136 |
| Contributors * | 1.15 (1.03–1.30) | yes | 0.091 |
| Pull requests opened | 0.88 (0.79–0.99) | yes | 0.089 |
| Commits in the first 6 months * | 1.06 (0.94–1.21) | no | 0.048 |
| Commits before creation (imported history) | 0.93 (0.86–1.00) | yes | 0.047 |
| Commits in the last 30 days of the window | 0.94 (0.84–1.04) | no | 0.046 |
| Issues and PRs closed | 0.95 (0.82–1.09) | no | 0.036 |
| Issues opened | 0.97 (0.86–1.07) | no | 0.024 |
| Contributors other than the owner * | 1.01 (0.90–1.16) | no | 0.008 |

\* On its own this feature goes with *less* abandonment. The positive sign only appears once the overlapping
features are held fixed (see finding 4).
Language is one feature in the model (mean abs. SHAP 0.075, see finding 5).

## Findings

1. **The strongest signal is silence at the end of month 6.** One standard deviation more days since the
   last commit (about 69 days) multiplies the odds of abandonment by 1.61 (1.44–1.79). On its own: of
   repositories whose last commit was within 14 days of the end of the window, 62% were abandoned; of those
   with no commit in the last 160 days, 94% (1,523 repositories, 39% of the training split).
2. **Sustained activity matters more than volume.** One standard deviation more active weeks (about 5.5
   weeks out of 26) multiplies the odds by 0.78 (0.69–0.87). Repositories with 0–1 active weeks were 93%
   abandoned; with more than 12 active weeks, 57%.
3. **Organisation ownership and outside engagement help.** Organisation-owned repositories: odds ratio 0.78
   (0.71–0.87), 70% abandoned versus 88% for personal repositories. Other people opening issues or pull requests:
   0.84 (0.73–0.96); pull requests opened: 0.88 (0.79–0.99). Observed: 75% abandoned when someone other than
   the owner opened an issue or PR, 90% when nobody did.
4. **Raw volume adds little once the pattern is known.** Commits, issues opened and closed, and commits in
   the last 30 days have intervals that include 1. Forks (1.20, 1.08–1.33) and contributors (1.15, 1.03–1.30)
   come out *positive*, although each goes with less abandonment on its own. These features overlap strongly
   (Spearman correlation on the log scale: commits vs active weeks 0.82, contributors vs non-owner
   contributors 0.83), so a coefficient measures what a feature adds beyond the others. **Do not read these two
   as "forks cause abandonment".** A possible explanation (not tested here) is that forks without matching
   issues or pull requests reflect attention that does not turn into collaboration.
5. **Language has a small effect.** Its overall weight is far below the top features (0.075 versus 0.433 for
   days since last commit). Intervals exclude zero for javascript, ruby, swift (higher risk) and rust, typescript (lower risk);
   the other languages are not distinguishable from the average. With 16 languages, a few intervals can miss
   zero by chance, and swift and rust have only 62 repositories each, so treat this as weak evidence.
6. **Imported history barely matters.** Commits before the creation date: 0.93 (0.86–1.00), borderline.

## What this means for the model

- The model mostly asks: *did the project keep going after the first weeks, and did anyone else join in?*
  This fits the test result: the single best rule (days since last commit) already reaches a ROC AUC of 0.74 on
  the test set, and the full model reaches 0.78. Most of the signal is in finding 1; the other features add a
  modest amount.
- The relationships are **associations in past repositories, not causes**. Pushing a commit on day 180 does not
  by itself make a repository survive.
- Intervals describe sampling uncertainty in the training split. They do not cover drift between creation
  periods (older repositories have had longer to go quiet), which is discussed with the test results.
