# Dashboard

Paste a GitHub link, get the model's abandonment forecast and an exact explanation of the score.

## Run

```
pip install -r requirements.txt
export GITHUB_TOKEN=ghp_...        # or put it in .env; needed only for the link lookup
uvicorn dashboard.server:app --port 8000
```

Open http://localhost:8000. The React app is already built in `dashboard/frontend/dist`, so Node is not needed to run it.
Without a token the 11 example repositories (held-out test split, with known outcomes) still work.

## How it works

- `POST /api/predict {url}`: link → GitHub API → the same collector used for training (`collect_repo`, `build_row`) → the 14 model inputs → `src.predict.predict_risk` on `models/final_model.pkl`.
  Refused: repositories younger than 182 days. Warned: created after 2023, under 5 stars, capped counts.
- Explanation: for a logistic regression the log-odds are a sum, so each feature's contribution is `coef × (value − training mean)` (after the same log and scaling steps), converted to percentage points against the average repository. It adds up exactly to the model's own score.
- `GET /api/performance`: final test metrics and figures from `reports/`.
- `dashboard/make_examples.py` rebuilds `reference.json` (training means, medians) and `examples.json`.
- `python -m dashboard.verify_live`: checks that live features match the CSV for a few repositories.

## Rebuild the frontend

```
cd dashboard/frontend && npm install && npm run build      # dev server: npm run dev (API on :8000)
```

## Pages

- **Forecast**: link in, probability and "why this score" out.
- **What the model learned**: step 5 results across the whole training set (odds ratios with intervals, observed rates by band, language effects, SHAP). Reads `reports/tables/interpretation_*.csv`; regenerate with `python -m src.interpret`.
- **Model performance**: final test metrics and figures.
