# PS4 — IDBI Early Warning System: Default Prediction Engine (Backend)

AI-powered probability-of-default (PD) scoring engine for IDBI Bank's retail/MSME loan
portfolio. Predicts default risk **12 months in advance** across Personal, Home, MSME and
Auto loans using a stacking ensemble (XGBoost + LightGBM + Logistic Regression), aligned to
RBI's incoming ECL framework (effective April 2027) and SMA/NPA classification norms.

## Why this exists

IDBI's current PD models achieve only 16-22% accuracy, work 3 months ahead at best, and rely
solely on structured data. This engine targets 90%+ AUC-ROC, a 12-month early-warning horizon,
and blends structured + behavioral (spending, payment discipline, digital activity) +
external (industry, geography, macro) signals into one common framework across loan types.

## Architecture

```
scripts/generate_data.py   -> synthetic 10,000-account loan book + 12-month time series
ml/feature_engineering.py  -> shared feature builder (used by BOTH training and live /predict)
ml/risk_logic.py           -> risk category / RAG / ECL stage / prescriptive actions / SHAP formatting
ml/train_model.py          -> trains XGBoost, LightGBM, Logistic Regression + stacking ensemble,
                               SHAP explanations, stratified + out-of-time validation, batch-scores
                               the full portfolio
app/main.py                -> FastAPI serving layer (9 endpoints, loads pre-trained models)
```

## Setup & Run

Requires Python 3.11 (XGBoost/LightGBM/SHAP wheels are not yet available for 3.14).

```bash
cd ps4-default-prediction/backend
py -3.11 -m venv venv
./venv/Scripts/pip install -r requirements.txt      # venv/bin/pip on macOS/Linux

# 1. Generate synthetic portfolio (10,000 accounts, 120,000 monthly snapshots)
./venv/Scripts/python scripts/generate_data.py

# 2. Train models, run SHAP + validation, batch-score the portfolio
./venv/Scripts/python ml/train_model.py

# 3. Serve the API
./venv/Scripts/python -m uvicorn app.main:app --reload --port 8000
```

Interactive API docs: http://127.0.0.1:8000/docs

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Live PD scoring for an arbitrary account (JSON body) |
| GET | `/portfolio` | Portfolio-level exposure, risk/SMA/ECL distribution, concentration |
| GET | `/portfolio/trends` | 12-month behavioral risk trend, overall + by loan type |
| GET | `/alerts` | Top-N highest-risk accounts (filterable by loan type, risk category, SMA, search) |
| GET | `/account/{account_id}` | Full deep-dive: SHAP drivers, payment history, risk trend |
| GET | `/model/performance` | AUC/PR-AUC/F1, confusion matrix, ROC/PR curves, feature importance, OOT validation |
| GET | `/stress-test` | Portfolio impact under Mild/Moderate/Severe macro stress scenarios |
| GET | `/ecl-summary` | Stage 1/2/3 exposure, provisioning estimate, stage transition matrix |

## Modeling notes

- **Class imbalance**: SMOTE oversampling on the training fold + `class_weight="balanced"`.
- **Validation**: stratified 80/20 holdout (primary metrics) + a genuine **out-of-time**
  holdout — trained on the oldest 75% of accounts by loan vintage (disbursement date),
  tested on the most recent 25% — the standard credit-risk OOT technique.
- **Explainability**: SHAP `TreeExplainer` on the XGBoost base learner is used as the
  interpretable surrogate for per-account risk drivers, since explaining the full stacking
  ensemble directly isn't straightforward. This also satisfies RBI's FREE-AI framework
  expectation of model explainability for credit-risk AI.
- **NPA accounts** (already 90+ DPD) are excluded from model *training* (predicting a
  default that has already happened is circular) but are still scored deterministically
  (PD=0.99, Stage 3) and included in all portfolio/ECL views.
- **ECL provisioning** uses loan-type-specific LGD assumptions (Home 25%, Auto 35%, MSME 50%,
  Personal 60%) — Stage 1 uses 12-month PD, Stage 2 uses an approximated lifetime PD
  (PD × 2.2, capped at 1.0), Stage 3 recognizes full LGD × exposure.
- Synthetic data is generated so that **~28% of defaults show minimal prior behavioral
  warning** and **~5% of healthy accounts show a temporary, self-resolving stress blip** —
  this keeps model performance realistic (recall/precision in the high-80s/90s, not a
  trivial 100%) rather than leaking the label directly into the features.
