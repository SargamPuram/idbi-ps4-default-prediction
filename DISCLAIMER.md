# Disclaimer

This repository was built as a **hackathon proof-of-concept** (IDBI Innovate
2026, Problem Statement 4 — Default Prediction Model) to demonstrate a
12-month early-warning approach to probability-of-default (PD) scoring. It is
a prototype, not a production banking system, and has not been through any
bank's model validation, audit, or regulatory sign-off process.

## 1. All data in this repository is synthetic

Every account, borrower and transaction record in this project — the
10,000-row loan book (`backend/data/accounts.csv`) and the 120,000-row
12-month behavioral time series (`backend/data/monthly_snapshots.csv`) — is
**synthetically generated** by `backend/scripts/generate_data.py`. No real
IDBI customer, account, credit bureau, or transaction data was used at any
point in building, training, or demonstrating this system.

Specifically, the generator (seeded, deterministic, `RNG_SEED = 42`):

- Fabricates borrower profiles (name, age, gender, city, employment type,
  industry sector, income, CIBIL-style credit score 300–900) using the
  `Faker` library with an `en_IN` locale — names and demographics are
  invented, not sampled from any real population.
- Fabricates loan terms (type, amount, tenure, interest rate, disbursement
  date) across four loan types (Personal, Home, MSME, Auto) from hand-picked
  ranges, not IDBI's actual product book.
- Fabricates behavioral and external signals (payment discipline score,
  spending volatility, income stability, digital activity score, industry
  and geography risk indices, macro-stress indicator, credit utilization)
  from parametric distributions, not real bureau or bank-statement data.
- Derives a **synthetic ground-truth default label** from a hand-authored
  logistic risk equation (`gen_behavioral_and_target` in
  `generate_data.py`), calibrated to an ~8% overall default rate, then
  generates a 12-month day-past-due / EMI-bounce / utilization time series
  per account consistent with that label — including a deliberate ~28% of
  defaulting accounts with muted prior warning signs and a ~5% "false
  alarm" slice of healthy accounts with a temporary, self-resolving stress
  blip, so the label isn't trivially leaked into the features.

Because the label is generated from the same equation family the model is
trained to recover, the reported accuracy reflects how well the model
learns a designed synthetic relationship — not real-world credit risk. It
demonstrates the *modeling approach* (feature engineering, ensembling,
explainability, out-of-time validation), not validated real-world predictive
power.

## 2. Model outputs are estimates for demonstration, not regulatory ECL

The 12-month probability-of-default, ECL Stage 1/2/3 classification, and
provisioning estimates produced by this system (see `backend/ml/risk_logic.py`
and the `/ecl-summary` endpoint) are **illustrative model outputs**, not
regulatory Expected Credit Loss / Ind AS 109 (IFRS 9) calculations. In
particular:

- Loss-Given-Default (LGD) assumptions used for provisioning (Home Loan 25%,
  Auto Loan 35%, MSME Loan 50%, Personal Loan 60%) are simplified,
  loan-type-level constants chosen for this demo, not IDBI's actual
  collateral-adjusted LGD models.
- The Stage 2 "lifetime PD" figure is an approximation (12-month PD × 2.2,
  capped at 1.0), not a modeled lifetime PD curve.
- SMA/NPA classification thresholds follow standard RBI day-past-due
  conventions but are applied here to synthetic DPD data only.
- This system does not implement macro-economic scenario weighting,
  auditor-reviewed staging overlays, or any of the governance controls a
  bank's actual ECL/IFRS 9 provisioning process requires.

**This tool is not a substitute for a bank's actual credit-risk model
validation, provisioning, or regulatory reporting process**, and its outputs
must not be used as the basis for real provisioning, capital adequacy, or
disclosure decisions.

## 3. Not investment, credit, or lending advice

Nothing in this repository, its outputs, or its documentation constitutes
investment advice, credit advice, a lending recommendation, or a guarantee
of any borrower's or portfolio's actual future performance. The
"recommended actions" surfaced per account (e.g., "schedule RM outreach",
"escalate to collections") are illustrative prescriptive suggestions tied to
a demo risk score, not vetted collections or underwriting policy.

## 4. Reported model performance is measured on synthetic, out-of-time data

The performance figures reported in `backend/reports/model_performance.json`
and surfaced via the `/model/performance` endpoint were measured entirely on
the synthetic dataset described above, using two holdouts:

- A stratified 80/20 random holdout, on which the primary **Stacking
  Ensemble** (XGBoost + LightGBM + Logistic Regression) achieved
  **AUC-ROC = 0.9945**, PR-AUC = 0.9567, F1 = 0.8882.
- A genuine **out-of-time** holdout (trained on the oldest 75% of accounts
  by loan-vintage/disbursement date, tested on the most recent 25% — the
  standard credit-risk OOT technique), on which the same ensemble
  architecture achieved **AUC-ROC = 0.9963**, PR-AUC = 0.97, F1 = 0.9005.

These numbers are accurately transcribed from the repository's own training
run output and are not rounded up or embellished. They demonstrate strong
recovery of a designed synthetic signal under a credit-risk-appropriate
validation methodology (stratified + out-of-time) — they are **not** a claim
of real-world predictive performance on IDBI's actual loan book, which has
not been tested. Real-world deployment would require retraining and
revalidating against real, permissioned bank data, followed by independent
model risk validation.

## 5. Scope limitation: structured data only

This prototype uses **structured data only** (loan bureau-style fields and a
synthetic behavioral time series). It does not yet incorporate unstructured
or public-domain signals (e.g., MCA filings, news sentiment, GST filings) —
this is an acknowledged gap and listed as future work, not a claimed
capability. See the README's "Known Limitations" and "Roadmap" sections for
detail.

## 6. General

This software is provided "as is", without warranty of any kind, for
hackathon evaluation and demonstration purposes only. It has not undergone
security review, penetration testing, or regulatory compliance review (RBI
FREE-AI, DPDP Act, or otherwise) beyond what is described in this repository.
Do not connect this prototype to real customer data or production banking
systems without a full security, privacy, and model-risk review.
