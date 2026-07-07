# PS4 — IDBI Early Warning System (Frontend)

React + Vite dashboard for the Default Prediction Engine. 5 pages: Portfolio Overview,
Early Warning Alerts, Account Deep Dive, Model Performance, ECL Regulatory View.

## Run

```bash
cd ps4-default-prediction/frontend
npm install
npm run dev        # http://localhost:5173
```

Requires the backend running at `http://localhost:8000` (see `../backend/README.md`).
Override the API URL via `.env` → `VITE_API_URL`.

## Stack

React 19 + React Router 7, Recharts for all charts, plain CSS (dark banking theme, IDBI
blue accent) — no UI framework dependency. `RiskGauge` (SVG semicircle) and `AnimatedNumber`
(count-up) are hand-built for the animated-metric requirement.

## Verified

All 5 routes load with zero console/page errors (checked headlessly via Playwright) and
render real data from the live backend — portfolio stats, risk distribution, SMA/ECL
breakdowns, alert table with filters, account deep-dive with SHAP drivers and payment
heatmap, ROC/PR curves + confusion matrix, and the ECL stage-transition matrix.
