"""
IDBI Early Warning System — Default Prediction Engine (PS4)

FastAPI serving layer over a pre-trained stacking ensemble (XGBoost + LightGBM +
Logistic Regression). Loads pre-scored portfolio data + the trained models on
startup; /predict runs live inference for an arbitrary account.

Run (from backend/):  uvicorn app.main:app --reload
"""

import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.feature_engineering import build_feature_matrix, engineer_snapshot_features, FEATURE_COLUMNS
from ml.risk_logic import ecl_stage, estimated_months_to_default, rag_status, recommended_action, risk_category, top_risk_drivers
from app.schemas import AccountFeaturesIn, PredictionOut

DATA_DIR = "data"
MODEL_DIR = "models"
REPORT_DIR = "reports"

LGD_BY_LOAN_TYPE = {  # loss-given-default assumptions used for ECL provisioning estimate
    "Home Loan": 0.25, "Auto Loan": 0.35, "MSME Loan": 0.50, "Personal Loan": 0.60,
}
TS_OVERRIDE_FIELDS = [
    "dpd_mean", "dpd_max", "dpd_last3_mean", "dpd_trend", "dpd_acceleration",
    "bounce_total", "bounce_last3", "utilization_trend",
]

STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading portfolio data and trained models...")
    accounts = pd.read_csv(f"{DATA_DIR}/scored_accounts.csv")
    accounts["disbursement_date"] = pd.to_datetime(accounts["disbursement_date"])
    snapshots = pd.read_csv(f"{DATA_DIR}/monthly_snapshots.csv")

    STATE["accounts"] = accounts
    STATE["snapshots"] = snapshots
    STATE["snap_features"] = engineer_snapshot_features(snapshots)
    STATE["stack_model"] = joblib.load(f"{MODEL_DIR}/stacking_ensemble.joblib")
    STATE["xgb_model"] = joblib.load(f"{MODEL_DIR}/xgb_model.joblib")
    STATE["explainer"] = shap.TreeExplainer(STATE["xgb_model"])
    with open(f"{MODEL_DIR}/feature_columns.json") as f:
        STATE["feature_columns"] = json.load(f)
    with open(f"{REPORT_DIR}/model_performance.json") as f:
        STATE["performance_report"] = json.load(f)
    STATE["transition_matrix"] = compute_transition_matrix(snapshots)
    print(f"Loaded {len(accounts)} accounts. Model ready.")
    yield
    STATE.clear()


app = FastAPI(
    title="IDBI Early Warning System — Default Prediction Engine",
    description="AI-powered probability-of-default scoring, ECL staging and portfolio early-warning for IDBI Bank (PS4 prototype).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def compute_transition_matrix(snapshots: pd.DataFrame) -> dict:
    snap = snapshots.sort_values(["account_id", "snapshot_month"]).copy()
    bins = [-1, 0, 30, 60, 90, 1000]
    labels = ["Regular", "SMA-0", "SMA-1", "SMA-2", "NPA"]
    snap["stage_bucket"] = pd.cut(snap["days_past_due"], bins=bins, labels=labels)
    snap["next_stage"] = snap.groupby("account_id")["stage_bucket"].shift(-1)
    trans = snap.dropna(subset=["next_stage"])
    matrix = pd.crosstab(trans["stage_bucket"], trans["next_stage"], normalize="index").round(4)
    matrix = matrix.reindex(index=labels, columns=labels, fill_value=0.0)
    return {row: matrix.loc[row].to_dict() for row in labels}


def crores(x) -> float:
    return round(float(x) / 1e7, 2)


def get_account_or_404(account_id: str) -> pd.Series:
    df = STATE["accounts"]
    match = df[df["account_id"] == account_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Account '{account_id}' not found")
    return match.iloc[0]


def parse_drivers(raw) -> list:
    try:
        return json.loads(raw) if isinstance(raw, str) else []
    except (json.JSONDecodeError, TypeError):
        return []


def monthly_behavioral_trend(snapshots: pd.DataFrame) -> list:
    g = snapshots.groupby("snapshot_month").agg(
        avg_dpd=("days_past_due", "mean"),
        pct_delinquent=("days_past_due", lambda s: float((s > 0).mean() * 100)),
        avg_utilization=("utilization_ratio", "mean"),
        bounce_rate=("emi_bounce_flag", "mean"),
    ).reset_index().sort_values("snapshot_month")
    g["portfolio_risk_score"] = (
        (g["avg_dpd"] / 90 * 40) + (g["bounce_rate"] * 100 * 0.3) + (g["avg_utilization"] * 30)
    ).clip(0, 100).round(2)
    for col in ["avg_dpd", "pct_delinquent", "avg_utilization", "bounce_rate"]:
        g[col] = g[col].round(4)
    return g.rename(columns={"snapshot_month": "month"}).to_dict(orient="records")


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "IDBI Early Warning System — Default Prediction Engine",
        "accounts_loaded": len(STATE["accounts"]) if "accounts" in STATE else 0,
        "model_loaded": "stack_model" in STATE,
    }


@app.post("/predict", response_model=PredictionOut)
def predict(payload: AccountFeaturesIn):
    d = payload.model_dump(exclude=set(TS_OVERRIDE_FIELDS))
    d["account_id"] = "TEMP-0001"
    row = pd.DataFrame([d])
    X = build_feature_matrix(row)

    overrides = payload.model_dump()
    for col in TS_OVERRIDE_FIELDS:
        if overrides.get(col) is not None:
            X.at[0, col] = overrides[col]

    pd_score = float(STATE["stack_model"].predict_proba(X)[:, 1][0])
    cat = risk_category(pd_score)
    ecl = ecl_stage(payload.current_sma_status, pd_score)
    shap_vals = STATE["explainer"].shap_values(X)[0]
    drivers = top_risk_drivers(shap_vals, X.iloc[0].to_dict(), STATE["feature_columns"], top_n=5)

    return PredictionOut(
        probability_of_default=round(pd_score, 4),
        risk_category=cat,
        rag_status=rag_status(pd_score),
        sma_classification=payload.current_sma_status,
        ecl_stage=ecl,
        estimated_months_to_default=estimated_months_to_default(pd_score, cat),
        recommended_action=recommended_action(cat, ecl, payload.loan_type),
        top_risk_drivers=drivers,
    )


@app.get("/portfolio")
def portfolio_overview():
    df = STATE["accounts"]
    ens_metrics = next(m for m in STATE["performance_report"]["model_comparison"] if m["model"] == "Stacking Ensemble")

    risk_dist = df.groupby("risk_category").agg(accounts=("account_id", "count"), exposure=("loan_amount", "sum")).reset_index()
    rag_dist = df.groupby("rag_status").agg(accounts=("account_id", "count"), exposure=("loan_amount", "sum")).reset_index()
    sma_dist = df.groupby("sma_classification").agg(accounts=("account_id", "count"), exposure=("loan_amount", "sum")).reset_index()
    ecl_dist = df.groupby("ecl_stage").agg(accounts=("account_id", "count"), exposure=("loan_amount", "sum")).reset_index()
    concentration = df.groupby(["loan_type", "city_tier"]).agg(
        accounts=("account_id", "count"), exposure=("loan_amount", "sum"), avg_pd=("probability_of_default", "mean"),
    ).reset_index()

    def fmt(rows):
        return [{**r, "exposure_cr": crores(r["exposure"])} for r in rows]

    return {
        "total_portfolio_exposure_cr": crores(df["loan_amount"].sum()),
        "total_accounts": int(len(df)),
        "predicted_default_rate_pct": round(float(df["probability_of_default"].mean()) * 100, 2),
        "accounts_at_risk": int((df["risk_category"].isin(["High", "Critical"])).sum()),
        "model_auc_roc": ens_metrics["auc_roc"],
        "risk_distribution": fmt(risk_dist.to_dict(orient="records")),
        "rag_distribution": fmt(rag_dist.to_dict(orient="records")),
        "sma_breakdown": fmt(sma_dist.to_dict(orient="records")),
        "ecl_staging": fmt(ecl_dist.to_dict(orient="records")),
        "risk_concentration": [
            {**r, "avg_pd": round(r["avg_pd"], 4), "exposure_cr": crores(r["exposure"])}
            for r in concentration.to_dict(orient="records")
        ],
        "monthly_risk_trend": monthly_behavioral_trend(STATE["snapshots"]),
    }


@app.get("/portfolio/trends")
def portfolio_trends():
    overall = monthly_behavioral_trend(STATE["snapshots"])

    snap = STATE["snapshots"].merge(STATE["accounts"][["account_id", "loan_type"]], on="account_id", how="left")
    by_loan_type = snap.groupby(["snapshot_month", "loan_type"]).agg(
        avg_dpd=("days_past_due", "mean"), avg_utilization=("utilization_ratio", "mean"),
        bounce_rate=("emi_bounce_flag", "mean"),
    ).reset_index()
    by_loan_type["portfolio_risk_score"] = (
        (by_loan_type["avg_dpd"] / 90 * 40) + (by_loan_type["bounce_rate"] * 100 * 0.3) + (by_loan_type["avg_utilization"] * 30)
    ).clip(0, 100).round(2)

    return {
        "overall_trend": overall,
        "trend_by_loan_type": by_loan_type.rename(columns={"snapshot_month": "month"}).round(4).to_dict(orient="records"),
    }


@app.get("/alerts")
def early_warning_alerts(
    loan_type: Optional[str] = None,
    risk_category: Optional[str] = Query(None, alias="risk_category"),
    sma_status: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
):
    df = STATE["accounts"]
    if loan_type:
        df = df[df["loan_type"] == loan_type]
    if risk_category:
        df = df[df["risk_category"] == risk_category]
    if sma_status:
        df = df[df["sma_classification"] == sma_status]
    if search:
        s = search.lower()
        df = df[df["account_id"].str.lower().str.contains(s) | df["borrower_name"].str.lower().str.contains(s)]

    df = df.sort_values("probability_of_default", ascending=False).head(limit)

    alerts = []
    for _, row in df.iterrows():
        drivers = parse_drivers(row["top_risk_drivers"])
        alerts.append({
            "account_id": row["account_id"],
            "borrower_name": row["borrower_name"],
            "loan_type": row["loan_type"],
            "outstanding_amount": float(row["loan_amount"]),
            "risk_score": round(float(row["probability_of_default"]) * 100, 1),
            "risk_category": row["risk_category"],
            "rag_status": row["rag_status"],
            "sma_classification": row["sma_classification"],
            "ecl_stage": row["ecl_stage"],
            "estimated_months_to_default": row["estimated_months_to_default"] if pd.notna(row["estimated_months_to_default"]) else None,
            "top_risk_factor": drivers[0]["explanation"] if drivers else "N/A",
        })
    return {"count": len(alerts), "alerts": alerts}


@app.get("/account/{account_id}")
def account_deep_dive(account_id: str):
    row = get_account_or_404(account_id)
    history = STATE["snapshots"][STATE["snapshots"]["account_id"] == account_id].sort_values("snapshot_month")

    payment_history = history[["snapshot_month", "days_past_due", "emi_bounce_flag"]].rename(
        columns={"snapshot_month": "month"}
    ).to_dict(orient="records")

    risk_trend = []
    for _, h in history.iterrows():
        indicator = min(100, (h["days_past_due"] / 90 * 40) + (h["utilization_ratio"] * 30) + (h["emi_bounce_flag"] * 30))
        risk_trend.append({"month": h["snapshot_month"], "risk_indicator": round(float(indicator), 2)})

    return {
        "account_id": row["account_id"],
        "borrower_name": row["borrower_name"],
        "age": int(row["age"]),
        "gender": row["gender"],
        "city": row["city"],
        "employment_type": row["employment_type"],
        "industry_sector": row["industry_sector"],
        "annual_income": float(row["annual_income"]),
        "credit_score": int(row["credit_score"]),
        "loan_type": row["loan_type"],
        "loan_amount": float(row["loan_amount"]),
        "interest_rate": float(row["interest_rate"]),
        "tenure_months": int(row["tenure_months"]),
        "disbursement_date": str(row["disbursement_date"]),
        "probability_of_default": float(row["probability_of_default"]),
        "risk_score": round(float(row["probability_of_default"]) * 100, 1),
        "risk_category": row["risk_category"],
        "rag_status": row["rag_status"],
        "sma_classification": row["sma_classification"],
        "ecl_stage": row["ecl_stage"],
        "estimated_months_to_default": row["estimated_months_to_default"] if pd.notna(row["estimated_months_to_default"]) else None,
        "recommended_action": row["recommended_action"],
        "top_risk_drivers": parse_drivers(row["top_risk_drivers"]),
        "payment_history_12m": payment_history,
        "utilization_balance_trend_12m": history[["snapshot_month", "utilization_ratio", "monthly_balance_avg"]].rename(
            columns={"snapshot_month": "month"}
        ).to_dict(orient="records"),
        "historical_risk_trend": risk_trend,
    }


@app.get("/model/performance")
def model_performance():
    return STATE["performance_report"]


@app.get("/stress-test")
def stress_test():
    accounts = STATE["accounts"].copy()
    snap_feat = STATE["snap_features"]
    baseline_default_rate = float(accounts["probability_of_default"].mean())
    baseline_exposure_at_risk = float(accounts.loc[accounts["risk_category"].isin(["High", "Critical"]), "loan_amount"].sum())

    scenarios = {
        "Mild": {"macro_shock": 1.5, "utilization_shock": 0.05, "income_stability_shock": -0.03},
        "Moderate": {"macro_shock": 3.5, "utilization_shock": 0.12, "income_stability_shock": -0.08},
        "Severe": {"macro_shock": 6.0, "utilization_shock": 0.22, "income_stability_shock": -0.15},
    }

    results = {"baseline": {
        "default_rate_pct": round(baseline_default_rate * 100, 2),
        "exposure_at_risk_cr": crores(baseline_exposure_at_risk),
    }}

    base_input = accounts.drop(columns=["probability_of_default", "risk_category", "rag_status", "ecl_stage",
                                          "estimated_months_to_default", "recommended_action", "top_risk_drivers"],
                                errors="ignore")
    for name, shock in scenarios.items():
        shocked = base_input.copy()
        shocked["macro_stress_indicator"] = (shocked["macro_stress_indicator"] + shock["macro_shock"]).clip(0, 10)
        shocked["utilization_ratio_avg"] = (shocked["utilization_ratio_avg"] + shock["utilization_shock"]).clip(0, 1.5)
        shocked["income_stability_index"] = (shocked["income_stability_index"] + shock["income_stability_shock"]).clip(0.05, 1.0)

        X = build_feature_matrix(shocked, snapshot_features=snap_feat)
        proba = STATE["stack_model"].predict_proba(X)[:, 1]
        shocked_cat = pd.Series(proba).apply(risk_category)
        exposure_at_risk = float(shocked.loc[shocked_cat.isin(["High", "Critical"]).values, "loan_amount"].sum())

        results[name] = {
            "default_rate_pct": round(float(proba.mean()) * 100, 2),
            "delta_vs_baseline_pct": round((float(proba.mean()) - baseline_default_rate) * 100, 2),
            "exposure_at_risk_cr": crores(exposure_at_risk),
            "accounts_high_or_critical": int(shocked_cat.isin(["High", "Critical"]).sum()),
        }

    return results


@app.get("/ecl-summary")
def ecl_summary():
    df = STATE["accounts"].copy()
    df["lgd"] = df["loan_type"].map(LGD_BY_LOAN_TYPE).fillna(0.5)

    def provision(row):
        if row["ecl_stage"] == "Stage 3":
            return row["loan_amount"] * row["lgd"]
        if row["ecl_stage"] == "Stage 2":
            lifetime_pd = min(1.0, row["probability_of_default"] * 2.2)
            return row["loan_amount"] * lifetime_pd * row["lgd"]
        return row["loan_amount"] * row["probability_of_default"] * row["lgd"]

    df["ecl_provision"] = df.apply(provision, axis=1)

    by_stage = df.groupby("ecl_stage").agg(
        accounts=("account_id", "count"), exposure=("loan_amount", "sum"), provision=("ecl_provision", "sum"),
    ).reset_index()
    by_loan_type = df.groupby(["ecl_stage", "loan_type"]).agg(
        accounts=("account_id", "count"), exposure=("loan_amount", "sum"), provision=("ecl_provision", "sum"),
    ).reset_index()

    return {
        "framework": "RBI Expected Credit Loss (ECL) Framework — effective 1 April 2027",
        "total_exposure_cr": crores(df["loan_amount"].sum()),
        "total_provision_cr": crores(df["ecl_provision"].sum()),
        "provision_coverage_ratio_pct": round(float(df["ecl_provision"].sum() / df["loan_amount"].sum()) * 100, 3),
        "stage_summary": [
            {**r, "exposure_cr": crores(r["exposure"]), "provision_cr": crores(r["provision"])}
            for r in by_stage.to_dict(orient="records")
        ],
        "stage_by_loan_type": [
            {**r, "exposure_cr": crores(r["exposure"]), "provision_cr": crores(r["provision"])}
            for r in by_loan_type.to_dict(orient="records")
        ],
        "stage_transition_matrix": STATE["transition_matrix"],
    }
