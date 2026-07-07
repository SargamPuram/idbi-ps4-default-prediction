"""
Shared feature engineering used by both the offline training pipeline
(ml/train_model.py) and the live FastAPI inference layer (app/main.py),
so a batch-scored account and a live /predict request go through the
exact same transform.
"""

import numpy as np
import pandas as pd

EMPLOYMENT_TYPES = ["Salaried", "Self-employed", "MSME"]
LOAN_TYPES = ["Personal Loan", "Home Loan", "MSME Loan", "Auto Loan"]
GENDERS = ["M", "F"]
INDUSTRY_SECTORS = [
    "IT Services", "Pharma & Healthcare", "FMCG Retail Trade", "Food Processing",
    "Auto Components", "Textiles", "Construction", "Real Estate",
    "Hospitality & Tourism", "Agriculture Processing", "Salaried - Corporate",
]
SMA_ORDER = {"Regular": 0, "SMA-0": 1, "SMA-1": 2, "SMA-2": 3, "NPA": 4}

ACCOUNT_NUMERIC_COLS = [
    "age", "annual_income", "credit_score", "interest_rate", "loan_amount", "tenure_months",
    "payment_discipline_score", "spending_volatility", "income_stability_index",
    "digital_activity_score", "industry_risk_score", "location_risk_index",
    "macro_stress_indicator", "utilization_ratio_avg",
]

SNAPSHOT_METRICS = [
    ("days_past_due", "dpd"),
    ("utilization_ratio", "utilization"),
    ("monthly_balance_avg", "balance"),
    ("payment_amount", "payment"),
    ("transaction_count", "txn_count"),
    ("debit_credit_ratio", "debit_credit_ratio"),
    ("savings_ratio_trend", "savings_ratio"),
    ("spending_velocity", "spending_velocity"),
]

ENGINEERED_TS_COLS = []
for _, short in SNAPSHOT_METRICS:
    ENGINEERED_TS_COLS += [f"{short}_mean", f"{short}_trend"]
ENGINEERED_TS_COLS += ["dpd_max", "dpd_last3_mean", "dpd_acceleration", "bounce_total", "bounce_last3"]

CATEGORICAL_BASE_COLS = ["employment_type", "loan_type", "gender", "industry_sector"]
ORDINAL_COLS = ["city_tier", "current_sma_status"]

FEATURE_COLUMNS = (
    ACCOUNT_NUMERIC_COLS
    + ENGINEERED_TS_COLS
    + ORDINAL_COLS
    + [f"employment_type__{v}" for v in EMPLOYMENT_TYPES]
    + [f"loan_type__{v}" for v in LOAN_TYPES]
    + [f"gender__{v}" for v in GENDERS]
    + [f"industry_sector__{v}" for v in INDUSTRY_SECTORS]
)


def _linreg_slope(y):
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n)
    return float(np.polyfit(x, y, 1)[0])


def _acceleration(y):
    n = len(y)
    if n < 3:
        return 0.0
    x = np.arange(n)
    coeffs = np.polyfit(x, y, 2)
    return float(coeffs[0] * 2)


def engineer_snapshot_features(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the 12-month time series per account_id into rolling stats, trends and acceleration."""
    snapshots = snapshots.sort_values(["account_id", "snapshot_month"])
    rows = []
    for account_id, g in snapshots.groupby("account_id", sort=False):
        rec = {"account_id": account_id}
        for col, short in SNAPSHOT_METRICS:
            y = g[col].to_numpy(dtype=float)
            rec[f"{short}_mean"] = float(y.mean())
            rec[f"{short}_trend"] = _linreg_slope(y)
        dpd = g["days_past_due"].to_numpy(dtype=float)
        rec["dpd_max"] = float(dpd.max())
        rec["dpd_last3_mean"] = float(dpd[-3:].mean())
        rec["dpd_acceleration"] = _acceleration(dpd)
        rec["bounce_total"] = float(g["emi_bounce_flag"].sum())
        rec["bounce_last3"] = float(g["emi_bounce_flag"].to_numpy()[-3:].sum())
        rows.append(rec)
    return pd.DataFrame(rows)


def build_feature_matrix(accounts: pd.DataFrame, snapshot_features: pd.DataFrame = None,
                          snapshots: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build the final numeric feature matrix (column order = FEATURE_COLUMNS) for a set of accounts.
    Either pass pre-aggregated `snapshot_features` (from engineer_snapshot_features) or raw `snapshots`.
    """
    df = accounts.copy()

    if snapshot_features is None:
        if snapshots is not None:
            snapshot_features = engineer_snapshot_features(snapshots)
        else:
            # Live /predict path: no time series supplied -> neutral defaults (median-ish healthy values)
            defaults = {f"{short}_mean": 0.0 for _, short in SNAPSHOT_METRICS}
            defaults.update({f"{short}_trend": 0.0 for _, short in SNAPSHOT_METRICS})
            defaults.update({"dpd_max": 0.0, "dpd_last3_mean": 0.0, "dpd_acceleration": 0.0,
                              "bounce_total": 0.0, "bounce_last3": 0.0})
            for k, v in defaults.items():
                df[k] = v
            snapshot_features = None

    if snapshot_features is not None:
        df = df.merge(snapshot_features, on="account_id", how="left")
        for col in ENGINEERED_TS_COLS:
            df[col] = df[col].fillna(0.0)

    df["current_sma_status"] = df["current_sma_status"].map(SMA_ORDER).fillna(0).astype(int)

    for col, levels in [
        ("employment_type", EMPLOYMENT_TYPES),
        ("loan_type", LOAN_TYPES),
        ("gender", GENDERS),
        ("industry_sector", INDUSTRY_SECTORS),
    ]:
        cat = pd.Categorical(df[col], categories=levels)
        dummies = pd.get_dummies(cat, prefix=col, prefix_sep="__")
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    X = df[FEATURE_COLUMNS].astype(float)
    return X
