"""ECL-aligned risk classification, prescriptive actions and SHAP-driven explanations."""

import numpy as np

FEATURE_LABELS = {
    "credit_score": "CIBIL-style credit score",
    "annual_income": "Annual income",
    "utilization_ratio_avg": "Credit utilization ratio",
    "payment_discipline_score": "Payment discipline score",
    "income_stability_index": "Income stability index",
    "digital_activity_score": "Digital engagement score",
    "industry_risk_score": "Industry/sector risk",
    "location_risk_index": "Geography risk index",
    "macro_stress_indicator": "Macro-economic stress indicator",
    "spending_volatility": "Spending volatility",
    "dpd_mean": "Average days-past-due (12m)",
    "dpd_max": "Peak days-past-due (12m)",
    "dpd_last3_mean": "Recent days-past-due (last 3 months)",
    "dpd_trend": "Days-past-due trend",
    "dpd_acceleration": "Days-past-due acceleration",
    "bounce_total": "EMI bounce count (12m)",
    "bounce_last3": "Recent EMI bounces (last 3 months)",
    "utilization_mean": "Average utilization (12m)",
    "utilization_trend": "Utilization trend",
    "balance_mean": "Average account balance (12m)",
    "balance_trend": "Account balance trend",
    "payment_mean": "Average EMI payment (12m)",
    "payment_trend": "Payment amount trend",
    "txn_count_mean": "Average monthly transactions (12m)",
    "txn_count_trend": "Transaction activity trend",
    "debit_credit_ratio_mean": "Debit/credit ratio",
    "debit_credit_ratio_trend": "Debit/credit ratio trend",
    "savings_ratio_mean": "Savings ratio",
    "spending_velocity_mean": "Spending velocity",
    "spending_velocity_trend": "Spending velocity trend",
    "current_sma_status": "Current SMA classification",
    "employment_type__Salaried": "Employment: Salaried",
    "employment_type__Self-employed": "Employment: Self-employed",
    "employment_type__MSME": "Employment: MSME owner",
    "loan_type__Personal Loan": "Loan type: Personal Loan",
    "loan_type__Home Loan": "Loan type: Home Loan",
    "loan_type__MSME Loan": "Loan type: MSME Loan",
    "loan_type__Auto Loan": "Loan type: Auto Loan",
}


def risk_category(pd_score: float) -> str:
    if pd_score >= 0.50:
        return "Critical"
    if pd_score >= 0.25:
        return "High"
    if pd_score >= 0.10:
        return "Medium"
    return "Low"


def rag_status(pd_score: float) -> str:
    if pd_score >= 0.25:
        return "Red"
    if pd_score >= 0.10:
        return "Amber"
    return "Green"


def ecl_stage(sma_status: str, pd_score: float) -> str:
    if sma_status == "NPA":
        return "Stage 3"
    if sma_status in ("SMA-1", "SMA-2") or pd_score >= 0.25:
        return "Stage 2"
    return "Stage 1"


def estimated_months_to_default(pd_score: float, category: str):
    if category in ("Low",):
        return None
    return int(np.clip(round(1 + 11 * (1 - pd_score)), 1, 12))


def recommended_action(category: str, ecl: str, loan_type: str) -> str:
    if ecl == "Stage 3":
        return (f"Account already impaired (NPA). Initiate structured recovery/restructuring "
                f"assessment for this {loan_type.lower()} and assign to collections within 48 hours.")
    if category == "Critical":
        return (f"Immediate intervention required: proactive outreach within 48-72 hours, "
                f"evaluate restructuring/top-up options, escalate to senior RM for this {loan_type.lower()}.")
    if category == "High":
        return ("Schedule proactive borrower outreach within 7 days; increase monitoring to weekly; "
                "flag for early-warning committee review.")
    if category == "Medium":
        return ("Schedule relationship manager check-in within 30 days; monitor utilization and "
                "payment trend for further deterioration.")
    return "Continue standard quarterly monitoring; no immediate action required."


def top_risk_drivers(shap_row: np.ndarray, feature_values: dict, feature_names: list, top_n: int = 5):
    order = np.argsort(-np.abs(shap_row))[:top_n]
    drivers = []
    for i in order:
        fname = feature_names[i]
        contrib = float(shap_row[i])
        label = FEATURE_LABELS.get(fname, fname.replace("_", " ").title())
        value = feature_values.get(fname)
        direction = "increasing risk" if contrib > 0 else "reducing risk"
        display_value = round(value, 2) if isinstance(value, (int, float)) else value
        drivers.append({
            "feature": fname,
            "label": label,
            "value": round(float(value), 3) if value is not None else None,
            "shap_contribution": round(contrib, 4),
            "direction": direction,
            "explanation": f"{label}: {display_value} — {direction}",
        })
    return drivers
