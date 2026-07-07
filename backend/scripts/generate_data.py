"""
Synthetic loan portfolio generator for IDBI Default Prediction Model (PS4).

Generates a realistic 10,000-account Indian retail/MSME loan book with:
  - data/accounts.csv            one row per account (profile + loan + behavioral + external + target)
  - data/monthly_snapshots.csv   12 monthly rows per account (time series)

The default outcome is generated from an underlying risk equation so the
downstream ML pipeline has real signal to learn from (not random noise).
"""

import numpy as np
import pandas as pd
from faker import Faker
from datetime import date, timedelta

RNG_SEED = 42
N_ACCOUNTS = 10_000
TODAY = date(2026, 7, 7)

rng = np.random.default_rng(RNG_SEED)
fake = Faker("en_IN")
Faker.seed(RNG_SEED)

# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------

TIER1_CITIES = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad"]
TIER2_CITIES = ["Jaipur", "Lucknow", "Nagpur", "Indore", "Bhopal", "Coimbatore", "Vadodara", "Surat", "Patna", "Chandigarh"]
TIER3_CITIES = ["Nashik", "Rajkot", "Varanasi", "Amritsar", "Ranchi", "Jodhpur", "Guwahati", "Dehradun", "Raipur", "Shimla"]

CITY_TIER_WEIGHTS = [0.45, 0.35, 0.20]
CITY_RISK_INDEX = {}  # populated below, tier1 lower risk, tier3 higher risk
for c in TIER1_CITIES:
    CITY_RISK_INDEX[c] = rng.uniform(1.0, 3.5)
for c in TIER2_CITIES:
    CITY_RISK_INDEX[c] = rng.uniform(3.0, 6.0)
for c in TIER3_CITIES:
    CITY_RISK_INDEX[c] = rng.uniform(5.0, 8.5)

EMPLOYMENT_TYPES = ["Salaried", "Self-employed", "MSME"]
EMPLOYMENT_WEIGHTS = [0.50, 0.30, 0.20]

LOAN_TYPES = ["Personal Loan", "Home Loan", "MSME Loan", "Auto Loan"]
LOAN_TYPE_WEIGHTS = [0.35, 0.25, 0.25, 0.15]

LOAN_AMOUNT_RANGE = {
    "Personal Loan": (50_000, 15_00_000),
    "Home Loan": (10_00_000, 2_00_00_000),
    "MSME Loan": (5_00_000, 2_00_00_000),
    "Auto Loan": (1_00_000, 25_00_000),
}
TENURE_RANGE_MONTHS = {
    "Personal Loan": (12, 60),
    "Home Loan": (60, 360),
    "MSME Loan": (12, 120),
    "Auto Loan": (12, 84),
}
BASE_RATE_RANGE = {
    "Personal Loan": (10.5, 18.0),
    "Home Loan": (8.0, 11.5),
    "MSME Loan": (9.5, 16.0),
    "Auto Loan": (8.5, 13.5),
}

INDUSTRY_SECTORS = {
    "IT Services": 2.5, "Pharma & Healthcare": 3.0, "FMCG Retail Trade": 4.0,
    "Food Processing": 4.5, "Auto Components": 5.0, "Textiles": 6.0,
    "Construction": 7.0, "Real Estate": 7.5, "Hospitality & Tourism": 7.5,
    "Agriculture Processing": 6.5,
}

# --------------------------------------------------------------------------
# Step 1: Borrower profile
# --------------------------------------------------------------------------

def gen_borrower_profiles(n):
    tiers = rng.choice([1, 2, 3], size=n, p=CITY_TIER_WEIGHTS)
    cities = np.array([
        rng.choice(TIER1_CITIES) if t == 1 else rng.choice(TIER2_CITIES) if t == 2 else rng.choice(TIER3_CITIES)
        for t in tiers
    ])
    ages = rng.integers(22, 66, size=n)
    employment = rng.choice(EMPLOYMENT_TYPES, size=n, p=EMPLOYMENT_WEIGHTS)

    # Income: lognormal, shifted by employment type and city tier
    base_income = rng.lognormal(mean=13.7, sigma=0.55, size=n)  # centered ~ 9L
    emp_multiplier = np.select(
        [employment == "Salaried", employment == "Self-employed", employment == "MSME"],
        [1.0, 1.15, 1.3],
    )
    tier_multiplier = np.select([tiers == 1, tiers == 2, tiers == 3], [1.25, 1.0, 0.8])
    annual_income = np.clip(base_income * emp_multiplier * tier_multiplier, 3_00_000, 50_00_000).round(-2)

    # Credit score: correlated with income (weakly) + noise, CIBIL-style 300-900
    income_z = (np.log(annual_income) - np.log(annual_income).mean()) / np.log(annual_income).std()
    credit_score = np.clip(680 + income_z * 45 + rng.normal(0, 90, size=n), 300, 900).round().astype(int)

    genders = rng.choice(["M", "F"], size=n, p=[0.62, 0.38])
    names = [fake.name_male() if g == "M" else fake.name_female() for g in genders]

    industry = np.array([
        rng.choice(list(INDUSTRY_SECTORS.keys())) if e in ("Self-employed", "MSME") else "Salaried - Corporate"
        for e in employment
    ])

    return pd.DataFrame({
        "borrower_name": names,
        "age": ages,
        "gender": genders,
        "city": cities,
        "city_tier": tiers,
        "employment_type": employment,
        "industry_sector": industry,
        "annual_income": annual_income.astype(int),
        "credit_score": credit_score,
    })


# --------------------------------------------------------------------------
# Step 2: Loan details
# --------------------------------------------------------------------------

def gen_loan_details(n, credit_score):
    loan_type = rng.choice(LOAN_TYPES, size=n, p=LOAN_TYPE_WEIGHTS)

    loan_amount = np.zeros(n)
    tenure_months = np.zeros(n, dtype=int)
    base_rate = np.zeros(n)
    for lt in LOAN_TYPES:
        mask = loan_type == lt
        lo, hi = LOAN_AMOUNT_RANGE[lt]
        loan_amount[mask] = rng.uniform(lo, hi, size=mask.sum())
        t_lo, t_hi = TENURE_RANGE_MONTHS[lt]
        tenure_months[mask] = rng.integers(t_lo, t_hi + 1, size=mask.sum())
        r_lo, r_hi = BASE_RATE_RANGE[lt]
        base_rate[mask] = rng.uniform(r_lo, r_hi, size=mask.sum())

    # Risk-adjusted interest rate: lower credit score -> higher rate premium
    risk_premium = np.clip((750 - credit_score) / 100, -1.5, 4.0)
    interest_rate = np.clip(base_rate + risk_premium * 0.6, 8.0, 18.0).round(2)

    days_back = rng.integers(30, 3 * 365, size=n)
    disbursement_date = [TODAY - timedelta(days=int(d)) for d in days_back]

    return pd.DataFrame({
        "loan_type": loan_type,
        "loan_amount": loan_amount.round(-2).astype(int),
        "interest_rate": interest_rate,
        "tenure_months": tenure_months,
        "disbursement_date": disbursement_date,
    })


# --------------------------------------------------------------------------
# Step 3: Behavioral & external features + risk latent -> default outcome
# --------------------------------------------------------------------------

def gen_behavioral_and_target(df):
    n = len(df)

    payment_discipline_score = np.clip(
        55 + (df["credit_score"] - 300) / 600 * 35 + rng.normal(0, 12, n), 0, 100
    )
    income_stability_index = np.clip(
        np.where(df["employment_type"] == "Salaried", rng.uniform(0.75, 0.98, n),
                  rng.uniform(0.35, 0.85, n)), 0.1, 1.0
    )
    spending_volatility = np.clip(rng.gamma(2.0, 8.0, n) * (1.3 - income_stability_index), 1, 100)
    digital_activity_score = np.clip(
        np.where(df["age"] < 40, rng.uniform(45, 100, n), rng.uniform(15, 80, n)), 0, 100
    )

    industry_risk_score = df["industry_sector"].map(lambda s: INDUSTRY_SECTORS.get(s, 2.0)).values
    industry_risk_score = np.clip(industry_risk_score + rng.normal(0, 0.6, n), 1, 10)
    location_risk_index = np.clip(
        df["city"].map(CITY_RISK_INDEX).values + rng.normal(0, 0.4, n), 1, 10
    )
    macro_stress_indicator = np.clip(rng.normal(4.5, 1.6, n), 0, 10)

    utilization_ratio_avg = np.clip(
        0.75 - (df["credit_score"] - 300) / 600 * 0.5 + rng.normal(0, 0.15, n), 0.02, 1.3
    )

    # ---- Composite risk latent score (drives default probability) ----
    z = (
        -0.032 * (df["credit_score"] - 650)
        - 0.028 * (payment_discipline_score - 60)
        - 3.2 * (income_stability_index - 0.6)
        + 1.9 * (utilization_ratio_avg - 0.5)
        + 0.16 * (industry_risk_score - 4.5)
        + 0.10 * (location_risk_index - 4.5)
        + 0.14 * (macro_stress_indicator - 4.5)
        - 0.012 * (digital_activity_score - 50)
        + 0.008 * (spending_volatility - 20)
        - 2.6
        + rng.normal(0, 1.15, n)
    )
    pd_true = 1 / (1 + np.exp(-z))

    # Calibrate to ~8% overall default rate via threshold search on the sampled draw
    target_rate = 0.08
    threshold = np.quantile(pd_true, 1 - target_rate)
    default_flag = (pd_true >= threshold).astype(int)

    # Not every defaulting account shows early behavioral warning signs in its 12m history —
    # ~28% default with little prior visible stress (genuinely hard cases). This keeps the
    # model's recall realistic instead of a trivial 100% giveaway.
    shows_early_warning = rng.random(n) < 0.72
    df_early_warning_flag = shows_early_warning

    # A small slice of otherwise-healthy accounts get a temporary, self-resolving stress
    # blip (cash-flow hiccup that recovers) — realistic false-alarm noise for the model.
    false_alarm_flag = (default_flag == 0) & (rng.random(n) < 0.05)

    months_to_default = np.full(n, np.nan)
    mtd_draw = np.clip(rng.beta(1.6, 2.2, n) * 12, 0.5, 12).round().astype(int)
    months_to_default[default_flag == 1] = mtd_draw[default_flag == 1]

    df = df.copy()
    df["payment_discipline_score"] = payment_discipline_score.round(1)
    df["spending_volatility"] = spending_volatility.round(2)
    df["income_stability_index"] = income_stability_index.round(3)
    df["digital_activity_score"] = digital_activity_score.round(1)
    df["industry_risk_score"] = industry_risk_score.round(2)
    df["location_risk_index"] = location_risk_index.round(2)
    df["macro_stress_indicator"] = macro_stress_indicator.round(2)
    df["utilization_ratio_avg"] = utilization_ratio_avg.round(3)
    df["pd_latent_true"] = pd_true.round(4)
    df["default_flag"] = default_flag
    df["months_to_default"] = months_to_default
    df["shows_early_warning"] = df_early_warning_flag
    df["false_alarm_flag"] = false_alarm_flag

    return df


def assign_sma_status(row):
    """Derive CURRENT SMA classification from most-recent-month DPD (set later from time series)."""
    dpd = row["current_dpd"]
    if dpd == 0:
        return "Regular"
    elif dpd <= 30:
        return "SMA-0"
    elif dpd <= 60:
        return "SMA-1"
    elif dpd <= 90:
        return "SMA-2"
    return "NPA"


# --------------------------------------------------------------------------
# Step 4: 12-month time series per account
# --------------------------------------------------------------------------

def gen_monthly_snapshots(df):
    n = len(df)
    records = []

    month_dates = [TODAY.replace(day=1) - pd.DateOffset(months=11 - m) for m in range(12)]

    for idx in range(n):
        acc_id = df.at[idx, "account_id"]
        will_default = df.at[idx, "default_flag"] == 1
        early_warning = bool(df.at[idx, "shows_early_warning"]) if will_default else False
        false_alarm = bool(df.at[idx, "false_alarm_flag"])
        income = df.at[idx, "annual_income"]
        monthly_income = income / 12
        base_balance = monthly_income * rng.uniform(0.3, 1.2)
        base_util = df.at[idx, "utilization_ratio_avg"]
        loan_amount = df.at[idx, "loan_amount"]
        tenure = df.at[idx, "tenure_months"]
        rate_monthly = df.at[idx, "interest_rate"] / 1200
        emi = loan_amount * rate_monthly * (1 + rate_monthly) ** tenure / (((1 + rate_monthly) ** tenure) - 1) if tenure > 0 else loan_amount / 12

        # Build a per-month stress curve depending on account category:
        #  - visible default:  genuine ramping deterioration (the true early-warning case)
        #  - silent default:   defaults with little/no visible history (hard, unpredictable case)
        #  - false alarm:      temporary cash-flow hiccup on an otherwise healthy account, recovers
        #  - healthy:          flat, no stress
        stress_curve = np.zeros(12)
        if will_default and early_warning:
            onset_month = rng.integers(3, 10)
            for m in range(12):
                stress_curve[m] = min(1.0, max(0, m - onset_month) / 6.0)
        elif will_default and not early_warning:
            onset_month = rng.integers(10, 12)
            for m in range(12):
                stress_curve[m] = min(0.35, max(0, m - onset_month) / 4.0)
        elif false_alarm:
            onset_month = rng.integers(3, 7)
            duration = rng.integers(2, 5)
            half = max(duration / 2, 1)
            for m in range(12):
                rel = m - onset_month
                if 0 <= rel < duration:
                    f = rel / half if rel < half else max(0.0, 1 - (rel - half) / half)
                    stress_curve[m] = min(1.0, f) * 0.6
                else:
                    stress_curve[m] = 0.0

        dpd_prev = 0
        bounce_cum = 0
        for m in range(12):
            stress_factor = stress_curve[m]

            # Days past due: ramps up under stress, otherwise mostly 0 with rare small blips
            if stress_factor > 0:
                dpd = np.clip(dpd_prev + rng.integers(3, 16) * stress_factor + rng.integers(0, 3), 0, 89)
            else:
                dpd = max(0, dpd_prev * 0.4 + rng.choice([0, 0, 0, 0, 0, 3, 5, 10], p=[0.72, 0.05, 0.05, 0.03, 0.03, 0.06, 0.03, 0.03]))
            dpd_prev = dpd

            bounced = 1 if (rng.random() < (0.02 + 0.5 * stress_factor)) else 0
            bounce_cum += bounced

            balance = base_balance * (1 - 0.35 * stress_factor) * rng.uniform(0.85, 1.15)
            payment_amount = emi * (1 - 0.6 * stress_factor * (rng.random() < 0.5)) * rng.uniform(0.95, 1.05)
            utilization = np.clip(base_util * (1 + 0.6 * stress_factor) + rng.normal(0, 0.05), 0.02, 1.5)
            txn_count = max(1, int(rng.normal(28 - 10 * stress_factor, 6)))
            debit_credit_ratio = np.clip(rng.normal(0.7 + 0.35 * stress_factor, 0.12), 0.1, 2.0)
            savings_ratio_trend = np.clip(rng.normal(0.15 - 0.12 * stress_factor, 0.05), -0.2, 0.5)
            spending_velocity = np.clip(rng.normal(1.0 + 0.5 * stress_factor, 0.15), 0.3, 3.0)

            records.append((
                acc_id, month_dates[m].date().isoformat(), round(balance, 2), round(payment_amount, 2),
                int(dpd), int(bounced), round(utilization, 3), txn_count,
                round(debit_credit_ratio, 3), round(savings_ratio_trend, 3), round(spending_velocity, 3),
                bounce_cum,
            ))

        df.at[idx, "current_dpd"] = int(round(dpd_prev))
        df.at[idx, "emi_bounce_count_12m"] = bounce_cum

    cols = [
        "account_id", "snapshot_month", "monthly_balance_avg", "payment_amount", "days_past_due",
        "emi_bounce_flag", "utilization_ratio", "transaction_count", "debit_credit_ratio",
        "savings_ratio_trend", "spending_velocity", "emi_bounce_count_cumulative",
    ]
    snapshots = pd.DataFrame.from_records(records, columns=cols)
    return df, snapshots


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    print(f"Generating {N_ACCOUNTS} synthetic loan accounts...")

    profiles = gen_borrower_profiles(N_ACCOUNTS)
    loans = gen_loan_details(N_ACCOUNTS, profiles["credit_score"].values)
    df = pd.concat([profiles, loans], axis=1)
    df.insert(0, "account_id", [f"IDBI-LN-{i:06d}" for i in range(1, N_ACCOUNTS + 1)])

    df = gen_behavioral_and_target(df)
    df["current_dpd"] = 0
    df["emi_bounce_count_12m"] = 0

    print("Generating 12-month time series per account (this takes a minute)...")
    df, snapshots = gen_monthly_snapshots(df)

    df["current_sma_status"] = df.apply(assign_sma_status, axis=1)
    # NPA accounts by definition already 90+ DPD -> treat as already defaulted, not "predicted" default
    df.loc[df["current_sma_status"] == "NPA", "default_flag"] = 1
    df.loc[(df["default_flag"] == 1) & (df["months_to_default"].isna()), "months_to_default"] = 1

    import os
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/accounts.csv", index=False)
    snapshots.to_csv("data/monthly_snapshots.csv", index=False)

    print(f"\nSaved data/accounts.csv ({len(df)} rows) and data/monthly_snapshots.csv ({len(snapshots)} rows)")
    print(f"Default rate: {df['default_flag'].mean() * 100:.2f}%")
    print(f"SMA distribution:\n{df['current_sma_status'].value_counts()}")
    print(f"Loan type distribution:\n{df['loan_type'].value_counts(normalize=True).round(3)}")


if __name__ == "__main__":
    main()
