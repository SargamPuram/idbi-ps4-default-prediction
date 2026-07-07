from typing import Optional

from pydantic import BaseModel, Field


class AccountFeaturesIn(BaseModel):
    age: int = Field(35, ge=18, le=80)
    gender: str = Field("M", description="M or F")
    city_tier: int = Field(1, ge=1, le=3)
    employment_type: str = Field("Salaried", description="Salaried, Self-employed or MSME")
    industry_sector: str = Field("Salaried - Corporate")
    annual_income: float = Field(900000, gt=0)
    credit_score: int = Field(720, ge=300, le=900)
    loan_type: str = Field("Personal Loan", description="Personal Loan, Home Loan, MSME Loan or Auto Loan")
    loan_amount: float = Field(500000, gt=0)
    interest_rate: float = Field(11.5, ge=5, le=24)
    tenure_months: int = Field(36, ge=3, le=360)
    payment_discipline_score: float = Field(75, ge=0, le=100)
    spending_volatility: float = Field(15, ge=0, le=100)
    income_stability_index: float = Field(0.8, ge=0, le=1)
    digital_activity_score: float = Field(60, ge=0, le=100)
    industry_risk_score: float = Field(3.0, ge=1, le=10)
    location_risk_index: float = Field(3.0, ge=1, le=10)
    macro_stress_indicator: float = Field(4.5, ge=0, le=10)
    utilization_ratio_avg: float = Field(0.4, ge=0, le=1.5)
    current_sma_status: str = Field("Regular", description="Regular, SMA-0, SMA-1 or SMA-2")

    # Optional 12-month behavioral summary — if omitted, neutral/no-history defaults are used
    dpd_mean: Optional[float] = None
    dpd_max: Optional[float] = None
    dpd_last3_mean: Optional[float] = None
    dpd_trend: Optional[float] = None
    dpd_acceleration: Optional[float] = None
    bounce_total: Optional[float] = None
    bounce_last3: Optional[float] = None
    utilization_trend: Optional[float] = None


class RiskDriver(BaseModel):
    feature: str
    label: str
    value: Optional[float]
    shap_contribution: float
    direction: str
    explanation: str


class PredictionOut(BaseModel):
    account_id: Optional[str] = None
    probability_of_default: float
    risk_category: str
    rag_status: str
    sma_classification: str
    ecl_stage: str
    estimated_months_to_default: Optional[int]
    recommended_action: str
    top_risk_drivers: list[RiskDriver]
