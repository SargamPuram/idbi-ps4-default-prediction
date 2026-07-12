"""
Tests for backend/ml/feature_engineering.py — the shared transform used by
both offline training and the live /predict endpoint, so a bug here would
silently affect every prediction. Focused on pure computation functions and
their edge cases (too-short series, all-zero series, missing account in the
snapshot table).
"""

import numpy as np
import pandas as pd
import pytest

from ml.feature_engineering import (
    ENGINEERED_TS_COLS,
    FEATURE_COLUMNS,
    _acceleration,
    _linreg_slope,
    build_feature_matrix,
    engineer_snapshot_features,
)


def _base_account_row(account_id="IDBI-LN-000001"):
    """A single account row with every column build_feature_matrix requires,
    mirroring the AccountFeaturesIn defaults used by the live /predict path."""
    return {
        "account_id": account_id,
        "age": 35,
        "gender": "M",
        "city_tier": 1,
        "employment_type": "Salaried",
        "industry_sector": "Salaried - Corporate",
        "annual_income": 900000.0,
        "credit_score": 720,
        "loan_type": "Personal Loan",
        "loan_amount": 500000.0,
        "interest_rate": 11.5,
        "tenure_months": 36,
        "payment_discipline_score": 75.0,
        "spending_volatility": 15.0,
        "income_stability_index": 0.8,
        "digital_activity_score": 60.0,
        "industry_risk_score": 3.0,
        "location_risk_index": 3.0,
        "macro_stress_indicator": 4.5,
        "utilization_ratio_avg": 0.4,
        "current_sma_status": "Regular",
    }


# --------------------------------------------------------------------------
# _linreg_slope — edge cases: empty/too-short series, flat series
# --------------------------------------------------------------------------

class TestLinregSlope:
    def test_empty_series_returns_zero(self):
        assert _linreg_slope(np.array([])) == 0.0

    def test_single_point_returns_zero(self):
        # n < 2: not enough points for a slope, must not raise.
        assert _linreg_slope(np.array([42.0])) == 0.0

    def test_two_points_gives_exact_slope(self):
        # x = [0, 1], y = [0, 10] -> slope = 10.0 exactly.
        assert _linreg_slope(np.array([0.0, 10.0])) == pytest.approx(10.0)

    def test_flat_series_has_zero_slope(self):
        assert _linreg_slope(np.array([5.0, 5.0, 5.0, 5.0])) == pytest.approx(0.0)

    def test_decreasing_series_has_negative_slope(self):
        assert _linreg_slope(np.array([10.0, 5.0, 0.0])) == pytest.approx(-5.0)


# --------------------------------------------------------------------------
# _acceleration — edge cases: too-short series (n < 3), perfect parabola
# --------------------------------------------------------------------------

class TestAcceleration:
    def test_two_points_returns_zero(self):
        # n < 3: a quadratic fit is underdetermined, must not raise.
        assert _acceleration(np.array([1.0, 2.0])) == 0.0

    def test_empty_series_returns_zero(self):
        assert _acceleration(np.array([])) == 0.0

    def test_perfect_parabola_recovers_known_acceleration(self):
        # y = x^2 over x = 0..4 -> quadratic coeff a=1 -> acceleration = 2*a = 2.0
        x = np.arange(5)
        y = (x ** 2).astype(float)
        assert _acceleration(y) == pytest.approx(2.0, abs=1e-6)

    def test_linear_series_has_zero_acceleration(self):
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        assert _acceleration(y) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# engineer_snapshot_features — edge case: all-zero DPD history (healthy
# account, the most common case in the synthetic portfolio)
# --------------------------------------------------------------------------

class TestEngineerSnapshotFeatures:
    def _flat_healthy_snapshots(self, account_id="IDBI-LN-000001", n_months=12):
        months = pd.date_range("2025-08-01", periods=n_months, freq="MS")
        return pd.DataFrame({
            "account_id": [account_id] * n_months,
            "snapshot_month": [m.strftime("%Y-%m-%d") for m in months],
            "days_past_due": [0] * n_months,
            "utilization_ratio": [0.3] * n_months,
            "monthly_balance_avg": [50000.0] * n_months,
            "payment_amount": [12000.0] * n_months,
            "transaction_count": [25] * n_months,
            "debit_credit_ratio": [0.7] * n_months,
            "savings_ratio_trend": [0.15] * n_months,
            "spending_velocity": [1.0] * n_months,
            "emi_bounce_flag": [0] * n_months,
        })

    def test_all_zero_dpd_gives_zero_max_and_zero_last3_mean(self):
        snapshots = self._flat_healthy_snapshots()
        feats = engineer_snapshot_features(snapshots)
        row = feats.iloc[0]
        assert row["dpd_max"] == 0.0
        assert row["dpd_last3_mean"] == 0.0
        assert row["bounce_total"] == 0.0
        assert row["bounce_last3"] == 0.0
        assert row["dpd_trend"] == pytest.approx(0.0)

    def test_ramping_dpd_is_captured_in_max_and_last3_mean(self):
        snapshots = self._flat_healthy_snapshots()
        # Deteriorate steadily over the last 3 months: 0,...,0,10,20,30
        snapshots.loc[snapshots.index[-3:], "days_past_due"] = [10, 20, 30]
        feats = engineer_snapshot_features(snapshots)
        row = feats.iloc[0]
        assert row["dpd_max"] == 30.0
        assert row["dpd_last3_mean"] == pytest.approx(20.0)
        assert row["dpd_trend"] > 0  # deteriorating trend is positive slope


# --------------------------------------------------------------------------
# build_feature_matrix — edge cases: no time-series supplied (live /predict
# with no history) and an account missing from the snapshot table
# --------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    def test_output_has_exactly_the_expected_feature_columns_in_order(self):
        accounts = pd.DataFrame([_base_account_row()])
        X = build_feature_matrix(accounts)
        assert list(X.columns) == FEATURE_COLUMNS

    def test_no_snapshots_supplied_fills_neutral_zero_defaults(self):
        # Mirrors the live /predict path when no 12-month history is provided.
        accounts = pd.DataFrame([_base_account_row()])
        X = build_feature_matrix(accounts)
        for col in ENGINEERED_TS_COLS:
            assert X.iloc[0][col] == 0.0

    def test_account_missing_from_snapshot_features_fills_zero_not_nan(self):
        accounts = pd.DataFrame([_base_account_row(account_id="IDBI-LN-999999")])
        # snapshot_features keyed to a *different* account_id -> left merge produces NaN
        # for the missing account, which must be filled with 0.0, not left as NaN.
        snapshot_features = pd.DataFrame([{
            "account_id": "IDBI-LN-000001",
            **{col: 5.0 for col in ENGINEERED_TS_COLS},
        }])
        X = build_feature_matrix(accounts, snapshot_features=snapshot_features)
        for col in ENGINEERED_TS_COLS:
            assert X.iloc[0][col] == 0.0
            assert not np.isnan(X.iloc[0][col])

    def test_unknown_categorical_level_does_not_raise_and_zero_fills(self):
        # An employment_type outside the known EMPLOYMENT_TYPES list should
        # not blow up the one-hot encoding; it should simply not set any
        # employment_type__* dummy to 1.
        row = _base_account_row()
        row["employment_type"] = "Unemployed"  # not in EMPLOYMENT_TYPES
        accounts = pd.DataFrame([row])
        X = build_feature_matrix(accounts)
        emp_cols = [c for c in FEATURE_COLUMNS if c.startswith("employment_type__")]
        assert X.iloc[0][emp_cols].sum() == 0.0
