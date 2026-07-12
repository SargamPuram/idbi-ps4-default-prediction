"""
Boundary-value tests for backend/ml/risk_logic.py.

risk_category(), rag_status() and ecl_stage() are all threshold-driven
classifiers (`if pd_score >= X: ...`), so the interesting bugs live exactly
at the threshold boundary (off-by-one in >= vs >, wrong threshold value).
Each boundary below is tested at the threshold value itself and at the
largest representable value just below it.
"""

import numpy as np
import pytest

from ml.risk_logic import (
    ecl_stage,
    estimated_months_to_default,
    rag_status,
    risk_category,
    top_risk_drivers,
)

JUST_BELOW = 1e-4  # small enough to stay clear of float rounding noise


# --------------------------------------------------------------------------
# risk_category — thresholds at 0.50 (Critical), 0.25 (High), 0.10 (Medium)
# --------------------------------------------------------------------------

class TestRiskCategoryBoundaries:
    def test_critical_threshold_at_050_is_critical(self):
        assert risk_category(0.50) == "Critical"

    def test_just_below_050_is_high_not_critical(self):
        assert risk_category(0.50 - JUST_BELOW) == "High"

    def test_high_threshold_at_025_is_high(self):
        assert risk_category(0.25) == "High"

    def test_just_below_025_is_medium_not_high(self):
        assert risk_category(0.25 - JUST_BELOW) == "Medium"

    def test_medium_threshold_at_010_is_medium(self):
        assert risk_category(0.10) == "Medium"

    def test_just_below_010_is_low_not_medium(self):
        assert risk_category(0.10 - JUST_BELOW) == "Low"

    def test_zero_is_low(self):
        assert risk_category(0.0) == "Low"

    def test_one_is_critical(self):
        assert risk_category(1.0) == "Critical"


# --------------------------------------------------------------------------
# rag_status — thresholds at 0.25 (Red), 0.10 (Amber)
# --------------------------------------------------------------------------

class TestRagStatusBoundaries:
    def test_red_threshold_at_025_is_red(self):
        assert rag_status(0.25) == "Red"

    def test_just_below_025_is_amber_not_red(self):
        assert rag_status(0.25 - JUST_BELOW) == "Amber"

    def test_amber_threshold_at_010_is_amber(self):
        assert rag_status(0.10) == "Amber"

    def test_just_below_010_is_green_not_amber(self):
        assert rag_status(0.10 - JUST_BELOW) == "Green"

    def test_zero_is_green(self):
        assert rag_status(0.0) == "Green"


# --------------------------------------------------------------------------
# ecl_stage — NPA always Stage 3; SMA-1/SMA-2 or pd>=0.25 -> Stage 2
# --------------------------------------------------------------------------

class TestEclStageBoundaries:
    def test_npa_is_always_stage3_even_with_low_pd(self):
        assert ecl_stage("NPA", 0.0) == "Stage 3"

    def test_regular_at_pd_025_is_stage2(self):
        assert ecl_stage("Regular", 0.25) == "Stage 2"

    def test_regular_just_below_pd_025_is_stage1(self):
        assert ecl_stage("Regular", 0.25 - JUST_BELOW) == "Stage 1"

    def test_sma1_with_zero_pd_is_stage2(self):
        # SMA-1/SMA-2 forces Stage 2 regardless of the PD score.
        assert ecl_stage("SMA-1", 0.0) == "Stage 2"

    def test_sma2_with_zero_pd_is_stage2(self):
        assert ecl_stage("SMA-2", 0.0) == "Stage 2"

    def test_sma0_with_low_pd_is_stage1(self):
        # SMA-0 is not in the Stage-2-forcing set, so a low PD stays Stage 1.
        assert ecl_stage("SMA-0", 0.0) == "Stage 1"

    def test_regular_with_high_pd_is_stage2(self):
        assert ecl_stage("Regular", 0.99) == "Stage 2"


# --------------------------------------------------------------------------
# estimated_months_to_default — None for "Low", else clip(round(1+11*(1-pd)), 1, 12)
# --------------------------------------------------------------------------

class TestEstimatedMonthsToDefault:
    def test_low_category_is_always_none(self):
        assert estimated_months_to_default(0.99, "Low") is None
        assert estimated_months_to_default(0.0, "Low") is None

    def test_pd_of_one_clips_to_one_month(self):
        assert estimated_months_to_default(1.0, "Critical") == 1

    def test_pd_of_zero_clips_to_twelve_months(self):
        assert estimated_months_to_default(0.0, "Critical") == 12

    def test_mid_pd_gives_expected_month(self):
        # round(1 + 11 * (1 - 0.25)) = round(1 + 8.25) = round(9.25) = 9
        assert estimated_months_to_default(0.25, "High") == 9


# --------------------------------------------------------------------------
# top_risk_drivers — sanity check on ordering/formatting (not threshold-based,
# but exercises the one remaining pure function in the module).
# --------------------------------------------------------------------------

class TestTopRiskDrivers:
    def test_orders_by_absolute_shap_contribution_descending(self):
        shap_row = np.array([0.01, -0.9, 0.3])
        feature_names = ["feat_a", "feat_b", "feat_c"]
        feature_values = {"feat_a": 1, "feat_b": 2, "feat_c": 3}

        drivers = top_risk_drivers(shap_row, feature_values, feature_names, top_n=2)

        assert len(drivers) == 2
        assert drivers[0]["feature"] == "feat_b"
        assert drivers[0]["direction"] == "reducing risk"
        assert drivers[1]["feature"] == "feat_c"
        assert drivers[1]["direction"] == "increasing risk"
