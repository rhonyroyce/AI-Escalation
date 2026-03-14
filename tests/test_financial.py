"""
Comprehensive test suite for financial calculations.

Tests the price catalog (escalation_ai.feedback.price_catalog) and the
financial metrics module (escalation_ai.financial.metrics).  These numbers
appear in executive reports and must be accurate.

Test categories:
  1. Cost Calculation Accuracy    — hand-verified formula checks
  2. Price Catalog Lookup         — category / sub-category / keyword resolution
  3. SLA Calculations             — penalty exposure for critical tickets
  4. Aggregation Tests            — summary statistics from known DataFrames
  5. Currency / Rounding          — no floating-point artifacts
  6. Zero / Edge Cases            — empty data, single category, negatives

Run:  pytest tests/test_financial.py -v
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from escalation_ai.feedback.price_catalog import PriceCatalog
from escalation_ai.financial.metrics import (
    FinancialMetrics,
    calculate_financial_metrics,
    calculate_roi_metrics,
    calculate_cost_avoidance,
    calculate_efficiency_metrics,
    calculate_financial_forecasts,
    _calculate_recurring_cost,
    _calculate_preventable_cost,
    _calculate_cost_avoidance,
    _calculate_risk_exposure,
    _calculate_recurrence_exposure,
    _calculate_customer_impact,
    _calculate_sla_penalty,
    _calculate_efficiency_score,
    _calculate_monthly_trend,
    _calculate_cost_velocity,
    _forecast_costs,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def catalog():
    """Create a PriceCatalog with in-memory data (no Excel file needed)."""
    cat = PriceCatalog.__new__(PriceCatalog)
    cat.catalog_path = "/dev/null"
    cat.is_loaded = True

    # Category Costs — matches template defaults ($20/hr, $0 material)
    cat.category_costs = {
        "Scheduling & Planning":           {"material_cost": 0, "labor_hours": 2.0,  "hourly_rate": 20},
        "Documentation & Reporting":       {"material_cost": 0, "labor_hours": 1.5,  "hourly_rate": 20},
        "Validation & QA":                 {"material_cost": 0, "labor_hours": 3.0,  "hourly_rate": 20},
        "Process Compliance":              {"material_cost": 0, "labor_hours": 2.5,  "hourly_rate": 20},
        "Configuration & Data Mismatch":   {"material_cost": 0, "labor_hours": 4.0,  "hourly_rate": 20},
        "Site Readiness":                  {"material_cost": 0, "labor_hours": 8.0,  "hourly_rate": 20},
        "Communication & Response":        {"material_cost": 0, "labor_hours": 0.5,  "hourly_rate": 20},
        "Nesting & Tool Errors":           {"material_cost": 0, "labor_hours": 3.0,  "hourly_rate": 20},
        "Unclassified":                    {"material_cost": 0, "labor_hours": 0.2,  "hourly_rate": 20},
    }

    # Sub-Category Costs (a representative subset)
    cat.sub_category_costs = {
        "Site Readiness": {
            "BH Not Ready":  {"material_cost": 0, "labor_hours": 10.0, "hourly_rate": 20},
            "MW Not Ready":  {"material_cost": 0, "labor_hours": 8.0,  "hourly_rate": 20},
        },
        "Documentation & Reporting": {
            "Missing Snapshot": {"material_cost": 0, "labor_hours": 1.5, "hourly_rate": 20},
        },
        "Scheduling & Planning": {
            "No TI Entry": {"material_cost": 0, "labor_hours": 3.0, "hourly_rate": 20},
        },
    }

    # Keyword Patterns (sorted by priority)
    cat.keyword_costs = [
        {"pattern": ".*BH.*not.*actual.*",           "category_override": "Site Readiness",                "material_cost": 0, "labor_hours": 10, "priority": 1},
        {"pattern": ".*port.*matrix.*mismatch.*",    "category_override": "Configuration & Data Mismatch", "material_cost": 0, "labor_hours": 6,  "priority": 1},
        {"pattern": ".*snapshot.*missing.*",         "category_override": "Documentation & Reporting",     "material_cost": 0, "labor_hours": 1.5, "priority": 3},
    ]

    # Severity Multipliers
    cat.severity_multipliers = {
        "critical": 2.5,
        "high":     1.75,
        "medium":   1.25,
        "low":      1.0,
    }

    # Origin Premiums
    cat.origin_premiums = {
        "external":  0.20,
        "vendor":    0.15,
        "customer":  0.10,
        "process":   0.05,
        "technical": 0.0,
    }

    # Business Multipliers
    cat.business_multipliers = {
        "revenue_at_risk":    2.5,
        "opportunity_cost":   0.35,
        "customer_impact":    1.5,
        "sla_penalty":        0.2,
        "prevention_rate":    0.8,
        "cost_avoidance_rate": 0.7,
    }

    return cat


def _make_df(n_tickets, **overrides):
    """Build a DataFrame with n_tickets rows and sensible defaults.

    Any column can be overridden by passing it as a keyword argument.
    Scalar values are broadcast; list/array values must have length n_tickets.
    """
    base = {
        "Financial_Impact": [100.0] * n_tickets,
        "AI_Category": ["Process Compliance"] * n_tickets,
        "Severity_Norm": ["Medium"] * n_tickets,
        "Origin_Norm": ["Technical"] * n_tickets,
        "Issue_Date": pd.date_range("2025-01-01", periods=n_tickets, freq="D"),
        "Resolution_Days": [1.0] * n_tickets,
        "AI_Recurrence_Risk": [0.1] * n_tickets,
        "Similar_Tickets_Found": [0] * n_tickets,
        "Engineer_Assigned": ["Eng_A"] * n_tickets,
        "Ticket_ID": [f"TK-{i:04d}" for i in range(n_tickets)],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# =====================================================================
# 1. COST CALCULATION ACCURACY
# =====================================================================
class TestCostCalculationAccuracy:
    """Verify the costing formula:
    Total = (Material + Labor_Hours × Hourly_Rate) × Severity_Mult × (1 + Origin_Premium)
    """

    # --- 1 ticket ---------------------------------------------------

    def test_category_cost_medium_technical_1_ticket(self, catalog):
        """Scheduling & Planning, Medium, Technical → (0+2×20)×1.25×1.0 = 50.0"""
        result = catalog.calculate_financial_impact(
            category="Scheduling & Planning", severity="Medium", origin="Technical"
        )
        expected = (0 + 2.0 * 20) * 1.25 * (1 + 0.0)  # 50.0
        assert result["total_impact"] == pytest.approx(expected)

    def test_category_cost_critical_external_1_ticket(self, catalog):
        """Site Readiness, Critical, External → (0+8×20)×2.5×1.20 = 480.0"""
        result = catalog.calculate_financial_impact(
            category="Site Readiness", severity="Critical", origin="External"
        )
        expected = (0 + 8.0 * 20) * 2.5 * (1 + 0.20)  # 480.0
        assert result["total_impact"] == pytest.approx(expected)

    def test_category_cost_low_technical_1_ticket(self, catalog):
        """Communication & Response, Low, Technical → (0+0.5×20)×1.0×1.0 = 10.0"""
        result = catalog.calculate_financial_impact(
            category="Communication & Response", severity="Low", origin="Technical"
        )
        expected = (0 + 0.5 * 20) * 1.0 * (1 + 0.0)  # 10.0
        assert result["total_impact"] == pytest.approx(expected)

    def test_high_severity_vendor_origin(self, catalog):
        """Validation & QA, High, Vendor → (0+3×20)×1.75×1.15 = 120.75"""
        result = catalog.calculate_financial_impact(
            category="Validation & QA", severity="High", origin="Vendor"
        )
        expected = (0 + 3.0 * 20) * 1.75 * (1 + 0.15)  # 120.75
        assert result["total_impact"] == pytest.approx(expected)

    def test_medium_severity_customer_origin(self, catalog):
        """Config & Data Mismatch, Medium, Customer → (0+4×20)×1.25×1.10 = 110.0"""
        result = catalog.calculate_financial_impact(
            category="Configuration & Data Mismatch", severity="Medium", origin="Customer"
        )
        expected = (0 + 4.0 * 20) * 1.25 * (1 + 0.10)  # 110.0
        assert result["total_impact"] == pytest.approx(expected)

    def test_process_origin_premium(self, catalog):
        """Process Compliance, Critical, Process → (0+2.5×20)×2.5×1.05 = 131.25"""
        result = catalog.calculate_financial_impact(
            category="Process Compliance", severity="Critical", origin="Process"
        )
        expected = (0 + 2.5 * 20) * 2.5 * (1 + 0.05)  # 131.25
        assert result["total_impact"] == pytest.approx(expected)

    # --- 10 tickets (verify linearity) --------------------------------

    def test_10_tickets_sum(self, catalog):
        """10 identical tickets should produce 10× the single-ticket cost."""
        single = catalog.calculate_financial_impact(
            category="Scheduling & Planning", severity="Medium", origin="Technical"
        )["total_impact"]
        total = sum(
            catalog.calculate_financial_impact(
                category="Scheduling & Planning", severity="Medium", origin="Technical"
            )["total_impact"]
            for _ in range(10)
        )
        assert total == pytest.approx(single * 10)

    # --- 1000 tickets (stress) ----------------------------------------

    def test_1000_tickets_sum(self, catalog):
        """1000 identical tickets → 1000× single cost, no accumulation error."""
        single = catalog.calculate_financial_impact(
            category="Site Readiness", severity="Critical", origin="External"
        )["total_impact"]
        total = sum(
            catalog.calculate_financial_impact(
                category="Site Readiness", severity="Critical", origin="External"
            )["total_impact"]
            for _ in range(1000)
        )
        assert total == pytest.approx(single * 1000)

    # --- Breakdown fields are consistent ------------------------------

    def test_breakdown_fields_consistent(self, catalog):
        """Verify the breakdown fields add up to total_impact."""
        result = catalog.calculate_financial_impact(
            category="Configuration & Data Mismatch", severity="High", origin="Vendor"
        )
        recomputed = (
            (result["material_cost"] + result["labor_hours"] * result["hourly_rate"])
            * result["severity_multiplier"]
            * (1 + result["origin_premium"])
        )
        assert result["total_impact"] == pytest.approx(recomputed)

    def test_labor_cost_equals_hours_times_rate(self, catalog):
        """labor_cost should always equal labor_hours × hourly_rate."""
        result = catalog.calculate_financial_impact(
            category="Nesting & Tool Errors", severity="Low", origin="Technical"
        )
        assert result["labor_cost"] == pytest.approx(result["labor_hours"] * result["hourly_rate"])

    def test_base_cost_equals_material_plus_labor(self, catalog):
        """base_cost = material_cost + labor_cost."""
        result = catalog.calculate_financial_impact(
            category="Validation & QA", severity="Medium", origin="Process"
        )
        assert result["base_cost"] == pytest.approx(result["material_cost"] + result["labor_cost"])


# =====================================================================
# 2. PRICE CATALOG LOOKUP
# =====================================================================
class TestPriceCatalogLookup:
    """Test lookup priority: keyword → sub-category → category → Unclassified."""

    # --- Known categories return correct costs -------------------------

    def test_known_category_lookup(self, catalog):
        """Known category 'Site Readiness' uses 8hr labor."""
        result = catalog.calculate_financial_impact(
            category="Site Readiness", severity="Low", origin="Technical"
        )
        assert result["labor_hours"] == 8.0
        assert result["hourly_rate"] == 20
        assert "category:Site Readiness" in result["source"]

    def test_known_subcategory_overrides_category(self, catalog):
        """Sub-category 'BH Not Ready' overrides category-level 8hr → 10hr."""
        result = catalog.calculate_financial_impact(
            category="Site Readiness", sub_category="BH Not Ready",
            severity="Low", origin="Technical"
        )
        assert result["labor_hours"] == 10.0
        assert "sub_category:Site Readiness/BH Not Ready" in result["source"]

    def test_keyword_overrides_category_and_subcategory(self, catalog):
        """Keyword 'BH not actualized' overrides even when category is wrong."""
        result = catalog.calculate_financial_impact(
            category="Documentation & Reporting",
            severity="Low", origin="Technical",
            description="BH not actualized in MB"
        )
        assert result["labor_hours"] == 10
        assert "keyword:" in result["source"]

    def test_keyword_match_is_case_insensitive(self, catalog):
        """Keyword matching should be case-insensitive."""
        result = catalog.calculate_financial_impact(
            category="Unclassified", severity="Low", origin="Technical",
            description="PORT MATRIX MISMATCH found"
        )
        assert result["labor_hours"] == 6
        assert "keyword:" in result["source"]

    # --- Unknown categories fall back to Unclassified -----------------

    def test_unknown_category_falls_back_to_unclassified(self, catalog):
        """Unknown category uses the 'Unclassified' row."""
        result = catalog.calculate_financial_impact(
            category="Nonexistent Category", severity="Low", origin="Technical"
        )
        assert result["labor_hours"] == 0.2
        assert "fallback:Unclassified" in result["source"]

    def test_unknown_category_uses_default_hourly_rate(self, catalog):
        """Unclassified fallback still uses the catalog's hourly rate ($20)."""
        result = catalog.calculate_financial_impact(
            category="Made Up Category", severity="Medium", origin="Technical"
        )
        assert result["hourly_rate"] == 20

    # --- Empty / None inputs don't crash ------------------------------

    def test_empty_category_string(self, catalog):
        """Empty string category → Unclassified fallback."""
        result = catalog.calculate_financial_impact(
            category="", severity="Medium", origin="Technical"
        )
        assert result["total_impact"] >= 0
        assert "fallback:Unclassified" in result["source"] or "no_match" in result["source"]

    def test_empty_description(self, catalog):
        """Empty description → no keyword match, falls through to category."""
        result = catalog.calculate_financial_impact(
            category="Scheduling & Planning", severity="Low", origin="Technical",
            description=""
        )
        assert result["keyword_match"] is None
        assert "category:Scheduling & Planning" in result["source"]

    def test_none_description(self, catalog):
        """None description → no keyword match."""
        result = catalog.calculate_financial_impact(
            category="Scheduling & Planning", severity="Low", origin="Technical",
            description=None
        )
        assert result["keyword_match"] is None

    # --- Unknown severity / origin default gracefully -----------------

    def test_unknown_severity_defaults_to_1x(self, catalog):
        """Unknown severity level should use multiplier 1.0."""
        result = catalog.calculate_financial_impact(
            category="Scheduling & Planning", severity="Unknown", origin="Technical"
        )
        assert result["severity_multiplier"] == 1.0

    def test_unknown_origin_defaults_to_no_premium(self, catalog):
        """Unknown origin should use 0% premium."""
        result = catalog.calculate_financial_impact(
            category="Scheduling & Planning", severity="Low", origin="Alien"
        )
        assert result["origin_premium"] == 0.0

    # --- Business multiplier lookup -----------------------------------

    def test_business_multiplier_known(self, catalog):
        """Known multiplier 'revenue_at_risk' returns 2.5."""
        assert catalog.get_business_multiplier("revenue_at_risk") == 2.5

    def test_business_multiplier_case_insensitive(self, catalog):
        """Business multiplier lookup is case-insensitive."""
        assert catalog.get_business_multiplier("REVENUE_AT_RISK") == 2.5

    def test_business_multiplier_unknown_returns_default(self, catalog):
        """Unknown multiplier returns the provided default."""
        assert catalog.get_business_multiplier("nonexistent", default=99.9) == 99.9

    # --- Benchmark costs ----------------------------------------------

    def test_get_benchmark_costs_returns_positive(self, catalog):
        """Benchmark costs should all be non-negative."""
        benchmarks = catalog.get_benchmark_costs()
        assert benchmarks["avg_per_ticket"] >= 0
        assert benchmarks["best_in_class"] >= 0
        assert benchmarks["laggard"] >= 0
        assert benchmarks["hourly_rate"] > 0


# =====================================================================
# 3. SLA CALCULATIONS
# =====================================================================
class TestSLACalculations:
    """Test SLA penalty exposure from _calculate_sla_penalty()."""

    def test_sla_penalty_critical_tickets_only(self):
        """Only Critical-severity tickets contribute to SLA penalty."""
        df = _make_df(
            4,
            Financial_Impact=[100, 200, 300, 400],
            Severity_Norm=["Critical", "High", "Medium", "Low"],
        )
        with patch("escalation_ai.financial.metrics._get_biz_multiplier", return_value=0.2):
            penalty = _calculate_sla_penalty(df)
        # Only the Critical ticket ($100) × 0.2 penalty rate = $20
        assert penalty == pytest.approx(20.0)

    def test_sla_penalty_multiple_critical(self):
        """Multiple Critical tickets sum before applying penalty rate."""
        df = _make_df(
            3,
            Financial_Impact=[100, 200, 300],
            Severity_Norm=["Critical", "Critical", "High"],
        )
        with patch("escalation_ai.financial.metrics._get_biz_multiplier", return_value=0.2):
            penalty = _calculate_sla_penalty(df)
        # Critical cost = 100 + 200 = 300; penalty = 300 × 0.2 = 60
        assert penalty == pytest.approx(60.0)

    def test_sla_penalty_no_critical_tickets(self):
        """No Critical tickets → $0 penalty."""
        df = _make_df(3, Severity_Norm=["High", "Medium", "Low"])
        with patch("escalation_ai.financial.metrics._get_biz_multiplier", return_value=0.2):
            penalty = _calculate_sla_penalty(df)
        assert penalty == pytest.approx(0.0)

    def test_sla_penalty_missing_severity_column(self):
        """Missing Severity_Norm column → $0 penalty (no crash)."""
        df = _make_df(3)
        df = df.drop(columns=["Severity_Norm"])
        penalty = _calculate_sla_penalty(df)
        assert penalty == 0.0

    def test_sla_penalty_missing_financial_impact_column(self):
        """Missing Financial_Impact column → $0 penalty."""
        df = pd.DataFrame({"Severity_Norm": ["Critical", "High"]})
        penalty = _calculate_sla_penalty(df)
        assert penalty == 0.0


# =====================================================================
# 4. AGGREGATION TESTS
# =====================================================================
class TestAggregation:
    """Test that summary statistics are correct from a known DataFrame."""

    def test_total_cost_sum(self):
        """total_cost should equal sum of Financial_Impact."""
        df = _make_df(5, Financial_Impact=[100, 200, 300, 400, 500])
        metrics = calculate_financial_metrics(df)
        assert metrics.total_cost == pytest.approx(1500.0)

    def test_total_tickets_count(self):
        """total_tickets should equal number of rows."""
        df = _make_df(7)
        metrics = calculate_financial_metrics(df)
        assert metrics.total_tickets == 7

    def test_avg_cost_per_ticket(self):
        """avg_cost_per_ticket = total / count."""
        df = _make_df(4, Financial_Impact=[100, 200, 300, 400])
        metrics = calculate_financial_metrics(df)
        assert metrics.avg_cost_per_ticket == pytest.approx(250.0)

    def test_median_cost(self):
        """Median of [100, 200, 300, 400, 500] = 300."""
        df = _make_df(5, Financial_Impact=[100, 200, 300, 400, 500])
        metrics = calculate_financial_metrics(df)
        assert metrics.median_cost == pytest.approx(300.0)

    def test_cost_by_category(self):
        """Cost grouped by AI_Category sums correctly."""
        df = _make_df(
            4,
            Financial_Impact=[100, 200, 150, 250],
            AI_Category=["Process Compliance", "Process Compliance",
                         "Site Readiness", "Site Readiness"],
        )
        metrics = calculate_financial_metrics(df)
        assert metrics.cost_by_category["Process Compliance"] == pytest.approx(300.0)
        assert metrics.cost_by_category["Site Readiness"] == pytest.approx(400.0)

    def test_cost_by_severity(self):
        """Cost grouped by Severity_Norm sums correctly."""
        df = _make_df(
            3,
            Financial_Impact=[100, 200, 300],
            Severity_Norm=["Critical", "Critical", "Low"],
        )
        metrics = calculate_financial_metrics(df)
        assert metrics.cost_by_severity["Critical"] == pytest.approx(300.0)
        assert metrics.cost_by_severity["Low"] == pytest.approx(300.0)

    def test_recurring_cost_threshold(self):
        """Only tickets with AI_Recurrence_Risk > 0.3 count as recurring."""
        df = _make_df(
            4,
            Financial_Impact=[100, 200, 300, 400],
            AI_Recurrence_Risk=[0.1, 0.5, 0.3, 0.8],
        )
        recurring = _calculate_recurring_cost(df)
        # Tickets with risk > 0.3: index 1 ($200) and index 3 ($400) = $600
        assert recurring == pytest.approx(600.0)

    def test_preventable_cost_categories(self):
        """Preventable categories: Process Compliance, Comm, Config, Scheduling, Doc."""
        df = _make_df(
            5,
            Financial_Impact=[100, 200, 300, 400, 500],
            AI_Category=[
                "Process Compliance",
                "Communication & Response",
                "Site Readiness",          # NOT preventable
                "Scheduling & Planning",
                "Validation & QA",         # NOT preventable
            ],
        )
        preventable = _calculate_preventable_cost(df)
        # Preventable: 100 + 200 + 400 = 700
        assert preventable == pytest.approx(700.0)

    def test_risk_exposure_weights(self):
        """Risk exposure applies severity-dependent weights."""
        df = _make_df(
            4,
            Financial_Impact=[1000, 1000, 1000, 1000],
            Severity_Norm=["Critical", "High", "Medium", "Low"],
        )
        exposure = _calculate_risk_exposure(df)
        # 1000×1.0 + 1000×0.7 + 1000×0.4 + 1000×0.2 = 2300
        assert exposure == pytest.approx(2300.0)

    def test_recurrence_exposure(self):
        """Recurrence exposure = sum(Financial_Impact × AI_Recurrence_Risk)."""
        df = _make_df(
            3,
            Financial_Impact=[100, 200, 300],
            AI_Recurrence_Risk=[0.5, 0.0, 1.0],
        )
        exposure = _calculate_recurrence_exposure(df)
        # 100×0.5 + 200×0.0 + 300×1.0 = 350
        assert exposure == pytest.approx(350.0)

    def test_customer_impact_cost(self):
        """Customer impact: External+Customer tickets × customer_impact multiplier."""
        df = _make_df(
            4,
            Financial_Impact=[100, 200, 300, 400],
            Origin_Norm=["External", "Customer", "Technical", "Process"],
        )
        with patch("escalation_ai.financial.metrics._get_biz_multiplier", return_value=1.5):
            impact = _calculate_customer_impact(df)
        # External($100) + Customer($200) = $300 × 1.5 = $450
        assert impact == pytest.approx(450.0)

    def test_cost_avoidance_four_levers(self):
        """calculate_cost_avoidance returns dict with all four levers."""
        df = _make_df(
            10,
            AI_Category=[
                "Process Compliance", "Scheduling & Planning",
                "Documentation & Reporting", "Nesting & Tool Errors",
                "Communication & Response", "Site Readiness",
                "Validation & QA", "Configuration & Data Mismatch",
                "Process Compliance", "Scheduling & Planning",
            ],
            AI_Recurrence_Risk=[0.5] * 10,
            Similar_Tickets_Found=[5] * 10,
        )
        avoidance = calculate_cost_avoidance(df)
        assert "recurring_issues" in avoidance
        assert "preventable_categories" in avoidance
        assert "knowledge_sharing" in avoidance
        assert "automation" in avoidance
        assert "total_avoidance" in avoidance
        assert avoidance["total_avoidance"] == pytest.approx(
            avoidance["recurring_issues"]
            + avoidance["preventable_categories"]
            + avoidance["knowledge_sharing"]
            + avoidance["automation"]
        )

    def test_efficiency_metrics_structure(self):
        """calculate_efficiency_metrics returns expected keys."""
        df = _make_df(5)
        result = calculate_efficiency_metrics(df)
        assert "cost_per_hour" in result
        assert "cost_per_engineer" in result
        assert "cost_per_category" in result
        assert "engineer_efficiency_scores" in result
        assert "outliers" in result

    def test_monthly_trend(self):
        """Monthly trend groups costs by month correctly."""
        dates = pd.to_datetime(["2025-01-15", "2025-01-20", "2025-02-10"])
        df = _make_df(
            3,
            Financial_Impact=[100, 200, 300],
            Issue_Date=dates,
        )
        trend = _calculate_monthly_trend(df)
        assert trend.get("2025-01") == pytest.approx(300.0)
        assert trend.get("2025-02") == pytest.approx(300.0)

    def test_roi_metrics_structure(self):
        """calculate_roi_metrics returns expected keys and positive values."""
        df = _make_df(
            20,
            Financial_Impact=[50 + i * 10 for i in range(20)],
            AI_Category=["Process Compliance"] * 10 + ["Site Readiness"] * 10,
            Issue_Date=pd.date_range("2025-01-01", periods=20, freq="5D"),
        )
        roi = calculate_roi_metrics(df)
        assert "total_investment_required" in roi
        assert "expected_annual_savings" in roi
        assert "roi_percentage" in roi
        assert "top_opportunities" in roi
        assert len(roi["top_opportunities"]) > 0

    def test_roi_annualization_factor(self):
        """Annualization factor = 365 / data_span_days."""
        df = _make_df(
            10,
            AI_Category=["Process Compliance"] * 10,
            Issue_Date=pd.date_range("2025-01-01", periods=10, freq="10D"),
        )
        roi = calculate_roi_metrics(df)
        # 10 tickets over 90 days → factor = 365/90 ≈ 4.06
        assert roi["data_span_days"] == 90
        assert roi["annualization_factor"] == pytest.approx(365 / 90, abs=0.01)


# =====================================================================
# 5. CURRENCY / ROUNDING
# =====================================================================
class TestCurrencyRounding:
    """Verify financial outputs are rounded to 2 decimal places."""

    def test_total_impact_rounded_to_2dp(self, catalog):
        """total_impact should have at most 2 decimal places."""
        result = catalog.calculate_financial_impact(
            category="Validation & QA", severity="High", origin="Vendor"
        )
        # (0+3×20)×1.75×1.15 = 120.75  — exact
        assert result["total_impact"] == round(result["total_impact"], 2)

    def test_material_cost_rounded(self, catalog):
        """material_cost field is rounded to 2dp."""
        result = catalog.calculate_financial_impact(
            category="Site Readiness", severity="Critical", origin="External"
        )
        assert result["material_cost"] == round(result["material_cost"], 2)

    def test_labor_cost_rounded(self, catalog):
        """labor_cost field is rounded to 2dp."""
        result = catalog.calculate_financial_impact(
            category="Site Readiness", severity="Critical", origin="External"
        )
        assert result["labor_cost"] == round(result["labor_cost"], 2)

    def test_base_cost_rounded(self, catalog):
        """base_cost field is rounded to 2dp."""
        result = catalog.calculate_financial_impact(
            category="Site Readiness", severity="Critical", origin="External"
        )
        assert result["base_cost"] == round(result["base_cost"], 2)

    def test_no_floating_point_artifacts(self, catalog):
        """Ensure no artifacts like $1234.5600000001."""
        result = catalog.calculate_financial_impact(
            category="Configuration & Data Mismatch", severity="High", origin="Customer"
        )
        impact_str = f"{result['total_impact']:.10f}"
        # After 2 decimal places, everything should be zeros
        decimal_part = impact_str.split(".")[1]
        assert decimal_part[2:] == "0" * 8, f"Floating-point artifact detected: {result['total_impact']}"

    def test_all_categories_round_cleanly(self, catalog):
        """Every category's cost should round to 2dp without artifacts."""
        for cat in catalog.category_costs:
            for sev in ["Critical", "High", "Medium", "Low"]:
                for origin in ["External", "Vendor", "Customer", "Process", "Technical"]:
                    result = catalog.calculate_financial_impact(
                        category=cat, severity=sev, origin=origin
                    )
                    assert result["total_impact"] == round(result["total_impact"], 2), (
                        f"Rounding issue for {cat}/{sev}/{origin}: {result['total_impact']}"
                    )


# =====================================================================
# 6. ZERO / EDGE CASES
# =====================================================================
class TestZeroEdgeCases:
    """Test with zero tickets, all-one-category, negatives, and empty data."""

    # --- Zero tickets → $0, not error ---------------------------------

    def test_empty_dataframe_returns_zero_metrics(self):
        """Empty DataFrame → all-zero FinancialMetrics, no crash."""
        df = pd.DataFrame()
        metrics = calculate_financial_metrics(df)
        assert metrics.total_cost == 0.0
        assert metrics.total_tickets == 0
        assert metrics.avg_cost_per_ticket == 0.0

    def test_empty_df_missing_financial_impact(self):
        """DataFrame without Financial_Impact column → zero metrics."""
        df = pd.DataFrame({"some_column": [1, 2, 3]})
        metrics = calculate_financial_metrics(df)
        assert metrics.total_cost == 0.0

    def test_empty_df_roi_metrics(self):
        """Empty DataFrame → zero ROI analysis."""
        df = pd.DataFrame()
        roi = calculate_roi_metrics(df)
        assert roi["total_investment_required"] == 0.0
        assert roi["top_opportunities"] == []

    def test_empty_df_cost_avoidance(self):
        """Empty DataFrame → zero cost avoidance."""
        df = pd.DataFrame()
        avoidance = calculate_cost_avoidance(df)
        assert avoidance["total_avoidance"] == 0.0

    def test_empty_df_efficiency_metrics(self):
        """Empty DataFrame → zero efficiency metrics."""
        df = pd.DataFrame()
        efficiency = calculate_efficiency_metrics(df)
        assert efficiency["cost_per_hour"] == 0.0
        assert efficiency["outliers"] == []

    def test_empty_df_recurring_cost(self):
        """Empty DataFrame → $0 recurring cost."""
        df = pd.DataFrame()
        assert _calculate_recurring_cost(df) == 0.0

    def test_empty_df_preventable_cost(self):
        """Empty DataFrame → $0 preventable cost."""
        df = pd.DataFrame()
        assert _calculate_preventable_cost(df) == 0.0

    # --- All tickets in one category ----------------------------------

    def test_all_one_category(self):
        """All tickets in one category → cost_by_category has one entry."""
        df = _make_df(10, AI_Category=["Site Readiness"] * 10)
        metrics = calculate_financial_metrics(df)
        assert len(metrics.cost_by_category) == 1
        assert "Site Readiness" in metrics.cost_by_category
        assert metrics.cost_by_category["Site Readiness"] == pytest.approx(1000.0)

    def test_all_one_severity(self):
        """All tickets Critical → critical_cost_ratio = 1.0."""
        df = _make_df(5, Severity_Norm=["Critical"] * 5)
        metrics = calculate_financial_metrics(df)
        assert metrics.critical_cost_ratio == pytest.approx(1.0)

    # --- Single ticket ------------------------------------------------

    def test_single_ticket(self):
        """Single ticket should produce valid metrics."""
        df = _make_df(1, Financial_Impact=[250.0])
        metrics = calculate_financial_metrics(df)
        assert metrics.total_cost == pytest.approx(250.0)
        assert metrics.total_tickets == 1
        assert metrics.avg_cost_per_ticket == pytest.approx(250.0)
        assert metrics.median_cost == pytest.approx(250.0)

    # --- Negative values ----------------------------------------------

    def test_negative_financial_impact_in_catalog(self, catalog):
        """Negative total_impact is clamped to $0 by the catalog."""
        # Inject a category with negative material to force negative base cost
        catalog.category_costs["Negative Test"] = {
            "material_cost": -1000, "labor_hours": 0, "hourly_rate": 20
        }
        result = catalog.calculate_financial_impact(
            category="Negative Test", severity="Low", origin="Technical"
        )
        assert result["total_impact"] >= 0, "Negative cost should be clamped to 0"

    def test_negative_values_in_dataframe(self):
        """Negative Financial_Impact values should not crash aggregation."""
        df = _make_df(3, Financial_Impact=[-100, 200, 300])
        metrics = calculate_financial_metrics(df)
        # Aggregation just sums — total = 400
        assert metrics.total_cost == pytest.approx(400.0)
        assert metrics.total_tickets == 3

    # --- Zero Financial_Impact ----------------------------------------

    def test_zero_financial_impact_tickets(self):
        """Tickets with $0 impact should be counted but not crash ratios."""
        df = _make_df(3, Financial_Impact=[0, 0, 0])
        metrics = calculate_financial_metrics(df)
        assert metrics.total_cost == pytest.approx(0.0)
        assert metrics.total_tickets == 3
        assert metrics.avg_cost_per_ticket == pytest.approx(0.0)
        # Ratios involving division by total_cost should not error
        assert metrics.cost_concentration_ratio == 0.0
        assert metrics.critical_cost_ratio == 0.0

    # --- Efficiency score bounds --------------------------------------

    def test_efficiency_score_bounds(self):
        """Efficiency score must be between 0 and 100."""
        df = _make_df(
            10,
            Financial_Impact=[5000] * 10,  # Very high cost to trigger penalties
            Severity_Norm=["Critical"] * 10,
            AI_Recurrence_Risk=[0.9] * 10,
        )
        metrics = calculate_financial_metrics(df)
        assert 0 <= metrics.cost_efficiency_score <= 100

    def test_efficiency_score_perfect_when_below_target(self):
        """Low-cost, low-risk tickets should score near 100."""
        df = _make_df(
            10,
            Financial_Impact=[10] * 10,  # Well below $500 target
            Severity_Norm=["Low"] * 10,
            AI_Recurrence_Risk=[0.0] * 10,
        )
        metrics = calculate_financial_metrics(df)
        assert metrics.cost_efficiency_score >= 90

    # --- Forecast edge cases ------------------------------------------

    def test_forecast_insufficient_data(self):
        """Fewer than 5 data points → falls back to mean × days."""
        df = _make_df(
            3,
            Financial_Impact=[100, 200, 300],
            Issue_Date=pd.date_range("2025-01-01", periods=3, freq="D"),
        )
        forecast = _forecast_costs(df, days=30)
        expected = 200.0 * 30  # mean(100,200,300) × 30
        assert forecast == pytest.approx(expected)

    def test_forecast_same_day_returns_zero(self):
        """All tickets on same day → cannot extrapolate → $0."""
        df = _make_df(
            10,
            Issue_Date=[pd.Timestamp("2025-06-01")] * 10,
        )
        forecast = _forecast_costs(df, days=30)
        assert forecast == pytest.approx(0.0)

    def test_cost_velocity_single_point(self):
        """Single data point → velocity = 0."""
        df = _make_df(1)
        velocity = _calculate_cost_velocity(df)
        assert velocity == pytest.approx(0.0)

    # --- Financial forecasts structure --------------------------------

    def test_financial_forecasts_structure(self):
        """calculate_financial_forecasts returns expected keys."""
        df = _make_df(
            30,
            Issue_Date=pd.date_range("2025-01-01", periods=30, freq="D"),
        )
        forecasts = calculate_financial_forecasts(df)
        assert "monthly_projection" in forecasts
        assert "annual_projection" in forecasts
        assert "trend" in forecasts
        assert "confidence" in forecasts
        assert "risk_scenarios" in forecasts

    def test_financial_forecasts_empty_df(self):
        """Empty DataFrame → forecasts with zero projections."""
        df = pd.DataFrame()
        forecasts = calculate_financial_forecasts(df)
        assert forecasts["annual_projection"] == 0.0
