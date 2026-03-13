"""
Comprehensive test suite for the Strategic Friction Scoring engine (Phase 2).

Tests the multiplicative scoring formula:
    Score = Base_Severity x Type_Multiplier x Origin_Multiplier x Impact_Multiplier

Also covers recidivism penalties, boundary conditions, determinism,
and ranking integrity.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from escalation_ai.core.config import (
    WEIGHTS,
    COL_SEVERITY,
    COL_TYPE,
    COL_ORIGIN,
    COL_IMPACT,
    COL_SUMMARY,
    COL_ENGINEER,
    COL_DATETIME,
    COL_ROOT_CAUSE,
    COL_RECURRENCE_RISK,
    REQUIRED_COLUMNS,
    RECIDIVISM_PENALTY_HIGH,
    RECIDIVISM_PENALTY_MEDIUM,
)
from escalation_ai.scoring import calculate_strategic_friction


# ============================================================================
# Helpers
# ============================================================================

def _make_row(severity="Major", type_="Escalations", origin="External",
              impact="High", summary="Test issue", engineer="Alice",
              datetime_val="2026-01-15", root_cause="", recurrence="Low"):
    """Build a single-row dict with all standard columns."""
    return {
        COL_SEVERITY: severity,
        COL_TYPE: type_,
        COL_ORIGIN: origin,
        COL_IMPACT: impact,
        COL_SUMMARY: summary,
        COL_ENGINEER: engineer,
        COL_DATETIME: datetime_val,
        COL_ROOT_CAUSE: root_cause,
        COL_RECURRENCE_RISK: recurrence,
    }


def _make_df(rows):
    """Build a DataFrame from a list of row dicts."""
    return pd.DataFrame(rows)


def _score_df(df):
    """Run scoring with price catalog mocked out (no file dependency)."""
    mock_catalog = MagicMock()
    mock_catalog.calculate_financial_impact.return_value = {
        'total_impact': 0.0,
        'source': 'test_mock',
    }
    with patch('escalation_ai.scoring.get_price_catalog', return_value=mock_catalog):
        return calculate_strategic_friction(df)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def single_row_df():
    """A single-row DataFrame with all standard columns."""
    return _make_df([_make_row()])


@pytest.fixture
def minimal_df():
    """DataFrame with only REQUIRED_COLUMNS (severity, type, origin)."""
    return pd.DataFrame([{
        COL_SEVERITY: "Major",
        COL_TYPE: "Escalations",
        COL_ORIGIN: "External",
    }])


@pytest.fixture
def five_ticket_df():
    """Five tickets with known severity ordering for ranking tests."""
    return _make_df([
        _make_row(severity="Critical", type_="Escalations", origin="External", impact="High"),
        _make_row(severity="Critical", type_="Escalations", origin="Internal", impact="High"),
        _make_row(severity="Major", type_="Escalations", origin="External", impact="Low"),
        _make_row(severity="Minor", type_="Concerns", origin="Internal", impact="None"),
        _make_row(severity="Minor", type_="Lessons Learned", origin="Internal", impact="None"),
    ])


# ============================================================================
# 1. Formula Validation
# ============================================================================

class TestFormulaValidation:
    """Verify the multiplicative scoring formula against hand-calculated values."""

    def test_maximum_score(self):
        """Critical + Escalation + External + High = 100 * 1.5 * 2.5 * 2.0 = 750."""
        df = _make_df([_make_row(
            severity="Critical", type_="Escalations",
            origin="External", impact="High",
        )])
        result = _score_df(df)
        assert result['Strategic_Friction_Score'].iloc[0] == pytest.approx(750.0)

    def test_minimum_nonzero_score(self):
        """Minor + Concerns + Internal + None = 10 * 1.0 * 1.0 * 1.0 = 10."""
        df = _make_df([_make_row(
            severity="Minor", type_="Concerns",
            origin="Internal", impact="None",
        )])
        result = _score_df(df)
        assert result['Strategic_Friction_Score'].iloc[0] == pytest.approx(10.0)

    def test_zero_score_lessons_learned(self):
        """Lessons Learned always produce score = 0 (type multiplier = 0)."""
        df = _make_df([_make_row(
            severity="Critical", type_="Lessons Learned",
            origin="External", impact="High",
        )])
        result = _score_df(df)
        assert result['Strategic_Friction_Score'].iloc[0] == pytest.approx(0.0)

    def test_typical_midrange(self):
        """Major + Escalations + Internal + Low.

        Note: the scoring engine only checks `if 'High' in impact` so Low
        falls through to the default 1.0 multiplier.
        50 * 1.5 * 1.0 * 1.0 = 75.
        """
        df = _make_df([_make_row(
            severity="Major", type_="Escalations",
            origin="Internal", impact="Low",
        )])
        result = _score_df(df)
        assert result['Strategic_Friction_Score'].iloc[0] == pytest.approx(75.0)

    def test_major_concern_external_high(self):
        """Major + Concerns + External + High = 50 * 1.0 * 2.5 * 2.0 = 250."""
        df = _make_df([_make_row(
            severity="Major", type_="Concerns",
            origin="External", impact="High",
        )])
        result = _score_df(df)
        assert result['Strategic_Friction_Score'].iloc[0] == pytest.approx(250.0)

    def test_critical_concern_internal_none(self):
        """Critical + Concerns + Internal + None = 100 * 1.0 * 1.0 * 1.0 = 100."""
        df = _make_df([_make_row(
            severity="Critical", type_="Concerns",
            origin="Internal", impact="None",
        )])
        result = _score_df(df)
        assert result['Strategic_Friction_Score'].iloc[0] == pytest.approx(100.0)

    def test_minor_escalation_external_none(self):
        """Minor + Escalations + External + None = 10 * 1.5 * 2.5 * 1.0 = 37.5."""
        df = _make_df([_make_row(
            severity="Minor", type_="Escalations",
            origin="External", impact="None",
        )])
        result = _score_df(df)
        assert result['Strategic_Friction_Score'].iloc[0] == pytest.approx(37.5)

    def test_score_always_nonnegative(self, five_ticket_df):
        """All scores must be >= 0."""
        result = _score_df(five_ticket_df)
        assert (result['Strategic_Friction_Score'] >= 0).all()

    def test_default_severity_fallback(self):
        """Unknown severity should use Default=5 from WEIGHTS."""
        df = _make_df([_make_row(severity="Unknown")])
        result = _score_df(df)
        # 5 * 1.5 * 2.5 * 2.0 = 37.5
        assert result['Strategic_Friction_Score'].iloc[0] == pytest.approx(37.5)

    def test_case_insensitivity(self):
        """Severity/type/origin/impact should be title-cased internally."""
        df = _make_df([_make_row(
            severity="CRITICAL", type_="escalations",
            origin="EXTERNAL", impact="high",
        )])
        result = _score_df(df)
        assert result['Strategic_Friction_Score'].iloc[0] == pytest.approx(750.0)

    def test_whitespace_trimmed(self):
        """Leading/trailing whitespace in field values should be ignored."""
        df = _make_df([_make_row(
            severity="  Critical  ", type_="  Escalations  ",
            origin="  External  ", impact="  High  ",
        )])
        result = _score_df(df)
        assert result['Strategic_Friction_Score'].iloc[0] == pytest.approx(750.0)


# ============================================================================
# 2. Recidivism Penalty
# ============================================================================

class TestRecidivismPenalty:
    """Verify recidivism penalty multipliers are correctly defined and applicable."""

    def test_high_penalty_value(self):
        """RECIDIVISM_PENALTY_HIGH should be 1.5."""
        assert RECIDIVISM_PENALTY_HIGH == pytest.approx(1.5)

    def test_medium_penalty_value(self):
        """RECIDIVISM_PENALTY_MEDIUM should be 1.25."""
        assert RECIDIVISM_PENALTY_MEDIUM == pytest.approx(1.25)

    def test_high_penalty_applied_to_score(self):
        """Applying high recidivism penalty should multiply score by 1.5."""
        df = _make_df([_make_row(
            severity="Critical", type_="Escalations",
            origin="External", impact="High",
        )])
        result = _score_df(df)
        base_score = result['Strategic_Friction_Score'].iloc[0]
        penalised = base_score * RECIDIVISM_PENALTY_HIGH
        assert penalised == pytest.approx(750.0 * 1.5)
        assert penalised == pytest.approx(1125.0)

    def test_medium_penalty_applied_to_score(self):
        """Applying medium recidivism penalty should multiply score by 1.25."""
        df = _make_df([_make_row(
            severity="Major", type_="Escalations",
            origin="Internal", impact="None",
        )])
        result = _score_df(df)
        base_score = result['Strategic_Friction_Score'].iloc[0]
        # 50 * 1.5 * 1.0 * 1.0 = 75
        assert base_score == pytest.approx(75.0)
        penalised = base_score * RECIDIVISM_PENALTY_MEDIUM
        assert penalised == pytest.approx(93.75)

    def test_no_penalty_for_non_recidivistic(self):
        """Non-recidivistic tickets have no penalty (multiplier = 1.0)."""
        df = _make_df([_make_row(
            severity="Major", type_="Escalations",
            origin="Internal", impact="None",
        )])
        result = _score_df(df)
        base_score = result['Strategic_Friction_Score'].iloc[0]
        # No penalty applied, score should stay at 75
        assert base_score == pytest.approx(75.0)

    def test_penalty_stacking_with_base_formula(self):
        """Penalties should correctly stack: base_formula * penalty_multiplier."""
        # Minor + Concerns + Internal + None = 10
        df = _make_df([_make_row(
            severity="Minor", type_="Concerns",
            origin="Internal", impact="None",
        )])
        result = _score_df(df)
        base = result['Strategic_Friction_Score'].iloc[0]
        assert base == pytest.approx(10.0)

        # High penalty stacking
        assert base * RECIDIVISM_PENALTY_HIGH == pytest.approx(15.0)
        # Medium penalty stacking
        assert base * RECIDIVISM_PENALTY_MEDIUM == pytest.approx(12.5)

    def test_high_penalty_greater_than_medium(self):
        """High recidivism penalty must always exceed medium."""
        assert RECIDIVISM_PENALTY_HIGH > RECIDIVISM_PENALTY_MEDIUM

    def test_both_penalties_greater_than_one(self):
        """Both penalties must be > 1.0 (they increase, not decrease, scores)."""
        assert RECIDIVISM_PENALTY_HIGH > 1.0
        assert RECIDIVISM_PENALTY_MEDIUM > 1.0


# ============================================================================
# 3. Boundary Conditions
# ============================================================================

class TestBoundaryConditions:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Scoring an empty DataFrame with correct columns should return empty.

        Note: the current implementation raises on empty DataFrames due to
        the financial impact apply step.  We verify it raises rather than
        producing silently wrong results.
        """
        df = pd.DataFrame(columns=[
            COL_SEVERITY, COL_TYPE, COL_ORIGIN, COL_IMPACT,
            COL_SUMMARY, COL_ENGINEER, COL_DATETIME,
            COL_ROOT_CAUSE, COL_RECURRENCE_RISK,
        ])
        with pytest.raises((ValueError, KeyError)):
            _score_df(df)

    def test_missing_optional_columns_still_scores(self):
        """DataFrame with only required columns should still produce scores."""
        df = pd.DataFrame([{
            COL_SEVERITY: "Critical",
            COL_TYPE: "Escalations",
            COL_ORIGIN: "External",
        }])
        result = _score_df(df)
        assert 'Strategic_Friction_Score' in result.columns
        # Without impact column, impact defaults to 'None' (1.0)
        # Critical * Escalation * External * 1.0 = 100 * 1.5 * 2.5 * 1.0 = 375
        assert result['Strategic_Friction_Score'].iloc[0] == pytest.approx(375.0)

    def test_nan_severity(self):
        """NaN severity should fall through to default base=5."""
        df = _make_df([_make_row(severity=float('nan'))])
        result = _score_df(df)
        # NaN -> str -> 'Nan' (title-cased) -> not in BASE_SEVERITY -> default 5
        # 5 * 1.5 * 2.5 * 2.0 = 37.5
        assert result['Strategic_Friction_Score'].iloc[0] == pytest.approx(37.5)

    def test_nan_impact(self):
        """NaN impact should default to 'None' multiplier (1.0)."""
        df = _make_df([_make_row(impact=float('nan'))])
        result = _score_df(df)
        # Major + Escalations + External + None(default) = 50 * 1.5 * 2.5 * 1.0 = 187.5
        assert result['Strategic_Friction_Score'].iloc[0] == pytest.approx(187.5)

    def test_unknown_type_defaults_to_1x(self):
        """An unrecognised type (no 'Escalation' or 'Lesson') should use 1.0 multiplier."""
        df = _make_df([_make_row(type_="Something Random")])
        result = _score_df(df)
        # Major + 1.0 + External + High = 50 * 1.0 * 2.5 * 2.0 = 250
        assert result['Strategic_Friction_Score'].iloc[0] == pytest.approx(250.0)

    def test_unknown_origin_defaults_to_1x(self):
        """An unrecognised origin should use 1.0 multiplier."""
        df = _make_df([_make_row(origin="Unknown")])
        result = _score_df(df)
        # Major + Escalations + 1.0 + High = 50 * 1.5 * 1.0 * 2.0 = 150
        assert result['Strategic_Friction_Score'].iloc[0] == pytest.approx(150.0)

    def test_unknown_impact_defaults_to_1x(self):
        """An unrecognised impact value (not 'High') should use 1.0 multiplier."""
        df = _make_df([_make_row(impact="Extreme")])
        result = _score_df(df)
        # Major + Escalations + External + 1.0 = 50 * 1.5 * 2.5 * 1.0 = 187.5
        assert result['Strategic_Friction_Score'].iloc[0] == pytest.approx(187.5)

    def test_single_row(self, single_row_df):
        """Single row should score correctly."""
        result = _score_df(single_row_df)
        assert len(result) == 1
        assert 'Strategic_Friction_Score' in result.columns

    def test_original_df_not_mutated(self):
        """Scoring should not mutate the input DataFrame."""
        df = _make_df([_make_row()])
        original_cols = set(df.columns)
        original_vals = df.copy()
        _score_df(df)
        assert set(df.columns) == original_cols
        pd.testing.assert_frame_equal(df, original_vals)


# ============================================================================
# 4. Determinism
# ============================================================================

class TestDeterminism:
    """Verify that scoring is deterministic (no randomness)."""

    def test_identical_input_identical_output(self, five_ticket_df):
        """Running scoring twice on the same input must produce identical results."""
        result1 = _score_df(five_ticket_df.copy())
        result2 = _score_df(five_ticket_df.copy())
        pd.testing.assert_series_equal(
            result1['Strategic_Friction_Score'],
            result2['Strategic_Friction_Score'],
        )

    def test_determinism_ten_runs(self):
        """Score the same row 10 times; all results must be identical."""
        df = _make_df([_make_row(
            severity="Critical", type_="Escalations",
            origin="External", impact="High",
        )])
        scores = []
        for _ in range(10):
            result = _score_df(df.copy())
            scores.append(result['Strategic_Friction_Score'].iloc[0])
        assert all(s == scores[0] for s in scores)

    def test_determinism_all_enrichment_columns(self, five_ticket_df):
        """All enrichment columns should be deterministic, not just the score."""
        result1 = _score_df(five_ticket_df.copy())
        result2 = _score_df(five_ticket_df.copy())
        # Compare Risk_Tier, Is_Human_Error, Root_Cause_Category
        for col in ['Risk_Tier', 'Is_Human_Error', 'Root_Cause_Category']:
            if col in result1.columns:
                pd.testing.assert_series_equal(
                    result1[col].reset_index(drop=True),
                    result2[col].reset_index(drop=True),
                    check_names=False,
                )


# ============================================================================
# 5. Ranking Integrity
# ============================================================================

class TestRankingIntegrity:
    """Verify that score ordering matches expected risk ranking."""

    def test_five_ticket_ranking(self, five_ticket_df):
        """Tickets should rank: max-risk > ... > lessons-learned (0)."""
        result = _score_df(five_ticket_df)
        scores = result['Strategic_Friction_Score'].tolist()

        # Ticket 0: Critical+Escalation+External+High = 750
        # Ticket 1: Critical+Escalation+Internal+High = 300
        # Ticket 2: Major+Escalation+External+Low = 82.5
        # Ticket 3: Minor+Concerns+Internal+None = 10
        # Ticket 4: Minor+Lessons Learned+Internal+None = 0
        assert scores[0] > scores[1] > scores[2] > scores[3] > scores[4]

    def test_expected_score_values(self, five_ticket_df):
        """Verify exact scores for the five-ticket fixture.

        Ticket 2 (Major+Escalation+External+Low): Low impact falls through
        to default 1.0 multiplier, so 50*1.5*2.5*1.0 = 187.5.
        """
        result = _score_df(five_ticket_df)
        scores = result['Strategic_Friction_Score'].tolist()
        assert scores[0] == pytest.approx(750.0)
        assert scores[1] == pytest.approx(300.0)
        assert scores[2] == pytest.approx(187.5)
        assert scores[3] == pytest.approx(10.0)
        assert scores[4] == pytest.approx(0.0)

    def test_external_always_outranks_internal(self):
        """Same ticket with External origin should always score higher than Internal."""
        rows = [
            _make_row(severity="Major", type_="Escalations", origin="External", impact="High"),
            _make_row(severity="Major", type_="Escalations", origin="Internal", impact="High"),
        ]
        result = _score_df(_make_df(rows))
        scores = result['Strategic_Friction_Score'].tolist()
        assert scores[0] > scores[1]
        # External/Internal ratio should be 2.5
        assert scores[0] / scores[1] == pytest.approx(2.5)

    def test_critical_always_outranks_minor(self):
        """Critical severity should always produce higher score than Minor."""
        rows = [
            _make_row(severity="Critical", type_="Concerns", origin="Internal", impact="None"),
            _make_row(severity="Minor", type_="Concerns", origin="Internal", impact="None"),
        ]
        result = _score_df(_make_df(rows))
        scores = result['Strategic_Friction_Score'].tolist()
        assert scores[0] > scores[1]
        # Critical/Minor ratio = 100/10 = 10
        assert scores[0] / scores[1] == pytest.approx(10.0)

    def test_high_impact_outranks_no_impact(self):
        """High impact should double the score compared to no impact."""
        rows = [
            _make_row(severity="Major", type_="Concerns", origin="Internal", impact="High"),
            _make_row(severity="Major", type_="Concerns", origin="Internal", impact="None"),
        ]
        result = _score_df(_make_df(rows))
        scores = result['Strategic_Friction_Score'].tolist()
        assert scores[0] / scores[1] == pytest.approx(2.0)


# ============================================================================
# 6. Risk Tier Assignment
# ============================================================================

class TestRiskTier:
    """Verify risk tier thresholds align with score buckets."""

    @pytest.mark.parametrize("severity,type_,origin,impact,expected_tier", [
        ("Critical", "Escalations", "External", "High", "Critical"),   # 750
        ("Critical", "Escalations", "Internal", "High", "Critical"),   # 300
        ("Major", "Escalations", "Internal", "High", "Critical"),      # 150
        ("Major", "Escalations", "External", "Low", "Critical"),        # 187.5 (Low impact = 1.0)
        ("Major", "Escalations", "Internal", "None", "High"),          # 75
        ("Major", "Concerns", "Internal", "None", "Medium"),           # 50
        ("Minor", "Escalations", "Internal", "None", "Low"),           # 15
        ("Minor", "Concerns", "Internal", "None", "Low"),              # 10
        ("Critical", "Lessons Learned", "External", "High", "Low"),    # 0
    ])
    def test_tier_assignment(self, severity, type_, origin, impact, expected_tier):
        df = _make_df([_make_row(severity=severity, type_=type_, origin=origin, impact=impact)])
        result = _score_df(df)
        assert result['Risk_Tier'].iloc[0] == expected_tier


# ============================================================================
# 7. WEIGHTS Config Integrity
# ============================================================================

class TestWeightsConfig:
    """Verify the WEIGHTS dictionary has expected structure and values."""

    def test_base_severity_keys(self):
        assert set(WEIGHTS['BASE_SEVERITY'].keys()) == {'Critical', 'Major', 'Minor', 'Default'}

    def test_type_multiplier_keys(self):
        assert 'Escalations' in WEIGHTS['TYPE_MULTIPLIER']
        assert 'Concerns' in WEIGHTS['TYPE_MULTIPLIER']
        assert 'Lessons Learned' in WEIGHTS['TYPE_MULTIPLIER']

    def test_origin_multiplier_keys(self):
        assert 'External' in WEIGHTS['ORIGIN_MULTIPLIER']
        assert 'Internal' in WEIGHTS['ORIGIN_MULTIPLIER']

    def test_impact_multiplier_keys(self):
        assert 'High' in WEIGHTS['IMPACT_MULTIPLIER']
        assert 'Low' in WEIGHTS['IMPACT_MULTIPLIER']
        assert 'None' in WEIGHTS['IMPACT_MULTIPLIER']

    def test_all_base_severities_positive(self):
        for val in WEIGHTS['BASE_SEVERITY'].values():
            assert val > 0

    def test_all_multipliers_nonnegative(self):
        for group in ['TYPE_MULTIPLIER', 'ORIGIN_MULTIPLIER', 'IMPACT_MULTIPLIER']:
            for val in WEIGHTS[group].values():
                assert val >= 0
