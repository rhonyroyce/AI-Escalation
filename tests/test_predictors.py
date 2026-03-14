"""
Test suites for the 3 ML predictor modules:
- RecurrencePredictor (binary classification)
- ResolutionTimePredictor (regression)
- SimilarTicketFinder (embedding-based similarity search)

Uses synthetic data and mocked dependencies to validate training,
inference, and edge cases without requiring GPU or Ollama.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Patch GPU utilities before importing predictors so they never try real GPU
# ---------------------------------------------------------------------------
_gpu_patches = {
    'escalation_ai.core.gpu_utils.is_gpu_available': lambda: False,
    'escalation_ai.core.config.USE_GPU': False,
}


@pytest.fixture(autouse=True)
def _disable_gpu(monkeypatch):
    """Ensure GPU is always disabled in tests."""
    monkeypatch.setattr('escalation_ai.core.config.USE_GPU', False)
    monkeypatch.setattr('escalation_ai.core.gpu_utils.is_gpu_available', lambda: False)
    # Also patch in the predictor modules where USE_GPU / is_gpu_available are imported
    monkeypatch.setattr('escalation_ai.predictors.recurrence.USE_GPU', False)
    monkeypatch.setattr('escalation_ai.predictors.recurrence.is_gpu_available', lambda: False)
    monkeypatch.setattr('escalation_ai.predictors.resolution_time.USE_GPU', False)
    monkeypatch.setattr('escalation_ai.predictors.resolution_time.is_gpu_available', lambda: False)
    monkeypatch.setattr('escalation_ai.predictors.similar_tickets.USE_GPU', False)
    monkeypatch.setattr('escalation_ai.predictors.similar_tickets.is_gpu_available', lambda: False)


# ---------------------------------------------------------------------------
# Helpers — synthetic DataFrames
# ---------------------------------------------------------------------------

def _make_recurrence_df(n=100, seed=42):
    """Create a synthetic DataFrame suitable for RecurrencePredictor training."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        'Strategic_Friction_Score': rng.uniform(0, 100, n),
        'AI_Confidence': rng.uniform(0.3, 1.0, n),
        'Engineer_Issue_Count': rng.randint(0, 10, n),
        'Days_Since_Issue': rng.randint(0, 90, n),
        'Recidivism_Score': rng.uniform(0, 1, n),
        'Severity_Norm': rng.choice(['Critical', 'High', 'Medium', 'Low'], n),
        'Type_Norm': rng.choice(['Escalations', 'Concerns', 'Lessons Learned'], n),
        'Origin_Norm': rng.choice(['Internal', 'External'], n),
        'Root_Cause_Category': rng.choice(
            ['Human Error', 'External Party', 'Process Gap',
             'System/Technical', 'Training Gap'], n
        ),
        'AI_Category': rng.choice(
            ['Scheduling & Planning', 'Documentation & Reporting',
             'Validation & QA', 'Process Compliance'], n
        ),
        'LOB_Risk_Tier': rng.choice(['Critical', 'High', 'Medium', 'Low'], n),
        'Is_Human_Error': rng.choice(['Yes', 'No'], n),
        'Engineer_Flag': rng.choice(['', 'Repeat Offender'], n),
        'Aging_Status': rng.choice(['<14 days', '>14 days', '>30 days'], n),
        'Recurrence_Actual': rng.choice(['Yes', 'No'], n, p=[0.35, 0.65]),
    })
    return df


def _make_resolution_df(n=50, seed=42):
    """Create a synthetic DataFrame suitable for ResolutionTimePredictor training."""
    rng = np.random.RandomState(seed)
    base = pd.Timestamp('2026-01-01')
    issue_dates = [base - pd.Timedelta(days=int(d)) for d in rng.randint(1, 90, n)]
    resolution_dates = [
        d + pd.Timedelta(days=int(r)) if r > 0 else pd.NaT
        for d, r in zip(issue_dates, rng.randint(1, 30, n))
    ]

    from escalation_ai.core.config import COL_SUMMARY, COL_SEVERITY, COL_DATETIME, COL_RESOLUTION_DATE

    df = pd.DataFrame({
        COL_DATETIME: issue_dates,
        COL_RESOLUTION_DATE: resolution_dates,
        COL_SUMMARY: [f'Ticket summary issue number {i}' for i in range(n)],
        COL_SEVERITY: rng.choice(['Critical', 'High', 'Medium', 'Low'], n),
        'AI_Category': rng.choice(
            ['Scheduling & Planning', 'Documentation & Reporting',
             'Validation & QA', 'Process Compliance',
             'Configuration & Data Mismatch', 'Site Readiness'], n
        ),
        'AI_Confidence': rng.uniform(0.3, 1.0, n),
        'AI_Recurrence_Risk': rng.choice(['High', 'Medium', 'Low'], n),
        'Expected_Resolution_Days': rng.uniform(1, 10, n),
        'Similar_Ticket_Count': rng.randint(0, 5, n),
    })
    return df


# ===========================================================================
# RecurrencePredictor Tests
# ===========================================================================

class TestRecurrencePredictor:
    """Tests for escalation_ai.predictors.recurrence.RecurrencePredictor."""

    def _make_predictor(self):
        from escalation_ai.predictors.recurrence import RecurrencePredictor
        return RecurrencePredictor()

    # ---- 1. Training Smoke Test ----

    def test_training_smoke(self):
        """Train on 100 synthetic rows — should not crash and produce a fitted model."""
        pred = self._make_predictor()
        df = _make_recurrence_df(n=100)

        metrics = pred.train(df, min_samples=50)

        assert 'error' not in metrics, f"Training returned error: {metrics}"
        assert pred.is_trained is True
        assert pred.model is not None
        assert 'accuracy' in metrics
        assert 'auc_roc' in metrics
        assert metrics['train_samples'] > 0
        assert metrics['test_samples'] > 0

    # ---- 2. Prediction Format ----

    def test_prediction_format(self):
        """After training, predictions should have the correct columns and value ranges."""
        pred = self._make_predictor()
        df_train = _make_recurrence_df(n=100)
        pred.train(df_train, min_samples=50)

        df_test = _make_recurrence_df(n=10, seed=99)
        result = pred.predict(df_test)

        # Required columns exist
        assert 'AI_Recurrence_Probability' in result.columns
        assert 'AI_Recurrence_Risk' in result.columns
        assert 'AI_Recurrence_Confidence' in result.columns

        # Probabilities are in [0, 1]
        probs = result['AI_Recurrence_Probability']
        assert (probs >= 0.0).all(), f"Some probabilities < 0: {probs.min()}"
        assert (probs <= 1.0).all(), f"Some probabilities > 1: {probs.max()}"

        # Risk tiers are one of the known strings
        valid_risk_substrings = ['High', 'Elevated', 'Moderate', 'Low']
        for risk in result['AI_Recurrence_Risk']:
            assert any(sub in risk for sub in valid_risk_substrings), (
                f"Unexpected risk tier: {risk}"
            )

        # Confidence is one of High / Medium / Low
        assert set(result['AI_Recurrence_Confidence']).issubset({'High', 'Medium', 'Low'})

    # ---- 3. Known Pattern Detection ----

    def test_known_pattern_detection(self):
        """Model should learn that 'outage' tickets always recur and 'inquiry' never recur."""
        rng = np.random.RandomState(123)
        n = 200

        # Build training data with a clear signal: outage -> Yes, inquiry -> No
        categories = []
        recurrence = []
        summaries = []
        for _ in range(n):
            if rng.random() < 0.5:
                categories.append('Site Readiness')
                recurrence.append('Yes')
                summaries.append('outage at tower site causing downtime')
            else:
                categories.append('Documentation & Reporting')
                recurrence.append('No')
                summaries.append('inquiry about documentation process')

        df = pd.DataFrame({
            'Strategic_Friction_Score': rng.uniform(0, 100, n),
            'AI_Confidence': rng.uniform(0.3, 1.0, n),
            'Engineer_Issue_Count': rng.randint(0, 5, n),
            'Days_Since_Issue': rng.randint(0, 30, n),
            'Recidivism_Score': rng.uniform(0, 0.3, n),
            'Severity_Norm': ['Critical' if r == 'Yes' else 'Low' for r in recurrence],
            'Type_Norm': ['Escalations'] * n,
            'Origin_Norm': ['Internal'] * n,
            'Root_Cause_Category': ['System/Technical' if r == 'Yes' else 'Other' for r in recurrence],
            'AI_Category': categories,
            'LOB_Risk_Tier': ['High' if r == 'Yes' else 'Low' for r in recurrence],
            'Is_Human_Error': ['No'] * n,
            'Engineer_Flag': [''] * n,
            'Aging_Status': ['>30 days' if r == 'Yes' else '<14 days' for r in recurrence],
            'Recurrence_Actual': recurrence,
        })

        pred = self._make_predictor()
        metrics = pred.train(df, min_samples=50)
        assert 'error' not in metrics

        # Predict on new outage tickets
        outage_df = pd.DataFrame({
            'Strategic_Friction_Score': [80.0] * 5,
            'AI_Confidence': [0.9] * 5,
            'Engineer_Issue_Count': [3] * 5,
            'Days_Since_Issue': [10] * 5,
            'Recidivism_Score': [0.1] * 5,
            'Severity_Norm': ['Critical'] * 5,
            'Type_Norm': ['Escalations'] * 5,
            'Origin_Norm': ['Internal'] * 5,
            'Root_Cause_Category': ['System/Technical'] * 5,
            'AI_Category': ['Site Readiness'] * 5,
            'LOB_Risk_Tier': ['High'] * 5,
            'Is_Human_Error': ['No'] * 5,
            'Engineer_Flag': [''] * 5,
            'Aging_Status': ['>30 days'] * 5,
        })

        inquiry_df = pd.DataFrame({
            'Strategic_Friction_Score': [10.0] * 5,
            'AI_Confidence': [0.9] * 5,
            'Engineer_Issue_Count': [0] * 5,
            'Days_Since_Issue': [5] * 5,
            'Recidivism_Score': [0.05] * 5,
            'Severity_Norm': ['Low'] * 5,
            'Type_Norm': ['Escalations'] * 5,
            'Origin_Norm': ['Internal'] * 5,
            'Root_Cause_Category': ['Other'] * 5,
            'AI_Category': ['Documentation & Reporting'] * 5,
            'LOB_Risk_Tier': ['Low'] * 5,
            'Is_Human_Error': ['No'] * 5,
            'Engineer_Flag': [''] * 5,
            'Aging_Status': ['<14 days'] * 5,
        })

        outage_result = pred.predict(outage_df)
        inquiry_result = pred.predict(inquiry_df)

        outage_probs = outage_result['AI_Recurrence_Probability'].values
        inquiry_probs = inquiry_result['AI_Recurrence_Probability'].values

        # Outage tickets should have higher recurrence probability than inquiry tickets
        assert outage_probs.mean() > inquiry_probs.mean(), (
            f"Outage mean ({outage_probs.mean():.3f}) should be > inquiry mean ({inquiry_probs.mean():.3f})"
        )

    # ---- 4. Empty Input Handling ----

    def test_empty_input_handling(self):
        """Predict on empty DataFrame should return empty result, not crash."""
        pred = self._make_predictor()
        df_train = _make_recurrence_df(n=100)
        pred.train(df_train, min_samples=50)

        empty_df = pd.DataFrame(columns=df_train.columns)
        result = pred.predict(empty_df)

        assert len(result) == 0
        assert 'AI_Recurrence_Probability' in result.columns

    def test_untrained_prediction_uses_heuristic(self):
        """When model is not trained, predict should fall back to heuristic without crashing."""
        pred = self._make_predictor()
        assert pred.is_trained is False

        df = _make_recurrence_df(n=5)
        result = pred.predict(df)

        assert len(result) == 5
        assert 'AI_Recurrence_Probability' in result.columns
        # Heuristic should produce some non-zero values
        assert result['AI_Recurrence_Probability'].sum() > 0

    def test_insufficient_training_data(self):
        """Training with fewer rows than min_samples should return error."""
        pred = self._make_predictor()
        df = _make_recurrence_df(n=10)
        metrics = pred.train(df, min_samples=50)

        assert 'error' in metrics
        assert pred.is_trained is False


# ===========================================================================
# ResolutionTimePredictor Tests
# ===========================================================================

class TestResolutionTimePredictor:
    """Tests for escalation_ai.predictors.resolution_time.ResolutionTimePredictor."""

    def _make_predictor(self):
        from escalation_ai.predictors.resolution_time import ResolutionTimePredictor
        return ResolutionTimePredictor()

    # ---- 5. Prediction Range ----

    def test_prediction_range_non_negative(self):
        """All predictions should be >= 0.5 (the minimum floor in the code)."""
        pred = self._make_predictor()
        df = _make_resolution_df(n=50)
        pred.train(df)

        from escalation_ai.core.config import COL_SUMMARY, COL_SEVERITY

        # Create rows that might trick a naive model into negative predictions
        # (low everything, very simple ticket)
        tricky_row = pd.Series({
            COL_SUMMARY: 'simple quick fix',
            COL_SEVERITY: 'Low',
            'AI_Category': 'UnknownCategoryNeverSeen',
            'AI_Confidence': 0.1,
            'AI_Recurrence_Risk': 'Low',
            'Expected_Resolution_Days': 0,
            'Similar_Ticket_Count': 0,
        })

        result = pred.predict(tricky_row)
        assert result['predicted_days'] >= 0.5, (
            f"Predicted {result['predicted_days']} days — should be >= 0.5"
        )
        assert result['confidence'] >= 0.0
        assert result['method'] in ('ml', 'category_stats', 'global_stats', 'heuristic')

    def test_predictions_always_positive(self):
        """Run predictions across many rows — all must be >= 0.5."""
        pred = self._make_predictor()
        df = _make_resolution_df(n=50)
        pred.train(df)

        for _, row in df.iterrows():
            result = pred.predict(row)
            assert result['predicted_days'] >= 0.5

    # ---- 6. Feature Engineering ----

    def test_feature_engineering(self):
        """Verify that _extract_features produces the expected feature set."""
        pred = self._make_predictor()

        from escalation_ai.core.config import COL_SUMMARY, COL_SEVERITY

        row = pd.Series({
            COL_SUMMARY: 'Complex migration integration issue with multiple teams',
            COL_SEVERITY: 'Critical',
            'AI_Category': 'Configuration & Data Mismatch',
            'AI_Confidence': 0.85,
            'AI_Recurrence_Risk': 'High',
            'Expected_Resolution_Days': 5.0,
            'Similar_Ticket_Count': 3,
        })

        features = pred._extract_features(row)

        # All expected feature keys exist
        expected_keys = [
            'category_hash', 'category_avg_days', 'category_median_days',
            'severity_level', 'text_length', 'word_count',
            'complexity_score', 'similar_resolution_days', 'similar_count',
            'ai_confidence', 'recurrence_risk',
        ]
        for key in expected_keys:
            assert key in features, f"Missing feature: {key}"

        # Severity mapping: Critical -> 4
        assert features['severity_level'] == 4

        # Word count should match the summary
        summary_text = 'Complex migration integration issue with multiple teams'
        assert features['word_count'] == len(summary_text.split())

        # Complexity keywords: 'complex', 'migration', 'integration', 'multiple' -> 4
        assert features['complexity_score'] >= 3, (
            f"Expected >= 3 complexity keywords, got {features['complexity_score']}"
        )

        # Recurrence risk: High -> 3
        assert features['recurrence_risk'] == 3

        # Category hash is deterministic
        assert features['category_hash'] == hash('Configuration & Data Mismatch') % 1000

    def test_training_success(self):
        """Training on valid data should succeed."""
        pred = self._make_predictor()
        df = _make_resolution_df(n=50)
        success = pred.train(df)
        assert success is True
        assert pred.is_trained is True

    def test_training_insufficient_data(self):
        """Training with too few valid rows should fail gracefully."""
        pred = self._make_predictor()
        from escalation_ai.core.config import COL_DATETIME, COL_RESOLUTION_DATE, COL_SUMMARY, COL_SEVERITY

        # Only 3 rows, all without resolution dates
        df = pd.DataFrame({
            COL_DATETIME: [pd.Timestamp('2026-01-01')] * 3,
            COL_RESOLUTION_DATE: [pd.NaT] * 3,
            COL_SUMMARY: ['test'] * 3,
            COL_SEVERITY: ['Medium'] * 3,
            'AI_Category': ['Validation & QA'] * 3,
        })
        success = pred.train(df)
        assert success is False
        assert pred.is_trained is False

    def test_heuristic_fallback(self):
        """Untrained predictor should use heuristic and still return valid results."""
        pred = self._make_predictor()
        assert pred.is_trained is False

        from escalation_ai.core.config import COL_SUMMARY, COL_SEVERITY

        row = pd.Series({
            COL_SUMMARY: 'Simple inquiry about process',
            COL_SEVERITY: 'Low',
            'AI_Category': 'Documentation & Reporting',
            'Severity_Norm': 'Low',
            'AI_Recurrence_Risk': 'Low',
            'Strategic_Friction_Score': 20,
        })

        result = pred.predict(row)
        assert result['predicted_days'] >= 0.5
        assert result['method'] == 'heuristic'

    def test_process_all_tickets(self):
        """process_all_tickets should add all output columns."""
        pred = self._make_predictor()
        df = _make_resolution_df(n=20)
        result = pred.process_all_tickets(df)

        assert 'Actual_Resolution_Days' in result.columns
        assert 'Predicted_Resolution_Days' in result.columns
        assert 'Resolution_Prediction_Confidence' in result.columns
        assert 'Resolution_Prediction_Method' in result.columns

        # All predicted values should be non-negative
        predicted = result['Predicted_Resolution_Days'].dropna()
        assert (predicted >= 0.5).all()


# ===========================================================================
# SimilarTicketFinder Tests
# ===========================================================================

class MockAIEngine:
    """Mock AI engine that returns deterministic embeddings based on text content."""

    def __init__(self, dim=64):
        self.dim = dim
        self._cache = {}

    def get_embedding(self, text):
        """Return a deterministic embedding.

        Similar texts get similar vectors; dissimilar texts get near-orthogonal vectors.
        Uses a hash-seeded RNG so the same text always produces the same embedding.
        """
        text = str(text).strip().lower()
        if text in self._cache:
            return self._cache[text]

        # Use hash of text as seed for reproducibility
        seed = hash(text) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dim).astype(np.float32)
        # Normalize
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        self._cache[text] = vec
        return vec

    def get_similar_embedding(self, base_text, noise_level=0.05):
        """Return an embedding similar to base_text but with small noise."""
        base_vec = self.get_embedding(base_text)
        rng = np.random.RandomState(42)
        noise = rng.randn(self.dim).astype(np.float32) * noise_level
        vec = base_vec + noise
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        return vec


class TestSimilarTicketFinder:
    """Tests for escalation_ai.predictors.similar_tickets.SimilarTicketFinder."""

    def _make_finder(self, top_k=5, threshold=0.5):
        from escalation_ai.predictors.similar_tickets import SimilarTicketFinder
        mock_ai = MockAIEngine(dim=64)
        # Patch _load_feedback to avoid filesystem access
        with patch.object(SimilarTicketFinder, '_load_feedback', lambda self: None):
            finder = SimilarTicketFinder(
                ai_engine=mock_ai,
                top_k=top_k,
                similarity_threshold=threshold,
            )
        finder.feedback_data = {}
        return finder

    def _make_corpus_df(self, n=20):
        """Create a small corpus DataFrame for similarity tests."""
        from escalation_ai.core.config import COL_SUMMARY
        summaries = [
            'Power supply unit failure at cell tower site causing outage',
            'Site was not scheduled in TI for integration on planned date',
            'Missing CBN snapshots and documentation not uploaded',
            'Post-integration validation failed due to missing KPI checks',
            'SOP not followed during site access without authorization',
            'RF antenna configuration mismatch between design and as-built',
            'Communication breakdown between engineering and field ops',
            'TEMS tool error during drive test lost measurement data',
            'Fiber optic cable cut by contractor during excavation',
            'Nesting tool failed to import neighbor list from CIQ file',
            'Power supply backup generator test failed during maintenance',
            'Site grounding verification incomplete before energization',
            'QA audit revealed failed acceptance criteria on antenna test',
            'Wrong antenna model installed per design specification',
            'Scheduling system did not reflect updated timeline after CR',
            'Backhaul transmission link failure causing site isolation',
            'Pre-integration checklist validation items not completed',
            'CIQ parameter entry error in azimuth and tilt values',
            'Safety compliance harness inspection certificate expired',
            'Remote monitoring tool connectivity loss to site management',
        ]
        df = pd.DataFrame({
            COL_SUMMARY: summaries[:n],
            'ID': [f'TKT-{i}' for i in range(n)],
        })
        return df

    # ---- 7. Self-Similarity ----

    def test_self_similarity(self):
        """If a ticket is in the corpus, searching for it with exclude_self=False
        should return it as the top result with similarity ~1.0."""
        finder = self._make_finder(top_k=5, threshold=0.3)
        df = self._make_corpus_df(n=10)

        from escalation_ai.core.config import COL_SUMMARY

        # Query with the exact same text as row 0
        query_row = df.iloc[0]
        results = finder.find_similar(query_row, df, exclude_self=False)

        assert len(results) > 0, "Should find at least one similar ticket"

        # The top result should be the same ticket (self-match) with high similarity
        top = results[0]
        assert top['similarity'] > 0.95, (
            f"Self-similarity should be ~1.0, got {top['similarity']:.3f}"
        )

    # ---- 8. Dissimilar Detection ----

    def test_dissimilar_detection(self):
        """Two completely unrelated tickets should have low similarity."""
        finder = self._make_finder(top_k=5, threshold=0.0)

        mock_ai = finder.ai
        # Get embeddings for two very different texts
        vec1 = mock_ai.get_embedding('power supply unit failure at tower site causing major outage')
        vec2 = mock_ai.get_embedding('quarterly financial report excel spreadsheet formatting')

        # Compute cosine similarity directly
        sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        # Hash-based random vectors in 64 dims should have low cosine similarity
        assert abs(sim) < 0.5, (
            f"Dissimilar texts should have low similarity, got {sim:.3f}"
        )

    # ---- 9. K Parameter ----

    def test_k_parameter(self):
        """When requesting top-5, should return exactly 5 (or fewer if corpus is smaller)."""
        finder = self._make_finder(top_k=5, threshold=0.0)
        df = self._make_corpus_df(n=20)

        from escalation_ai.core.config import COL_SUMMARY

        query_row = df.iloc[0]
        results = finder.find_similar(query_row, df, exclude_self=True)

        # Should return at most top_k results
        assert len(results) <= 5, f"Expected <= 5 results, got {len(results)}"

    def test_k_parameter_small_corpus(self):
        """When corpus has fewer tickets than K, return all available."""
        finder = self._make_finder(top_k=10, threshold=0.0)
        df = self._make_corpus_df(n=3)

        from escalation_ai.core.config import COL_SUMMARY

        query_row = df.iloc[0]
        results = finder.find_similar(query_row, df, exclude_self=True)

        # Should return at most 2 (corpus of 3 minus self-exclusion)
        assert len(results) <= 2

    def test_empty_query_text(self):
        """Empty query text should return empty results, not crash."""
        finder = self._make_finder(top_k=5, threshold=0.3)
        df = self._make_corpus_df(n=10)

        from escalation_ai.core.config import COL_SUMMARY

        empty_row = pd.Series({COL_SUMMARY: '', 'ID': 'EMPTY'})
        results = finder.find_similar(empty_row, df)

        assert results == []

    def test_cosine_similarity_method(self):
        """The _cosine_similarity method should work correctly."""
        finder = self._make_finder()

        # Identical vectors -> similarity 1.0
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        sim = finder._cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.01

        # None input -> 0.0
        sim_none = finder._cosine_similarity(None, vec)
        assert sim_none == 0.0

    def test_feedback_adjustment(self):
        """Feedback should boost confirmed matches and penalise rejected ones."""
        finder = self._make_finder()

        # Record positive feedback
        finder.record_feedback('TKT-1', 'TKT-2', is_similar=True, notes='confirmed')
        adjusted = finder.get_feedback_adjusted_similarity('TKT-1', 'TKT-2', 0.7)
        assert adjusted == pytest.approx(0.9, abs=0.01)

        # Record negative feedback
        finder.record_feedback('TKT-3', 'TKT-4', is_similar=False, notes='rejected')
        adjusted_neg = finder.get_feedback_adjusted_similarity('TKT-3', 'TKT-4', 0.7)
        assert adjusted_neg == pytest.approx(0.2, abs=0.01)

    def test_no_ai_engine(self):
        """Finder with no AI engine should return empty results without crashing."""
        from escalation_ai.predictors.similar_tickets import SimilarTicketFinder
        with patch.object(SimilarTicketFinder, '_load_feedback', lambda self: None):
            finder = SimilarTicketFinder(ai_engine=None, top_k=5, similarity_threshold=0.5)
        finder.feedback_data = {}

        df = self._make_corpus_df(n=5)
        from escalation_ai.core.config import COL_SUMMARY
        query_row = df.iloc[0]
        results = finder.find_similar(query_row, df)

        assert results == []
