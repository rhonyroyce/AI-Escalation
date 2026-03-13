"""
Comprehensive test suite for the classification engine (Phase 1).

Tests the three-tier hybrid classifier:
  Tier 1: Regex pattern matching
  Tier 2: Keyword/phrase scoring
  Tier 3: Embedding similarity (mocked Ollama)

Also covers sub-category assignment, confidence scores, edge cases,
and graceful degradation when Ollama is unavailable.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from escalation_ai.classification import (
    pattern_classify,
    keyword_classify,
    get_sub_category,
    classify_rows,
    clear_centroid_cache,
)
from escalation_ai.core.config import (
    ANCHORS,
    CATEGORY_KEYWORDS,
    SUB_CATEGORIES,
    MIN_CLASSIFICATION_CONFIDENCE,
)


# ============================================================================
# Constants
# ============================================================================

ALL_CATEGORIES = list(ANCHORS.keys())
EMBED_DIM = 768  # Standard nomic-embed-text dimension


# ============================================================================
# Helpers
# ============================================================================

def _random_embedding(seed=42):
    """Return a deterministic random unit vector of EMBED_DIM dimensions."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(EMBED_DIM)
    return vec / np.linalg.norm(vec)


def _zero_embedding():
    return np.zeros(EMBED_DIM)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear module-level caches before each test for isolation."""
    clear_centroid_cache()
    # Also reset compiled patterns cache
    from escalation_ai.classification import _COMPILED_PATTERNS
    _COMPILED_PATTERNS.clear()
    yield
    clear_centroid_cache()


@pytest.fixture
def mock_ai():
    """Create a mock OllamaBrain that returns deterministic embeddings."""
    ai = MagicMock()
    ai._embed_dim = EMBED_DIM

    # Each category gets a unique direction so cosine similarity works
    category_seeds = {cat: i + 10 for i, cat in enumerate(sorted(ALL_CATEGORIES))}

    def fake_get_embedding(text):
        if pd.isna(text) or text == "":
            return _zero_embedding()
        return _random_embedding(seed=hash(text) % 2**31)

    def fake_get_embeddings_batch(texts, batch_size=None):
        return [fake_get_embedding(t) for t in texts]

    def fake_get_dim():
        return EMBED_DIM

    ai.get_embedding = MagicMock(side_effect=fake_get_embedding)
    ai.get_embeddings_batch = MagicMock(side_effect=fake_get_embeddings_batch)
    ai.get_dim = MagicMock(side_effect=fake_get_dim)

    return ai


# ============================================================================
# 1. KEYWORD MATCHING TESTS (Tier 1 & 2)
# ============================================================================

class TestPatternClassify:
    """Tier 1: Regex pattern matching — high confidence (0.95)."""

    @pytest.mark.parametrize("text, expected_category", [
        # Scheduling & Planning patterns
        ("Site was not scheduled in Ti for integration", "Scheduling & Planning"),
        ("FE logged but not scheduled for support", "Scheduling & Planning"),
        ("Ti entry missing for the weekend site", "Scheduling & Planning"),
        ("ticket was in closeout status already", "Scheduling & Planning"),
        # Documentation & Reporting patterns
        ("CBN snapshot missing from report", "Documentation & Reporting"),
        ("wrong site ID was used in the email", "Documentation & Reporting"),
        ("E911 missing from completion package", "Documentation & Reporting"),
        # Validation & QA patterns
        ("missed to check the VSWR readings", "Validation & QA"),
        ("degradation not detected during postcheck", "Validation & QA"),
        ("incomplete validation of the BH fields", "Validation & QA"),
        # Process Compliance patterns
        ("released without BH actualized in the system", "Process Compliance"),
        ("escalated to NTAC while not supposed to", "Process Compliance"),
        ("without backhaul acceptance from PAG", "Process Compliance"),
        # Configuration & Data Mismatch patterns
        ("port matrix mismatch found on site", "Configuration & Data Mismatch"),
        ("RET naming issue detected in sector Alpha", "Configuration & Data Mismatch"),
        ("TAC mismatch showing RIOT red", "Configuration & Data Mismatch"),
        # Site Readiness patterns
        ("BH not actualized in MB system", "Site Readiness"),
        ("MW not ready for integration", "Site Readiness"),
        ("material missing from the site", "Site Readiness"),
        # Communication & Response patterns
        ("delay in reply from the support team", "Communication & Response"),
        ("FE waited for hours at the site", "Communication & Response"),
        ("communication gap between PM and FE", "Communication & Response"),
        # Nesting & Tool Errors patterns
        ("site nested as NSA but only SA allowed", "Nesting & Tool Errors"),
        ("RIOT red due to config mismatch", "Nesting & Tool Errors"),
        ("nest extended without approval", "Nesting & Tool Errors"),
    ])
    def test_pattern_matches_correct_category(self, text, expected_category):
        result = pattern_classify(text)
        assert result is not None, f"Expected pattern match for: {text!r}"
        category, confidence = result
        assert category == expected_category
        assert confidence == 0.95

    def test_pattern_returns_none_for_unrelated_text(self):
        result = pattern_classify("The weather today is sunny and warm")
        assert result is None


class TestKeywordClassify:
    """Tier 2: Keyword/phrase scoring — confidence 0.50-0.85."""

    @pytest.mark.parametrize("text, expected_category", [
        # Scheduling — load enough keywords to hit score >= 5
        (
            "Site schedule was not followed, FE logged for IX on the wrong date, "
            "TI entry missing, holiday weekend calendar conflict",
            "Scheduling & Planning",
        ),
        (
            "The schedule was not in TI, site was not planned and the calendar "
            "had the wrong date for the unplanned closeout",
            "Scheduling & Planning",
        ),
        # Documentation & Reporting
        (
            "Missing snapshot, CBN output missing, forgot to attach the RTT report, "
            "wrong attachment in the EOD email summary",
            "Documentation & Reporting",
        ),
        (
            "Screenshot missing from report, email subject wrong, "
            "E911 summary table not in mail, logs incomplete",
            "Documentation & Reporting",
        ),
        # Validation & QA
        (
            "Engineer missed to check VSWR and RSSI during precheck validation. "
            "Degradation not detected, issue not escalated, incomplete validation",
            "Validation & QA",
        ),
        (
            "Audit found missed to report KPI fail, cells in unsync not detected "
            "during postcheck, wrong pass fail captured",
            "Validation & QA",
        ),
        # Process Compliance
        (
            "Released without BH actualized, escalated to NTAC wrong distro, "
            "not following process, skipped step bypassed validation without approval",
            "Process Compliance",
        ),
        (
            "Proceeded without backhaul, wrong distro, SOP guideline bypass, "
            "process step skipped, improper escalation procedure",
            "Process Compliance",
        ),
        # Configuration & Data Mismatch
        (
            "Port matrix mismatch, RET naming issue, TAC mismatch, SCF mismatch "
            "CIQ mismatch, RFDS mismatch detected on site configuration",
            "Configuration & Data Mismatch",
        ),
        (
            "RET naming wrong, port matrix mismatch, need updated port matrix, "
            "One-T OneT config not matching RIOT red",
            "Configuration & Data Mismatch",
        ),
        # Site Readiness
        (
            "BH not actualized, backhaul not ready, MW not ready, material missing, "
            "SFP missing, site was down, cancelled due to pending actualization",
            "Site Readiness",
        ),
        (
            "Site not ready, microwave not ready, BH not ready in MB, "
            "transmission not ready, FE on site but BH pending",
            "Site Readiness",
        ),
        # Communication & Response
        (
            "Delayed reply, delayed response, FE waited for hours, "
            "GC query not replied, no reply from PM, follow-up required, "
            "communication gap, multiple follow-ups",
            "Communication & Response",
        ),
        (
            "Delay in reply, waited for hours, replied when PM asked, "
            "reminder sent, proactive update missing, questioned over delays",
            "Communication & Response",
        ),
        # Nesting & Tool Errors
        (
            "Nested as NSA, wrong nest type, nest extended, not allowed in market, "
            "RIOT mismatch, tool not updated, FCI not updated, site not unnested",
            "Nesting & Tool Errors",
        ),
        (
            "Nested as NSI, nest extension, market guideline, RIOT red, "
            "TAC mismatch RIOT, RF config mismatch, OSS mismatch, lemming validation",
            "Nesting & Tool Errors",
        ),
    ])
    def test_keyword_matches_correct_category(self, text, expected_category):
        result = keyword_classify(text)
        assert result is not None, f"Expected keyword match for: {text!r}"
        category, confidence = result
        assert category == expected_category

    def test_keyword_confidence_range(self):
        """Confidence should be between 0.50 and 0.85 for keyword matches."""
        text = (
            "Missing snapshot, CBN output missing, forgot to attach the RTT report, "
            "wrong attachment in the EOD email summary table"
        )
        result = keyword_classify(text)
        assert result is not None
        _, confidence = result
        assert 0.50 <= confidence <= 0.85

    def test_keyword_returns_none_for_weak_text(self):
        """Single keyword hit should NOT reach the score threshold of 5."""
        result = keyword_classify("schedule")
        assert result is None


# ============================================================================
# 2. CONFIDENCE SCORE TESTS
# ============================================================================

class TestConfidenceScores:
    """Verify AI_Confidence is always in [0, 1] and ordering makes sense."""

    def test_pattern_confidence_is_0_95(self):
        result = pattern_classify("site was not scheduled in Ti")
        assert result is not None
        assert result[1] == 0.95

    def test_keyword_confidence_bounded(self):
        """All keyword matches produce confidence in [0.50, 0.85]."""
        for cat, kw_data in CATEGORY_KEYWORDS.items():
            # Build a text with many keywords to ensure a match
            phrases = kw_data.get("phrases", [])[:5]
            primary = kw_data.get("primary", [])[:5]
            text = " ".join(phrases + primary)
            result = keyword_classify(text)
            if result is not None:
                _, conf = result
                assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of [0,1] for {cat}"
                assert conf <= 0.85, f"Keyword confidence {conf} > 0.85 for {cat}"

    def test_high_confidence_beats_ambiguous(self):
        """Exact pattern hit (0.95) should score higher than keyword-only."""
        pattern_result = pattern_classify("site was not scheduled in Ti for integration")
        keyword_text = "schedule TI planned"
        keyword_result = keyword_classify(keyword_text)

        assert pattern_result is not None
        if keyword_result is not None:
            assert pattern_result[1] > keyword_result[1]


# ============================================================================
# 3. EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Unusual inputs should never crash the classifier."""

    def test_empty_string(self):
        assert pattern_classify("") is None
        assert keyword_classify("") is None
        assert get_sub_category("", "Scheduling & Planning") == "General"

    def test_none_input(self):
        assert pattern_classify(None) is None
        assert keyword_classify(None) is None
        assert get_sub_category(None, "Scheduling & Planning") == "General"

    def test_very_long_input(self):
        """10,000 characters of text should classify without error."""
        long_text = "schedule TI planned FE logged " * 500  # ~15,000 chars
        # Should not raise
        pattern_classify(long_text)
        result = keyword_classify(long_text)
        # With that many keyword hits, should get a match
        assert result is not None
        cat, conf = result
        assert cat in ALL_CATEGORIES
        assert 0.0 <= conf <= 1.0

    def test_special_characters_only(self):
        text = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        assert pattern_classify(text) is None
        assert keyword_classify(text) is None

    def test_non_english_input(self):
        text = "El circuito de aprovisionamiento se retrasó 3 semanas"
        # Should not crash
        pattern_classify(text)
        keyword_classify(text)

    def test_numeric_only_input(self):
        text = "12345 67890 11111"
        assert pattern_classify(text) is None
        assert keyword_classify(text) is None


# ============================================================================
# 4. SUB-CATEGORY ASSIGNMENT
# ============================================================================

class TestSubCategoryAssignment:
    """Verify sub-category drill-down from the SUB_CATEGORIES config."""

    @pytest.mark.parametrize("text, category, expected_sub", [
        # Scheduling & Planning -> No TI Entry
        (
            "Site not schedule in Ti, TI entry missing for integration",
            "Scheduling & Planning",
            "No TI Entry",
        ),
        # Scheduling & Planning -> Weekend Schedule Issue
        (
            "not site schedule on weekend, over weekend issue",
            "Scheduling & Planning",
            "Weekend Schedule Issue",
        ),
        # Documentation & Reporting -> Missing Snapshot
        (
            "CBN snapshot missing, validation snapshot missing from report",
            "Documentation & Reporting",
            "Missing Snapshot",
        ),
        # Documentation & Reporting -> Wrong Site ID
        (
            "wrong site ID in email, Different site ID was used",
            "Documentation & Reporting",
            "Wrong Site ID",
        ),
        # Validation & QA -> Missed Issue
        (
            "Engineer did not identified the fiber issue, missed to report SFP issue",
            "Validation & QA",
            "Missed Issue",
        ),
        # Configuration & Data Mismatch -> Port Matrix Mismatch
        (
            "port matrix mismatch, need updated port matrix, Incorrect Port matrix",
            "Configuration & Data Mismatch",
            "Port Matrix Mismatch",
        ),
        # Site Readiness -> BH Not Ready
        (
            "BH not ready, backhaul not ready, BH not actualized",
            "Site Readiness",
            "BH Not Ready",
        ),
        # Communication & Response -> Delayed Response
        (
            "Delay in reply, Delayed reply to GC, delayed response from team",
            "Communication & Response",
            "Delayed Response",
        ),
        # Nesting & Tool Errors -> Wrong Nest Type
        (
            "nested as NSA but only SA is allowed in this market, wrong nest type",
            "Nesting & Tool Errors",
            "Wrong Nest Type",
        ),
    ])
    def test_sub_category_assigned_correctly(self, text, category, expected_sub):
        result = get_sub_category(text, category)
        assert result == expected_sub

    def test_unknown_category_returns_general(self):
        assert get_sub_category("some text", "Nonexistent Category") == "General"

    def test_no_matching_keywords_returns_general(self):
        assert get_sub_category("sunny weather", "Scheduling & Planning") == "General"


# ============================================================================
# 5. MOCK OLLAMA FOR EMBEDDING TESTS (Tier 3 fallback)
# ============================================================================

class TestEmbeddingClassification:
    """Test the full classify_rows pipeline with mocked Ollama."""

    @patch("escalation_ai.classification.get_feedback_learner")
    def test_classify_rows_returns_valid_columns(self, mock_feedback, mock_ai):
        """classify_rows should add AI_Category, AI_Confidence, AI_Sub_Category."""
        # Configure feedback learner mock to be a no-op
        learner = MagicMock()
        learner.stats = {"loaded": 0, "custom_categories": 0}
        mock_feedback.return_value = learner

        df = pd.DataFrame({
            "Combined_Text": [
                "BH not actualized, backhaul not ready, site not ready, MW not ready",
                "missing snapshot, CBN output missing, forgot to attach report",
                "purely random text with no telecom keywords whatsoever",
            ]
        })

        result = classify_rows(df, mock_ai, show_progress=False)

        assert "AI_Category" in result.columns
        assert "AI_Confidence" in result.columns
        assert "AI_Sub_Category" in result.columns
        assert len(result) == 3

        # All confidence scores should be between 0 and 1
        for conf in result["AI_Confidence"]:
            assert 0.0 <= conf <= 1.0

        # All categories should be valid or "Unclassified"
        valid = set(ALL_CATEGORIES) | {"Unclassified"}
        for cat in result["AI_Category"]:
            assert cat in valid, f"Unexpected category: {cat}"

    @patch("escalation_ai.classification.get_feedback_learner")
    def test_embedding_fallback_produces_valid_result(self, mock_feedback, mock_ai):
        """When Tier 1 & 2 miss, Tier 3 (embedding) should still classify."""
        learner = MagicMock()
        learner.stats = {"loaded": 0, "custom_categories": 0}
        mock_feedback.return_value = learner

        # Text with no obvious keyword matches
        df = pd.DataFrame({
            "Combined_Text": ["generic issue with the system infrastructure component"]
        })

        result = classify_rows(df, mock_ai, show_progress=False)

        # Should get some category (possibly Unclassified if similarity is low)
        cat = result["AI_Category"].iloc[0]
        valid = set(ALL_CATEGORIES) | {"Unclassified"}
        assert cat in valid


# ============================================================================
# 6. OLLAMA FAILURE GRACEFUL DEGRADATION
# ============================================================================

class TestOllamaFailure:
    """When Ollama is unreachable, the classifier should not crash."""

    @patch("escalation_ai.classification.get_feedback_learner")
    def test_classify_rows_with_zero_embeddings(self, mock_feedback):
        """Simulate Ollama failure by returning zero vectors for everything."""
        learner = MagicMock()
        learner.stats = {"loaded": 0, "custom_categories": 0}
        mock_feedback.return_value = learner

        ai = MagicMock()
        ai._embed_dim = EMBED_DIM
        ai.get_dim.return_value = EMBED_DIM
        # All embeddings return zero vectors (simulates connection failure)
        ai.get_embeddings_batch.return_value = [_zero_embedding() for _ in range(3)]
        ai.get_embedding.return_value = _zero_embedding()

        df = pd.DataFrame({
            "Combined_Text": [
                "BH not actualized, backhaul not ready, site not ready",
                "missing snapshot from the report",
                "random text",
            ]
        })

        # Should NOT raise an exception
        result = classify_rows(df, ai, show_progress=False)

        assert len(result) == 3
        assert "AI_Category" in result.columns
        assert "AI_Confidence" in result.columns

        # All results should be valid
        valid = set(ALL_CATEGORIES) | {"Unclassified"}
        for cat in result["AI_Category"]:
            assert cat in valid

    @patch("escalation_ai.core.ai_engine.requests.post")
    def test_ollama_connection_error_returns_zero_vector(self, mock_post):
        """OllamaBrain.get_embedding should return zero vector on ConnectionError."""
        from escalation_ai.core.ai_engine import OllamaBrain

        mock_post.side_effect = ConnectionError("Ollama is down")

        with patch.object(OllamaBrain, "get_dim", return_value=EMBED_DIM):
            brain = OllamaBrain()
            brain._embed_dim = EMBED_DIM
            vec = brain.get_embedding("test text")

        assert isinstance(vec, np.ndarray)
        assert len(vec) == EMBED_DIM
        assert np.all(vec == 0)

    @patch("escalation_ai.core.ai_engine.requests.post")
    def test_ollama_batch_connection_error_returns_zero_vectors(self, mock_post):
        """get_embeddings_batch should return zero vectors on ConnectionError."""
        from escalation_ai.core.ai_engine import OllamaBrain

        mock_post.side_effect = ConnectionError("Ollama is down")

        with patch("escalation_ai.core.ai_engine.get_optimal_embedding_batch_size", return_value=5):
            brain = OllamaBrain()
            brain._embed_dim = EMBED_DIM
            results = brain.get_embeddings_batch(["text1", "text2", "text3"])

        assert len(results) == 3
        for vec in results:
            assert isinstance(vec, np.ndarray)
            assert len(vec) == EMBED_DIM
            assert np.all(vec == 0)


# ============================================================================
# 7. INTEGRATION: Pattern + Keyword consistency with config
# ============================================================================

class TestConfigConsistency:
    """Verify that every category in ANCHORS has keywords and patterns."""

    def test_all_anchor_categories_have_keywords(self):
        for cat in ANCHORS:
            assert cat in CATEGORY_KEYWORDS, f"Category {cat!r} in ANCHORS but not CATEGORY_KEYWORDS"

    def test_all_anchor_categories_have_sub_categories(self):
        for cat in ANCHORS:
            assert cat in SUB_CATEGORIES, f"Category {cat!r} in ANCHORS but not SUB_CATEGORIES"

    def test_all_keyword_categories_have_patterns(self):
        for cat, kw_data in CATEGORY_KEYWORDS.items():
            assert "patterns" in kw_data, f"Category {cat!r} has no 'patterns' key"
            assert len(kw_data["patterns"]) > 0, f"Category {cat!r} has empty patterns list"

    def test_min_classification_confidence_is_valid(self):
        assert 0.0 < MIN_CLASSIFICATION_CONFIDENCE < 1.0
