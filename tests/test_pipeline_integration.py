"""
End-to-end integration test for the 7-phase Escalation AI pipeline.

Runs the full EscalationPipeline with synthetic data and mocked Ollama API
to catch regression bugs without requiring a live LLM server.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.fixtures.synthetic_tickets import generate_synthetic_tickets


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBED_DIM = 768  # Dimensionality for fake embeddings
NUM_TICKETS = 50

VALID_CATEGORIES = {
    "Scheduling & Planning",
    "Documentation & Reporting",
    "Validation & QA",
    "Process Compliance",
    "Configuration & Data Mismatch",
    "Site Readiness",
    "Communication & Response",
    "Nesting & Tool Errors",
    "Unclassified",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_deterministic_embedding(text, dim=EMBED_DIM):
    """Generate a deterministic fake embedding from text via hashing."""
    # Use hash of text to seed a PRNG for reproducible vectors
    seed = hash(text) % (2**31)
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float64)
    # Normalize to unit length (like real embeddings)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _mock_ollama_post(url, json=None, timeout=None, **kwargs):
    """Mock requests.post for Ollama embed and generate endpoints."""
    response = MagicMock()
    response.status_code = 200

    if '/api/embed' in url:
        payload = json or {}
        input_data = payload.get('input', '')

        # Handle batch embedding (list of strings)
        if isinstance(input_data, list):
            embeddings = [
                _make_deterministic_embedding(str(t)).tolist()
                for t in input_data
            ]
            response.json.return_value = {'embeddings': embeddings}
        else:
            # Single string embedding
            embedding = _make_deterministic_embedding(str(input_data)).tolist()
            response.json.return_value = {'embedding': embedding}

    elif '/api/generate' in url:
        response.json.return_value = {
            'response': (
                "## Executive Summary\n\n"
                "This is a mock executive summary for integration testing.\n"
                "Total tickets analyzed: 50. Key finding: scheduling issues "
                "represent the largest category of escalations.\n\n"
                "### Recommendations\n"
                "1. Improve TI scheduling compliance\n"
                "2. Enhance documentation processes\n"
                "3. Strengthen vendor management protocols\n"
            )
        }
    else:
        response.status_code = 404
        response.json.return_value = {'error': 'not found'}

    return response


def _mock_ollama_get(url, timeout=None, **kwargs):
    """Mock requests.get for Ollama health check."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {'models': []}
    return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_excel(tmp_path):
    """Generate the synthetic Excel file in a temp directory."""
    excel_path = tmp_path / "synthetic_input.xlsx"
    generate_synthetic_tickets(excel_path, n_rows=NUM_TICKETS)
    return excel_path


@pytest.fixture
def pipeline_patches():
    """Context manager that patches all external dependencies."""
    patches = [
        # Mock all requests to Ollama
        patch('requests.post', side_effect=_mock_ollama_post),
        patch('requests.get', side_effect=_mock_ollama_get),

        # Mock tkinter (prevent GUI popups)
        patch('tkinter.Tk'),
        patch('tkinter.filedialog.askopenfilename', return_value=''),
        patch('tkinter.filedialog.asksaveasfilename', return_value=''),
        patch('tkinter.messagebox.showerror'),
        patch('tkinter.messagebox.showinfo'),

        # Mock orchestrator-level tkinter references
        patch(
            'escalation_ai.pipeline.orchestrator.messagebox',
            MagicMock()
        ),

        # Force CPU path — avoids cuML/cupy CUDA compilation errors in CI/test
        patch('escalation_ai.core.gpu_utils._check_cuml', return_value=False),
        patch('escalation_ai.core.gpu_utils._CUML_AVAILABLE', False),

        # Mock subprocess/os.system if used
        patch('os.system', return_value=0),
    ]

    started = []
    for p in patches:
        started.append(p.start())

    yield started

    for p in patches:
        p.stop()


@pytest.fixture
def pipeline_instance(synthetic_excel, pipeline_patches):
    """Create and initialize an EscalationPipeline with all mocks active."""
    from escalation_ai.pipeline.orchestrator import EscalationPipeline

    pipe = EscalationPipeline()
    pipe.show_progress = False  # Suppress tqdm output

    # Initialize: connects to (mocked) Ollama, loads feedback/pricing
    result = pipe.initialize()
    assert result is True, "Pipeline initialization failed with mocked Ollama"

    # Load synthetic data
    result = pipe.load_data(str(synthetic_excel))
    assert result is True, "Pipeline data loading failed"
    assert len(pipe.df) == NUM_TICKETS

    return pipe


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFullPipelineRun:
    """Test 1: Full pipeline runs to completion without exception."""

    def test_run_all_phases_completes(self, pipeline_instance):
        """Run all 6 analysis phases — must not raise."""
        pipe = pipeline_instance
        result_df = pipe.run_all_phases()
        assert result_df is not None
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == NUM_TICKETS

    def test_generate_executive_summary_completes(self, pipeline_instance):
        """Run phases + executive summary (Phase 7) — must not raise."""
        pipe = pipeline_instance
        pipe.run_all_phases()
        summary = pipe.generate_executive_summary()
        assert summary is not None
        assert isinstance(summary, str)
        assert len(summary) > 0


class TestOutputSchema:
    """Test 2: Verify the output DataFrame contains all expected enrichment columns."""

    @pytest.fixture(autouse=True)
    def _run_pipeline(self, pipeline_instance):
        """Run the full pipeline once for all schema tests."""
        self.pipe = pipeline_instance
        self.df = self.pipe.run_all_phases()

    def test_ai_category_column(self):
        assert 'AI_Category' in self.df.columns
        assert self.df['AI_Category'].notna().all(), "AI_Category has nulls"
        invalid = set(self.df['AI_Category'].unique()) - VALID_CATEGORIES
        assert not invalid, f"Unknown categories: {invalid}"

    def test_ai_confidence_column(self):
        assert 'AI_Confidence' in self.df.columns
        confidence = self.df['AI_Confidence'].astype(float)
        assert (confidence >= 0).all(), "AI_Confidence has values < 0"
        assert (confidence <= 1).all(), "AI_Confidence has values > 1"

    def test_ai_sub_category_column(self):
        assert 'AI_Sub_Category' in self.df.columns
        assert self.df['AI_Sub_Category'].notna().all(), "AI_Sub_Category has nulls"

    def test_strategic_friction_score_column(self):
        assert 'Strategic_Friction_Score' in self.df.columns
        scores = self.df['Strategic_Friction_Score'].astype(float)
        assert (scores > 0).any(), "All friction scores are 0"

    def test_learning_status_column(self):
        assert 'Learning_Status' in self.df.columns
        assert self.df['Learning_Status'].notna().all(), "Learning_Status has nulls"

    def test_recidivism_score_column(self):
        assert 'Recidivism_Score' in self.df.columns
        scores = pd.to_numeric(self.df['Recidivism_Score'], errors='coerce')
        assert scores.notna().all(), "Recidivism_Score has non-numeric values"

    def test_ai_recurrence_probability_column(self):
        assert 'AI_Recurrence_Probability' in self.df.columns
        probs = self.df['AI_Recurrence_Probability'].astype(float)
        assert (probs >= 0).all(), "AI_Recurrence_Probability has values < 0"
        assert (probs <= 1).all(), "AI_Recurrence_Probability has values > 1"

    def test_ai_recurrence_risk_tier_column(self):
        # Pipeline uses 'AI_Recurrence_Risk' (not 'AI_Recurrence_Risk_Tier')
        assert 'AI_Recurrence_Risk' in self.df.columns
        assert self.df['AI_Recurrence_Risk'].notna().all()

    def test_predicted_resolution_days_column(self):
        # Pipeline uses 'Predicted_Resolution_Days' (not 'AI_Predicted_Resolution_Hours')
        assert 'Predicted_Resolution_Days' in self.df.columns
        days = self.df['Predicted_Resolution_Days'].astype(float)
        assert (days >= 0).all(), "Predicted resolution days has negative values"


class TestOutputRowCount:
    """Test 3: Input 50 rows, output should still be 50 rows."""

    def test_no_rows_lost_or_duplicated(self, pipeline_instance):
        pipe = pipeline_instance
        input_count = len(pipe.df)
        result_df = pipe.run_all_phases()
        assert len(result_df) == input_count, (
            f"Row count changed: input={input_count}, output={len(result_df)}"
        )


class TestIdempotency:
    """Test 5: Pipeline produces identical output on two runs with same input."""

    def test_deterministic_output(self, synthetic_excel, pipeline_patches):
        """Run pipeline twice on same input — outputs must match."""
        from escalation_ai.pipeline.orchestrator import EscalationPipeline

        results = []
        for run_idx in range(2):
            pipe = EscalationPipeline()
            pipe.show_progress = False

            assert pipe.initialize() is True
            assert pipe.load_data(str(synthetic_excel)) is True

            df = pipe.run_all_phases()
            results.append(df.copy())

        df1, df2 = results

        # Compare key output columns (skip 'embedding' which contains numpy arrays)
        compare_cols = [
            'AI_Category', 'AI_Confidence', 'AI_Sub_Category',
            'Strategic_Friction_Score', 'Learning_Status', 'Recidivism_Score',
            'AI_Recurrence_Probability', 'AI_Recurrence_Risk',
            'Predicted_Resolution_Days',
        ]

        for col in compare_cols:
            if col in df1.columns and col in df2.columns:
                # Convert to comparable types
                s1 = df1[col].astype(str).tolist()
                s2 = df2[col].astype(str).tolist()
                assert s1 == s2, (
                    f"Column '{col}' differs between runs:\n"
                    f"  Run 1 sample: {s1[:5]}\n"
                    f"  Run 2 sample: {s2[:5]}"
                )
