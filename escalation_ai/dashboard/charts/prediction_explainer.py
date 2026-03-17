"""
SHAP-based Prediction Explainability — Waterfall charts for Random Forest outputs.

Provides interactive SHAP waterfall visualizations that explain WHY the
recurrence and resolution time models made specific predictions for individual
tickets.  Uses TreeExplainer for efficient exact SHAP values on tree-based
models (Random Forest / Gradient Boosting).

Integration:
    Called from the Distributions tab in analytics_view.py to add a
    "Why this prediction?" section beneath the existing prediction charts.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from escalation_ai.dashboard.shared_helpers import create_plotly_theme

logger = logging.getLogger(__name__)

# Human-readable labels for encoded feature names
_FEATURE_LABELS = {
    'Strategic_Friction_Score': 'Friction Score',
    'AI_Confidence': 'AI Confidence',
    'Engineer_Issue_Count': 'Engineer Issue Count',
    'Days_Since_Issue': 'Days Since Issue',
    'Recidivism_Score': 'Recidivism Score',
    'Severity_Norm_encoded': 'Severity',
    'Type_Norm_encoded': 'Issue Type',
    'Origin_Norm_encoded': 'Origin',
    'Root_Cause_Category_encoded': 'Root Cause',
    'AI_Category_encoded': 'AI Category',
    'LOB_Risk_Tier_encoded': 'LOB Risk Tier',
    'Is_Human_Error_encoded': 'Human Error',
    'Is_Repeat_Offender': 'Repeat Offender',
    'Is_Aged': 'Ticket Aging',
    # Resolution model features
    'category_hash': 'Category',
    'category_avg_days': 'Category Avg Days',
    'category_median_days': 'Category Median Days',
    'severity_level': 'Severity Level',
    'text_length': 'Description Length',
    'word_count': 'Word Count',
    'complexity_score': 'Complexity Score',
    'similar_resolution_days': 'Similar Ticket Resolution',
    'similar_count': 'Similar Ticket Count',
    'ai_confidence': 'AI Confidence',
    'recurrence_risk': 'Recurrence Risk',
}


def _friendly_name(feature: str) -> str:
    """Map internal feature name to a human-readable label."""
    if feature in _FEATURE_LABELS:
        return _FEATURE_LABELS[feature]
    if feature.startswith('emb_dim_'):
        return f'Embedding Dim {feature.split("_")[-1]}'
    return feature.replace('_', ' ').title()


def _build_waterfall(shap_values: np.ndarray, feature_names: list[str],
                     base_value: float, title: str, top_n: int = 10) -> go.Figure:
    """Build a Plotly horizontal waterfall from SHAP values for one sample."""
    importance = pd.Series(shap_values, index=feature_names)
    # Keep top N by absolute magnitude
    top = importance.reindex(importance.abs().nlargest(top_n).index)
    # Sort so largest positive is at top
    top = top.sort_values(ascending=True)

    labels = [_friendly_name(f) for f in top.index]
    values = top.values

    colors = ['#ef4444' if v > 0 else '#22c55e' for v in values]

    theme = create_plotly_theme()

    fig = go.Figure(go.Bar(
        y=labels,
        x=values,
        orientation='h',
        marker_color=colors,
        hovertemplate='<b>%{y}</b><br>SHAP value: %{x:.4f}<extra></extra>',
    ))

    fig.update_layout(
        **theme,
        title=dict(text=title, font=dict(size=14)),
        xaxis_title='SHAP Value (impact on prediction)',
        yaxis_title='',
        height=max(300, top_n * 35 + 80),
        showlegend=False,
    )

    # Add annotation for base value
    fig.add_annotation(
        text=f"Base value: {base_value:.3f}",
        xref="paper", yref="paper",
        x=1.0, y=-0.08,
        showarrow=False,
        font=dict(size=11, color='#94a3b8'),
    )

    return fig


def _get_sklearn_model(wrapper_model):
    """Extract the underlying sklearn/cuML model from the GPU wrapper."""
    if hasattr(wrapper_model, 'model'):
        return wrapper_model.model
    return wrapper_model


def render_recurrence_explanation(df: pd.DataFrame, row_index: int = 0):
    """Show SHAP waterfall for a single ticket's recurrence prediction.

    Loads the saved recurrence model from disk, computes SHAP values for the
    selected ticket, and renders a horizontal bar chart of the top contributing
    features.

    Args:
        df: The processed DataFrame with all feature columns.
        row_index: Index into df for the ticket to explain.
    """
    try:
        import shap
        import joblib
        from pathlib import Path
        from escalation_ai.core.config import RECURRENCE_MODEL_PATH, RECURRENCE_ENCODERS_PATH
    except ImportError:
        st.info("Install SHAP for prediction explainability: `pip install shap`")
        return

    model_path = Path(RECURRENCE_MODEL_PATH)
    encoders_path = Path(RECURRENCE_ENCODERS_PATH)

    if not model_path.exists():
        st.info("Recurrence model not found. Run the full pipeline first to train the model.")
        return

    try:
        data = joblib.load(model_path)
        model_wrapper = data['model']
        feature_columns = data['feature_columns']
        encoders = joblib.load(encoders_path) if encoders_path.exists() else {}

        sklearn_model = _get_sklearn_model(model_wrapper)

        # Re-prepare features using the same logic as the predictor
        from escalation_ai.predictors.recurrence import RecurrencePredictor
        predictor = RecurrencePredictor()
        predictor.encoders = encoders
        predictor.feature_columns = feature_columns

        X, feat_names = predictor._prepare_features(df, fit_encoders=False)

        if X.shape[0] == 0 or X.shape[1] == 0:
            st.warning("Cannot explain predictions: feature matrix is empty.")
            return

        # Compute SHAP values using TreeExplainer (exact for tree models)
        explainer = shap.TreeExplainer(sklearn_model)
        shap_values = explainer.shap_values(X[[row_index]])

        # Handle different SHAP output formats across versions:
        # - SHAP <0.50: list of [class_0_array, class_1_array]
        # - SHAP >=0.50: ndarray of shape (n_samples, n_features, n_classes)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # class 1 = recurrence
            base = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        elif shap_values.ndim == 3:
            sv = shap_values[0, :, 1]  # sample 0, all features, class 1
            base = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        else:
            sv = shap_values[0]
            base = explainer.expected_value
            if isinstance(base, (list, np.ndarray)):
                base = base[0]

        prob = df.iloc[row_index].get('AI_Recurrence_Probability', None)
        title_suffix = f" (predicted: {prob:.0%})" if prob is not None else ""

        fig = _build_waterfall(sv, feat_names, float(base),
                               f"Top Factors Driving Recurrence Prediction{title_suffix}")
        st.plotly_chart(fig, use_container_width=True)

        # Show interpretation hint
        st.caption("Red bars push the prediction toward recurrence; "
                   "green bars push toward non-recurrence.")

    except Exception as e:
        logger.warning(f"SHAP recurrence explanation failed: {e}", exc_info=True)
        st.warning(f"Explainability unavailable: {type(e).__name__}: {e}")


def render_resolution_explanation(df: pd.DataFrame, row_index: int = 0):
    """Show SHAP waterfall for a single ticket's resolution time prediction.

    Uses the resolution time predictor's feature extraction to build the feature
    matrix, then computes SHAP values against the trained Random Forest regressor.

    Args:
        df: The processed DataFrame.
        row_index: Index into df for the ticket to explain.
    """
    try:
        import shap
        from escalation_ai.predictors.resolution_time import resolution_time_predictor
    except ImportError:
        st.info("Install SHAP for prediction explainability: `pip install shap`")
        return

    if resolution_time_predictor is None or not resolution_time_predictor.is_trained:
        st.info("Resolution model not trained. Run the full pipeline first.")
        return

    try:
        predictor = resolution_time_predictor
        sklearn_model = _get_sklearn_model(predictor.model)
        feature_columns = predictor.feature_columns

        # Build feature matrix for all rows (SHAP needs background data)
        feature_rows = []
        for _, row in df.iterrows():
            features = predictor._extract_features(row)
            feature_rows.append([features.get(c, 0) for c in feature_columns])

        X = np.array(feature_rows)
        if X.shape[0] == 0:
            st.warning("Cannot explain predictions: no valid feature data.")
            return

        explainer = shap.TreeExplainer(sklearn_model)
        shap_values = explainer.shap_values(X[[row_index]])

        sv = shap_values[0] if shap_values.ndim > 1 else shap_values
        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = base[0]

        predicted = df.iloc[row_index].get('Predicted_Resolution_Days', None)
        title_suffix = f" (predicted: {predicted:.1f} days)" if predicted is not None else ""

        fig = _build_waterfall(sv, feature_columns, float(base),
                               f"Top Factors Driving Resolution Time{title_suffix}")
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Red bars increase predicted resolution time; "
                   "green bars decrease it.")

    except Exception as e:
        logger.warning(f"SHAP resolution explanation failed: {e}", exc_info=True)
        st.warning(f"Explainability unavailable: {type(e).__name__}: {e}")


def render_prediction_explainer(df: pd.DataFrame):
    """Render the full SHAP prediction explainability section.

    Provides a ticket selector and expandable SHAP explanations for both
    the recurrence and resolution time models.

    Args:
        df: The processed DataFrame with prediction columns.
    """
    st.markdown("### Prediction Explainability")
    st.markdown("*Understand WHY the model made a specific prediction using SHAP analysis*")

    # Build ticket selector with meaningful labels
    has_recurrence = 'AI_Recurrence_Probability' in df.columns
    has_resolution = 'Predicted_Resolution_Days' in df.columns

    if not has_recurrence and not has_resolution:
        st.info("No prediction columns found. Run the full pipeline to generate predictions.")
        return

    # Create display labels for the selectbox
    id_col = None
    for col in ['Identity', 'Ticket_ID', 'ID']:
        if col in df.columns:
            id_col = col
            break

    if id_col:
        options = {i: f"{row[id_col]} — {row.get('AI_Category', 'N/A')}" for i, row in df.iterrows()}
    else:
        options = {i: f"Ticket #{i+1} — {row.get('AI_Category', 'N/A')}" for i, row in df.iterrows()}

    selected_idx = st.selectbox(
        "Select ticket to explain",
        options=list(options.keys()),
        format_func=lambda x: options[x],
        key="shap_ticket_selector",
    )

    # Show ticket summary
    row = df.iloc[selected_idx] if isinstance(selected_idx, int) else df.loc[selected_idx]
    summary_cols = ['AI_Category', 'Severity_Norm', 'AI_Recurrence_Risk', 'Predicted_Resolution_Days']
    summary_items = {col: row.get(col, 'N/A') for col in summary_cols if col in df.columns}
    if summary_items:
        cols = st.columns(len(summary_items))
        for col_ui, (label, value) in zip(cols, summary_items.items()):
            display_label = label.replace('_', ' ').replace('AI ', '').replace('Predicted ', '')
            with col_ui:
                st.metric(display_label, str(value))

    # Recurrence SHAP
    if has_recurrence:
        with st.expander("Why this recurrence prediction?", expanded=True):
            # Convert to positional index for array slicing
            pos_idx = df.index.get_loc(selected_idx) if selected_idx in df.index else selected_idx
            render_recurrence_explanation(df, pos_idx)

    # Resolution SHAP
    if has_resolution:
        with st.expander("Why this resolution time prediction?"):
            pos_idx = df.index.get_loc(selected_idx) if selected_idx in df.index else selected_idx
            render_resolution_explanation(df, pos_idx)
