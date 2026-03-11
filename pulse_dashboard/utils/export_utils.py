"""
Pulse Dashboard - Export Utilities
===================================
Helper functions for CSV and PNG download buttons.
Used by: 1_Executive_Summary.py, 6_Project_Details.py, 3_Trends.py.
"""
import io
from typing import Optional

import streamlit as st
import pandas as pd


def download_csv_button(
    df: pd.DataFrame,
    filename: str = "export.csv",
    label: str = "Download CSV",
    key: Optional[str] = None,
) -> None:
    """Render a Streamlit download button for a DataFrame as CSV."""
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label=label,
        data=csv_buffer.getvalue(),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def download_chart_png(
    fig,
    filename: str = "chart.png",
    label: str = "Download Chart",
    key: Optional[str] = None,
    width: int = 1200,
    height: int = 600,
) -> None:
    """Render a Streamlit download button for a Plotly figure as PNG.

    Requires the kaleido package. Shows a helpful message if missing.
    """
    try:
        img_bytes = fig.to_image(format="png", width=width, height=height)
        st.download_button(
            label=label,
            data=img_bytes,
            file_name=filename,
            mime="image/png",
            key=key,
        )
    except Exception as e:
        if "kaleido" in str(e).lower() or "orca" in str(e).lower():
            st.caption("PNG export requires kaleido: `pip install kaleido`")
        else:
            st.caption(f"Chart export unavailable: {e}")
