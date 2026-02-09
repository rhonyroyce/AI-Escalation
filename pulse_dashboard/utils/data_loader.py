"""
Pulse Dashboard - Data Loading & Cleaning

Handles:
- Excel loading from file uploader or default path
- Data cleaning (\xa0, mixed types, text normalization)
- Derived column generation (Pulse_Status, Year_Week, Effort)
- Caching via Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from utils.styles import get_pulse_status, get_pulse_color, SCORE_DIMENSIONS

DROP_COLUMNS = ['Project Pulse', 'Unnamed: 28']
SCORE_COLS = SCORE_DIMENSIONS  # ['Design', 'IX', 'PAG', 'RF Opt', 'Field', 'CSAT', 'PM Performance', 'Potential']
ALL_SCORE_COLS = SCORE_COLS + ['Total Score']
TEXT_COLS = ['Comments', 'Pain Points', 'Resolution Plan']
REQUIRED_COLS = ['Year', 'Wk', 'Region', 'Area', 'Project', 'PM Name', 'Total Score']


def get_default_file_path() -> Path | None:
    """Return Path to ProjectPulse.xlsx at project root, or None."""
    root = Path(__file__).parent.parent.parent
    candidate = root / 'ProjectPulse.xlsx'
    return candidate if candidate.exists() else None


def _clean_text_series(s: pd.Series) -> pd.Series:
    """Clean a text column: replace \xa0, strip whitespace, keep \n for line breaks."""
    return (
        s.astype(str)
        .str.replace('\xa0', ' ', regex=False)
        .str.replace('\t', '', regex=False)
        .str.strip()
        .replace({'nan': np.nan, '': np.nan, 'None': np.nan})
    )


@st.cache_data
def load_pulse_data(file) -> pd.DataFrame:
    """Load and clean pulse data from Excel file or path.

    Args:
        file: UploadedFile from st.file_uploader, or a file path string/Path.

    Returns:
        Cleaned DataFrame with derived columns.

    Raises:
        ValueError: If 'Project Pulse' sheet not found or required columns missing.
    """
    # Read the Excel file
    try:
        xl = pd.ExcelFile(file)
    except Exception as e:
        raise ValueError(f"Cannot open file: {e}")

    if 'Project Pulse' not in xl.sheet_names:
        raise ValueError(
            f"Sheet 'Project Pulse' not found. Available sheets: {xl.sheet_names}"
        )

    df = pd.read_excel(xl, sheet_name='Project Pulse')

    # Drop useless columns
    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Check required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Fix score columns: replace \xa0 and coerce to numeric
    for col in ALL_SCORE_COLS:
        if col in df.columns:
            df[col] = df[col].replace('\xa0', np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Clean text fields
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = _clean_text_series(df[col])

    # Also clean supplementary text columns if present
    extra_text = ['Issue', 'Pending Action', 'Owner',
                  'Concerns (External)', 'Concerns (Internal)',
                  'Escalations (External)', 'Escalations (Internal)',
                  'Potential Escalations']
    for col in extra_text:
        if col in df.columns:
            df[col] = _clean_text_series(df[col])

    # Add derived columns
    df['Pulse_Status'] = df['Total Score'].apply(get_pulse_status)
    df['Pulse_Color'] = df['Total Score'].apply(get_pulse_color)

    # Year_Week for correct time series sorting (handles year boundary)
    df['Year_Week'] = df['Year'].astype(str) + '-W' + df['Wk'].astype(str).str.zfill(2)

    # Effort: count of dimensions scoring below 2
    df['Effort'] = (df[SCORE_COLS] < 2).sum(axis=1)

    return df
