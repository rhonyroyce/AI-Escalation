"""
Pulse Dashboard - Data Loading & Cleaning
==========================================

This module is the single entry point for all data ingestion in the Pulse
Dashboard.  It reads the "ProjectPulse.xlsx" workbook (either uploaded by the
user through Streamlit's file_uploader widget or loaded from a default location
on disk), validates its structure, cleans the raw cell values, and derives
several calculated columns that the rest of the dashboard depends on.

Data Flow
---------
1. Excel file  -->  pd.ExcelFile (validates the file can be opened)
2. Sheet "Project Pulse"  -->  raw DataFrame
3. Drop junk columns ('Project Pulse' header row artifact, 'Unnamed: 28')
4. Validate that every REQUIRED column is present
5. Coerce score columns to integers (handle \xa0 non-breaking spaces from Excel)
6. Clean free-text columns (strip whitespace, normalise blanks to NaN)
7. Derive helper columns:
   - Pulse_Status  : categorical label  ('Red' | 'Yellow' | 'Green' | 'Dark Green')
   - Pulse_Color   : hex color string for the status
   - Year_Week     : 'YYYY-Wnn' string for correct chronological sorting
   - Effort        : integer count of "struggling" dimensions (score < 2)

Caching
-------
`load_pulse_data` is decorated with `@st.cache_data` so that repeated calls
with the same file (same bytes / same path) return a cached DataFrame instead
of re-reading and re-cleaning the Excel file on every Streamlit rerun.

Column Reference
----------------
Score dimensions (each 0-3):
    Design, IX, PAG, RF Opt, Field, CSAT, PM Performance, Potential
    - 0 = Escalation / Critical
    - 1 = Issues / Complaint
    - 2 = On-track / BAU
    - 3 = Exceptional / Appreciation

Total Score (0-24): sum of the 8 dimensions above.

Pulse Status thresholds (defined in styles.py):
    Red         :  1-13   (Critical)
    Yellow      : 14-15   (At Risk)
    Green       : 16-19   (On Track)
    Dark Green  : 20-24   (Exceptional)

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

# Import scoring helpers and dimension list from the shared styles module.
# get_pulse_status maps Total Score -> status label ('Red', 'Yellow', etc.)
# get_pulse_color  maps Total Score -> hex color string
# SCORE_DIMENSIONS is the canonical list of 8 scoring dimension column names
from utils.styles import get_pulse_status, get_pulse_color, SCORE_DIMENSIONS

# ── Column Lists ─────────────────────────────────────────────────────────────
# These lists define which columns get special treatment during cleaning.

# DROP_COLUMNS: artefact columns created by Excel formatting.
# 'Project Pulse' is a merged-cell title row that pandas reads as a column,
# and 'Unnamed: 28' is a trailing empty column from extra formatting.
DROP_COLUMNS = ['Project Pulse', 'Unnamed: 28']

# SCORE_COLS: the 8 individual dimension scores (each rated 0-3).
# Aliased from SCORE_DIMENSIONS so both names can be used interchangeably.
SCORE_COLS = SCORE_DIMENSIONS  # ['Design', 'IX', 'PAG', 'RF Opt', 'Field', 'CSAT', 'PM Performance', 'Potential']

# ALL_SCORE_COLS: the 8 dimensions PLUS the aggregate Total Score column.
# All of these need numeric coercion because Excel may store them as text
# or contain non-breaking spaces (\xa0) in empty cells.
ALL_SCORE_COLS = SCORE_COLS + ['Total Score']

# TEXT_COLS: primary free-text narrative columns that PMs fill in each week.
# These contain multi-line comments, bullet points, and sometimes pasted text
# with stray \xa0 or tab characters that need cleaning.
TEXT_COLS = ['Comments', 'Pain Points', 'Resolution Plan']

# REQUIRED_COLS: minimum set of columns that MUST exist for the dashboard to
# function.  If any are missing the loader raises ValueError so the user sees
# a clear error rather than a cryptic KeyError downstream.
REQUIRED_COLS = ['Year', 'Wk', 'Region', 'Area', 'Project', 'PM Name', 'Total Score']


def get_default_file_path() -> Path | None:
    """Return the Path to ProjectPulse.xlsx at the project root, or None.

    The project root is determined by navigating two directories up from this
    file's location:
        utils/data_loader.py  ->  utils/  ->  pulse_dashboard/  ->  project root

    This allows the dashboard to auto-load the workbook without requiring the
    user to upload it every time, which is convenient during development and
    for recurring weekly use.

    Returns:
        Path object if the file exists at the expected location, else None.
    """
    # __file__ is pulse_dashboard/utils/data_loader.py
    # .parent      = pulse_dashboard/utils/
    # .parent      = pulse_dashboard/
    # .parent      = project root (AI-Escalation/)
    root = Path(__file__).parent.parent.parent
    candidate = root / 'ProjectPulse.xlsx'
    return candidate if candidate.exists() else None


def _clean_text_series(s: pd.Series) -> pd.Series:
    """Clean a text column: replace \\xa0, strip whitespace, keep \\n for line breaks.

    Excel exports often contain non-breaking spaces (\\xa0 / &nbsp;) and stray
    tab characters that look invisible but break string comparisons and display.
    This helper normalises a whole Series in one pass.

    The pipeline:
    1. Cast everything to str (handles mixed int/float/NaN cells).
    2. Replace \\xa0 (non-breaking space) with regular space.
    3. Remove tab characters entirely (they serve no purpose in our text).
    4. Strip leading/trailing whitespace from each cell.
    5. Map the string representations of missing values ('nan', '', 'None')
       back to proper np.nan so that pandas .isna() / .dropna() work correctly.

    Newlines (\\n) are intentionally preserved because PMs use them for
    multi-line bullet lists in Comments / Pain Points / Resolution Plan.

    Args:
        s: A pandas Series containing raw text from an Excel column.

    Returns:
        A cleaned Series with consistent whitespace and proper NaN values.
    """
    return (
        s.astype(str)
        .str.replace('\xa0', ' ', regex=False)   # Non-breaking space -> regular space
        .str.replace('\t', '', regex=False)       # Remove tab characters
        .str.strip()                              # Trim leading/trailing whitespace
        .replace({'nan': np.nan, '': np.nan, 'None': np.nan})  # Restore proper NaN
    )


@st.cache_data
def load_pulse_data(file) -> pd.DataFrame:
    """Load and clean pulse data from Excel file or path.

    This is the primary data loading function for the entire dashboard.
    It is cached by Streamlit so that the expensive Excel parse + clean
    pipeline only runs once per unique file.  When the user uploads a new
    file or the default file changes on disk, the cache key changes and
    the function re-executes.

    Args:
        file: Either a Streamlit UploadedFile object (from st.file_uploader),
              or a string/Path pointing to a local .xlsx file.  Both are
              accepted because pd.ExcelFile() handles both transparently.

    Returns:
        Cleaned DataFrame with all original columns plus derived columns:
        - Pulse_Status (str): 'Red', 'Yellow', 'Green', or 'Dark Green'
        - Pulse_Color (str): hex color code matching the status
        - Year_Week (str): 'YYYY-Wnn' sortable time key
        - Effort (int): count of dimensions scoring below 2

    Raises:
        ValueError: If the file cannot be opened, if the required
                    'Project Pulse' sheet is not found, or if any
                    column listed in REQUIRED_COLS is missing.
    """
    # ── Step 1: Open the Excel file ──────────────────────────────────────
    # pd.ExcelFile defers actual sheet parsing until read_excel() is called,
    # but it does validate the file is a readable Excel workbook.
    try:
        xl = pd.ExcelFile(file)
    except Exception as e:
        raise ValueError(f"Cannot open file: {e}")

    # ── Step 2: Validate sheet existence ─────────────────────────────────
    # The workbook must contain a sheet named exactly 'Project Pulse'.
    # Other sheets (e.g. lookup tables, historical data) are ignored.
    if 'Project Pulse' not in xl.sheet_names:
        raise ValueError(
            f"Sheet 'Project Pulse' not found. Available sheets: {xl.sheet_names}"
        )

    # Parse the target sheet into a DataFrame.
    df = pd.read_excel(xl, sheet_name='Project Pulse')

    # ── Step 3: Drop artefact columns ────────────────────────────────────
    # These columns are side-effects of how the Excel workbook is formatted
    # (merged header cells, trailing empty columns).  They carry no data and
    # would clutter downstream processing if left in.
    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # ── Step 4: Validate required columns ────────────────────────────────
    # Fail fast with a clear message rather than letting the dashboard crash
    # later with an opaque KeyError when a page tries to access a column.
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── Step 5: Coerce score columns to integers ─────────────────────────
    # Score columns in the Excel file can contain:
    #   - Actual integers (0, 1, 2, 3 for dimensions; 0-24 for Total Score)
    #   - Non-breaking space '\xa0' in cells that the PM left blank
    #   - Text strings if Excel auto-formatted the cell
    # We replace \xa0 with NaN, coerce everything to numeric, fill remaining
    # NaN with 0 (a blank score means "not scored" which defaults to worst),
    # and cast to int for clean display and arithmetic.
    for col in ALL_SCORE_COLS:
        if col in df.columns:
            df[col] = df[col].replace('\xa0', np.nan)                    # Non-breaking space -> NaN
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)  # Force integer

    # ── Step 6: Clean primary text fields ────────────────────────────────
    # These are the three main narrative columns that PMs write each week.
    # Clean them to remove invisible characters and normalise empty values.
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = _clean_text_series(df[col])

    # Also clean supplementary text columns if they exist in this workbook.
    # Not all workbook versions include these columns, so we check each one.
    # These columns capture escalation details, concerns, and action items.
    extra_text = ['Issue', 'Pending Action', 'Owner',
                  'Concerns (External)', 'Concerns (Internal)',
                  'Escalations (External)', 'Escalations (Internal)',
                  'Potential Escalations']
    for col in extra_text:
        if col in df.columns:
            df[col] = _clean_text_series(df[col])

    # ── Step 7: Derive calculated columns ────────────────────────────────

    # Pulse_Status: categorical bucket based on Total Score thresholds.
    # Used for color coding, filtering, and grouping throughout the dashboard.
    # Thresholds: Red (1-13), Yellow (14-15), Green (16-19), Dark Green (20-24)
    df['Pulse_Status'] = df['Total Score'].apply(get_pulse_status)

    # Pulse_Color: hex color string corresponding to the Pulse_Status.
    # Pre-computed here so chart code can use it directly without re-mapping.
    df['Pulse_Color'] = df['Total Score'].apply(get_pulse_color)

    # Year_Week: composite sort key that handles year boundaries correctly.
    # Format: "2025-W03" ensures lexicographic sort matches chronological order.
    # Without the year prefix, week 52 of one year would sort after week 01
    # of the next year, breaking time-series charts.
    # .str.zfill(2) pads single-digit weeks: 1 -> "01", 9 -> "09", 12 -> "12"
    df['Year_Week'] = df['Year'].astype(str) + '-W' + df['Wk'].astype(str).str.zfill(2)

    # Effort: how many of the 8 dimensions are scoring below 2 (the "on-track"
    # threshold).  A score < 2 means that dimension is flagged as having issues
    # or an escalation.  A high Effort value signals that the project needs
    # attention across multiple areas simultaneously, making it a useful triage
    # metric for managers deciding where to allocate support.
    df['Effort'] = (df[SCORE_COLS] < 2).sum(axis=1)

    return df
