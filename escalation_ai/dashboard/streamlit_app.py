"""
Escalation AI - Executive Intelligence Dashboard

McKinsey-grade executive dashboard with:
- C-Suite Executive Summary with strategic recommendations
- Financial Impact Analysis with ROI calculations
- Predictive Intelligence with 30/60/90 day forecasts
- Competitive Benchmarking vs industry standards
- Root Cause Analysis with Pareto & driver trees
- Action Tracker with RACI and progress monitoring
- Executive Presentation Mode with auto-cycling slides
- Real-time KPI metrics with pulse indicators
- Interactive Plotly charts
- Category drift visualization
- Smart alert monitoring
- What-If scenario simulator
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sys
import json
import time
import base64
import io
import zipfile
from streamlit_js_eval import streamlit_js_eval

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import advanced charts
from escalation_ai.dashboard.advanced_plotly_charts import (
    chart_sla_funnel, chart_engineer_quadrant, chart_cost_waterfall,
    chart_time_heatmap, chart_aging_analysis, chart_health_gauge,
    chart_resolution_consistency, chart_recurrence_patterns,
    # Sub-category drill-down charts
    chart_category_sunburst as advanced_category_sunburst,
    chart_category_treemap, chart_subcategory_breakdown,
    chart_category_financial_drilldown, chart_subcategory_comparison_table
)

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Escalation AI | Executive Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'presentation_mode' not in st.session_state:
    st.session_state.presentation_mode = False
if 'current_slide' not in st.session_state:
    st.session_state.current_slide = 0
if 'action_items' not in st.session_state:
    st.session_state.action_items = []
if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None

# ============================================================================
# PRICE CATALOG - LOAD COSTS FROM EXCEL
# ============================================================================

def _get_price_catalog_mtime():
    """Get modification time of price_catalog.xlsx for cache invalidation."""
    import os
    paths = [
        '/home/k8s/Projects/AI-Escalation/price_catalog.xlsx',
        'price_catalog.xlsx',
    ]
    for p in paths:
        if os.path.exists(p):
            return os.path.getmtime(p)
    return 0

@st.cache_data(ttl=60)  # Refresh every 60 seconds, or when file changes
def load_price_catalog(_mtime=None):
    """Load pricing data from price_catalog.xlsx for all cost calculations.

    Args:
        _mtime: File modification time (used for cache invalidation)
    """
    import os

    catalog_data = {
        'category_costs': {},
        'severity_multipliers': {'critical': 2.5, 'high': 1.75, 'medium': 1.25, 'low': 1.0, 'major': 1.75, 'minor': 1.0},
        'origin_premiums': {'vendor': 0.15, 'process': 0.05, 'external': 0.20, 'customer': 0.10, 'technical': 0.0, 'internal': 0.0},
        'avg_cost_per_ticket': 0,
        'benchmark_best': 0,
        'benchmark_avg': 0,
        'benchmark_laggard': 0,
        'hourly_rate': 20,
    }

    # Find price_catalog.xlsx
    price_paths = [
        '/home/k8s/Projects/AI-Escalation/price_catalog.xlsx',
        'price_catalog.xlsx',
        Path(__file__).parent.parent.parent / 'price_catalog.xlsx',
    ]

    price_catalog_path = None
    for p in price_paths:
        if os.path.exists(p):
            price_catalog_path = p
            break

    if not price_catalog_path:
        print("⚠ price_catalog.xlsx not found, using minimal defaults")
        catalog_data['avg_cost_per_ticket'] = 500
        catalog_data['benchmark_best'] = 300
        catalog_data['benchmark_avg'] = 500
        catalog_data['benchmark_laggard'] = 1000
        return catalog_data

    try:
        xl = pd.ExcelFile(price_catalog_path)

        # Load Category Costs
        if 'Category Costs' in xl.sheet_names:
            cat_df = pd.read_excel(xl, sheet_name='Category Costs')
            for _, row in cat_df.iterrows():
                cat = str(row['Category']).strip()
                labor_hrs = float(row.get('Labor_Hours', 2) or 2)
                hourly_rate = float(row.get('Hourly_Rate', 20) or 20)
                delay_cost = float(row.get('Delay_Cost_Per_Hour', 100) or 100)
                material = float(row.get('Material_Cost', 0) or 0)
                # Base cost = Material + Labor + Delay
                base_cost = material + (labor_hrs * hourly_rate) + (labor_hrs * delay_cost)
                catalog_data['category_costs'][cat] = {
                    'base_cost': base_cost,
                    'labor_hours': labor_hrs,
                    'hourly_rate': hourly_rate,
                    'delay_cost': delay_cost,
                    'material': material
                }
                catalog_data['hourly_rate'] = hourly_rate

        # Load Severity Multipliers
        if 'Severity Multipliers' in xl.sheet_names:
            sev_df = pd.read_excel(xl, sheet_name='Severity Multipliers')
            for _, row in sev_df.iterrows():
                sev = str(row['Severity_Level']).strip().lower()
                mult = float(row.get('Cost_Multiplier', 1.0) or 1.0)
                catalog_data['severity_multipliers'][sev] = mult

        # Load Origin Premiums
        if 'Origin Premiums' in xl.sheet_names:
            orig_df = pd.read_excel(xl, sheet_name='Origin Premiums')
            for _, row in orig_df.iterrows():
                origin = str(row['Origin_Type']).strip().lower()
                prem = float(row.get('Premium_Percentage', 0) or 0)
                catalog_data['origin_premiums'][origin] = prem

        # Calculate benchmarks from category costs
        if catalog_data['category_costs']:
            all_base_costs = [c['base_cost'] for c in catalog_data['category_costs'].values()]
            avg_base = sum(all_base_costs) / len(all_base_costs)
            avg_sev_mult = sum(catalog_data['severity_multipliers'].values()) / len(catalog_data['severity_multipliers'])
            catalog_data['avg_cost_per_ticket'] = avg_base * avg_sev_mult
            catalog_data['benchmark_best'] = min(all_base_costs)
            catalog_data['benchmark_avg'] = avg_base * avg_sev_mult
            catalog_data['benchmark_laggard'] = max(all_base_costs) * max(catalog_data['severity_multipliers'].values())

        print(f"✓ Loaded price_catalog.xlsx: {len(catalog_data['category_costs'])} categories, avg=${catalog_data['avg_cost_per_ticket']:.0f}/ticket")

    except Exception as e:
        print(f"⚠ Error loading price_catalog.xlsx: {e}")
        catalog_data['avg_cost_per_ticket'] = 500
        catalog_data['benchmark_best'] = 300
        catalog_data['benchmark_avg'] = 500
        catalog_data['benchmark_laggard'] = 1000

    return catalog_data


def get_catalog_cost(category: str = None, severity: str = 'Medium', origin: str = 'Technical') -> float:
    """Calculate cost for a ticket using price_catalog.xlsx data."""
    catalog = load_price_catalog(_mtime=_get_price_catalog_mtime())

    # Get base cost from category
    if category and category in catalog['category_costs']:
        base_cost = catalog['category_costs'][category]['base_cost']
    else:
        base_cost = catalog['avg_cost_per_ticket'] / 1.5  # Approximate base without multipliers

    # Apply severity multiplier
    sev_key = str(severity).lower() if severity else 'medium'
    sev_mult = catalog['severity_multipliers'].get(sev_key, 1.25)

    # Apply origin premium
    orig_key = str(origin).lower() if origin else 'technical'
    orig_prem = catalog['origin_premiums'].get(orig_key, 0.0)

    return base_cost * sev_mult * (1 + orig_prem)


def get_benchmark_costs() -> dict:
    """Get benchmark costs from price_catalog.xlsx."""
    catalog = load_price_catalog(_mtime=_get_price_catalog_mtime())
    return {
        'best_in_class': catalog['benchmark_best'],
        'industry_avg': catalog['benchmark_avg'],
        'laggard': catalog['benchmark_laggard'],
        'avg_per_ticket': catalog['avg_cost_per_ticket'],
        'hourly_rate': catalog['hourly_rate'],
    }

# ============================================================================
# EXCEL API REFRESH UTILITY
# ============================================================================

def is_excel_available():
    """Check if Microsoft Excel is available on this system."""
    import platform
    if platform.system() != 'Windows':
        return False
    try:
        import win32com.client
        return True
    except ImportError:
        try:
            import xlwings
            return True
        except ImportError:
            return False


def refresh_excel_connections(file_path: str, timeout_seconds: int = 120) -> tuple[bool, str]:
    """
    Refresh all data connections in an Excel file.

    Args:
        file_path: Path to the Excel file
        timeout_seconds: Maximum time to wait for refresh

    Returns:
        Tuple of (success: bool, message: str)
    """
    import platform

    if platform.system() != 'Windows':
        return False, "Excel refresh only available on Windows"

    # Try xlwings first (more reliable)
    try:
        import xlwings as xw

        # Open Excel in the background
        app = xw.App(visible=False, add_book=False)
        app.display_alerts = False
        app.screen_updating = False

        try:
            # Open the workbook
            wb = app.books.open(file_path)

            # Check if there are any connections
            has_connections = False
            try:
                if wb.api.Connections.Count > 0:
                    has_connections = True
            except:
                pass

            try:
                if wb.api.Queries.Count > 0:
                    has_connections = True
            except:
                pass

            if not has_connections:
                wb.close()
                app.quit()
                return False, "No API connections found in Excel file"

            # Refresh all connections
            wb.api.RefreshAll()

            # Wait for refresh to complete
            import time
            start_time = time.time()
            while time.time() - start_time < timeout_seconds:
                try:
                    # Check if any query is still refreshing
                    still_refreshing = False
                    for conn in wb.api.Connections:
                        if conn.OLEDBConnection.Refreshing:
                            still_refreshing = True
                            break
                except:
                    pass

                if not still_refreshing:
                    break
                time.sleep(1)

            # Save the workbook
            wb.save()
            wb.close()
            app.quit()

            return True, "Successfully refreshed Excel data connections"

        except Exception as e:
            try:
                wb.close()
            except:
                pass
            app.quit()
            raise e

    except ImportError:
        pass
    except Exception as e:
        # Try win32com as fallback
        pass

    # Fallback to win32com
    try:
        import win32com.client
        import pythoncom

        pythoncom.CoInitialize()

        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False

        try:
            wb = excel.Workbooks.Open(file_path)

            # Check for connections
            has_connections = wb.Connections.Count > 0

            if not has_connections:
                wb.Close(SaveChanges=False)
                excel.Quit()
                pythoncom.CoUninitialize()
                return False, "No API connections found in Excel file"

            # Refresh all
            wb.RefreshAll()

            # Wait for background queries
            import time
            time.sleep(5)  # Initial wait

            # Try to wait for refresh completion
            start_time = time.time()
            while time.time() - start_time < timeout_seconds:
                try:
                    excel.CalculateUntilAsyncQueriesDone()
                    break
                except:
                    time.sleep(2)

            # Save and close
            wb.Save()
            wb.Close(SaveChanges=True)
            excel.Quit()
            pythoncom.CoUninitialize()

            return True, "Successfully refreshed Excel data connections"

        except Exception as e:
            try:
                wb.Close(SaveChanges=False)
            except:
                pass
            excel.Quit()
            pythoncom.CoUninitialize()
            return False, f"Error refreshing Excel: {str(e)}"

    except ImportError:
        return False, "Neither xlwings nor pywin32 installed. Install with: pip install xlwings pywin32"
    except Exception as e:
        return False, f"Error: {str(e)}"


def load_excel_with_refresh(file_path: str, force_refresh: bool = True) -> tuple[pd.DataFrame, str, bool]:
    """
    Load Excel file, optionally refreshing API connections first.

    Args:
        file_path: Path to Excel file
        force_refresh: Whether to attempt refresh

    Returns:
        Tuple of (dataframe, message, was_refreshed)
    """
    was_refreshed = False
    message = ""

    if force_refresh and is_excel_available():
        success, msg = refresh_excel_connections(file_path)
        was_refreshed = success
        message = msg
    elif force_refresh:
        message = "Excel not available - loading file as-is"

    # Load the Excel file
    try:
        # Try Scored Data sheet first
        df = pd.read_excel(file_path, sheet_name="Scored Data")
    except:
        try:
            # Try first sheet
            df = pd.read_excel(file_path, sheet_name=0)
        except Exception as e:
            return None, f"Error loading Excel: {str(e)}", False

    return df, message, was_refreshed


# ============================================================================
# ACTION ITEMS PERSISTENCE (JSON)
# ============================================================================

ACTION_ITEMS_FILE = Path(__file__).parent / 'action_items.json'

def load_action_items():
    """Load action items from JSON file."""
    if ACTION_ITEMS_FILE.exists():
        try:
            with open(ACTION_ITEMS_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None

def save_action_items(items):
    """Save action items to JSON file."""
    try:
        with open(ACTION_ITEMS_FILE, 'w') as f:
            json.dump(items, f, indent=2)
    except IOError as e:
        st.warning(f"Could not save action items: {e}")

# ============================================================================
# CUSTOM CSS - EXECUTIVE STYLING
# ============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

.stApp {
    font-family: 'Inter', sans-serif;
}

/* Force Plotly charts to have adequate height and prevent overlap */
.stPlotlyChart {
    min-height: 400px !important;
}

.stPlotlyChart > div {
    min-height: 400px !important;
}

/* Ensure columns don't overlap */
.stColumns {
    gap: 1rem;
}

.stColumn {
    padding: 0 0.5rem;
}

/* Executive Glassmorphism cards */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 24px;
    margin: 10px 0;
}

/* Executive Summary Card */
.exec-card {
    background: linear-gradient(145deg, rgba(0, 40, 85, 0.9) 0%, rgba(0, 20, 40, 0.95) 100%);
    border-radius: 20px;
    padding: 32px;
    border: 1px solid rgba(0, 150, 255, 0.2);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    margin: 16px 0;
}

/* Strategic Recommendation Cards */
.strategy-card {
    background: linear-gradient(135deg, rgba(0, 102, 204, 0.1) 0%, rgba(0, 51, 102, 0.2) 100%);
    border-left: 5px solid #00BFFF;
    border-radius: 0 16px 16px 0;
    padding: 20px 24px;
    margin: 12px 0;
}

.strategy-card.high-priority {
    border-left-color: #FF6B6B;
    background: linear-gradient(135deg, rgba(255, 107, 107, 0.1) 0%, rgba(139, 0, 0, 0.15) 100%);
}

.strategy-card.medium-priority {
    border-left-color: #FFB347;
    background: linear-gradient(135deg, rgba(255, 179, 71, 0.1) 0%, rgba(255, 140, 0, 0.15) 100%);
}

/* KPI Cards - Enhanced */
.kpi-container {
    background: linear-gradient(135deg, rgba(0, 102, 204, 0.15) 0%, rgba(0, 51, 102, 0.25) 100%);
    border-radius: 16px;
    padding: 24px;
    border-left: 4px solid #0066CC;
    text-align: center;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.kpi-container:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 102, 204, 0.3);
}

.kpi-container.critical {
    background: linear-gradient(135deg, rgba(220, 53, 69, 0.15) 0%, rgba(139, 0, 0, 0.25) 100%);
    border-left-color: #DC3545;
}

.kpi-container.warning {
    background: linear-gradient(135deg, rgba(255, 193, 7, 0.15) 0%, rgba(255, 140, 0, 0.25) 100%);
    border-left-color: #FFC107;
}

.kpi-container.success {
    background: linear-gradient(135deg, rgba(40, 167, 69, 0.15) 0%, rgba(0, 100, 0, 0.25) 100%);
    border-left-color: #28A745;
}

/* Executive KPI - Larger */
.exec-kpi {
    background: linear-gradient(145deg, rgba(0, 102, 204, 0.2) 0%, rgba(0, 40, 80, 0.4) 100%);
    border-radius: 24px;
    padding: 40px;
    text-align: center;
    border: 2px solid rgba(0, 191, 255, 0.3);
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.2);
}

.exec-kpi-value {
    font-size: 4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00BFFF 0%, #0066CC 50%, #004080 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.1;
}

.exec-kpi-value.money {
    background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.exec-kpi-value.alert {
    background: linear-gradient(135deg, #FF6B6B 0%, #DC3545 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.kpi-value {
    font-size: 2.8rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(135deg, #00BFFF 0%, #0066CC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.kpi-label {
    font-size: 0.85rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 8px;
}

.kpi-delta {
    font-size: 0.9rem;
    font-weight: 600;
    margin-top: 4px;
}

.delta-up { color: #DC3545; }
.delta-down { color: #28A745; }

/* Pulse Indicator */
.pulse-dot {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse 2s infinite;
}

.pulse-dot.green { background: #28A745; box-shadow: 0 0 8px #28A745; }
.pulse-dot.yellow { background: #FFC107; box-shadow: 0 0 8px #FFC107; }
.pulse-dot.red { background: #DC3545; box-shadow: 0 0 8px #DC3545; }

@keyframes pulse {
    0% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.2); }
    100% { opacity: 1; transform: scale(1); }
}

/* Main header - Executive */
.main-header {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00BFFF 0%, #0066CC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.exec-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #FFFFFF 0%, #00BFFF 50%, #0066CC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
    text-align: center;
}

.sub-header {
    color: #888;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* Benchmark Meter */
.benchmark-container {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
}

.benchmark-bar {
    height: 24px;
    background: linear-gradient(90deg, #28A745 0%, #FFC107 50%, #DC3545 100%);
    border-radius: 12px;
    position: relative;
    margin: 10px 0;
}

.benchmark-marker {
    position: absolute;
    width: 4px;
    height: 32px;
    background: white;
    top: -4px;
    border-radius: 2px;
    box-shadow: 0 0 8px rgba(255,255,255,0.5);
}

/* Action Item Cards */
.action-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    border: 1px solid rgba(255,255,255,0.1);
}

.action-card.completed {
    border-left: 4px solid #28A745;
    opacity: 0.7;
}

.action-card.in-progress {
    border-left: 4px solid #0066CC;
}

.action-card.blocked {
    border-left: 4px solid #DC3545;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1929 0%, #001e3c 100%);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* Hide streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* Chart container fix - prevent overlap */
[data-testid="stPlotlyChart"] {
    min-height: 360px;
    max-height: 450px;
    overflow: visible !important;
}

[data-testid="column"] {
    overflow: visible !important;
}

/* Ensure charts scale properly */
.js-plotly-plot {
    width: 100% !important;
}

/* Chart styling */
.chart-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #E0E0E0;
    margin-bottom: 12px;
}

/* Alert badges */
.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

.badge-critical { background: #DC3545; color: white; }
.badge-warning { background: #FFC107; color: #212529; }
.badge-success { background: #28A745; color: white; }
.badge-info { background: #0066CC; color: white; }

/* Priority Tags */
.priority-p1 { background: linear-gradient(135deg, #DC3545, #8B0000); color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; }
.priority-p2 { background: linear-gradient(135deg, #FFC107, #FF8C00); color: #212529; padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; }
.priority-p3 { background: linear-gradient(135deg, #0066CC, #004080); color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(255,255,255,0.05);
    padding: 4px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 12px 24px;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0066CC 0%, #004C97 100%);
}

/* Slider styling */
.stSlider > div > div {
    background: linear-gradient(90deg, #0066CC, #00BFFF);
}

/* Metric delta fix */
[data-testid="stMetricDelta"] {
    font-size: 0.9rem;
}

/* Presentation Mode */
.presentation-slide {
    min-height: 80vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 40px;
}

/* Confidence Score */
.confidence-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(0, 102, 204, 0.2);
    border: 1px solid rgba(0, 191, 255, 0.3);
}

/* Impact Cards */
.impact-positive {
    background: linear-gradient(135deg, rgba(40, 167, 69, 0.15) 0%, rgba(0, 100, 0, 0.25) 100%);
    border-color: #28A745;
}

.impact-negative {
    background: linear-gradient(135deg, rgba(220, 53, 69, 0.15) 0%, rgba(139, 0, 0, 0.25) 100%);
    border-color: #DC3545;
}

/* Table Styling */
.exec-table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
}

.exec-table th {
    background: rgba(0, 102, 204, 0.3);
    padding: 12px 16px;
    text-align: left;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.8rem;
    letter-spacing: 1px;
}

.exec-table td {
    padding: 12px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

.exec-table tr:hover {
    background: rgba(0, 102, 204, 0.1);
}

/* ============================================
   EXCEL-STYLE DASHBOARD CSS
   ============================================ */

/* Dashboard Title Header */
.excel-dashboard-header {
    background: linear-gradient(135deg, #0a2540 0%, #003366 100%);
    padding: 20px 30px;
    border-radius: 12px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.excel-title {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 2px;
    margin: 0;
}

.excel-title-accent {
    color: #dc3545;
    font-weight: 800;
}

.excel-subtitle {
    color: #87ceeb;
    font-size: 0.9rem;
    margin: 4px 0 0 0;
}

/* Filter Sidebar Styling */
.excel-filter-section {
    background: linear-gradient(180deg, #003366 0%, #002244 100%);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}

.excel-filter-title {
    color: #ffffff;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.excel-clear-btn {
    background: #dc3545;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 6px;
    font-weight: 600;
    cursor: pointer;
    width: 100%;
    margin-top: 10px;
}

/* KPI Cards - Excel Style */
.excel-kpi-card {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.excel-kpi-card.primary {
    background: linear-gradient(145deg, rgba(0, 102, 204, 0.15) 0%, rgba(0, 51, 102, 0.25) 100%);
    border-left: 4px solid #0066cc;
}

.excel-kpi-card.accent {
    background: linear-gradient(145deg, rgba(220, 53, 69, 0.15) 0%, rgba(139, 0, 0, 0.25) 100%);
    border-left: 4px solid #dc3545;
}

.excel-kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    line-height: 1.2;
}

.excel-kpi-value.large {
    font-size: 2.5rem;
}

.excel-kpi-value.money {
    color: #4ade80;
}

.excel-kpi-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 8px;
}

.excel-kpi-sublabel {
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 4px;
}

/* Chart Card Container - Excel Style */
.excel-chart-card {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.02) 0%, rgba(255, 255, 255, 0.005) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
}

.excel-chart-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

/* Legend Container */
.excel-legend {
    padding: 10px;
    font-size: 0.75rem;
}

.excel-legend-item {
    display: flex;
    align-items: center;
    margin: 6px 0;
    color: #94a3b8;
}

.excel-legend-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
}

.excel-legend-dot.blue { background: #3b82f6; }
.excel-legend-dot.red { background: #dc3545; }
.excel-legend-dot.green { background: #22c55e; }
.excel-legend-dot.orange { background: #f97316; }

/* Progress Bar - Excel Style */
.excel-progress-container {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    height: 24px;
    overflow: hidden;
    margin: 8px 0;
}

.excel-progress-bar {
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 8px;
    font-size: 0.7rem;
    font-weight: 600;
    color: white;
}

.excel-progress-bar.male {
    background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
}

.excel-progress-bar.female {
    background: linear-gradient(90deg, #dc3545 0%, #ef4444 100%);
}

/* Comparison Grid */
.excel-comparison-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
}

/* Donut Chart Value Overlay */
.excel-donut-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
}

/* Horizontal Bar Chart Labels */
.excel-bar-row {
    display: flex;
    align-items: center;
    margin: 4px 0;
}

.excel-bar-value {
    font-size: 0.7rem;
    color: #94a3b8;
    min-width: 60px;
    text-align: right;
    padding-right: 8px;
}

.excel-bar-name {
    font-size: 0.75rem;
    color: #cbd5e1;
    margin-left: 8px;
}

/* Product Revenue Cards */
.excel-product-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}

.excel-product-icon {
    font-size: 1.5rem;
    margin-bottom: 6px;
}

.excel-product-value {
    font-size: 0.9rem;
    font-weight: 600;
    color: #ffffff;
}

.excel-product-label {
    font-size: 0.7rem;
    color: #64748b;
}

/* Stores Analysis Scatter */
.excel-scatter-container {
    position: relative;
    height: 200px;
}

/* Quarter Donut Grid */
.excel-quarter-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
}

.excel-quarter-card {
    text-align: center;
    padding: 8px;
}

.excel-quarter-label {
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard transformations to a loaded dataframe.

    Note: Similarity columns (Best_Match_Similarity, Resolution_Consistency)
    are generated by the pipeline's similar_tickets.py module during analysis.
    """
    # Map column names to expected format
    if 'tickets_data_engineer_name' in df.columns and 'Engineer' not in df.columns:
        df['Engineer'] = df['tickets_data_engineer_name']
    if 'tickets_data_lob' in df.columns and 'LOB' not in df.columns:
        df['LOB'] = df['tickets_data_lob']

    # Use AI_Recurrence_Probability as the numeric recurrence risk
    # (AI_Recurrence_Risk in file is string like "Elevated (50-70%)")
    if 'AI_Recurrence_Probability' in df.columns:
        df['AI_Recurrence_Risk'] = pd.to_numeric(df['AI_Recurrence_Probability'], errors='coerce').fillna(0.15)

    # Ensure numeric columns are numeric
    numeric_cols = ['Strategic_Friction_Score', 'Financial_Impact', 'Predicted_Resolution_Days',
                   'AI_Confidence', 'Resolution_Prediction_Confidence', 'Best_Match_Similarity']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Map AI_Origin from available origin columns (for heatmap visualization)
    if 'AI_Origin' not in df.columns:
        for col in ['Origin_Norm', 'tickets_data_escalation_origin', 'tickets_data_origin', 'Origin']:
            if col in df.columns:
                df['AI_Origin'] = df[col]
                break

    # ALWAYS recalculate Financial_Impact using PriceCatalog
    # This ensures the dashboard reflects the current price_catalog.xlsx configuration
    df = _calculate_financial_impact_from_catalog(df)

    return df


def _calculate_financial_impact_from_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Financial_Impact for each row using the PriceCatalog."""
    try:
        import os
        from pathlib import Path
        from ..feedback.price_catalog import get_price_catalog

        # Ensure we're in the right directory for price_catalog.xlsx
        project_root = Path(__file__).parent.parent.parent
        price_catalog_path = project_root / 'price_catalog.xlsx'

        if not price_catalog_path.exists():
            raise FileNotFoundError(f"price_catalog.xlsx not found at {price_catalog_path}")

        # Change to project root temporarily if needed
        original_cwd = os.getcwd()
        os.chdir(project_root)

        try:
            price_catalog = get_price_catalog()
            price_catalog.load_catalog()  # Always reload to ensure latest values
        finally:
            os.chdir(original_cwd)

        def calc_impact(row):
            category = row.get('AI_Category', 'Unclassified')
            severity = row.get('Severity_Norm', row.get('tickets_data_severity', 'Medium'))
            origin = row.get('Origin_Norm', row.get('tickets_data_origin', 'Internal'))
            # Use tickets_data_issue_summary (the actual column name) for keyword pattern matching
            description = str(row.get('tickets_data_issue_summary', row.get('Summary', '')))

            result = price_catalog.calculate_financial_impact(
                category=str(category) if pd.notna(category) else 'Unclassified',
                severity=str(severity) if pd.notna(severity) else 'Medium',
                origin=str(origin) if pd.notna(origin) else 'Technical',
                description=description
            )
            return result['total_impact']

        df['Financial_Impact'] = df.apply(calc_impact, axis=1)
        print(f"✓ Calculated Financial_Impact using price_catalog.xlsx: ${df['Financial_Impact'].sum():,.0f} total")

    except Exception as e:
        import traceback
        print(f"⚠ ERROR calculating Financial_Impact from price_catalog: {e}")
        print(f"⚠ Full traceback:\n{traceback.format_exc()}")
        print("⚠ Using Strategic_Friction_Score × 15 as fallback")
        # Fallback: use friction score as proxy if price catalog fails
        if 'Strategic_Friction_Score' in df.columns:
            df['Financial_Impact'] = df['Strategic_Friction_Score'] * 15  # Reasonable multiplier
        else:
            df['Financial_Impact'] = 500  # Minimal fallback

    return df


@st.cache_data
def _load_excel_raw(file_path: str) -> pd.DataFrame:
    """Load raw Excel data (cached). Financial_Impact calculated separately."""
    try:
        return pd.read_excel(file_path, sheet_name="Scored Data")
    except:
        return pd.read_excel(file_path, sheet_name=0)


def load_excel_file(file_path: str) -> tuple:
    """Load an Excel file and return processed dataframe with fresh Financial_Impact."""
    try:
        df = _load_excel_raw(file_path)
        df = process_dataframe(df)  # This recalculates Financial_Impact from price_catalog
        return df, file_path
    except Exception as e:
        return None, str(e)


@st.cache_data
def _find_data_file():
    """Find the data file path (cached). Returns path only, not data."""
    project_root = Path(__file__).parent.parent.parent

    search_paths = [
        Path("Strategic_Report.xlsx"),
        project_root / "Strategic_Report.xlsx",
        Path.cwd() / "Strategic_Report.xlsx",
    ]

    for path in search_paths:
        if path.exists():
            return str(path), "Scored Data"

    # Look for other processed Excel files
    data_files = list(project_root.glob("Escalation_Analysis_*.xlsx"))
    if data_files:
        latest = max(data_files, key=lambda x: x.stat().st_mtime)
        return str(latest), "Detailed Analysis"

    return None, None


def load_data():
    """Load the most recent analysis data with fresh Financial_Impact from price_catalog."""
    file_path, sheet_name = _find_data_file()

    if file_path is None:
        return None, "No data file found"

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df = process_dataframe(df)  # Recalculates Financial_Impact from price_catalog
        print(f"✅ Loaded {len(df)} records from {file_path}")
        return df, file_path
    except Exception as e:
        st.warning(f"Could not load {file_path}: {e}")
        # Generate sample data as fallback
        st.warning("⚠️ No data file found. Showing sample data. Please ensure Strategic_Report.xlsx is in the project root.")
        return generate_sample_data(), "Sample Data"


def generate_sample_data():
    """Generate realistic sample data for demo."""
    np.random.seed(42)
    n = 250

    # 8-category system for telecom escalation analysis
    categories = [
        'Scheduling & Planning', 'Documentation & Reporting', 'Validation & QA',
        'Process Compliance', 'Configuration & Data Mismatch',
        'Site Readiness', 'Communication & Response', 'Nesting & Tool Errors'
    ]

    # Sub-categories for each main category (detailed sub-types from Embedding.md)
    sub_categories = {
        'Scheduling & Planning': ['No TI Entry', 'Schedule Not Followed', 'Weekend Schedule Issue',
                                   'Ticket Status Issue', 'Premature Scheduling'],
        'Documentation & Reporting': ['Missing Snapshot', 'Missing Attachment', 'Incorrect Reporting',
                                       'Wrong Site ID', 'Incomplete Snapshot', 'Missing Information',
                                       'Wrong Attachment', 'Incorrect Status'],
        'Validation & QA': ['Incomplete Validation', 'Missed Issue', 'Missed Check', 'No Escalation',
                            'Missed Degradation', 'Wrong Tool Usage', 'Incomplete Testing'],
        'Process Compliance': ['Process Violation', 'Wrong Escalation', 'Wrong Bucket', 'Missed Step',
                               'Missing Ticket', 'Process Non-Compliance'],
        'Configuration & Data Mismatch': ['Port Matrix Mismatch', 'RET Naming', 'RET Swap',
                                           'TAC Mismatch', 'CIQ/SCF Mismatch', 'RFDS Mismatch',
                                           'Missing Documents', 'Config Error'],
        'Site Readiness': ['BH Not Ready', 'MW Not Ready', 'Material Missing', 'Site Down',
                           'BH Status Issue', 'Site Complexity'],
        'Communication & Response': ['Delayed Response', 'Delayed Deliverable', 'No Proactive Update',
                                      'No Communication', 'Training Issue'],
        'Nesting & Tool Errors': ['Wrong Nest Type', 'Improper Extension', 'Missing Nesting',
                                   'HW Issue', 'Rework', 'Post-OA Degradation', 'Delayed Audit']
    }

    engineers = ['Alice Chen', 'Bob Smith', 'Carlos Rodriguez', 'Diana Patel',
                 'Eric Johnson', 'Fatima Ahmed', 'George Kim', 'Hannah Lee']

    lobs = ['Network Operations', 'Field Services', 'Customer Support',
            'Infrastructure', 'Enterprise', 'Residential']

    regions = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West Coast', 'Central']

    root_causes = ['Training Gap', 'Process Failure', 'Tool Limitation', 'Resource Constraint',
                   'Vendor Issue', 'Communication Breakdown', 'Technical Debt', 'Policy Conflict']

    # Generate dates over 90 days
    dates = pd.date_range(end=datetime.now(), periods=n, freq='8H')

    # Generate categories
    cat_values = np.random.choice(categories, n, p=[0.18, 0.15, 0.14, 0.12, 0.12, 0.11, 0.10, 0.08])

    # Generate sub-categories based on main category
    sub_cat_values = [np.random.choice(sub_categories[cat]) for cat in cat_values]

    severities = np.random.choice(['Critical', 'Major', 'Minor'], n, p=[0.15, 0.45, 0.40])
    origins = np.random.choice(['External', 'Internal'], n, p=[0.35, 0.65])

    df = pd.DataFrame({
        'AI_Category': cat_values,
        'AI_Sub_Category': sub_cat_values,
        'AI_Confidence': np.clip(np.random.beta(8, 2, n), 0.4, 0.99),
        'Strategic_Friction_Score': np.clip(np.random.exponential(45, n), 5, 200),
        'AI_Recurrence_Risk': np.clip(np.random.beta(2, 5, n), 0, 1),
        'AI_Recurrence_Probability': np.clip(np.random.beta(2, 5, n), 0, 1),
        'Predicted_Resolution_Days': np.clip(np.random.exponential(2.5, n), 0.5, 15),
        'tickets_data_severity': severities,
        'Severity_Norm': severities,
        'tickets_data_escalation_origin': origins,
        'Origin_Norm': origins,
        'tickets_data_issue_datetime': dates,
        'Engineer': np.random.choice(engineers, n),
        'LOB': np.random.choice(lobs, n),
        'Region': np.random.choice(regions, n),
        'Root_Cause': np.random.choice(root_causes, n),
        'Customer_Impact_Score': np.clip(np.random.exponential(50, n), 5, 100),
        'SLA_Breached': np.random.choice([True, False], n, p=[0.12, 0.88]),
        'Repeat_Customer': np.random.choice([True, False], n, p=[0.25, 0.75]),
        'Contract_Value': np.clip(np.random.exponential(50000, n), 5000, 500000),
        'Customer_Tenure_Years': np.clip(np.random.exponential(3, n), 0.5, 15),
        'NPS_Impact': np.random.choice([-3, -2, -1, 0], n, p=[0.1, 0.2, 0.4, 0.3]),
    })

    # Calculate Financial_Impact using PriceCatalog
    df = _calculate_financial_impact_from_catalog(df)

    # Derived metrics
    df['Revenue_At_Risk'] = df['Contract_Value'] * df['AI_Recurrence_Risk'] * 0.15
    df['Churn_Probability'] = np.clip(df['Customer_Impact_Score'] / 100 * df['AI_Recurrence_Risk'], 0, 0.5)

    return df


# ============================================================================
# INDUSTRY BENCHMARKS
# ============================================================================

INDUSTRY_BENCHMARKS = {
    'resolution_days': {'best_in_class': 1.2, 'industry_avg': 2.8, 'laggard': 5.5},
    'recurrence_rate': {'best_in_class': 8, 'industry_avg': 18, 'laggard': 32},
    'sla_breach_rate': {'best_in_class': 3, 'industry_avg': 12, 'laggard': 25},
    'first_contact_resolution': {'best_in_class': 72, 'industry_avg': 55, 'laggard': 38},
    'cost_per_escalation': 'FROM_CATALOG',  # Loaded from price_catalog.xlsx at runtime
    'customer_satisfaction': {'best_in_class': 92, 'industry_avg': 78, 'laggard': 62},
}

# ============================================================================
# CHART FUNCTIONS
# ============================================================================

def create_plotly_theme():
    """Get consistent Plotly theme settings."""
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#E0E0E0'),
        margin=dict(l=40, r=40, t=50, b=40),
    )


# ============================================================================
# CHART INSIGHT SYSTEM - Hover tooltips with explanations and data insights
# ============================================================================

# Chart descriptions explaining what each visualization signifies
CHART_DESCRIPTIONS = {
    'trend_timeline': {
        'title': 'Escalation Trend (7-Day Moving Average)',
        'description': 'Tracks escalation volume and friction scores over time using a 7-day rolling average to smooth out daily fluctuations.',
        'what_it_shows': 'Shows patterns in escalation frequency and intensity. Rising trends indicate growing issues; falling trends suggest improvements.',
        'how_to_read': 'Blue area = friction intensity, Orange line = ticket count. Watch for correlation between volume spikes and friction increases.',
    },
    'severity_distribution': {
        'title': 'Severity Distribution',
        'description': 'Breakdown of escalations by severity level (Critical, Major, Minor).',
        'what_it_shows': 'Helps prioritize resources and identify if too many tickets are being marked Critical.',
        'how_to_read': 'Healthy distribution: <15% Critical, 30-40% Major, 50-60% Minor. High Critical % suggests either severe issues or over-escalation.',
    },
    'friction_by_category': {
        'title': 'Strategic Friction by Category',
        'description': 'Total friction score accumulated by each issue category.',
        'what_it_shows': 'Identifies which categories cause the most organizational friction and should be prioritized for process improvements.',
        'how_to_read': 'Longer bars = more cumulative friction. Focus on top categories for maximum impact on reducing escalation burden.',
    },
    'recurrence_risk': {
        'title': 'Average Recurrence Risk',
        'description': 'AI-predicted probability that similar issues will recur based on historical patterns.',
        'what_it_shows': 'High recurrence indicates systemic issues that need root cause fixes, not just symptom treatment.',
        'how_to_read': 'Green (<30%) = Well-managed. Yellow (30-60%) = Needs attention. Red (>60%) = Systemic problem requiring intervention.',
    },
    'category_sunburst': {
        'title': 'Category & Sub-Category Sunburst',
        'description': 'Hierarchical view of categories and their sub-categories showing ticket distribution.',
        'what_it_shows': 'Reveals the composition of each category and helps identify specific sub-categories driving volume.',
        'how_to_read': 'Click on segments to drill down. Larger segments = more tickets. Inner ring = categories, outer = sub-categories.',
    },
    'engineer_performance': {
        'title': 'Engineer Performance Matrix',
        'description': 'Compares engineers by friction score and ticket volume handled.',
        'what_it_shows': 'Identifies high performers (low friction, high volume) and those needing support (high friction).',
        'how_to_read': 'Ideal: Lower-left quadrant (low friction, efficient). Investigate high-friction engineers for training needs.',
    },
    'resolution_distribution': {
        'title': 'Resolution Time Distribution',
        'description': 'Histogram of predicted resolution times across all tickets.',
        'what_it_shows': 'Helps set realistic SLA targets and identify if resolution times follow expected patterns.',
        'how_to_read': 'Narrow distribution = consistent performance. Long tail = some tickets take disproportionately long.',
    },
    'pareto_analysis': {
        'title': 'Pareto Analysis (80/20 Rule)',
        'description': 'Shows which categories account for 80% of the total friction.',
        'what_it_shows': 'Classic Pareto principle - focus on the vital few categories that drive most of the impact.',
        'how_to_read': 'Bars = category friction. Line = cumulative %. Categories before line crosses 80% are your priority focus.',
    },
    'benchmark_gauge': {
        'title': 'Competitive Benchmark',
        'description': 'Compares your metrics against industry standards.',
        'what_it_shows': 'Positions your performance relative to best-in-class, industry average, and laggard benchmarks.',
        'how_to_read': 'Green zone = Best-in-class. Yellow = Industry average. Red = Below average. Needle shows your position.',
    },
    'risk_heatmap': {
        'title': 'Risk Heatmap',
        'description': 'Matrix showing risk levels across categories and severity.',
        'what_it_shows': 'Identifies dangerous combinations of category and severity that need immediate attention.',
        'how_to_read': 'Darker colors = higher risk. Focus on dark red cells for critical risk areas.',
    },
    # Similarity Search Charts
    'similarity_count': {
        'title': 'Similar Ticket Count Distribution',
        'description': 'Shows how many similar historical tickets were found for each current issue.',
        'what_it_shows': 'Zero matches may indicate new issue types. High counts suggest well-documented problem patterns.',
        'how_to_read': 'Bars show frequency. Zero-match tickets need manual review. High-match tickets have good historical data for predictions.',
    },
    'resolution_consistency': {
        'title': 'Resolution Consistency Analysis',
        'description': 'Compares how current tickets are being resolved vs how similar historical tickets were resolved.',
        'what_it_shows': 'Inconsistent resolutions suggest either evolving best practices or engineers deviating from proven solutions.',
        'how_to_read': 'Green = consistent with history. Red = different approach being taken. High inconsistency warrants investigation.',
    },
    'similarity_score': {
        'title': 'Similarity Score Distribution',
        'description': 'Quality of best matches found for each ticket (0 = no match, 1 = identical).',
        'what_it_shows': 'Higher scores mean more confident predictions. Low scores suggest unique or poorly documented issues.',
        'how_to_read': 'Green zone (>0.7) = high confidence. Yellow (0.5-0.7) = moderate. Red (<0.5) = low confidence.',
    },
    'inconsistent_resolution': {
        'title': 'Inconsistent Resolutions by Category',
        'description': 'Categories where current resolutions differ from similar historical cases.',
        'what_it_shows': 'Identifies categories with resolution approach inconsistency - may indicate training gaps or process changes.',
        'how_to_read': 'Longer bars = more inconsistent cases. Focus on top categories for standardization opportunities.',
    },
    'similarity_effectiveness': {
        'title': 'Similarity Search Effectiveness',
        'description': 'Heatmap showing average similar ticket matches by category and origin.',
        'what_it_shows': 'Green areas have good historical coverage. Red areas lack similar historical data.',
        'how_to_read': 'Higher values = better knowledge base coverage. Low values need historical data enrichment.',
    },
}


def get_chart_insight(chart_key: str, df: pd.DataFrame) -> dict:
    """
    Generate data-driven insights for a specific chart based on current data.

    Returns dict with: description, what_it_shows, how_to_read, current_insight
    """
    base_info = CHART_DESCRIPTIONS.get(chart_key, {
        'title': 'Chart',
        'description': 'Visualization of escalation data.',
        'what_it_shows': 'Data patterns and trends.',
        'how_to_read': 'Analyze the visual patterns.',
    })

    # Generate data-driven insight based on chart type
    current_insight = ""

    try:
        if chart_key == 'trend_timeline':
            if 'tickets_data_issue_datetime' in df.columns:
                df_temp = df.copy()
                df_temp['date'] = pd.to_datetime(df_temp['tickets_data_issue_datetime']).dt.date
                daily = df_temp.groupby('date').size()
                if len(daily) >= 7:
                    recent_avg = daily.tail(7).mean()
                    older_avg = daily.head(7).mean() if len(daily) >= 14 else recent_avg
                    trend = "increasing" if recent_avg > older_avg * 1.1 else "decreasing" if recent_avg < older_avg * 0.9 else "stable"
                    current_insight = f"📊 Current trend is **{trend}**. Recent 7-day avg: {recent_avg:.1f} tickets/day vs earlier: {older_avg:.1f}"

        elif chart_key == 'severity_distribution':
            if 'tickets_data_severity' in df.columns:
                severity = df['tickets_data_severity'].value_counts(normalize=True) * 100
                critical_pct = severity.get('Critical', 0)
                if critical_pct > 25:
                    current_insight = f"⚠️ **High Alert**: {critical_pct:.0f}% Critical tickets - review escalation criteria or investigate systemic issues."
                elif critical_pct > 15:
                    current_insight = f"🟡 **Elevated**: {critical_pct:.0f}% Critical tickets - slightly above healthy threshold of 15%."
                else:
                    current_insight = f"✅ **Healthy**: {critical_pct:.0f}% Critical tickets - within normal range."

        elif chart_key == 'friction_by_category':
            if 'AI_Category' in df.columns and 'Strategic_Friction_Score' in df.columns:
                friction = df.groupby('AI_Category')['Strategic_Friction_Score'].sum().sort_values(ascending=False)
                top_cat = friction.index[0]
                top_pct = (friction.iloc[0] / friction.sum()) * 100
                current_insight = f"🎯 **{top_cat}** drives {top_pct:.0f}% of total friction. Prioritize this category for maximum impact."

        elif chart_key == 'recurrence_risk':
            if 'AI_Recurrence_Probability' in df.columns:
                avg_risk = df['AI_Recurrence_Probability'].mean() * 100
                if avg_risk > 60:
                    current_insight = f"🔴 **Critical**: {avg_risk:.0f}% avg recurrence risk indicates systemic issues requiring root cause analysis."
                elif avg_risk > 30:
                    current_insight = f"🟡 **Elevated**: {avg_risk:.0f}% avg recurrence risk - some patterns need investigation."
                else:
                    current_insight = f"✅ **Good**: {avg_risk:.0f}% avg recurrence risk - issues are generally being resolved effectively."

        elif chart_key == 'engineer_performance':
            if 'Engineer' in df.columns and 'Strategic_Friction_Score' in df.columns:
                eng_stats = df.groupby('Engineer').agg({
                    'Strategic_Friction_Score': 'mean',
                    'AI_Category': 'count'
                }).rename(columns={'AI_Category': 'count'})
                top_performer = eng_stats[eng_stats['count'] >= 5].nsmallest(1, 'Strategic_Friction_Score')
                if not top_performer.empty:
                    current_insight = f"⭐ **Top Performer**: {top_performer.index[0]} - lowest friction with {top_performer['count'].iloc[0]} tickets."

        elif chart_key == 'category_sunburst':
            if 'AI_Category' in df.columns:
                cat_counts = df['AI_Category'].value_counts()
                top_3 = cat_counts.head(3)
                top_3_pct = (top_3.sum() / len(df)) * 100
                current_insight = f"📈 Top 3 categories ({', '.join(top_3.index[:3])}) account for **{top_3_pct:.0f}%** of all tickets."

        elif chart_key == 'pareto_analysis':
            if 'AI_Category' in df.columns and 'Strategic_Friction_Score' in df.columns:
                friction = df.groupby('AI_Category')['Strategic_Friction_Score'].sum().sort_values(ascending=False)
                cumsum = friction.cumsum() / friction.sum() * 100
                cats_for_80 = (cumsum <= 80).sum() + 1
                current_insight = f"📊 **{cats_for_80} categories** drive 80% of total friction. Focus improvement efforts here for maximum impact."

        elif chart_key == 'risk_heatmap':
            if 'AI_Category' in df.columns and 'tickets_data_severity' in df.columns:
                high_risk = df[(df['tickets_data_severity'] == 'Critical')]
                if len(high_risk) > 0:
                    top_critical_cat = high_risk['AI_Category'].value_counts().idxmax()
                    critical_count = high_risk['AI_Category'].value_counts().iloc[0]
                    current_insight = f"🔴 **{top_critical_cat}** has the most Critical tickets ({critical_count}). This is your highest risk area."

        elif chart_key == 'resolution_distribution':
            if 'Predicted_Resolution_Days' in df.columns:
                avg_days = df['Predicted_Resolution_Days'].mean()
                median_days = df['Predicted_Resolution_Days'].median()
                if avg_days > median_days * 1.5:
                    current_insight = f"⚠️ Mean ({avg_days:.1f}d) >> Median ({median_days:.1f}d) - some tickets take disproportionately long. Investigate outliers."
                else:
                    current_insight = f"✅ Resolution times are consistent. Avg: {avg_days:.1f} days, Median: {median_days:.1f} days."

    except Exception:
        pass  # Silently fail and return base info only

    return {
        **base_info,
        'current_insight': current_insight
    }


def render_spectacular_header(title: str, subtitle: str, icon: str = "📊"):
    """Render a spectacular gradient header with timestamp."""
    from datetime import datetime
    current_time = datetime.now().strftime("%b %d, %Y at %I:%M %p")

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0a1628 0%, #1a365d 50%, #0a1628 100%);
                padding: 25px 35px; border-radius: 16px; margin-bottom: 20px;
                border: 1px solid rgba(59, 130, 246, 0.3);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255,255,255,0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="font-size: 2rem; font-weight: 800; margin: 0;
                           background: linear-gradient(135deg, #ffffff 0%, #60a5fa 50%, #3b82f6 100%);
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {icon} {title}
                </h1>
                <p style="color: #94a3b8; font-size: 0.9rem; margin: 8px 0 0 0; letter-spacing: 1px;">
                    {subtitle}
                </p>
            </div>
            <div style="text-align: right;">
                <div style="color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;">Last Updated</div>
                <div style="color: #60a5fa; font-size: 1rem; font-weight: 600; margin-top: 4px;">{current_time}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_chart_with_insight(chart_key: str, chart_fig, df: pd.DataFrame, container=None):
    """
    Render a chart with an expandable insight tooltip.

    Args:
        chart_key: Key from CHART_DESCRIPTIONS
        chart_fig: Plotly figure object
        df: DataFrame for generating insights
        container: Optional streamlit container (defaults to st)
    """
    if container is None:
        container = st

    insight = get_chart_insight(chart_key, df)

    # Add chart info popover in top-right corner style
    with container.container():
        # Header row with title and info button
        title_col, info_col = st.columns([10, 1])

        with info_col:
            with st.popover("ℹ️"):
                st.markdown(f"### {insight.get('title', 'Chart')}")
                st.markdown(f"*{insight.get('description', '')}*")
                st.divider()
                st.markdown(f"**📊 What it shows:**")
                st.markdown(insight.get('what_it_shows', ''))
                st.markdown(f"**📖 How to read:**")
                st.markdown(insight.get('how_to_read', ''))
                if insight.get('current_insight'):
                    st.divider()
                    st.markdown(f"**💡 Current Data Insight:**")
                    st.markdown(insight['current_insight'])

        # Render the chart
        st.plotly_chart(chart_fig, use_container_width=True)


def chart_friction_by_category(df):
    """Interactive bar chart of friction by category."""
    friction = df.groupby('AI_Category')['Strategic_Friction_Score'].sum().sort_values(ascending=True)
    
    # Create gradient colors
    colors = px.colors.sequential.Blues_r[:len(friction)]
    
    fig = go.Figure(go.Bar(
        x=friction.values,
        y=friction.index,
        orientation='h',
        marker=dict(
            color=friction.values,
            colorscale='Blues',
            line=dict(width=0)
        ),
        hovertemplate='<b>%{y}</b><br>Friction: %{x:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(text='Strategic Friction by Category', font=dict(size=16)),
        xaxis_title='Total Friction Score',
        yaxis_title='',
        height=450,
        showlegend=False
    )
    
    return fig


def chart_severity_distribution(df):
    """Donut chart of severity distribution."""
    severity_counts = df['tickets_data_severity'].value_counts()
    
    colors = {'Critical': '#DC3545', 'Major': '#FFC107', 'Minor': '#28A745'}
    
    fig = go.Figure(go.Pie(
        labels=severity_counts.index,
        values=severity_counts.values,
        hole=0.6,
        marker=dict(colors=[colors.get(s, '#6C757D') for s in severity_counts.index]),
        textinfo='label+percent',
        textfont=dict(size=12),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>'
    ))
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(text='Severity Distribution', font=dict(size=16)),
        height=450,
        showlegend=True,
        legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center')
    )
    
    # Add center text
    fig.add_annotation(
        text=f"<b>{len(df)}</b><br>Total",
        x=0.5, y=0.5,
        font=dict(size=20, color='#E0E0E0'),
        showarrow=False
    )
    
    return fig


def chart_trend_timeline(df):
    """Animated area chart of escalations over time."""
    df_temp = df.copy()
    df_temp['date'] = pd.to_datetime(df_temp['tickets_data_issue_datetime']).dt.date
    daily = df_temp.groupby('date').agg({
        'Strategic_Friction_Score': 'sum',
        'AI_Category': 'count'
    }).rename(columns={'AI_Category': 'count'}).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    
    # 7-day rolling average
    daily['friction_ma'] = daily['Strategic_Friction_Score'].rolling(7, min_periods=1).mean()
    daily['count_ma'] = daily['count'].rolling(7, min_periods=1).mean()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Friction area
    fig.add_trace(go.Scatter(
        x=daily['date'],
        y=daily['friction_ma'],
        name='Friction Score',
        fill='tozeroy',
        fillcolor='rgba(0, 102, 204, 0.3)',
        line=dict(color='#0066CC', width=2),
        hovertemplate='Date: %{x}<br>Friction: %{y:,.0f}<extra></extra>'
    ), secondary_y=False)
    
    # Count line
    fig.add_trace(go.Scatter(
        x=daily['date'],
        y=daily['count_ma'],
        name='Escalation Count',
        line=dict(color='#FF6B6B', width=2, dash='dot'),
        hovertemplate='Date: %{x}<br>Count: %{y:.1f}<extra></extra>'
    ), secondary_y=True)
    
    fig.update_layout(
        **{
            **create_plotly_theme(),
            'margin': dict(l=60, r=60, t=80, b=60),
        },
        title=dict(text='Escalation Trend (7-Day Moving Average)', font=dict(size=16), y=0.95),
        height=450,
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center', yanchor='bottom'),
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text='Friction Score', secondary_y=False, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(title_text='Count', secondary_y=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def chart_recurrence_risk(df):
    """Gauge chart for average recurrence risk."""
    # AI_Recurrence_Risk is now guaranteed to be numeric from load_data()
    avg_risk = df['AI_Recurrence_Risk'].mean() * 100 if 'AI_Recurrence_Risk' in df.columns else 15

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_risk,
        number={'suffix': '%', 'font': {'size': 40}},
        delta={'reference': 15, 'relative': False, 'suffix': '%'},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': '#0066CC'},
            'bgcolor': 'rgba(255,255,255,0.1)',
            'steps': [
                {'range': [0, 20], 'color': 'rgba(40, 167, 69, 0.3)'},
                {'range': [20, 40], 'color': 'rgba(255, 193, 7, 0.3)'},
                {'range': [40, 100], 'color': 'rgba(220, 53, 69, 0.3)'}
            ],
            'threshold': {
                'line': {'color': '#FF6B6B', 'width': 4},
                'thickness': 0.8,
                'value': avg_risk  # Point needle at actual value
            }
        }
    ))
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(text='Avg Recurrence Risk', font=dict(size=16)),
        height=450
    )
    
    return fig


def chart_resolution_distribution(df):
    """Histogram of predicted resolution times."""
    fig = go.Figure(go.Histogram(
        x=df['Predicted_Resolution_Days'],
        nbinsx=20,
        marker=dict(
            color='rgba(0, 191, 255, 0.7)',
            line=dict(color='#00BFFF', width=1)
        ),
        hovertemplate='Days: %{x:.1f}<br>Count: %{y}<extra></extra>'
    ))
    
    # Add mean line
    mean_days = df['Predicted_Resolution_Days'].mean()
    fig.add_vline(x=mean_days, line_dash="dash", line_color="#FF6B6B",
                  annotation_text=f"Avg: {mean_days:.1f} days")
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(text='Resolution Time Distribution', font=dict(size=16)),
        xaxis_title='Predicted Days',
        yaxis_title='Count',
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def chart_category_sunburst(df):
    """Interactive sunburst chart of categories and sub-categories with drill-down."""
    # Check if AI_Sub_Category column exists
    sub_cat_col = 'AI_Sub_Category' if 'AI_Sub_Category' in df.columns else None
    cost_col = 'Financial_Impact' if 'Financial_Impact' in df.columns else None

    if sub_cat_col:
        # Category → Sub-Category drill-down
        sunburst_data = df.groupby(['AI_Category', sub_cat_col]).size().reset_index(name='count')
        path_cols = ['AI_Category', sub_cat_col]
        title_text = 'Category & Sub-Category Drill-Down<br><span style="font-size:12px">Click to expand categories</span>'
    else:
        # Fallback to Category → Severity
        sunburst_data = df.groupby(['AI_Category', 'tickets_data_severity']).size().reset_index(name='count')
        path_cols = ['AI_Category', 'tickets_data_severity']
        title_text = 'Category & Severity Breakdown'

    # Add financial data if available
    if cost_col and sub_cat_col:
        cost_data = df.groupby(['AI_Category', sub_cat_col])[cost_col].sum().reset_index()
        sunburst_data = sunburst_data.merge(cost_data, on=['AI_Category', sub_cat_col], how='left')
        sunburst_data[cost_col] = sunburst_data[cost_col].fillna(0)

    fig = px.sunburst(
        sunburst_data,
        path=path_cols,
        values='count',
        color='count',
        color_continuous_scale='Blues'
    )

    fig.update_layout(
        **create_plotly_theme(),
        title=dict(text=title_text, font=dict(size=16)),
        height=500
    )

    # Enhanced hover template
    if cost_col and sub_cat_col:
        fig.update_traces(
            hovertemplate='<b>%{label}</b><br>Tickets: %{value}<extra></extra>'
        )
    else:
        fig.update_traces(
            hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
        )

    return fig


def chart_engineer_performance(df):
    """Horizontal bar chart of engineer friction."""
    if 'Engineer' not in df.columns:
        return None
    
    # Build agg dict with available columns
    agg_dict = {'AI_Category': 'count'}
    if 'Strategic_Friction_Score' in df.columns:
        agg_dict['Strategic_Friction_Score'] = 'mean'
    if 'AI_Recurrence_Risk' in df.columns:
        agg_dict['AI_Recurrence_Risk'] = 'mean'
    
    eng_stats = df.groupby('Engineer').agg(agg_dict).rename(columns={'AI_Category': 'ticket_count'})
    if 'Strategic_Friction_Score' in eng_stats.columns:
        eng_stats = eng_stats.sort_values('Strategic_Friction_Score')
        x_vals = eng_stats['Strategic_Friction_Score']
        x_title = 'Average Friction Score'
        text_vals = [f"{v:.0f} ({c} tickets)" for v, c in zip(eng_stats['Strategic_Friction_Score'], eng_stats['ticket_count'])]
    else:
        eng_stats = eng_stats.sort_values('ticket_count')
        x_vals = eng_stats['ticket_count']
        x_title = 'Ticket Count'
        text_vals = [f"{c} tickets" for c in eng_stats['ticket_count']]
    
    fig = go.Figure(go.Bar(
        x=x_vals,
        y=eng_stats.index,
        orientation='h',
        marker=dict(
            color=x_vals,
            colorscale='RdYlGn_r',
            line=dict(width=0)
        ),
        text=text_vals,
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Value: %{x:.1f}<extra></extra>'
    ))
    
    # Get theme without margin, then add custom margin
    theme = create_plotly_theme()
    theme.pop('margin', None)
    
    theme.pop('margin', None)  # Remove margin from theme to avoid conflict
    fig.update_layout(
        **theme,
        title=dict(text='Engineer Performance', font=dict(size=16)),
        xaxis_title=x_title,
        height=400,
        margin=dict(l=150, r=100, t=60, b=40)  # Room for names and values
    )
    
    return fig


# ============================================================================
# EXECUTIVE CHARTS
# ============================================================================

def chart_pareto_analysis(df):
    """Pareto chart showing 80/20 rule for escalation causes."""
    category_friction = df.groupby('AI_Category')['Strategic_Friction_Score'].sum().sort_values(ascending=False)
    cumulative_pct = category_friction.cumsum() / category_friction.sum() * 100
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    colors = ['#FF6B6B' if pct <= 80 else '#6C757D' for pct in cumulative_pct]
    
    fig.add_trace(go.Bar(
        x=category_friction.index,
        y=category_friction.values,
        name='Friction Score',
        marker_color=colors,
        hovertemplate='<b>%{x}</b><br>Friction: %{y:,.0f}<extra></extra>'
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=category_friction.index,
        y=cumulative_pct.values,
        name='Cumulative %',
        mode='lines+markers',
        line=dict(color='#00BFFF', width=3),
        marker=dict(size=8),
        hovertemplate='%{x}<br>Cumulative: %{y:.1f}%<extra></extra>'
    ), secondary_y=True)
    
    fig.add_hline(y=80, line_dash="dash", line_color="#FFC107", 
                  annotation_text="80% Threshold", secondary_y=True)
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(text='🎯 Pareto Analysis: Focus on the Vital Few', font=dict(size=18)),
        height=400,
        xaxis_tickangle=-45,
        legend=dict(orientation='h', y=1.15)
    )
    
    fig.update_yaxes(title_text='Friction Score', secondary_y=False)
    fig.update_yaxes(title_text='Cumulative %', secondary_y=True, range=[0, 105])
    
    return fig


def chart_driver_tree(df):
    """Create a driver tree showing impact decomposition."""
    total_friction = df['Strategic_Friction_Score'].sum()
    
    # Level 1: By Origin
    origin_data = df.groupby('tickets_data_escalation_origin')['Strategic_Friction_Score'].sum()
    
    # Level 2: By Severity within Origin
    severity_origin = df.groupby(['tickets_data_escalation_origin', 'tickets_data_severity'])['Strategic_Friction_Score'].sum()
    
    labels = ['Total Friction']
    parents = ['']
    values = [total_friction]
    
    for origin in origin_data.index:
        labels.append(origin)
        parents.append('Total Friction')
        values.append(origin_data[origin])
        
        for severity in ['Critical', 'Major', 'Minor']:
            if (origin, severity) in severity_origin.index:
                labels.append(f"{origin} - {severity}")
                parents.append(origin)
                values.append(severity_origin[(origin, severity)])
    
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues='total',
        marker=dict(
            colors=values,
            colorscale='RdYlBu_r',
            showscale=True
        ),
        textinfo='label+value+percent parent',
        hovertemplate='<b>%{label}</b><br>Friction: %{value:,.0f}<br>%{percentParent:.1%} of parent<extra></extra>'
    ))
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(text='🌳 Friction Driver Tree', font=dict(size=18)),
        height=500
    )
    
    return fig


def chart_forecast_projection(df):
    """Create 30/60/90 day forecast visualization."""
    df_temp = df.copy()
    df_temp['date'] = pd.to_datetime(df_temp['tickets_data_issue_datetime']).dt.date
    daily = df_temp.groupby('date').agg({
        'Strategic_Friction_Score': 'sum',
        'AI_Category': 'count'
    }).rename(columns={'AI_Category': 'count'}).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    
    # Calculate trend
    daily['day_num'] = (daily['date'] - daily['date'].min()).dt.days
    z = np.polyfit(daily['day_num'], daily['count'], 1)
    slope = z[0]
    
    # Forecast
    last_date = daily['date'].max()
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=90, freq='D')
    forecast_day_nums = np.arange(daily['day_num'].max() + 1, daily['day_num'].max() + 91)
    forecast_values = np.polyval(z, forecast_day_nums)
    
    # Add uncertainty cone
    std = daily['count'].std()
    
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(
        x=daily['date'], y=daily['count'],
        mode='lines',
        name='Historical',
        line=dict(color='#0066CC', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 102, 204, 0.2)'
    ))
    
    # Forecast cone (upper)
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values + 2*std,
        mode='lines',
        name='Upper Bound',
        line=dict(color='rgba(255, 107, 107, 0.3)', width=0),
        showlegend=False
    ))
    
    # Forecast cone (lower)
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=np.maximum(0, forecast_values - 2*std),
        mode='lines',
        name='Forecast Range',
        line=dict(color='rgba(255, 107, 107, 0.3)', width=0),
        fill='tonexty',
        fillcolor='rgba(255, 107, 107, 0.2)'
    ))
    
    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast_values,
        mode='lines',
        name='Forecast',
        line=dict(color='#FF6B6B', width=2, dash='dash')
    ))
    
    # Add 30/60/90 markers
    for days, label in [(30, '30D'), (60, '60D'), (90, '90D')]:
        if days <= len(forecast_dates):
            fig.add_vline(x=forecast_dates[days-1], line_dash="dot", line_color="#FFC107")
            fig.add_annotation(x=forecast_dates[days-1], y=forecast_values[days-1]*1.1,
                             text=label, showarrow=False, font=dict(color='#FFC107'))
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(text='📈 90-Day Escalation Forecast', font=dict(size=18)),
        height=400,
        xaxis_title='Date',
        yaxis_title='Daily Escalations',
        legend=dict(orientation='h', y=1.1)
    )
    
    return fig, slope


def chart_risk_heatmap(df):
    """Create risk heatmap by category and severity."""
    pivot = df.pivot_table(
        values='Strategic_Friction_Score',
        index='AI_Category',
        columns='tickets_data_severity',
        aggfunc='sum',
        fill_value=0
    )
    
    # Reorder columns
    cols = ['Critical', 'Major', 'Minor']
    pivot = pivot[[c for c in cols if c in pivot.columns]]
    
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn_r',
        text=pivot.values.astype(int),
        texttemplate='%{text}',
        textfont=dict(size=12),
        hovertemplate='<b>%{y}</b><br>Severity: %{x}<br>Friction: %{z:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(text='🔥 Risk Heatmap: Category × Severity', font=dict(size=18)),
        height=450,
        xaxis_title='Severity',
        yaxis_title=''
    )
    
    return fig


def chart_benchmark_gauge(metric_name, current_value, benchmark_data, unit=''):
    """Create a benchmark gauge showing position vs industry."""
    best = benchmark_data['best_in_class']
    avg = benchmark_data['industry_avg']
    laggard = benchmark_data['laggard']
    
    # Determine if lower is better
    lower_better = best < laggard
    
    if lower_better:
        min_val, max_val = best * 0.5, laggard * 1.2
    else:
        min_val, max_val = laggard * 0.8, best * 1.1
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_value,
        number={'suffix': unit, 'font': {'size': 32}},
        delta={'reference': avg, 'relative': False, 'suffix': unit,
               'increasing': {'color': '#DC3545' if lower_better else '#28A745'},
               'decreasing': {'color': '#28A745' if lower_better else '#DC3545'}},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': '#0066CC'},
            'steps': [
                {'range': [min_val, best if lower_better else laggard], 'color': 'rgba(40, 167, 69, 0.3)' if lower_better else 'rgba(220, 53, 69, 0.3)'},
                {'range': [best if lower_better else laggard, avg], 'color': 'rgba(255, 193, 7, 0.3)'},
                {'range': [avg, max_val], 'color': 'rgba(220, 53, 69, 0.3)' if lower_better else 'rgba(40, 167, 69, 0.3)'}
            ],
            'threshold': {
                'line': {'color': '#00BFFF', 'width': 4},
                'thickness': 0.75,
                'value': current_value
            }
        }
    ))
    
    # Add benchmark annotations
    fig.add_annotation(x=0.15, y=-0.15, text=f"Best: {best}{unit}", showarrow=False, font=dict(size=10, color='#28A745'))
    fig.add_annotation(x=0.5, y=-0.15, text=f"Avg: {avg}{unit}", showarrow=False, font=dict(size=10, color='#FFC107'))
    fig.add_annotation(x=0.85, y=-0.15, text=f"Laggard: {laggard}{unit}", showarrow=False, font=dict(size=10, color='#DC3545'))
    
    # Get theme without margin, then add specific margin
    theme = create_plotly_theme()
    theme.pop('margin', None)  # Remove margin from theme
    
    fig.update_layout(
        **theme,
        title=dict(text=metric_name, font=dict(size=14)),
        height=300,
        margin=dict(t=30, b=60, l=20, r=20)
    )
    
    return fig

# ============================================================================
# SIMILARITY SEARCH CHARTS
# ============================================================================

def chart_similarity_count_distribution(df):
    """Histogram showing distribution of similar ticket counts."""
    if 'Similar_Ticket_Count' not in df.columns:
        return None

    counts = df['Similar_Ticket_Count'].dropna()
    if len(counts) == 0:
        return None

    fig = go.Figure()

    # Create histogram with color gradient
    fig.add_trace(go.Histogram(
        x=counts,
        nbinsx=min(20, int(counts.max()) + 1),
        marker=dict(
            color=counts,
            colorscale='Viridis',
            line=dict(color='white', width=1)
        ),
        name='Count'
    ))

    # Add statistics
    avg_count = counts.mean()
    zero_matches = (counts == 0).sum()

    fig.add_vline(x=avg_count, line_dash="dash", line_color="#FF6600",
                  annotation_text=f"Avg: {avg_count:.1f}")

    theme = create_plotly_theme()
    theme.pop('margin', None)

    fig.update_layout(
        **theme,
        title=f'Similar Ticket Count Distribution<br><sub>{zero_matches} tickets with no matches</sub>',
        xaxis_title='Number of Similar Tickets Found',
        yaxis_title='Frequency',
        height=400,
        margin=dict(l=50, r=30, t=80, b=50)
    )

    return fig


def chart_resolution_consistency(df):
    """Pie chart showing resolution consistency breakdown."""
    if 'Resolution_Consistency' not in df.columns:
        return None

    consistency = df['Resolution_Consistency'].value_counts()
    if len(consistency) == 0:
        return None

    colors = {
        'Consistent': '#28A745',
        'Inconsistent': '#DC3545',
        'No Similar Data': '#6C757D'
    }

    fig = go.Figure(data=[go.Pie(
        labels=consistency.index,
        values=consistency.values,
        hole=0.4,
        marker=dict(colors=[colors.get(l, '#0066CC') for l in consistency.index]),
        textinfo='label+percent',
        textposition='outside'
    )])

    theme = create_plotly_theme()
    theme.pop('margin', None)

    fig.update_layout(
        **theme,
        title='Resolution Consistency Analysis',
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def chart_similarity_score_distribution(df):
    """Histogram of best match similarity scores."""
    if 'Best_Match_Similarity' not in df.columns:
        return None

    scores = df['Best_Match_Similarity'].dropna()
    scores = scores[scores > 0]  # Filter zeros

    if len(scores) == 0:
        return None

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=20,
        marker=dict(
            color=scores,
            colorscale='RdYlGn',
            cmin=0,
            cmax=1,
            line=dict(color='white', width=1)
        ),
        name='Scores'
    ))

    # Threshold lines
    fig.add_vline(x=0.7, line_dash="dash", line_color="#28A745",
                  annotation_text="High (0.7)")
    fig.add_vline(x=0.5, line_dash="dot", line_color="#FFC107",
                  annotation_text="Medium (0.5)")

    theme = create_plotly_theme()
    theme.pop('margin', None)

    high_conf = (scores >= 0.7).sum()
    fig.update_layout(
        **theme,
        title=f'Similarity Score Distribution<br><sub>{high_conf} high-confidence matches ({high_conf/len(scores)*100:.0f}%)</sub>',
        xaxis_title='Best Match Similarity Score',
        yaxis_title='Frequency',
        height=400,
        margin=dict(l=50, r=30, t=80, b=50)
    )

    return fig


def chart_inconsistent_by_category(df):
    """Bar chart showing inconsistent resolutions by category."""
    if 'Inconsistent_Resolution' not in df.columns or 'AI_Category' not in df.columns:
        return None

    # Filter to inconsistent only
    inconsistent = df[df['Inconsistent_Resolution'] == True]
    if len(inconsistent) == 0:
        return None

    by_cat = inconsistent.groupby('AI_Category').size().sort_values(ascending=True)

    fig = go.Figure(go.Bar(
        x=by_cat.values,
        y=by_cat.index,
        orientation='h',
        marker=dict(
            color=by_cat.values,
            colorscale='Reds',
            line=dict(color='white', width=1)
        ),
        text=by_cat.values,
        textposition='outside'
    ))

    theme = create_plotly_theme()
    theme.pop('margin', None)

    fig.update_layout(
        **theme,
        title='Inconsistent Resolutions by Category',
        xaxis_title='Count of Inconsistent Resolutions',
        yaxis_title='',
        height=400,
        margin=dict(l=150, r=60, t=60, b=50)
    )

    return fig


def chart_similarity_effectiveness_heatmap(df):
    """Heatmap showing similarity effectiveness by category and origin."""
    if 'Similar_Ticket_Count' not in df.columns or 'AI_Category' not in df.columns:
        return None

    # Check for origin column
    origin_col = None
    for col in ['tickets_data_origin', 'Origin', 'tickets_data_source']:
        if col in df.columns:
            origin_col = col
            break

    if origin_col is None:
        return None

    # Calculate effectiveness (avg similar count) by category and origin
    effectiveness = df.groupby(['AI_Category', origin_col])['Similar_Ticket_Count'].mean().unstack(fill_value=0)

    if effectiveness.empty:
        return None

    fig = go.Figure(data=go.Heatmap(
        z=effectiveness.values,
        x=effectiveness.columns,
        y=effectiveness.index,
        colorscale='RdYlGn',
        text=np.round(effectiveness.values, 1),
        texttemplate='%{text:.1f}',
        textfont={"size": 10},
        hoverongaps=False
    ))

    theme = create_plotly_theme()
    theme.pop('margin', None)

    fig.update_layout(
        **theme,
        title='Similarity Search Effectiveness<br><sub>Avg similar tickets found by Category & Origin</sub>',
        xaxis_title='Origin',
        yaxis_title='Category',
        height=450,
        margin=dict(l=150, r=30, t=80, b=80)
    )

    return fig


def chart_expected_vs_predicted_resolution(df):
    """Grouped bar comparing expected (from similar tickets) vs AI predicted resolution."""
    if 'Expected_Resolution_Days' not in df.columns or 'Predicted_Resolution_Days' not in df.columns:
        # Try alternate column name
        if 'Similar_Expected_Days' in df.columns:
            expected_col = 'Similar_Expected_Days'
        else:
            return None
    else:
        expected_col = 'Expected_Resolution_Days'

    if 'AI_Category' not in df.columns:
        return None

    # Aggregate by category
    comparison = df.groupby('AI_Category').agg({
        expected_col: 'mean',
        'Predicted_Resolution_Days': 'mean'
    }).dropna()

    if comparison.empty:
        return None

    comparison = comparison.sort_values('Predicted_Resolution_Days', ascending=False).head(10)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Expected (Similar Tickets)',
        x=comparison.index,
        y=comparison[expected_col],
        marker_color='#0066CC',
        text=[f'{v:.1f}d' for v in comparison[expected_col]],
        textposition='outside'
    ))

    fig.add_trace(go.Bar(
        name='AI Predicted',
        x=comparison.index,
        y=comparison['Predicted_Resolution_Days'],
        marker_color='#FF6600',
        text=[f'{v:.1f}d' for v in comparison['Predicted_Resolution_Days']],
        textposition='outside'
    ))

    theme = create_plotly_theme()
    theme.pop('margin', None)

    fig.update_layout(
        **theme,
        barmode='group',
        title='Resolution Time: Similar Tickets vs AI Prediction',
        xaxis_title='Category',
        yaxis_title='Days',
        xaxis_tickangle=-45,
        height=400,
        margin=dict(l=50, r=30, t=60, b=100),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )

    return fig


# ============================================================================
# LESSONS LEARNED EFFECTIVENESS CHARTS (Comprehensive 6-Pillar Scorecard)
# ============================================================================

# Import the comprehensive scorecard
try:
    from escalation_ai.analysis.lessons_learned import (
        LearningEffectivenessScorecard, create_scorecard, CategoryScorecard
    )
    SCORECARD_AVAILABLE = True
except ImportError:
    SCORECARD_AVAILABLE = False

# Cache for scorecard results
_scorecard_cache: Dict[int, Any] = {}


def get_comprehensive_scorecard(df) -> Optional[Any]:
    """Get or create cached comprehensive scorecard."""
    if not SCORECARD_AVAILABLE:
        return None

    # Use hash of dataframe shape and categories as cache key
    cache_key = hash((len(df), tuple(sorted(df['AI_Category'].unique())) if 'AI_Category' in df.columns else ()))

    if cache_key not in _scorecard_cache:
        try:
            _scorecard_cache[cache_key] = create_scorecard(df)
        except Exception as e:
            return None

    return _scorecard_cache[cache_key]


def chart_scorecard_radar(df, category: str = None):
    """
    Radar chart showing 6-pillar scorecard for a category or overall average.
    """
    scorecard = get_comprehensive_scorecard(df)
    if not scorecard or not scorecard.category_scorecards:
        return None

    pillar_names = [
        'Learning Velocity',
        'Impact Management',
        'Knowledge Quality',
        'Process Maturity',
        'Knowledge Transfer',
        'Outcome Effectiveness'
    ]
    pillar_keys = [
        'learning_velocity', 'impact_management', 'knowledge_quality',
        'process_maturity', 'knowledge_transfer', 'outcome_effectiveness'
    ]

    fig = go.Figure()

    if category and category in scorecard.category_scorecards:
        # Single category radar
        cat_scorecard = scorecard.category_scorecards[category]
        scores = [cat_scorecard.pillars[k].score for k in pillar_keys]

        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],  # Close the shape
            theta=pillar_names + [pillar_names[0]],
            fill='toself',
            fillcolor='rgba(0, 102, 204, 0.3)',
            line=dict(color='#0066CC', width=2),
            name=category,
            hovertemplate='%{theta}: %{r:.0f}<extra></extra>'
        ))
        title = f'Learning Effectiveness Scorecard: {category}<br><sub>Grade: {cat_scorecard.overall_grade} ({cat_scorecard.overall_score:.0f}/100)</sub>'
    else:
        # Average across all categories
        avg_scores = []
        for key in pillar_keys:
            pillar_scores = [sc.pillars[key].score for sc in scorecard.category_scorecards.values()]
            avg_scores.append(np.mean(pillar_scores))

        fig.add_trace(go.Scatterpolar(
            r=avg_scores + [avg_scores[0]],
            theta=pillar_names + [pillar_names[0]],
            fill='toself',
            fillcolor='rgba(0, 102, 204, 0.3)',
            line=dict(color='#0066CC', width=2),
            name='Organization Average',
            hovertemplate='%{theta}: %{r:.0f}<extra></extra>'
        ))
        avg_overall = np.mean([sc.overall_score for sc in scorecard.category_scorecards.values()])
        title = f'Organization Learning Effectiveness<br><sub>Average Score: {avg_overall:.0f}/100</sub>'

    theme = create_plotly_theme()
    theme.pop('margin', None)

    fig.update_layout(
        **theme,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=11))
        ),
        showlegend=False,
        title=dict(text=title, x=0.5, xanchor='center'),
        height=450,
        margin=dict(l=80, r=80, t=100, b=50)
    )

    return fig


def chart_scorecard_comparison(df, categories: List[str] = None):
    """
    Radar chart comparing multiple categories.
    """
    scorecard = get_comprehensive_scorecard(df)
    if not scorecard or not scorecard.category_scorecards:
        return None

    pillar_names = [
        'Learning Velocity', 'Impact Management', 'Knowledge Quality',
        'Process Maturity', 'Knowledge Transfer', 'Outcome Effectiveness'
    ]
    pillar_keys = [
        'learning_velocity', 'impact_management', 'knowledge_quality',
        'process_maturity', 'knowledge_transfer', 'outcome_effectiveness'
    ]

    if not categories:
        # Use top 3 and bottom 2 by default
        sorted_cats = sorted(
            scorecard.category_scorecards.items(),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        categories = [c[0] for c in sorted_cats[:3]] + [c[0] for c in sorted_cats[-2:]]

    colors = ['#0066CC', '#28A745', '#FFC107', '#DC3545', '#6C757D']

    fig = go.Figure()

    for i, cat in enumerate(categories[:5]):
        if cat not in scorecard.category_scorecards:
            continue
        cat_scorecard = scorecard.category_scorecards[cat]
        scores = [cat_scorecard.pillars[k].score for k in pillar_keys]

        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],
            theta=pillar_names + [pillar_names[0]],
            name=f'{cat} ({cat_scorecard.overall_grade})',
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate=f'{cat}<br>%{{theta}}: %{{r:.0f}}<extra></extra>'
        ))

    theme = create_plotly_theme()
    theme.pop('margin', None)

    fig.update_layout(
        **theme,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=10))
        ),
        title='Category Comparison: 6-Pillar Scorecard',
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
        height=500,
        margin=dict(l=80, r=80, t=80, b=100)
    )

    return fig


def chart_pillar_breakdown(df, pillar_name: str):
    """
    Bar chart showing sub-factor scores for a specific pillar across categories.
    """
    scorecard = get_comprehensive_scorecard(df)
    if not scorecard or not scorecard.category_scorecards:
        return None

    pillar_key = pillar_name.lower().replace(' ', '_')

    # Get all categories and their sub-scores for this pillar
    data = []
    for cat, sc in scorecard.category_scorecards.items():
        if pillar_key in sc.pillars:
            pillar = sc.pillars[pillar_key]
            for sub_name, sub_score in pillar.sub_scores.items():
                data.append({
                    'Category': cat,
                    'Sub-Factor': sub_name.replace('_', ' ').title(),
                    'Score': sub_score
                })

    if not data:
        return None

    plot_df = pd.DataFrame(data)

    fig = px.bar(
        plot_df,
        x='Category',
        y='Score',
        color='Sub-Factor',
        barmode='group',
        title=f'{pillar_name} - Sub-Factor Breakdown',
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    theme = create_plotly_theme()
    theme.pop('margin', None)

    fig.update_layout(
        **theme,
        xaxis_tickangle=-45,
        height=450,
        margin=dict(l=50, r=30, t=80, b=100),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )

    return fig


def chart_learning_grades(df):
    """
    Bar chart showing learning effectiveness grades (A-F) by category.
    Uses comprehensive scorecard if available, falls back to simple calculation.
    """
    # Check for required columns
    required_cols = ['AI_Category']
    if not all(col in df.columns for col in required_cols):
        return None

    # Try comprehensive scorecard first
    scorecard = get_comprehensive_scorecard(df)
    if scorecard and scorecard.category_scorecards:
        sorted_items = sorted(
            scorecard.category_scorecards.items(),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        categories = [item[0] for item in sorted_items]
        scores = [item[1].overall_score for item in sorted_items]
        grades = [item[1].overall_grade for item in sorted_items]
    else:
        # Fall back to simple calculation
        grades_data = _calculate_learning_grades(df)
        if not grades_data:
            return None
        sorted_items = sorted(grades_data.items(), key=lambda x: x[1]['score'], reverse=True)
        categories = [item[0] for item in sorted_items]
        scores = [item[1]['score'] for item in sorted_items]
        grades = [item[1]['grade'] for item in sorted_items]

    # Grade colors (extended for +/- grades)
    grade_colors = {
        'A+': '#1B5E20', 'A': '#28A745', 'A-': '#43A047',
        'B+': '#558B2F', 'B': '#7CB342', 'B-': '#9CCC65',
        'C+': '#F9A825', 'C': '#FFC107', 'C-': '#FFCA28',
        'D+': '#EF6C00', 'D': '#FF9800', 'D-': '#FFA726',
        'F': '#DC3545'
    }
    colors = [grade_colors.get(g, '#6C757D') for g in grades]

    fig = go.Figure(go.Bar(
        x=scores,
        y=categories,
        orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=1)),
        text=[f'{g} ({s:.0f})' for g, s in zip(grades, scores)],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Score: %{x:.0f}<br>Grade: %{text}<extra></extra>'
    ))

    theme = create_plotly_theme()
    theme.pop('margin', None)

    fig.update_layout(
        **theme,
        title='Learning Effectiveness Grades by Category<br><sub>Comprehensive 6-Pillar Assessment</sub>',
        xaxis_title='Learning Effectiveness Score (0-100)',
        yaxis_title='',
        xaxis=dict(range=[0, 110]),
        height=max(400, len(categories) * 35),
        margin=dict(l=200, r=80, t=80, b=50)
    )

    return fig


def chart_lesson_completion_rate(df):
    """Grouped bar chart showing documented vs completed lessons by category."""
    # Find lesson columns
    lesson_title_col = None
    lesson_status_col = None
    for col in ['tickets_data_lessons_learned_title', 'Lesson_Title', 'lessons_learned_title']:
        if col in df.columns:
            lesson_title_col = col
            break
    for col in ['tickets_data_lessons_learned_status', 'Lesson_Status', 'lessons_learned_status']:
        if col in df.columns:
            lesson_status_col = col
            break

    if not lesson_title_col or 'AI_Category' not in df.columns:
        return None

    # Calculate documented and completed counts per category
    lessons_data = {}
    for cat in df['AI_Category'].dropna().unique():
        cat_df = df[df['AI_Category'] == cat]
        documented = cat_df[lesson_title_col].notna().sum()
        completed = 0
        if lesson_status_col:
            completed = cat_df[lesson_status_col].str.lower().str.contains('complete|done|closed', na=False).sum()
        if documented > 0:
            lessons_data[cat] = {'documented': documented, 'completed': completed}

    if not lessons_data:
        return None

    # Sort by documented count
    sorted_items = sorted(lessons_data.items(), key=lambda x: x[1]['documented'], reverse=True)[:15]
    categories = [item[0] for item in sorted_items]
    documented = [item[1]['documented'] for item in sorted_items]
    completed = [item[1]['completed'] for item in sorted_items]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Documented',
        x=categories,
        y=documented,
        marker_color='#0066CC',
        text=documented,
        textposition='outside'
    ))

    fig.add_trace(go.Bar(
        name='Completed',
        x=categories,
        y=completed,
        marker_color='#28A745',
        text=completed,
        textposition='outside'
    ))

    theme = create_plotly_theme()
    theme.pop('margin', None)

    fig.update_layout(
        **theme,
        barmode='group',
        title='Lesson Documentation & Completion by Category',
        xaxis_title='Category',
        yaxis_title='Number of Lessons',
        xaxis_tickangle=-45,
        height=450,
        margin=dict(l=50, r=30, t=60, b=120),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )

    return fig


def chart_recurrence_vs_lessons(df):
    """
    Scatter plot showing correlation between lesson completion and recurrence reduction.
    Quadrant analysis: ideal = high completion, low recurrence.
    """
    if 'AI_Category' not in df.columns:
        return None

    # Find lesson columns
    lesson_title_col = None
    lesson_status_col = None
    for col in ['tickets_data_lessons_learned_title', 'Lesson_Title', 'lessons_learned_title']:
        if col in df.columns:
            lesson_title_col = col
            break
    for col in ['tickets_data_lessons_learned_status', 'Lesson_Status', 'lessons_learned_status']:
        if col in df.columns:
            lesson_status_col = col
            break

    # Calculate metrics per category
    correlation_data = {}
    for cat in df['AI_Category'].dropna().unique():
        cat_df = df[df['AI_Category'] == cat]
        ticket_count = len(cat_df)
        if ticket_count < 3:
            continue

        # Recurrence rate from AI prediction or similarity
        recurrence_rate = 0
        if 'AI_Recurrence_Probability' in cat_df.columns:
            recurrence_rate = cat_df['AI_Recurrence_Probability'].mean() * 100
        elif 'Similar_Ticket_Count' in cat_df.columns:
            recurrence_rate = (cat_df['Similar_Ticket_Count'] > 0).mean() * 100

        # Lesson completion
        completion_rate = 0
        if lesson_title_col:
            documented = cat_df[lesson_title_col].notna().sum()
            if documented > 0 and lesson_status_col:
                completed = cat_df[lesson_status_col].str.lower().str.contains('complete|done|closed', na=False).sum()
                completion_rate = (completed / documented) * 100

        correlation_data[cat] = {
            'recurrence_rate': recurrence_rate,
            'lesson_completion': completion_rate,
            'ticket_count': ticket_count
        }

    if not correlation_data:
        return None

    categories = list(correlation_data.keys())
    recurrence = [correlation_data[c]['recurrence_rate'] for c in categories]
    completion = [correlation_data[c]['lesson_completion'] for c in categories]
    ticket_counts = [correlation_data[c]['ticket_count'] for c in categories]

    # Size based on ticket count
    sizes = [max(15, min(60, t * 2)) for t in ticket_counts]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=completion,
        y=recurrence,
        mode='markers+text',
        marker=dict(
            size=sizes,
            color=recurrence,
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title='Recurrence %'),
            line=dict(color='white', width=1)
        ),
        text=[c[:12] + '..' if len(c) > 12 else c for c in categories],
        textposition='top center',
        textfont=dict(size=9),
        hovertemplate='<b>%{text}</b><br>Completion: %{x:.0f}%<br>Recurrence: %{y:.0f}%<extra></extra>'
    ))

    # Add quadrant lines
    fig.add_hline(y=30, line_dash="dash", line_color="#CCCCCC", line_width=1)
    fig.add_vline(x=50, line_dash="dash", line_color="#CCCCCC", line_width=1)

    # Quadrant annotations
    fig.add_annotation(x=75, y=50, text="⚠️ Process Issue", showarrow=False,
                      font=dict(color='#FF9800', size=10))
    fig.add_annotation(x=25, y=50, text="🔴 NEEDS ATTENTION", showarrow=False,
                      font=dict(color='#DC3545', size=10, weight='bold'))
    fig.add_annotation(x=75, y=15, text="✅ IDEAL", showarrow=False,
                      font=dict(color='#28A745', size=10, weight='bold'))
    fig.add_annotation(x=25, y=15, text="Natural Resolution", showarrow=False,
                      font=dict(color='#6C757D', size=10))

    theme = create_plotly_theme()
    theme.pop('margin', None)

    fig.update_layout(
        **theme,
        title='Recurrence Rate vs Lesson Completion<br><sub>Bubble size = ticket volume</sub>',
        xaxis_title='Lesson Completion Rate (%)',
        yaxis_title='Recurrence Rate (%)',
        xaxis=dict(range=[-5, 105]),
        yaxis=dict(range=[-5, max(recurrence) * 1.2 if recurrence else 100]),
        height=500,
        margin=dict(l=60, r=30, t=80, b=60)
    )

    return fig


def chart_learning_heatmap(df):
    """Heatmap showing learning effectiveness scores by Category and LOB."""
    if 'AI_Category' not in df.columns:
        return None

    # Find LOB column
    lob_col = None
    for col in ['tickets_data_lob', 'LOB', 'tickets_data_market']:
        if col in df.columns:
            lob_col = col
            break

    if not lob_col:
        return None

    # Calculate learning scores by category and LOB
    grades_data = _calculate_learning_grades(df)
    if not grades_data:
        return None

    # Build matrix
    categories = list(grades_data.keys())
    lobs = df[lob_col].dropna().unique()

    matrix = []
    for cat in categories:
        row = []
        for lob in lobs:
            subset = df[(df['AI_Category'] == cat) & (df[lob_col] == lob)]
            if len(subset) >= 2:
                subset_grades = _calculate_learning_grades(subset)
                if cat in subset_grades:
                    row.append(subset_grades[cat]['score'])
                else:
                    row.append(np.nan)
            else:
                row.append(np.nan)
        matrix.append(row)

    matrix = np.array(matrix)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[str(l)[:15] for l in lobs],
        y=[c[:20] for c in categories],
        colorscale='RdYlGn',
        text=np.round(matrix, 0),
        texttemplate='%{text:.0f}',
        textfont={"size": 9},
        hoverongaps=False,
        colorbar=dict(title='Score')
    ))

    theme = create_plotly_theme()
    theme.pop('margin', None)

    fig.update_layout(
        **theme,
        title='Learning Effectiveness by Category & LOB<br><sub>Score 0-100 (higher = better)</sub>',
        xaxis_title='LOB / Market',
        yaxis_title='Category',
        height=max(400, len(categories) * 30),
        margin=dict(l=200, r=30, t=80, b=80)
    )

    return fig


def chart_at_risk_categories(df):
    """Bar chart highlighting categories with poor learning (D/F grades) that need attention."""
    grades_data = _calculate_learning_grades(df)
    if not grades_data:
        return None

    # Filter to D and F grades only
    at_risk = {k: v for k, v in grades_data.items() if v['grade'] in ['D', 'F']}
    if not at_risk:
        return None

    # Sort by score ascending (worst first)
    sorted_items = sorted(at_risk.items(), key=lambda x: x[1]['score'])
    categories = [item[0] for item in sorted_items]
    scores = [item[1]['score'] for item in sorted_items]
    grades = [item[1]['grade'] for item in sorted_items]
    recurrence = [item[1].get('recurrence_rate', 0) for item in sorted_items]

    colors = ['#DC3545' if g == 'F' else '#FF9800' for g in grades]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=scores,
        y=categories,
        orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=1)),
        text=[f'{g} - {r:.0f}% recurrence' for g, r in zip(grades, recurrence)],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Score: %{x:.0f}<br>%{text}<extra></extra>'
    ))

    theme = create_plotly_theme()
    theme.pop('margin', None)

    fig.update_layout(
        **theme,
        title='⚠️ At-Risk Categories (D/F Grades)<br><sub>These categories need immediate attention</sub>',
        xaxis_title='Learning Effectiveness Score',
        yaxis_title='',
        xaxis=dict(range=[0, 60]),
        height=max(300, len(categories) * 40),
        margin=dict(l=200, r=100, t=80, b=50)
    )

    return fig


def _calculate_learning_grades(df) -> Dict[str, Dict]:
    """
    Calculate learning effectiveness grades for each category.
    Returns dict: {category: {score, grade, recurrence_rate, lesson_completion, consistency}}
    """
    if 'AI_Category' not in df.columns:
        return {}

    # Find lesson columns
    lesson_title_col = None
    lesson_status_col = None
    for col in ['tickets_data_lessons_learned_title', 'Lesson_Title', 'lessons_learned_title']:
        if col in df.columns:
            lesson_title_col = col
            break
    for col in ['tickets_data_lessons_learned_status', 'Lesson_Status', 'lessons_learned_status']:
        if col in df.columns:
            lesson_status_col = col
            break

    grades_data = {}

    for cat in df['AI_Category'].dropna().unique():
        cat_df = df[df['AI_Category'] == cat]
        ticket_count = len(cat_df)

        if ticket_count < 2:
            continue

        # 1. Recurrence rate (from AI prediction or similarity)
        recurrence_rate = 0
        if 'AI_Recurrence_Probability' in cat_df.columns:
            recurrence_rate = cat_df['AI_Recurrence_Probability'].mean() * 100
        elif 'Similar_Ticket_Count' in cat_df.columns:
            recurrence_rate = (cat_df['Similar_Ticket_Count'] > 0).mean() * 100

        # 2. Lesson completion rate
        lesson_completion = 0
        lessons_documented = 0
        if lesson_title_col:
            lessons_documented = cat_df[lesson_title_col].notna().sum()
            if lessons_documented > 0 and lesson_status_col:
                completed = cat_df[lesson_status_col].str.lower().str.contains('complete|done|closed', na=False).sum()
                lesson_completion = (completed / lessons_documented) * 100

        # 3. Resolution consistency
        consistency = 50  # Default
        if 'Resolution_Consistency' in cat_df.columns:
            consistent = cat_df['Resolution_Consistency'].str.contains('consistent|Consistent', na=False).sum()
            consistency = (consistent / ticket_count) * 100

        # 4. Has any lesson documented (bonus)
        has_lessons = 1 if lessons_documented > 0 else 0

        # Calculate weighted score
        # Lower recurrence = better, so invert it
        recurrence_score = max(0, 100 - recurrence_rate)
        score = (
            recurrence_score * 0.35 +
            lesson_completion * 0.30 +
            consistency * 0.25 +
            has_lessons * 10  # Bonus points for documenting
        )

        # Assign grade
        if score >= 80:
            grade = 'A'
        elif score >= 65:
            grade = 'B'
        elif score >= 50:
            grade = 'C'
        elif score >= 35:
            grade = 'D'
        else:
            grade = 'F'

        grades_data[cat] = {
            'score': score,
            'grade': grade,
            'recurrence_rate': recurrence_rate,
            'lesson_completion': lesson_completion,
            'consistency': consistency,
            'ticket_count': ticket_count,
            'lessons_documented': lessons_documented
        }

    return grades_data


def generate_ai_lesson_recommendations(df, use_ollama: bool = True) -> List[Dict]:
    """
    Generate AI-powered recommendations for improving learning effectiveness.
    Uses Ollama for inference, falls back to rule-based if unavailable.
    """
    grades_data = _calculate_learning_grades(df)
    if not grades_data:
        return []

    recommendations = []

    # Get at-risk categories (D and F grades)
    at_risk = [(cat, data) for cat, data in grades_data.items() if data['grade'] in ['D', 'F']]
    at_risk.sort(key=lambda x: x[1]['score'])

    if use_ollama:
        try:
            import requests
            from ..core.config import OLLAMA_BASE_URL, GEN_MODEL

            # Build context for AI
            context_lines = [
                "Learning Effectiveness Analysis Summary:",
                f"Total categories analyzed: {len(grades_data)}",
                f"At-risk categories (D/F grades): {len(at_risk)}",
                "",
                "Categories needing attention:"
            ]

            for cat, data in at_risk[:10]:
                context_lines.append(
                    f"- {cat}: Grade {data['grade']} (Score: {data['score']:.0f}), "
                    f"Recurrence: {data['recurrence_rate']:.0f}%, "
                    f"Lesson Completion: {data['lesson_completion']:.0f}%"
                )

            context = "\n".join(context_lines)

            prompt = f"""You are an expert in operational excellence and continuous improvement.
Analyze this learning effectiveness data and provide specific, actionable recommendations.

{context}

For each at-risk category, provide:
1. Root cause analysis (why is learning failing?)
2. Specific action to improve (who should do what)
3. Expected impact if addressed

Format each recommendation as:
[PRIORITY: HIGH/MEDIUM] Category: Recommendation

Be specific and actionable. Focus on the worst performing categories first."""

            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": GEN_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 1000}
                },
                timeout=60
            )

            if response.status_code == 200:
                ai_text = response.json().get("response", "").strip()
                if ai_text:
                    recommendations.append({
                        'type': 'ai',
                        'content': ai_text,
                        'categories': [cat for cat, _ in at_risk[:10]]
                    })
                    return recommendations

        except Exception as e:
            pass  # Fall back to rule-based

    # Rule-based recommendations
    for cat, data in at_risk[:5]:
        if data['recurrence_rate'] > 50 and data['lesson_completion'] < 30:
            rec = f"🔴 HIGH PRIORITY: {cat} has {data['recurrence_rate']:.0f}% recurrence with only {data['lesson_completion']:.0f}% lesson completion. Mandate lesson documentation for all resolved tickets in this category."
        elif data['recurrence_rate'] > 30:
            rec = f"🟡 MEDIUM: {cat} shows {data['recurrence_rate']:.0f}% recurrence. Review root causes and ensure lessons learned are being applied."
        else:
            rec = f"📋 {cat}: Improve lesson documentation rate (currently {data['lessons_documented']} documented)."

        recommendations.append({
            'type': 'rule',
            'priority': 'HIGH' if data['grade'] == 'F' else 'MEDIUM',
            'category': cat,
            'content': rec,
            'score': data['score'],
            'grade': data['grade']
        })

    return recommendations


# ============================================================================
# SYSTEMIC ISSUES ANALYSIS
# ============================================================================

# Recommended fixes mapping based on sub-category analysis
SYSTEMIC_ISSUE_FIXES = {
    # Scheduling & Planning sub-categories
    "No TI Entry": {
        "root_cause": "PM/FE coordination gap; reactive scheduling",
        "fix": "Enforce TI entry 24hrs before FE dispatch; auto-block support if no TI record"
    },
    "Schedule Not Followed": {
        "root_cause": "Communication breakdown; schedule changes not propagated",
        "fix": "Implement real-time schedule sync between TI and FE dispatch systems"
    },
    "Weekend Schedule Issue": {
        "root_cause": "Weekend coverage gaps; approval workflow delays",
        "fix": "Pre-approve weekend schedules on Thursday; dedicated weekend coordinator"
    },
    "Ticket Status Issue": {
        "root_cause": "Ticket lifecycle mismanagement; bucket confusion",
        "fix": "Auto-validate ticket status before FE dispatch; status change notifications"
    },
    "Premature Scheduling": {
        "root_cause": "BH/MW readiness not verified before scheduling",
        "fix": "Hard dependency check: no schedule without BH actualization confirmed"
    },

    # Documentation & Reporting sub-categories
    "Missing Snapshot": {
        "root_cause": "Manual process reliance; no validation gate",
        "fix": "Implement mandatory snapshot upload with form validation before submission"
    },
    "Missing Attachment": {
        "root_cause": "Checklist not enforced; time pressure",
        "fix": "Required attachment checklist with blocking validation; template pre-population"
    },
    "Wrong Site ID": {
        "root_cause": "Manual entry errors; copy-paste mistakes",
        "fix": "Auto-populate site ID from ticket; validation against site database"
    },
    "Incomplete Snapshot": {
        "root_cause": "Training gap; unclear requirements",
        "fix": "Snapshot completeness checker with sector count validation"
    },
    "Missing Information": {
        "root_cause": "Template gaps; unclear requirements",
        "fix": "Dynamic form with required fields based on ticket type"
    },

    # Configuration & Data Mismatch sub-categories
    "Port Matrix Mismatch": {
        "root_cause": "Outdated documents; no version control",
        "fix": "Implement PMX validation gate in precheck; flag mismatches automatically"
    },
    "RET Naming": {
        "root_cause": "Manual naming errors; inconsistent conventions",
        "fix": "Automated RET naming validation against site config; naming convention enforcer"
    },
    "TAC Mismatch": {
        "root_cause": "Config drift between tools; manual updates",
        "fix": "Real-time TAC sync between RIOT and OSS; alert on mismatch"
    },
    "CIQ/SCF Mismatch": {
        "root_cause": "Multiple document versions; no single source of truth",
        "fix": "Centralized config repository with version control; auto-diff on updates"
    },

    # Site Readiness sub-categories
    "BH Not Ready": {
        "root_cause": "Pressure to support despite readiness gaps",
        "fix": "Hard-stop in workflow - no precheck release without MB actualization confirmed"
    },
    "MW Not Ready": {
        "root_cause": "MW team coordination gap; status not updated",
        "fix": "MW readiness API check before scheduling; real-time status dashboard"
    },
    "Material Missing": {
        "root_cause": "Inventory tracking gaps; last-minute discoveries",
        "fix": "Material availability check 48hrs before scheduled date; auto-postpone if missing"
    },

    # Process Compliance sub-categories
    "Process Violation": {
        "root_cause": "Schedule pressure overriding compliance",
        "fix": "System-enforced workflow gates; no bypass without manager approval audit trail"
    },
    "Wrong Escalation": {
        "root_cause": "Unclear escalation paths; distro list confusion",
        "fix": "Smart escalation routing based on ticket type; dynamic distro selection"
    },
    "Missing Ticket": {
        "root_cause": "Manual ticket creation; process shortcuts",
        "fix": "Auto-ticket creation triggers; mandatory ticket linkage validation"
    },

    # Validation & QA sub-categories
    "Missed Issue": {
        "root_cause": "Checklist fatigue; time pressure",
        "fix": "AI-assisted anomaly detection; automated red-flag alerts"
    },
    "Incomplete Validation": {
        "root_cause": "Partial checks; validation shortcuts",
        "fix": "Mandatory validation checklist with photo evidence; no progress without completion"
    },
    "Skipped Validation": {
        "root_cause": "Process shortcuts under pressure",
        "fix": "Workflow enforcement - validation step cannot be bypassed"
    },

    # Communication & Response sub-categories
    "Delayed Response": {
        "root_cause": "No SLA tracking; workload imbalance",
        "fix": "Response time SLA alerts at 50%, 75%, 90% thresholds; auto-escalation"
    },
    "No Proactive Communication": {
        "root_cause": "Reactive culture; no communication standards",
        "fix": "Mandatory status updates every 2hrs for active tickets; auto-reminders"
    },

    # Nesting & Tool Errors sub-categories
    "Wrong Nest Type": {
        "root_cause": "Market guideline confusion; tool limitations",
        "fix": "Market-specific nest type validation; block invalid combinations"
    },
    "Missing Nesting": {
        "root_cause": "Manual nesting process; oversight",
        "fix": "Auto-nest trigger on ticket creation; nesting status validation gate"
    }
}


def get_top_systemic_issues(df, top_n: int = 3):
    """
    Analyze data to identify top systemic issues with root causes and recommended fixes.

    Returns a list of dicts with: rank, issue, root_cause, recommended_fix, impact_count, financial_impact
    """
    issues = []

    # Check if we have sub-category data
    if 'AI_Sub_Category' not in df.columns or 'AI_Category' not in df.columns:
        # Fallback to category-level analysis
        if 'AI_Category' in df.columns:
            cat_counts = df['AI_Category'].value_counts()
            for i, (cat, count) in enumerate(cat_counts.head(top_n).items()):
                cost = df[df['AI_Category'] == cat]['Financial_Impact'].sum() if 'Financial_Impact' in df.columns else count * get_benchmark_costs()['avg_per_ticket']
                issues.append({
                    'rank': i + 1,
                    'issue': cat,
                    'root_cause': "Multiple contributing factors identified",
                    'recommended_fix': f"Establish {cat} improvement initiative; conduct root cause deep-dive",
                    'count': count,
                    'financial_impact': cost
                })
        return issues

    # Sub-category level analysis
    subcat_stats = df.groupby(['AI_Category', 'AI_Sub_Category']).agg({
        'AI_Category': 'count'
    }).rename(columns={'AI_Category': 'count'})

    if 'Financial_Impact' in df.columns:
        financial_stats = df.groupby(['AI_Category', 'AI_Sub_Category'])['Financial_Impact'].sum()
        subcat_stats['financial_impact'] = financial_stats
    else:
        subcat_stats['financial_impact'] = subcat_stats['count'] * get_benchmark_costs()['avg_per_ticket']

    subcat_stats = subcat_stats.reset_index()
    subcat_stats = subcat_stats.sort_values('count', ascending=False)

    # Get top N issues
    for i, row in subcat_stats.head(top_n).iterrows():
        sub_cat = row['AI_Sub_Category']
        category = row['AI_Category']

        # Get fix from mapping or generate generic
        if sub_cat in SYSTEMIC_ISSUE_FIXES:
            fix_info = SYSTEMIC_ISSUE_FIXES[sub_cat]
            root_cause = fix_info['root_cause']
            recommended_fix = fix_info['fix']
        else:
            # Generate generic fix based on category
            root_cause = f"Process gap in {category}"
            recommended_fix = f"Review {sub_cat} procedures; implement validation controls"

        issues.append({
            'rank': len(issues) + 1,
            'issue': sub_cat,
            'category': category,
            'root_cause': root_cause,
            'recommended_fix': recommended_fix,
            'count': row['count'],
            'financial_impact': row['financial_impact']
        })

    return issues


def generate_systemic_issue_initiatives(df):
    """
    Generate action items from top systemic issues for the Initiative Status.
    Returns list of initiative dicts ready to add to action_items.
    """
    issues = get_top_systemic_issues(df, top_n=5)
    initiatives = []

    for issue in issues:
        priority = 'P1' if issue['rank'] <= 2 else 'P2'

        initiatives.append({
            'title': f"Fix: {issue['issue']}",
            'priority': priority,
            'description': f"Root Cause: {issue['root_cause']}\n\nRecommended Fix: {issue['recommended_fix']}\n\nImpact: {issue['count']} tickets, ${issue['financial_impact']:,.0f}",
            'impact': f"{issue['count']} tickets, ${issue['financial_impact']:,.0f} impact",
            'confidence': 90 if issue['rank'] == 1 else 85,
            'timeline': '30 days' if priority == 'P1' else '60 days',
            'from_systemic_analysis': True
        })

    return initiatives


# ============================================================================
# WHAT-IF SIMULATOR
# ============================================================================

def generate_strategic_recommendations(df):
    """Generate AI-powered strategic recommendations with confidence scores."""
    recommendations = []
    
    # Analyze data patterns
    category_friction = df.groupby('AI_Category')['Strategic_Friction_Score'].sum().sort_values(ascending=False)
    top_category = category_friction.index[0]
    top_category_pct = category_friction.iloc[0] / category_friction.sum() * 100
    
    # Get safe values
    avg_recurrence = df['AI_Recurrence_Risk'].mean() * 100 if 'AI_Recurrence_Risk' in df.columns else 15
    critical_pct = (df['tickets_data_severity'] == 'Critical').mean() * 100 if 'tickets_data_severity' in df.columns else 10
    avg_resolution = df['Predicted_Resolution_Days'].mean() if 'Predicted_Resolution_Days' in df.columns else 5
    
    sla_breach_rate = df['SLA_Breached'].mean() * 100 if 'SLA_Breached' in df.columns else 12
    
    # Recommendation 1: Category Focus
    if top_category_pct > 15:
        recommendations.append({
            'priority': 'P1',
            'title': f'Establish {top_category} Tiger Team',
            'description': f'{top_category} accounts for {top_category_pct:.0f}% of total friction. Create dedicated cross-functional team to address root causes.',
            'impact': f'Reduce total friction by up to {top_category_pct * 0.4:.0f}%',
            'confidence': 92,
            'timeline': '30 days',
            'investment': '$25,000 - $50,000',
            'roi': '340%'
        })
    
    # Recommendation 2: Recurrence Prevention
    if avg_recurrence > 20:
        recommendations.append({
            'priority': 'P1',
            'title': 'Implement Predictive Maintenance Program',
            'description': f'Recurrence risk at {avg_recurrence:.0f}% indicates systemic issues. Deploy ML-based early warning system.',
            'impact': f'Reduce recurring escalations by 35-50%',
            'confidence': 87,
            'timeline': '60 days',
            'investment': '$75,000 - $120,000',
            'roi': '280%'
        })
    
    # Recommendation 3: SLA Improvement
    if sla_breach_rate > 10:
        recommendations.append({
            'priority': 'P2',
            'title': 'SLA Recovery Initiative',
            'description': f'Current SLA breach rate of {sla_breach_rate:.1f}% exceeds industry benchmark. Implement escalation fast-track protocol.',
            'impact': 'Reduce SLA breaches to <5%',
            'confidence': 85,
            'timeline': '45 days',
            'investment': '$30,000 - $60,000',
            'roi': '420%'
        })
    
    # Recommendation 4: Resolution Optimization
    if avg_resolution > 2.5:
        recommendations.append({
            'priority': 'P2',
            'title': 'Resolution Time Optimization',
            'description': f'Average {avg_resolution:.1f} day resolution time above benchmark. Implement automated triage and parallel processing.',
            'impact': f'Reduce resolution time by {min(40, (avg_resolution - 1.5) / avg_resolution * 100):.0f}%',
            'confidence': 88,
            'timeline': '90 days',
            'investment': '$50,000 - $100,000',
            'roi': '250%'
        })
    
    # Recommendation 5: Training Investment
    if critical_pct > 12:
        recommendations.append({
            'priority': 'P2',
            'title': 'Targeted Skill Development Program',
            'description': f'Critical severity rate of {critical_pct:.0f}% suggests training gaps. Deploy category-specific certification program.',
            'impact': 'Reduce critical escalations by 25-40%',
            'confidence': 79,
            'timeline': '120 days',
            'investment': '$40,000 - $80,000',
            'roi': '200%'
        })
    
    # Recommendation 6: Process Automation
    recommendations.append({
        'priority': 'P3',
        'title': 'Intelligent Process Automation',
        'description': 'Deploy RPA for repetitive escalation handling tasks. Integrate with existing ticketing systems.',
        'impact': 'Reduce manual effort by 30-45%',
        'confidence': 83,
        'timeline': '180 days',
        'investment': '$100,000 - $200,000',
        'roi': '180%'
    })
    
    return recommendations


def render_executive_summary(df):
    """Render the C-Suite Executive Summary page."""
    render_spectacular_header("Executive Intelligence Brief", "Strategic insights for leadership decision-making", "🎯")
    
    # Top-line executive KPIs
    st.markdown("### 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_impact = df['Financial_Impact'].sum() if 'Financial_Impact' in df.columns else len(df) * get_benchmark_costs()['avg_per_ticket']
    # Revenue at risk should be a percentage of financial impact, not a multiplier
    # Using 20% as reasonable estimate for churn risk impact
    revenue_at_risk = df['Revenue_At_Risk'].sum() if 'Revenue_At_Risk' in df.columns else total_impact * 0.20
    cost_per_esc = total_impact / len(df)
    
    with col1:
        st.markdown(f"""
        <div class="exec-kpi">
            <p class="exec-kpi-value money">${total_impact/1000:,.0f}K</p>
            <p class="kpi-label">Total Operational Cost</p>
            <p class="kpi-delta">90-day period</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="exec-kpi">
            <p class="exec-kpi-value alert">${revenue_at_risk/1000:,.0f}K</p>
            <p class="kpi-label">Revenue at Risk</p>
            <p class="kpi-delta delta-up">Due to churn risk</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        potential_savings = total_impact * 0.35
        st.markdown(f"""
        <div class="exec-kpi">
            <p class="exec-kpi-value money">${potential_savings/1000:,.0f}K</p>
            <p class="kpi-label">Savings Opportunity</p>
            <p class="kpi-delta delta-down">35% reduction achievable</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Get safe values
        recurrence_rate = df['AI_Recurrence_Risk'].mean() * 100 if 'AI_Recurrence_Risk' in df.columns else 15
        friction_mean = df['Strategic_Friction_Score'].mean() if 'Strategic_Friction_Score' in df.columns else 50
        health_score = max(20, 100 - recurrence_rate - (friction_mean / 2))
        pulse_color = 'green' if health_score > 70 else 'yellow' if health_score > 50 else 'red'
        st.markdown(f"""
        <div class="exec-kpi">
            <p class="exec-kpi-value">{health_score:.0f}</p>
            <p class="kpi-label">Operational Health Score</p>
            <p><span class="pulse-dot {pulse_color}"></span>{'Healthy' if health_score > 70 else 'At Risk' if health_score > 50 else 'Critical'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Strategic Recommendations
    st.markdown("### 🎯 Strategic Recommendations")
    st.markdown("*AI-generated insights with confidence scoring*")
    
    recommendations = generate_strategic_recommendations(df)
    
    for i, rec in enumerate(recommendations[:4]):  # Top 4 recommendations
        priority_class = f"priority-{rec['priority'].lower()}"
        card_class = 'high-priority' if rec['priority'] == 'P1' else 'medium-priority' if rec['priority'] == 'P2' else ''
        
        st.markdown(f"""
        <div class="strategy-card {card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div>
                    <span class="{priority_class}">{rec['priority']}</span>
                    <strong style="font-size: 1.1rem; margin-left: 12px;">{rec['title']}</strong>
                </div>
                <div class="confidence-badge">
                    <span style="margin-right: 6px;">🎯</span> {rec['confidence']}% confidence
                </div>
            </div>
            <p style="color: #B0B0B0; margin: 8px 0;">{rec['description']}</p>
            <div style="display: flex; gap: 24px; margin-top: 12px;">
                <div><strong style="color: #28A745;">Impact:</strong> <span style="color: #E0E0E0;">{rec['impact']}</span></div>
                <div><strong style="color: #0066CC;">Timeline:</strong> <span style="color: #E0E0E0;">{rec['timeline']}</span></div>
                <div><strong style="color: #FFC107;">Investment:</strong> <span style="color: #E0E0E0;">{rec['investment']}</span></div>
                <div><strong style="color: #00BFFF;">ROI:</strong> <span style="color: #E0E0E0;">{rec['roi']}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

    # Key Charts Row
    col1, col2 = st.columns(2)

    with col1:
        render_chart_with_insight('pareto_analysis', chart_pareto_analysis(df), df)

    with col2:
        forecast_fig, slope = chart_forecast_projection(df)
        st.plotly_chart(forecast_fig, use_container_width=True)
        
        trend_direction = "increasing" if slope > 0 else "decreasing"
        trend_color = "#DC3545" if slope > 0 else "#28A745"
        st.markdown(f"""
        <div style="text-align: center; padding: 12px; background: rgba(0,0,0,0.2); border-radius: 8px;">
            <span style="color: {trend_color}; font-weight: 600;">
                {'📈' if slope > 0 else '📉'} Trend: {abs(slope):.2f} escalations/day {trend_direction}
            </span>
        </div>
        """, unsafe_allow_html=True)


def render_financial_analysis(df):
    """Render the Enhanced Financial Impact Analysis page with advanced metrics."""
    from escalation_ai.financial import (
        calculate_financial_metrics,
        calculate_roi_metrics,
        calculate_cost_avoidance,
        calculate_efficiency_metrics,
        calculate_financial_forecasts,
        generate_financial_insights,
        create_financial_waterfall,
        create_roi_opportunity_chart,
        create_cost_avoidance_breakdown,
        create_cost_trend_forecast,
        create_efficiency_scorecard,
        create_category_cost_comparison,
        create_engineer_cost_efficiency_matrix,
        create_financial_kpi_cards,
        create_insights_table
    )

    render_spectacular_header("Financial Impact Analysis", "Comprehensive financial metrics, ROI analysis, and cost optimization", "💰")

    # Ensure Financial_Impact column exists - calculate from price_catalog
    if 'Financial_Impact' not in df.columns:
        df = df.copy()
        df = _calculate_financial_impact_from_catalog(df)

    # Calculate comprehensive metrics
    with st.spinner('Calculating advanced financial metrics...'):
        financial_metrics = calculate_financial_metrics(df)
        roi_metrics = calculate_roi_metrics(df)
        cost_avoidance = calculate_cost_avoidance(df)
        efficiency_metrics = calculate_efficiency_metrics(df)
        forecasts = calculate_financial_forecasts(df)
        insights = generate_financial_insights(df)

    # KPI Cards
    st.markdown("### 📊 Key Financial Indicators")
    kpi_data = create_financial_kpi_cards(financial_metrics)

    # Display in 3x2 grid
    for row_idx in range(2):
        cols = st.columns(3)
        for col_idx, col in enumerate(cols):
            kpi_idx = row_idx * 3 + col_idx
            if kpi_idx < len(kpi_data):
                kpi = kpi_data[kpi_idx]
                with col:
                    # Proper color logic: green = good, red = bad
                    delta_color = "off"  # Default: no color

                    # Get numeric delta for proper color calculation
                    delta_value = kpi.get('delta')
                    delta_display = kpi.get('delta_text')

                    # For costs: lower is better (inverse)
                    if 'Cost' in kpi['title'] or 'Revenue at Risk' in kpi['title']:
                        delta_color = "inverse"

                    # For positive metrics: higher is better (normal)
                    elif 'ROI' in kpi['title'] or 'Efficiency' in kpi['title'] or 'Avoidance' in kpi['title']:
                        delta_color = "normal"

                    # Use numeric delta if available for proper coloring, otherwise use text
                    display_delta = delta_value if delta_value is not None else delta_display

                    st.metric(
                        label=f"{kpi['icon']} {kpi['title']}",
                        value=kpi['value'],
                        delta=display_delta,
                        delta_color=delta_color
                    )

    st.markdown("---")

    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "💹 ROI Opportunities",
        "💡 Cost Avoidance",
        "📈 Trends & Forecast",
        "🎯 Insights & Actions"
    ])

    with tab1:
        st.markdown("### Financial Impact Overview")

        col1, col2 = st.columns(2)

        with col1:
            # Financial waterfall
            try:
                waterfall_data = {
                    'total_cost': financial_metrics.total_cost,
                    'recurring_issue_cost': financial_metrics.recurring_issue_cost,
                    'preventable_cost': financial_metrics.preventable_cost,
                    'customer_impact_cost': financial_metrics.customer_impact_cost,
                    'sla_penalty_exposure': financial_metrics.sla_penalty_exposure
                }
                fig = create_financial_waterfall(waterfall_data)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating waterfall chart: {e}")

        with col2:
            # Efficiency scorecard
            try:
                fig = create_efficiency_scorecard(efficiency_metrics, financial_metrics)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating efficiency scorecard: {e}")

        # Category cost comparison
        st.markdown("### Cost Analysis by Category")
        try:
            fig = create_category_cost_comparison(df)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating category comparison: {e}")

        # Cost concentration
        st.markdown("### Cost Concentration (Pareto Analysis)")
        from escalation_ai.financial.visualizations import create_cost_concentration_chart
        try:
            fig = create_cost_concentration_chart(df)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating concentration chart: {e}")

        st.info(f"""
        **Cost Concentration**: {financial_metrics.cost_concentration_ratio*100:.0f}% of total costs come from the top 20% of tickets.
        {'🔴 High concentration - focus on top cost drivers' if financial_metrics.cost_concentration_ratio > 0.8 else '🟢 Good cost distribution'}
        """)

    with tab2:
        st.markdown("### 💹 ROI Investment Opportunities")

        if roi_metrics['top_opportunities']:
            # ROI opportunity chart
            try:
                fig = create_roi_opportunity_chart(roi_metrics)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating ROI chart: {e}")

            # ROI opportunity table
            st.markdown("### Top ROI Opportunities")
            roi_df = pd.DataFrame(roi_metrics['top_opportunities']).round(2)
            roi_df_display = pd.DataFrame({
                'Category': roi_df['category'],
                'Incidents': roi_df['incident_count'],
                'Total Cost': roi_df['total_cost'].apply(lambda x: f"${x:,.0f}"),
                'Investment': roi_df['investment_required'].apply(lambda x: f"${x:,.0f}"),
                'Annual Savings': roi_df['annual_savings'].apply(lambda x: f"${x:,.0f}"),
                'ROI %': roi_df['roi_percentage'].apply(lambda x: f"{x:.0f}%"),
                'Payback (mo)': roi_df['payback_months'].apply(lambda x: f"{x:.1f}")
            })

            st.dataframe(roi_df_display, use_container_width=True, hide_index=True)

            # ROI summary
            st.success(f"""
            **Investment Summary:**
            - Total Investment: **${roi_metrics['total_investment_required']:,.0f}**
            - Expected Annual Savings: **${roi_metrics['expected_annual_savings']:,.0f}**
            - Overall ROI: **{roi_metrics['roi_percentage']:.0f}%**
            - Payback Period: **{roi_metrics['payback_months']:.1f} months**
            """)
        else:
            st.info("Not enough recurring patterns to identify ROI opportunities. Need at least 3 similar incidents per category.")

    with tab3:
        st.markdown("### 💡 Cost Avoidance Potential")

        col1, col2 = st.columns(2)

        with col1:
            # Cost avoidance breakdown
            try:
                fig = create_cost_avoidance_breakdown(cost_avoidance)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating cost avoidance chart: {e}")

        with col2:
            # Cost avoidance details
            st.markdown("#### Avoidance Opportunities")

            avoidance_items = [
                ("🔄 Recurring Issues", cost_avoidance['recurring_issues'], "Fix root causes to prevent repeat incidents"),
                ("📋 Preventable Categories", cost_avoidance['preventable_categories'], "Improve processes and documentation"),
                ("📚 Knowledge Sharing", cost_avoidance['knowledge_sharing'], "Leverage similar ticket solutions"),
                ("🤖 Automation", cost_avoidance['automation'], "Automate repetitive tasks")
            ]

            for label, value, description in avoidance_items:
                st.markdown(f"""
                <div style="padding: 15px; background: #0a1929; border-left: 4px solid #2ca02c; margin-bottom: 10px;">
                    <div style="font-size: 1.1rem; font-weight: 600;">{label}</div>
                    <div style="font-size: 1.5rem; color: #2ca02c; font-weight: 700;">${value:,.0f}</div>
                    <div style="color: #999; font-size: 0.9rem;">{description}</div>
                </div>
                """, unsafe_allow_html=True)

            st.success(f"**Total Avoidance Potential: ${cost_avoidance['total_avoidance']:,.0f}**")

    with tab4:
        st.markdown("### 📈 Cost Trends & Financial Forecast")

        # Check if we have required data
        has_dates = any(col for col in df.columns if 'date' in col.lower() or 'time' in col.lower())

        if not has_dates:
            st.warning("""
            **No date information available in this dataset.**

            To enable trend analysis and forecasting:
            1. Regenerate the report with the latest pipeline
            2. Ensure your input data has an 'Issue Date' or 'Created Date' column

            The current report was generated without date information.
            """)

            # Show basic cost summary instead
            if 'AI_Category' in df.columns:
                st.markdown("#### Current Cost Summary by Category")
                cost_summary = df.groupby('AI_Category')['Financial_Impact'].agg(['sum', 'mean', 'count'])
                cost_summary.columns = ['Total Cost', 'Avg Cost', 'Count']
                cost_summary = cost_summary.sort_values('Total Cost', ascending=False)
                st.dataframe(cost_summary.style.format({
                    'Total Cost': '${:,.0f}',
                    'Avg Cost': '${:,.0f}'
                }), use_container_width=True)
        else:
            # Forecast chart
            try:
                fig = create_cost_trend_forecast(df, forecasts)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating trend forecast: {e}")
                st.info("Try regenerating the report with the latest pipeline.")

        # Forecast metrics (only show if we have forecast data)
        if forecasts.get('monthly_projection'):
            col1, col2, col3 = st.columns(3)
        else:
            st.info("Run the pipeline with date information to see forecasts.")
            col1, col2, col3 = st.columns(3)

        with col1:
            trend_icon = "📈" if forecasts['trend'] == 'increasing' else "📉" if forecasts['trend'] == 'decreasing' else "➡️"
            st.metric("Cost Trend", f"{trend_icon} {forecasts['trend'].title()}",
                     delta=f"{forecasts['confidence'].title()} confidence")

        with col2:
            st.metric("30-Day Projection", f"${financial_metrics.cost_forecast_30d:,.0f}")

        with col3:
            st.metric("Annual Projection", f"${forecasts.get('annual_projection', 0):,.0f}")

        # Risk scenarios
        if forecasts.get('risk_scenarios'):
            st.markdown("#### 📊 Financial Scenarios")
            scenarios_df = pd.DataFrame({
                'Scenario': ['Best Case (20% reduction)', 'Expected', 'Worst Case (30% increase)'],
                'Annual Cost': [
                    f"${forecasts['risk_scenarios']['best_case']:,.0f}",
                    f"${forecasts['risk_scenarios']['expected']:,.0f}",
                    f"${forecasts['risk_scenarios']['worst_case']:,.0f}"
                ],
                'Monthly': [
                    f"${forecasts['risk_scenarios']['best_case']/12:,.0f}",
                    f"${forecasts['risk_scenarios']['expected']/12:,.0f}",
                    f"${forecasts['risk_scenarios']['worst_case']/12:,.0f}"
                ]
            })
            st.dataframe(scenarios_df, use_container_width=True, hide_index=True)

    with tab5:
        st.markdown("### 🎯 Financial Insights & Action Items")

        if insights:
            # Display insights as cards
            for insight in insights:
                priority_colors = {
                    'high': '#d62728',
                    'medium': '#ff7f0e',
                    'low': '#2ca02c'
                }
                color = priority_colors.get(insight['priority'], '#999')

                st.markdown(f"""
                <div style="padding: 20px; background: #0a1929; border-left: 5px solid {color}; margin-bottom: 15px;">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div style="flex: 1;">
                            <div style="font-size: 0.8rem; color: {color}; font-weight: 600; text-transform: uppercase;">
                                {insight['priority']} Priority
                            </div>
                            <div style="font-size: 1.2rem; font-weight: 700; margin: 8px 0;">
                                {insight['title']}
                            </div>
                            <div style="color: #bbb; margin-bottom: 10px;">
                                {insight['description']}
                            </div>
                            <div style="background: #001e3c; padding: 10px; border-radius: 5px;">
                                <strong>💡 Recommendation:</strong> {insight['recommendation']}
                            </div>
                        </div>
                        <div style="text-align: right; margin-left: 20px;">
                            <div style="font-size: 0.8rem; color: #888;">Potential Savings</div>
                            <div style="font-size: 1.8rem; color: #2ca02c; font-weight: 700;">
                                ${insight.get('potential_savings', 0):,.0f}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Insights summary table
            st.markdown("### 📋 Insights Summary Table")
            insights_df = create_insights_table(insights)
            st.dataframe(insights_df, use_container_width=True, hide_index=True)
        else:
            st.info("No significant financial insights identified. Continue monitoring.")

        # Engineer efficiency matrix
        if 'Engineer_Assigned' in df.columns and 'Resolution_Days' in df.columns:
            st.markdown("### 👥 Engineer Cost Efficiency Analysis")
            try:
                fig = create_engineer_cost_efficiency_matrix(df)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating engineer efficiency matrix: {e}")

    # Bottom section: Interactive ROI Calculator
    st.markdown("---")
    st.markdown("### 💹 ROI Scenario Calculator")

    col1, col2 = st.columns([1, 2])

    with col1:
        reduction_pct = st.slider("Target Friction Reduction %", 10, 50, 25)
        investment = st.number_input("Proposed Investment ($)", 50000, 500000, 100000, step=25000)
        timeline_months = st.slider("Implementation Timeline (months)", 3, 18, 6)

    with col2:
        # Calculate ROI
        total_cost = financial_metrics.total_cost
        annual_savings = (total_cost * 4) * (reduction_pct / 100)  # Annualized
        roi = ((annual_savings - investment) / investment) * 100 if investment > 0 else 0
        payback_months = investment / (annual_savings / 12) if annual_savings > 0 else float('inf')
        npv = sum([(annual_savings - investment if i == 0 else annual_savings) / (1.08 ** i) for i in range(3)])

        st.markdown(f"""
        <div class="exec-card">
            <h4 style="color: #00BFFF; margin-bottom: 20px;">📈 Projected Financial Outcomes</h4>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px;">
                <div style="text-align: center;">
                    <p style="font-size: 2rem; font-weight: 700; color: #28A745; margin: 0;">${annual_savings:,.0f}</p>
                    <p style="color: #888; font-size: 0.85rem;">Annual Savings</p>
                </div>
                <div style="text-align: center;">
                    <p style="font-size: 2rem; font-weight: 700; color: #00BFFF; margin: 0;">{roi:.0f}%</p>
                    <p style="color: #888; font-size: 0.85rem;">First Year ROI</p>
                </div>
                <div style="text-align: center;">
                    <p style="font-size: 2rem; font-weight: 700; color: #FFC107; margin: 0;">{f'{payback_months:.1f}' if payback_months != float('inf') else 'N/A'}</p>
                    <p style="color: #888; font-size: 0.85rem;">Payback (Months)</p>
                </div>
                <div style="text-align: center;">
                    <p style="font-size: 2rem; font-weight: 700; color: #4CAF50; margin: 0;">${npv:,.0f}</p>
                    <p style="color: #888; font-size: 0.85rem;">3-Year NPV</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_benchmarking(df):
    """Render the Competitive Benchmarking page."""
    render_spectacular_header("Competitive Benchmarking", "How you compare against industry standards", "🏆")
    
    # Get safe values with defaults
    recurrence_rate = df['AI_Recurrence_Risk'].mean() if 'AI_Recurrence_Risk' in df.columns else 0.15
    resolution_days = df['Predicted_Resolution_Days'].mean() if 'Predicted_Resolution_Days' in df.columns else 5
    
    # Calculate current metrics
    current_metrics = {
        'resolution_days': resolution_days,
        'recurrence_rate': recurrence_rate * 100,
        'sla_breach_rate': df['SLA_Breached'].mean() * 100 if 'SLA_Breached' in df.columns else 12,
        'first_contact_resolution': 100 - (recurrence_rate * 100 * 2),  # Proxy
        'cost_per_escalation': df['Financial_Impact'].mean() if 'Financial_Impact' in df.columns else get_benchmark_costs()['avg_per_ticket'],
        'customer_satisfaction': 100 - (df['Customer_Impact_Score'].mean() * 0.3) if 'Customer_Impact_Score' in df.columns else 75,
    }
    
    # Benchmark gauges
    col1, col2, col3 = st.columns(3)
    
    gauge_configs = [
        ('Resolution Time', 'resolution_days', current_metrics['resolution_days'], ' days'),
        ('Recurrence Rate', 'recurrence_rate', current_metrics['recurrence_rate'], '%'),
        ('SLA Breach Rate', 'sla_breach_rate', current_metrics['sla_breach_rate'], '%'),
        ('First Contact Resolution', 'first_contact_resolution', current_metrics['first_contact_resolution'], '%'),
        ('Cost per Escalation', 'cost_per_escalation', current_metrics['cost_per_escalation'], '$'),
        ('Customer Satisfaction', 'customer_satisfaction', current_metrics['customer_satisfaction'], '%'),
    ]
    
    for i, (name, key, value, unit) in enumerate(gauge_configs):
        col = [col1, col2, col3][i % 3]
        with col:
            fig = chart_benchmark_gauge(name, value, INDUSTRY_BENCHMARKS[key], unit)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Competitive Position Summary
    st.markdown("### 📊 Competitive Position Summary")
    
    position_data = []
    for name, key, value, unit in gauge_configs:
        bench = INDUSTRY_BENCHMARKS[key]
        lower_better = bench['best_in_class'] < bench['laggard']
        
        if lower_better:
            if value <= bench['best_in_class']:
                position = "Best-in-Class"
                gap = 0
                color = "#28A745"
            elif value <= bench['industry_avg']:
                position = "Above Average"
                gap = value - bench['best_in_class']
                color = "#28A745"
            elif value <= bench['laggard']:
                position = "Below Average"
                gap = value - bench['industry_avg']
                color = "#FFC107"
            else:
                position = "Laggard"
                gap = value - bench['laggard']
                color = "#DC3545"
        else:
            if value >= bench['best_in_class']:
                position = "Best-in-Class"
                gap = 0
                color = "#28A745"
            elif value >= bench['industry_avg']:
                position = "Above Average"
                gap = bench['best_in_class'] - value
                color = "#28A745"
            elif value >= bench['laggard']:
                position = "Below Average"
                gap = bench['industry_avg'] - value
                color = "#FFC107"
            else:
                position = "Laggard"
                gap = bench['laggard'] - value
                color = "#DC3545"
        
        position_data.append({
            'Metric': name,
            'Current': f"{value:.1f}{unit}",
            'Best-in-Class': f"{bench['best_in_class']}{unit}",
            'Industry Avg': f"{bench['industry_avg']}{unit}",
            'Position': position,
            'Gap to Best': f"{gap:.1f}{unit}" if gap > 0 else "—",
            'Color': color
        })
    
    # Display as a proper dataframe with styling
    display_df = pd.DataFrame(position_data)
    display_df = display_df[['Metric', 'Current', 'Best-in-Class', 'Industry Avg', 'Position', 'Gap to Best']]
    
    # Use Streamlit's dataframe with custom styling
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Metric": st.column_config.TextColumn("Metric", width="medium"),
            "Current": st.column_config.TextColumn("Current", width="small"),
            "Best-in-Class": st.column_config.TextColumn("Best-in-Class", width="small"),
            "Industry Avg": st.column_config.TextColumn("Industry Avg", width="small"),
            "Position": st.column_config.TextColumn("Position", width="medium"),
            "Gap to Best": st.column_config.TextColumn("Gap to Best", width="small"),
        }
    )


def render_root_cause(df):
    """Render Root Cause Analysis page."""
    render_spectacular_header("Root Cause Analysis", "Identify and quantify the drivers of escalation friction", "🔬")

    col1, col2 = st.columns(2)

    with col1:
        render_chart_with_insight('pareto_analysis', chart_pareto_analysis(df), df)

    with col2:
        st.plotly_chart(chart_driver_tree(df), use_container_width=True)
    
    st.markdown("---")
    
    # Root cause breakdown
    st.markdown("### 🎯 Root Cause Impact Quantification")
    
    if 'Root_Cause' in df.columns:
        root_cause_analysis = df.groupby('Root_Cause').agg({
            'Strategic_Friction_Score': 'sum',
            'Financial_Impact': 'sum',
            'AI_Category': 'count'
        }).rename(columns={'AI_Category': 'count'}).sort_values('Strategic_Friction_Score', ascending=False)
        
        root_cause_analysis['Friction %'] = root_cause_analysis['Strategic_Friction_Score'] / root_cause_analysis['Strategic_Friction_Score'].sum() * 100
        root_cause_analysis['Avg Cost'] = root_cause_analysis['Financial_Impact'] / root_cause_analysis['count']
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Friction by Root Cause', 'Cost Impact by Root Cause'),
                            horizontal_spacing=0.15)  # Add spacing between subplots
        
        fig.add_trace(go.Bar(
            y=root_cause_analysis.index,
            x=root_cause_analysis['Strategic_Friction_Score'],
            orientation='h',
            marker_color=px.colors.sequential.Blues_r[:len(root_cause_analysis)],
            name='Friction'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            y=root_cause_analysis.index,
            x=root_cause_analysis['Financial_Impact'],
            orientation='h',
            marker_color=px.colors.sequential.Reds_r[:len(root_cause_analysis)],
            name='Cost'
        ), row=1, col=2)
        
        fig.update_layout(
            **{
                **create_plotly_theme(),
                'margin': dict(l=150, r=40, t=60, b=40),  # More left margin for labels
            },
            height=500,  # Taller to prevent overlap
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # Risk heatmap
    render_chart_with_insight('risk_heatmap', chart_risk_heatmap(df), df)


def render_action_tracker(df):
    """Render the Action Tracker page."""
    render_spectacular_header("Action Tracker", "Strategic initiatives monitoring and accountability", "📋")

    # =========================================================================
    # TOP 5 SYSTEMIC ISSUES TABLE
    # =========================================================================
    st.markdown("### 🎯 Top 5 Systemic Issues")
    st.markdown("*Data-driven analysis of highest-impact recurring problems with recommended fixes*")

    systemic_issues = get_top_systemic_issues(df, top_n=5)

    if systemic_issues:
        # Create styled table
        issues_data = []
        for issue in systemic_issues:
            issues_data.append({
                'Rank': issue['rank'],
                'Issue': issue['issue'],
                'Root Cause': issue['root_cause'],
                'Recommended Fix': issue['recommended_fix'],
                'Tickets': issue['count'],
                'Impact': f"${issue['financial_impact']:,.0f}"
            })

        issues_df = pd.DataFrame(issues_data)

        # Custom styling for the table
        st.markdown("""
        <style>
        .systemic-table {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 10px;
            padding: 5px;
        }
        .systemic-table th {
            background-color: #0f3460 !important;
            color: #e94560 !important;
            font-weight: bold;
            text-align: left;
        }
        .systemic-table td {
            color: #eee !important;
            border-bottom: 1px solid #333;
        }
        .systemic-table tr:hover td {
            background-color: #1f4068 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.dataframe(
            issues_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Issue": st.column_config.TextColumn("Issue", width="medium"),
                "Root Cause": st.column_config.TextColumn("Root Cause", width="large"),
                "Recommended Fix": st.column_config.TextColumn("Recommended Fix", width="large"),
                "Tickets": st.column_config.NumberColumn("Tickets", width="small"),
                "Impact": st.column_config.TextColumn("Impact", width="small"),
            }
        )

        # Quick action buttons to convert issues to initiatives
        st.markdown("##### Quick Actions")
        col_actions = st.columns(5)
        for idx, issue in enumerate(systemic_issues):
            with col_actions[idx]:
                btn_key = f"add_issue_{idx}_{issue['issue'][:10]}"
                if st.button(f"➕ #{issue['rank']}", key=btn_key, help=f"Add '{issue['issue']}' to initiatives"):
                    # Check if already exists
                    existing_titles = {item['title'] for item in st.session_state.action_items}
                    new_title = f"Fix: {issue['issue']}"
                    if new_title not in existing_titles:
                        new_id = max((item['id'] for item in st.session_state.action_items), default=-1) + 1
                        st.session_state.action_items.append({
                            'id': new_id,
                            'title': new_title,
                            'priority': 'P1' if issue['rank'] <= 2 else 'P2',
                            'owner': 'Unassigned',
                            'due_date': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
                            'status': 'Not Started',
                            'progress': 0,
                            'notes': f"Root Cause: {issue['root_cause']}\n\nRecommended Fix: {issue['recommended_fix']}\n\nImpact: {issue['count']} tickets, ${issue['financial_impact']:,.0f}",
                            'ai_generated': True,
                            'from_systemic': True
                        })
                        save_action_items(st.session_state.action_items)
                        st.toast(f"Added '{issue['issue']}' to initiatives!", icon="✅")
                        st.rerun()
                    else:
                        st.toast(f"'{issue['issue']}' already in initiatives", icon="ℹ️")
    else:
        st.info("No systemic issues identified. Run classification pipeline with sub-category analysis enabled.")

    st.markdown("---")

    # =========================================================================
    # INITIATIVE MANAGEMENT
    # =========================================================================

    # Initialize action items - load from JSON, then merge with AI recommendations
    if not st.session_state.action_items:
        saved_items = load_action_items() or []
        st.session_state.action_items = saved_items

    # Always generate fresh AI recommendations and merge new ones
    recommendations = generate_strategic_recommendations(df)

    # Also get systemic issue initiatives
    systemic_initiatives = generate_systemic_issue_initiatives(df)
    all_recommendations = recommendations[:3] + systemic_initiatives[:2]  # Mix both types

    existing_titles = {item['title'] for item in st.session_state.action_items}

    new_items_added = False
    for rec in all_recommendations:
        if rec['title'] not in existing_titles:
            # This is a new AI recommendation - add it
            new_id = max((item['id'] for item in st.session_state.action_items), default=-1) + 1
            st.session_state.action_items.append({
                'id': new_id,
                'title': rec['title'],
                'priority': rec['priority'],
                'owner': 'Unassigned',
                'due_date': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
                'status': 'Not Started',
                'progress': 0,
                'notes': rec['description'],
                'ai_generated': True,
                'from_systemic': rec.get('from_systemic_analysis', False)
            })
            new_items_added = True

    if new_items_added:
        save_action_items(st.session_state.action_items)
        st.toast("🤖 New AI recommendations added!", icon="✨")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    total_actions = len(st.session_state.action_items)
    completed = sum(1 for a in st.session_state.action_items if a['status'] == 'Completed')
    in_progress = sum(1 for a in st.session_state.action_items if a['status'] == 'In Progress')
    blocked = sum(1 for a in st.session_state.action_items if a['status'] == 'Blocked')

    with col1:
        st.metric("Total Initiatives", total_actions)
    with col2:
        st.metric("Completed", completed, delta=f"{completed/total_actions*100:.0f}%" if total_actions > 0 else "0%")
    with col3:
        st.metric("In Progress", in_progress)
    with col4:
        st.metric("Blocked", blocked, delta_color="inverse" if blocked > 0 else "normal")

    st.markdown("---")

    # Add new action
    with st.expander("➕ Add New Initiative"):
        col1, col2 = st.columns(2)
        with col1:
            new_title = st.text_input("Initiative Title")
            new_priority = st.selectbox("Priority", ['P1', 'P2', 'P3'])
        with col2:
            new_owner = st.text_input("Owner")
            new_due = st.date_input("Due Date", value=datetime.now() + timedelta(days=30))
        
        new_notes = st.text_area("Description/Notes")
        
        if st.button("Add Initiative"):
            st.session_state.action_items.append({
                'id': len(st.session_state.action_items),
                'title': new_title,
                'priority': new_priority,
                'owner': new_owner,
                'due_date': new_due.strftime('%Y-%m-%d'),
                'status': 'Not Started',
                'progress': 0,
                'notes': new_notes
            })
            save_action_items(st.session_state.action_items)
            st.rerun()
    
    # Action items list
    st.markdown("### 📝 Initiative Status")
    
    # Track items to delete (can't modify list while iterating)
    items_to_delete = []
    
    for i, action in enumerate(st.session_state.action_items):
        status_class = 'completed' if action['status'] == 'Completed' else 'in-progress' if action['status'] == 'In Progress' else 'blocked' if action['status'] == 'Blocked' else ''
        priority_class = f"priority-{action['priority'].lower()}"
        
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 0.5])
            
            with col1:
                ai_badge = "🤖 " if action.get('ai_generated') else ""
                st.markdown(f"""
                <div class="action-card {status_class}">
                    <span class="{priority_class}">{action['priority']}</span>
                    <strong style="margin-left: 12px;">{ai_badge}{action['title']}</strong>
                    <p style="color: #888; font-size: 0.85rem; margin: 8px 0 0 0;">{action['notes'][:100]}{'...' if len(action['notes']) > 100 else ''}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                new_status = st.selectbox(
                    "Status", 
                    ['Not Started', 'In Progress', 'Completed', 'Blocked'],
                    index=['Not Started', 'In Progress', 'Completed', 'Blocked'].index(action['status']),
                    key=f"status_{i}"
                )
                if st.session_state.action_items[i]['status'] != new_status:
                    st.session_state.action_items[i]['status'] = new_status
                    save_action_items(st.session_state.action_items)
            
            with col3:
                new_owner = st.text_input("Owner", value=action['owner'], key=f"owner_{i}")
                if st.session_state.action_items[i]['owner'] != new_owner:
                    st.session_state.action_items[i]['owner'] = new_owner
                    save_action_items(st.session_state.action_items)
            
            with col4:
                progress = st.slider("Progress", 0, 100, action['progress'], key=f"progress_{i}")
                if st.session_state.action_items[i]['progress'] != progress:
                    st.session_state.action_items[i]['progress'] = progress
                    save_action_items(st.session_state.action_items)
            
            with col5:
                st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                col_close, col_del = st.columns(2)
                with col_close:
                    if action['status'] != 'Completed':
                        if st.button("✅", key=f"close_{i}", help="Mark as Completed"):
                            st.session_state.action_items[i]['status'] = 'Completed'
                            st.session_state.action_items[i]['progress'] = 100
                            save_action_items(st.session_state.action_items)
                            st.rerun()
                with col_del:
                    if st.button("🗑️", key=f"delete_{i}", help="Delete Initiative"):
                        items_to_delete.append(i)
        
        st.markdown("---")
    
    # Process deletions after loop
    if items_to_delete:
        for idx in sorted(items_to_delete, reverse=True):
            st.session_state.action_items.pop(idx)
        save_action_items(st.session_state.action_items)
        st.rerun()


def render_presentation_mode(df):
    """Render Executive Presentation Mode with auto-cycling slides."""
    render_spectacular_header("Executive Presentation", "Auto-cycling executive slides", "📽️")
    
    slides = [
        "executive_summary",
        "financial_impact", 
        "benchmarking",
        "recommendations",
        "forecast"
    ]
    
    slide_titles = [
        "Executive Summary",
        "Financial Impact",
        "Competitive Benchmarking", 
        "Strategic Recommendations",
        "90-Day Forecast"
    ]
    
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button("⬅️ Previous"):
            st.session_state.current_slide = (st.session_state.current_slide - 1) % len(slides)
            st.rerun()
    
    with col2:
        st.markdown(f"<h3 style='text-align: center;'>Slide {st.session_state.current_slide + 1} of {len(slides)}: {slide_titles[st.session_state.current_slide]}</h3>", unsafe_allow_html=True)
    
    with col3:
        if st.button("Next ➡️"):
            st.session_state.current_slide = (st.session_state.current_slide + 1) % len(slides)
            st.rerun()
    
    auto_play = st.checkbox("Auto-advance slides (10 seconds)")
    
    st.markdown("---")
    
    current = slides[st.session_state.current_slide]
    
    if current == "executive_summary":
        # Condensed executive summary
        total_impact = df['Financial_Impact'].sum() if 'Financial_Impact' in df.columns else len(df) * get_benchmark_costs()['avg_per_ticket']
        savings = total_impact * 0.35
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Cost", f"${total_impact:,.0f}")
        with col2:
            st.metric("Savings Opportunity", f"${savings:,.0f}")
        with col3:
            st.metric("Escalations", f"{len(df):,}")
        
        st.plotly_chart(chart_pareto_analysis(df), use_container_width=True)
    
    elif current == "financial_impact":
        col1, col2 = st.columns(2)
        with col1:
            cost_by_cat = df.groupby('AI_Category')['Financial_Impact'].sum().sort_values(ascending=False)
            fig = go.Figure(go.Pie(
                labels=cost_by_cat.index,
                values=cost_by_cat.values,
                hole=0.5,
                marker=dict(colors=px.colors.sequential.Reds_r)
            ))
            fig.update_layout(**create_plotly_theme(), title='Cost Distribution', height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.plotly_chart(chart_driver_tree(df), use_container_width=True)
    
    elif current == "benchmarking":
        # Get safe values
        recurrence_rate = df['AI_Recurrence_Risk'].mean() if 'AI_Recurrence_Risk' in df.columns else 0.15
        resolution_days = df['Predicted_Resolution_Days'].mean() if 'Predicted_Resolution_Days' in df.columns else 5
        
        current_metrics = {
            'resolution_days': resolution_days,
            'recurrence_rate': recurrence_rate * 100,
            'cost_per_escalation': df['Financial_Impact'].mean() if 'Financial_Impact' in df.columns else get_benchmark_costs()['avg_per_ticket'],
        }
        
        col1, col2, col3 = st.columns(3)
        for i, (key, title) in enumerate([('resolution_days', 'Resolution Time'), ('recurrence_rate', 'Recurrence Rate'), ('cost_per_escalation', 'Cost/Escalation')]):
            unit = ' days' if 'days' in key else '%' if 'rate' in key else '$'
            with [col1, col2, col3][i]:
                fig = chart_benchmark_gauge(title, current_metrics[key], INDUSTRY_BENCHMARKS[key], unit)
                st.plotly_chart(fig, use_container_width=True)
    
    elif current == "recommendations":
        recommendations = generate_strategic_recommendations(df)
        for rec in recommendations[:3]:
            st.markdown(f"""
            <div class="strategy-card {'high-priority' if rec['priority'] == 'P1' else ''}">
                <span class="priority-{rec['priority'].lower()}">{rec['priority']}</span>
                <strong style="margin-left: 12px;">{rec['title']}</strong>
                <p style="color: #888;">{rec['description']}</p>
                <p><strong>ROI:</strong> {rec['roi']} | <strong>Timeline:</strong> {rec['timeline']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif current == "forecast":
        forecast_fig, slope = chart_forecast_projection(df)
        st.plotly_chart(forecast_fig, use_container_width=True)
        
        trend = "📈 Escalations trending UP" if slope > 0 else "📉 Escalations trending DOWN"
        color = "#DC3545" if slope > 0 else "#28A745"
        st.markdown(f"<h3 style='text-align: center; color: {color};'>{trend} ({abs(slope):.2f}/day)</h3>", unsafe_allow_html=True)
    
    if auto_play:
        time.sleep(10)
        st.session_state.current_slide = (st.session_state.current_slide + 1) % len(slides)
        st.rerun()


def render_whatif_simulator(df):
    """Render the What-If Scenario Simulator page."""
    render_spectacular_header("What-If Scenario Simulator", "Adjust parameters to simulate impact on escalation metrics", "🔮")
    
    # Get safe values with defaults
    recurrence_rate = df['AI_Recurrence_Risk'].mean() if 'AI_Recurrence_Risk' in df.columns else 0.15
    avg_resolution = df['Predicted_Resolution_Days'].mean() if 'Predicted_Resolution_Days' in df.columns else 5
    friction_sum = df['Strategic_Friction_Score'].sum() if 'Strategic_Friction_Score' in df.columns else 3000
    cost_sum = df['Financial_Impact'].sum() if 'Financial_Impact' in df.columns else 375000
    
    # Current baseline metrics
    baseline = {
        'avg_resolution': avg_resolution,
        'recurrence_rate': recurrence_rate * 100,
        'monthly_friction': friction_sum / 3,  # Assume 3 months data
        'monthly_cost': cost_sum / 3
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Scenario Parameters")
        
        # Staffing
        st.markdown("**👥 Staffing Changes**")
        engineer_change = st.slider("Add/Remove Engineers", -3, 5, 0, 
                                     help="Positive = add engineers, Negative = reduce")
        
        # Training
        st.markdown("**📚 Training Impact**")
        training_effect = st.slider("Error Reduction from Training", 0, 50, 0,
                                     help="Expected % reduction in human errors")
        
        # Volume
        st.markdown("**📈 Volume Changes**")
        volume_change = st.slider("Escalation Volume Change %", -30, 50, 0,
                                   help="Positive = more escalations expected")
        
        # Process
        st.markdown("**⚙️ Process Improvements**")
        process_improvement = st.slider("Process Efficiency Gain %", 0, 40, 0,
                                         help="Expected efficiency improvement")
    
    with col2:
        st.markdown("#### 📈 Projected Impact")
        
        # Calculate projections (simplified model)
        # Staffing effect: more engineers = faster resolution, less recurrence
        resolution_factor = 1 - (engineer_change * 0.08)  # Each engineer reduces time by 8%
        recurrence_factor = 1 - (engineer_change * 0.05)  # Each engineer reduces recurrence by 5%
        
        # Training effect
        recurrence_factor *= (1 - training_effect / 100 * 0.5)  # Training reduces recurrence
        resolution_factor *= (1 - training_effect / 100 * 0.2)  # Training speeds resolution
        
        # Volume effect
        cost_factor = 1 + (volume_change / 100)
        friction_factor = 1 + (volume_change / 100 * 0.8)
        
        # Process effect
        resolution_factor *= (1 - process_improvement / 100)
        friction_factor *= (1 - process_improvement / 100 * 0.7)
        
        # Calculate projected values
        projected = {
            'avg_resolution': baseline['avg_resolution'] * max(0.3, resolution_factor),
            'recurrence_rate': baseline['recurrence_rate'] * max(0.2, recurrence_factor),
            'monthly_friction': baseline['monthly_friction'] * max(0.3, friction_factor),
            'monthly_cost': baseline['monthly_cost'] * max(0.3, cost_factor * resolution_factor)
        }
        
        # Display comparison
        metrics = [
            ('Resolution Time', 'avg_resolution', 'days', True),
            ('Recurrence Rate', 'recurrence_rate', '%', True),
            ('Monthly Friction', 'monthly_friction', 'pts', True),
            ('Monthly Cost', 'monthly_cost', '$', True)
        ]
        
        for label, key, unit, lower_better in metrics:
            base_val = baseline[key]
            proj_val = projected[key]
            delta = ((proj_val - base_val) / base_val) * 100
            
            if unit == '$':
                base_str = f"${base_val:,.0f}"
                proj_str = f"${proj_val:,.0f}"
            elif unit == 'days':
                base_str = f"{base_val:.1f} {unit}"
                proj_str = f"{proj_val:.1f} {unit}"
            else:
                base_str = f"{base_val:.1f}{unit}"
                proj_str = f"{proj_val:.1f}{unit}"
            
            is_improvement = (delta < 0) if lower_better else (delta > 0)
            delta_color = "delta-down" if is_improvement else "delta-up"
            arrow = "↓" if delta < 0 else "↑"
            
            st.markdown(f"""
            <div class="kpi-container {'success' if is_improvement else 'warning'}">
                <div style="font-size: 0.8rem; color: #888;">{label}</div>
                <div style="font-size: 1.5rem; font-weight: 600;">{base_str} → {proj_str}</div>
                <div class="{delta_color}">{arrow} {abs(delta):.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    # ROI Calculation
    st.markdown("---")
    st.markdown("#### 💰 Return on Investment")
    
    # Costs of changes (derived from price_catalog hourly rate)
    hourly_rate = get_benchmark_costs()['hourly_rate']
    annual_salary = hourly_rate * 2000  # 2000 work hours per year
    engineer_cost = max(0, engineer_change) * annual_salary
    training_cost_per_engineer = hourly_rate * 25  # ~25 hours of training per level
    training_cost = training_effect * training_cost_per_engineer * len(df['Engineer'].unique()) if 'Engineer' in df.columns else training_effect * (training_cost_per_engineer * 10)
    process_cost = process_improvement * (hourly_rate * 100)  # ~100 hours of process work per level
    
    total_investment = engineer_cost + training_cost + process_cost
    
    # Savings (annualized)
    monthly_savings = baseline['monthly_cost'] - projected['monthly_cost']
    annual_savings = monthly_savings * 12
    
    if total_investment > 0:
        roi = (annual_savings / total_investment) * 100
        payback_months = total_investment / monthly_savings if monthly_savings > 0 else float('inf')
    else:
        roi = float('inf') if annual_savings > 0 else 0
        payback_months = 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Investment", f"${total_investment:,.0f}")
    with col2:
        st.metric("Annual Savings", f"${annual_savings:,.0f}", 
                  delta=f"{annual_savings/baseline['monthly_cost']/12*100:.0f}% of current cost")
    with col3:
        if payback_months < 24:
            st.metric("Payback Period", f"{payback_months:.1f} months")
        else:
            st.metric("ROI", f"{roi:.0f}%" if roi != float('inf') else "∞")

# ============================================================================
# DRIFT DETECTION
# ============================================================================

def render_drift_page(df):
    """Render the Category Drift Detection page."""
    try:
        render_spectacular_header("Category Drift Detection", "Analyze how escalation patterns are changing over time", "📊")

        # Debug: Show available columns
        if st.checkbox("Show debug info", value=False):
            st.write("Available columns:", df.columns.tolist())
            st.write("DataFrame shape:", df.shape)

        # Find date column
        date_col = None
        for col in ['Issue_Date', 'Issue Date', 'tickets_data_issue_datetime', 'Created_Date', 'Date', 'Timestamp']:
            if col in df.columns:
                date_col = col
                break

        if not date_col:
            st.warning("""
            **No date information available for drift detection.**

            To enable drift analysis:
            1. Regenerate the report with the latest pipeline (`python run.py`)
            2. Ensure your input data has an 'Issue Date' or 'Created Date' column

            **Current dataset does not contain date information.**
            """)

            # Show what we DO have
            st.info(f"Dataset contains {len(df)} records with {len(df.columns)} columns.")
            return

        # Check for AI_Category column
        if 'AI_Category' not in df.columns:
            st.error("Missing 'AI_Category' column required for drift detection.")
            st.info("Available columns: " + ", ".join(df.columns.tolist()))
            return

        # Split data into baseline and recent
        df_temp = df.copy()
        df_temp['date'] = pd.to_datetime(df_temp[date_col], errors='coerce')
        df_temp = df_temp.dropna(subset=['date'])
        df_temp = df_temp.sort_values('date')

        if len(df_temp) < 10:
            st.warning(f"Not enough data points for drift detection. Found {len(df_temp)} tickets with dates (need at least 10).")
            return

        st.success(f"✓ Found {len(df_temp)} tickets with valid dates for drift analysis")

        split_idx = int(len(df_temp) * 0.6)
        baseline_df = df_temp.iloc[:split_idx]
        current_df = df_temp.iloc[split_idx:]

        # Calculate distributions
        baseline_dist = baseline_df['AI_Category'].value_counts(normalize=True)
        current_dist = current_df['AI_Category'].value_counts(normalize=True)

    except Exception as e:
        st.error(f"""
        **Error in Drift Detection:**

        {str(e)}

        Please try:
        1. Regenerating the report with `python run.py`
        2. Ensuring your input data has proper date columns
        """)

        import traceback
        if st.checkbox("Show technical details"):
            st.code(traceback.format_exc())
        return

    # Create comparison chart
    all_cats = sorted(set(baseline_dist.index) | set(current_dist.index))
    
    comparison_data = pd.DataFrame({
        'Category': all_cats,
        'Baseline': [baseline_dist.get(c, 0) * 100 for c in all_cats],
        'Current': [current_dist.get(c, 0) * 100 for c in all_cats]
    })
    comparison_data['Change'] = comparison_data['Current'] - comparison_data['Baseline']
    comparison_data = comparison_data.sort_values('Change')
    
    # Grouped bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Baseline (First 60%)',
        x=comparison_data['Category'],
        y=comparison_data['Baseline'],
        marker_color='rgba(100, 149, 237, 0.7)'
    ))
    
    fig.add_trace(go.Bar(
        name='Current (Last 40%)',
        x=comparison_data['Category'],
        y=comparison_data['Current'],
        marker_color='rgba(255, 107, 107, 0.7)'
    ))
    
    fig.update_layout(
        **create_plotly_theme(),
        title='Category Distribution: Baseline vs Current',
        barmode='group',
        xaxis_tickangle=-45,
        height=400,
        legend=dict(orientation='h', y=1.1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Change analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Emerging Categories")
        emerging = comparison_data[comparison_data['Change'] > 2].sort_values('Change', ascending=False)
        for _, row in emerging.iterrows():
            st.markdown(f"""
            <div class="kpi-container warning">
                <b>{row['Category']}</b><br>
                <span class="delta-up">↑ {row['Change']:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        if emerging.empty:
            st.info("No significantly emerging categories detected.")
    
    with col2:
        st.markdown("#### 📉 Declining Categories")
        declining = comparison_data[comparison_data['Change'] < -2].sort_values('Change')
        for _, row in declining.iterrows():
            st.markdown(f"""
            <div class="kpi-container success">
                <b>{row['Category']}</b><br>
                <span class="delta-down">↓ {abs(row['Change']):.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        if declining.empty:
            st.info("No significantly declining categories detected.")

# ============================================================================
# ALERTS PAGE
# ============================================================================

def render_alerts_page(df):
    """Render the Smart Alerts page."""
    render_spectacular_header("Smart Alert Thresholds", "Real-time monitoring of key metrics against dynamic thresholds", "⚠️")

    # Find date column
    date_col = None
    for col in ['Issue_Date', 'Issue Date', 'tickets_data_issue_datetime', 'Created_Date', 'Date', 'Timestamp']:
        if col in df.columns:
            date_col = col
            break

    if not date_col:
        st.warning("No date information available. Showing current state alerts only.")
        date_col = None

    # Calculate current metrics
    df_temp = df.copy()
    if date_col:
        try:
            df_temp['date'] = pd.to_datetime(df_temp[date_col], errors='coerce').dt.date
        except:
            date_col = None
    
    # Build agg dict with available columns
    agg_dict = {'AI_Category': 'count'}
    if 'Strategic_Friction_Score' in df.columns:
        agg_dict['Strategic_Friction_Score'] = 'sum'
    if 'AI_Recurrence_Risk' in df.columns:
        agg_dict['AI_Recurrence_Risk'] = 'mean'
    
    daily = df_temp.groupby('date').agg(agg_dict).rename(columns={'AI_Category': 'count'})
    
    # Ensure Strategic_Friction_Score exists
    if 'Strategic_Friction_Score' not in daily.columns:
        daily['Strategic_Friction_Score'] = 50
    
    # Calculate thresholds (simplified)
    metrics_config = {
        'Daily Escalations': {
            'values': daily['count'],
            'current': daily['count'].iloc[-1] if len(daily) > 0 else 0,
            'warning': daily['count'].quantile(0.75),
            'critical': daily['count'].quantile(0.90)
        },
        'Daily Friction': {
            'values': daily['Strategic_Friction_Score'],
            'current': daily['Strategic_Friction_Score'].iloc[-1] if len(daily) > 0 else 0,
            'warning': daily['Strategic_Friction_Score'].quantile(0.75),
            'critical': daily['Strategic_Friction_Score'].quantile(0.90)
        },
        'Recurrence Risk': {
            'values': daily['AI_Recurrence_Risk'] * 100,
            'current': daily['AI_Recurrence_Risk'].iloc[-1] * 100 if len(daily) > 0 else 0,
            'warning': 25,
            'critical': 40
        }
    }
    
    cols = st.columns(3)
    
    for i, (metric_name, config) in enumerate(metrics_config.items()):
        with cols[i]:
            current = config['current']
            warning = config['warning']
            critical = config['critical']
            
            if current >= critical:
                status = 'critical'
                badge_class = 'badge-critical'
                status_text = 'CRITICAL'
            elif current >= warning:
                status = 'warning'
                badge_class = 'badge-warning'
                status_text = 'WARNING'
            else:
                status = 'success'
                badge_class = 'badge-success'
                status_text = 'NORMAL'
            
            st.markdown(f"""
            <div class="kpi-container {status}">
                <span class="badge {badge_class}">{status_text}</span>
                <h3 style="margin: 10px 0; font-size: 2rem;">{current:.1f}</h3>
                <p style="color: #888; margin: 0;">{metric_name}</p>
                <small>Warning: {warning:.1f} | Critical: {critical:.1f}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Threshold timeline chart
    st.markdown("---")
    st.markdown("#### 📈 Metric Timeline with Thresholds")
    
    selected_metric = st.selectbox("Select Metric", list(metrics_config.keys()))
    config = metrics_config[selected_metric]
    
    fig = go.Figure()
    
    dates = daily.index
    values = config['values']
    
    # Add threshold zones
    fig.add_hrect(y0=0, y1=config['warning'], fillcolor="rgba(40,167,69,0.1)", line_width=0)
    fig.add_hrect(y0=config['warning'], y1=config['critical'], fillcolor="rgba(255,193,7,0.1)", line_width=0)
    fig.add_hrect(y0=config['critical'], y1=values.max()*1.2, fillcolor="rgba(220,53,69,0.1)", line_width=0)
    
    # Add threshold lines
    fig.add_hline(y=config['warning'], line_dash="dash", line_color="#FFC107", 
                  annotation_text="Warning")
    fig.add_hline(y=config['critical'], line_dash="dash", line_color="#DC3545",
                  annotation_text="Critical")
    
    # Add values
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode='lines+markers',
        line=dict(color='#0066CC', width=2),
        marker=dict(size=6),
        hovertemplate='%{x}<br>Value: %{y:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        **create_plotly_theme(),
        height=400,
        showlegend=False,
        xaxis_title='Date',
        yaxis_title=selected_metric
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PDF EXPORT FUNCTIONALITY
# ============================================================================

def generate_executive_pdf_report(df):
    """Generate comprehensive PDF reference guide with scoring methodology, assumptions, and usage guide."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.graphics.shapes import Drawing
        from reportlab.graphics.charts.barcharts import VerticalBarChart
        from reportlab.graphics.charts.piecharts import Pie

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=0.5*inch, leftMargin=0.5*inch,
                               topMargin=0.4*inch, bottomMargin=0.4*inch)
        styles = getSampleStyleSheet()

        # Compact styles
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=22, spaceAfter=6,
                                     alignment=TA_CENTER, textColor=colors.HexColor('#0066CC'))
        heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14, spaceBefore=8,
                                       spaceAfter=4, textColor=colors.HexColor('#0066CC'))
        subheading_style = ParagraphStyle('SubHeading', parent=styles['Heading3'], fontSize=11, spaceBefore=6,
                                          spaceAfter=3, textColor=colors.HexColor('#333333'))
        body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=9, spaceAfter=3, leading=11)
        formula_style = ParagraphStyle('Formula', parent=styles['Normal'], fontSize=8, spaceAfter=2, leading=10,
                                       leftIndent=15, textColor=colors.HexColor('#0066CC'), fontName='Courier')
        note_style = ParagraphStyle('Note', parent=styles['Normal'], fontSize=8, spaceAfter=2, leading=10,
                                    textColor=colors.HexColor('#666666'), fontName='Helvetica-Oblique')
        bullet_style = ParagraphStyle('Bullet', parent=styles['Normal'], fontSize=9, spaceAfter=2, leading=11, leftIndent=15)

        chart_colors = [colors.HexColor('#0066CC'), colors.HexColor('#28A745'), colors.HexColor('#DC3545'),
                        colors.HexColor('#FFC107'), colors.HexColor('#17A2B8'), colors.HexColor('#6C757D')]

        story = []

        # ===== COVER PAGE =====
        story.append(Spacer(1, 0.8*inch))
        story.append(Paragraph("ESCALATION AI", title_style))
        story.append(Paragraph("Reference Guide & Scoring Methodology",
                              ParagraphStyle('Sub', parent=body_style, fontSize=12, alignment=TA_CENTER, textColor=colors.HexColor('#666666'))))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(f"Version 2.2.0 | {datetime.now().strftime('%B %d, %Y')}",
                              ParagraphStyle('Date', parent=body_style, fontSize=10, alignment=TA_CENTER)))
        story.append(Spacer(1, 0.3*inch))

        # Table of Contents
        story.append(Paragraph("<b>Contents</b>", subheading_style))
        toc = ["1. Quick Start Guide", "2. Scoring Methodology", "3. Financial Analysis Assumptions",
               "4. Data Overview & Charts", "5. Key Metrics Reference", "6. Glossary"]
        for item in toc:
            story.append(Paragraph(f"  {item}", body_style))
        story.append(PageBreak())

        # ===== 1. QUICK START GUIDE =====
        story.append(Paragraph("1. Quick Start Guide", heading_style))
        story.append(Paragraph("How to use the Escalation AI Dashboard effectively:", body_style))

        guide_data = [
            ['Tab', 'Purpose', 'Key Actions'],
            ['Overview', 'High-level KPIs and trends', 'Monitor daily metrics, identify spikes'],
            ['Analysis', 'Deep dive into categories', 'Filter by category, severity, time'],
            ['Similarity', 'Find related tickets', 'Identify patterns, recurring issues'],
            ['Financial', 'Cost impact analysis', 'Track costs, ROI calculations'],
            ['Lessons', 'Knowledge management', 'Review lessons, track effectiveness'],
            ['Planning', 'Action recommendations', 'Prioritize actions, assign owners'],
            ['Reports', 'Export & documentation', 'Generate reports, download data'],
        ]
        guide_table = Table(guide_data, colWidths=[1*inch, 2.2*inch, 2.5*inch])
        guide_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066CC')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ]))
        story.append(guide_table)

        story.append(Paragraph("<b>Pro Tips:</b>", subheading_style))
        tips = [
            "• Use filters to narrow down to specific time periods or categories",
            "• Export data regularly for trend analysis over longer periods",
            "• Review Lessons Learned tab weekly to track learning effectiveness",
            "• Use Planning tab to create actionable improvement roadmaps",
        ]
        for tip in tips:
            story.append(Paragraph(tip, bullet_style))

        # ===== 2. SCORING METHODOLOGY =====
        story.append(Paragraph("2. Scoring Methodology", heading_style))

        # 2.1 Strategic Friction Score
        story.append(Paragraph("2.1 Strategic Friction Score (0-200)", subheading_style))
        story.append(Paragraph("Measures the operational impact and urgency of each escalation.", body_style))
        story.append(Paragraph("Formula: SFS = (Severity × 40) + (Impact × 30) + (Duration × 20) + (Recurrence × 10)", formula_style))

        friction_data = [
            ['Component', 'Weight', 'Calculation', 'Range'],
            ['Severity Score', '40%', 'Critical=40, High=30, Medium=20, Low=10', '10-40'],
            ['Business Impact', '30%', 'Based on customer tier and revenue impact', '0-30'],
            ['Duration Factor', '20%', 'Days open × 2 (capped at 20)', '0-20'],
            ['Recurrence Risk', '10%', 'ML-predicted probability × 10', '0-10'],
        ]
        friction_table = Table(friction_data, colWidths=[1.3*inch, 0.7*inch, 2.5*inch, 0.8*inch])
        friction_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#DC3545')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(friction_table)

        # 2.2 Learning Effectiveness Score
        story.append(Paragraph("2.2 Learning Effectiveness Score (0-100)", subheading_style))
        story.append(Paragraph("Evaluates how well lessons learned are being applied to prevent recurrence.", body_style))
        story.append(Paragraph("Formula: LES = (Recurrence × 35) + (Completion × 30) + (Consistency × 25) + (Doc × 10)", formula_style))

        learning_data = [
            ['Component', 'Weight', 'Description', 'Points'],
            ['Recurrence Prevention', '35%', 'Reduction in similar issues after lesson', '0-35'],
            ['Lesson Completion', '30%', 'Actions from lessons fully implemented', '0-30'],
            ['Resolution Consistency', '25%', 'Similar issues resolved consistently', '0-25'],
            ['Documentation Bonus', '10%', 'Well-documented with clear steps', '0-10'],
        ]
        learning_table = Table(learning_data, colWidths=[1.5*inch, 0.7*inch, 2.3*inch, 0.8*inch])
        learning_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#28A745')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(learning_table)

        # 2.3 Similarity Score
        story.append(Paragraph("2.3 Similarity Score (0-100%)", subheading_style))
        story.append(Paragraph("Identifies related tickets using NLP and pattern matching.", body_style))
        story.append(Paragraph("Method: Cosine similarity on TF-IDF vectors of ticket descriptions", formula_style))

        sim_data = [
            ['Score Range', 'Interpretation', 'Action'],
            ['90-100%', 'Near duplicate', 'Merge or reference existing solution'],
            ['70-89%', 'Highly similar', 'Apply similar resolution approach'],
            ['50-69%', 'Related pattern', 'Review for common root cause'],
            ['Below 50%', 'Unique issue', 'Investigate independently'],
        ]
        sim_table = Table(sim_data, colWidths=[1.2*inch, 1.8*inch, 2.3*inch])
        sim_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#17A2B8')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(sim_table)

        # 2.4 AI Recurrence Risk
        story.append(Paragraph("2.4 AI Recurrence Risk (0-100%)", subheading_style))
        story.append(Paragraph("ML model predicting likelihood of issue recurring within 30 days.", body_style))
        story.append(Paragraph("Model: Random Forest Classifier | Features: Category, Severity, Resolution, History", formula_style))
        story.append(Paragraph("Note: Confidence intervals based on historical accuracy of 87%.", note_style))

        # ===== 3. FINANCIAL ANALYSIS (from price_catalog.xlsx) =====
        story.append(Paragraph("3. Financial Analysis (from price_catalog.xlsx)", heading_style))

        # Load actual pricing data from price_catalog.xlsx
        import os
        price_catalog_path = None
        for path in ['/home/k8s/Projects/AI-Escalation/price_catalog.xlsx', 'price_catalog.xlsx',
                     os.path.join(os.path.dirname(__file__), '..', '..', 'price_catalog.xlsx')]:
            if os.path.exists(path):
                price_catalog_path = path
                break

        if price_catalog_path:
            try:
                import pandas as pd
                xl = pd.ExcelFile(price_catalog_path)

                # 3.1 Category Costs
                story.append(Paragraph("3.1 Category Base Costs", subheading_style))
                cat_costs = pd.read_excel(xl, sheet_name='Category Costs')
                cost_data = [['Category', 'Labor Hrs', 'Rate/Hr', 'Delay Cost', 'Notes']]
                for _, row in cat_costs.iterrows():
                    cost_data.append([
                        str(row['Category'])[:22],
                        f"{row['Labor_Hours']:.1f}",
                        f"${row['Hourly_Rate']:.0f}",
                        f"${row['Delay_Cost_Per_Hour']:.0f}",
                        str(row['Notes'])[:30] if pd.notna(row['Notes']) else ''
                    ])
                cost_table = Table(cost_data, colWidths=[1.8*inch, 0.7*inch, 0.6*inch, 0.7*inch, 1.9*inch])
                cost_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#28A745')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 7),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
                    ('TOPPADDING', (0, 0), (-1, -1), 2),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                ]))
                story.append(cost_table)

                # 3.2 Severity Multipliers
                story.append(Paragraph("3.2 Severity Multipliers", subheading_style))
                sev_mult = pd.read_excel(xl, sheet_name='Severity Multipliers')
                sev_data = [['Severity', 'Multiplier', 'Description']]
                for _, row in sev_mult.iterrows():
                    sev_data.append([row['Severity_Level'], f"{row['Cost_Multiplier']:.2f}x", row['Description']])
                sev_table = Table(sev_data, colWidths=[1.2*inch, 1*inch, 3.5*inch])
                sev_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#DC3545')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
                    ('TOPPADDING', (0, 0), (-1, -1), 2),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                ]))
                story.append(sev_table)

                # 3.3 Origin Premiums
                story.append(Paragraph("3.3 Origin Premiums", subheading_style))
                origin_prem = pd.read_excel(xl, sheet_name='Origin Premiums')
                origin_data = [['Origin Type', 'Premium %', 'Description']]
                for _, row in origin_prem.iterrows():
                    origin_data.append([row['Origin_Type'], f"{row['Premium_Percentage']*100:.0f}%", row['Description'][:40]])
                origin_table = Table(origin_data, colWidths=[1.2*inch, 0.9*inch, 3.6*inch])
                origin_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FFC107')),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
                    ('TOPPADDING', (0, 0), (-1, -1), 2),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                ]))
                story.append(origin_table)

                # 3.4 Cost Formula
                story.append(Paragraph("3.4 Cost Calculation Formula", subheading_style))
                story.append(Paragraph("Total_Impact = (Material + Labor × Rate + Delay) × Severity_Mult × (1 + Origin_Premium)", formula_style))

            except Exception as e:
                story.append(Paragraph(f"Note: Could not load price_catalog.xlsx: {str(e)}", note_style))
        else:
            story.append(Paragraph("Note: price_catalog.xlsx not found. Using default assumptions.", note_style))

        # ===== 4. DATA OVERVIEW =====
        story.append(Paragraph("4. Data Overview", heading_style))

        # Calculate metrics
        total_cost = df['Financial_Impact'].sum() if 'Financial_Impact' in df.columns else len(df) * get_benchmark_costs()['avg_per_ticket']
        avg_resolution = df['Predicted_Resolution_Days'].mean() if 'Predicted_Resolution_Days' in df.columns else 3.0
        recurrence_rate = df['AI_Recurrence_Risk'].mean() * 100 if 'AI_Recurrence_Risk' in df.columns else 20
        type_col = 'tickets_data_type'
        escalations_count = len(df[df[type_col].astype(str).str.contains('Escalation', case=False, na=False)]) if type_col in df.columns else len(df)
        concerns_count = len(df[df[type_col].astype(str).str.contains('Concern', case=False, na=False)]) if type_col in df.columns else 0
        lessons_count = len(df[df[type_col].astype(str).str.contains('Lesson', case=False, na=False)]) if type_col in df.columns else 0

        # Summary metrics
        story.append(Paragraph(f"<b>Total Records:</b> {len(df):,} | <b>Total Cost:</b> ${total_cost:,.0f} | "
                              f"<b>Avg Resolution:</b> {avg_resolution:.1f} days | <b>Recurrence:</b> {recurrence_rate:.1f}%", body_style))

        # Pie chart - Record Types
        if escalations_count + concerns_count + lessons_count > 0:
            story.append(Paragraph("Record Type Distribution", subheading_style))
            type_drawing = Drawing(400, 140)
            pie = Pie()
            pie.x, pie.y, pie.width, pie.height = 80, 10, 120, 120
            pie.data = [escalations_count, concerns_count, lessons_count]
            pie.labels = [f'Escalations ({escalations_count})', f'Concerns ({concerns_count})', f'Lessons ({lessons_count})']
            pie.slices.strokeWidth = 0.5
            for i, c in enumerate(chart_colors[:3]):
                pie.slices[i].fillColor = c
            type_drawing.add(pie)
            story.append(type_drawing)

        # Category friction bar chart
        if 'AI_Category' in df.columns and 'Strategic_Friction_Score' in df.columns:
            story.append(Paragraph("Top Categories by Friction Score", subheading_style))
            category_friction = df.groupby('AI_Category')['Strategic_Friction_Score'].sum().sort_values(ascending=False).head(5)
            bar_drawing = Drawing(450, 130)
            bc = VerticalBarChart()
            bc.x, bc.y, bc.height, bc.width = 50, 25, 90, 360
            bc.data = [list(category_friction.values)]
            bc.categoryAxis.categoryNames = [c[:12] + '..' if len(c) > 12 else c for c in category_friction.index]
            bc.categoryAxis.labels.fontSize = 7
            bc.categoryAxis.labels.angle = 20
            bc.categoryAxis.labels.boxAnchor = 'ne'
            bc.valueAxis.labels.fontSize = 7
            bc.bars[0].fillColor = colors.HexColor('#0066CC')
            bar_drawing.add(bc)
            story.append(bar_drawing)

        story.append(PageBreak())

        # ===== 5. KEY METRICS REFERENCE =====
        story.append(Paragraph("5. Key Metrics Reference", heading_style))

        # Get cost thresholds from price_catalog
        benchmarks = get_benchmark_costs()
        cost_good = f"< ${benchmarks['best_in_class']:,.0f}"
        cost_warn = f"${benchmarks['best_in_class']:,.0f}-{benchmarks['industry_avg']:,.0f}"
        cost_crit = f"> ${benchmarks['industry_avg']:,.0f}"

        metrics_data = [
            ['Metric', 'Definition', 'Good', 'Warning', 'Critical'],
            ['Resolution Time', 'Days to resolve', '< 2 days', '2-5 days', '> 5 days'],
            ['Recurrence Rate', '30-day repeat %', '< 10%', '10-20%', '> 20%'],
            ['Friction Score', 'Impact severity', '< 50', '50-100', '> 100'],
            ['Learning Score', 'Lesson effectiveness', '> 80', '50-80', '< 50'],
            ['Similarity Score', 'Pattern match %', 'Review > 70%', '—', '—'],
            ['Cost/Ticket', 'Average cost', cost_good, cost_warn, cost_crit],
        ]
        metrics_table = Table(metrics_data, colWidths=[1.2*inch, 1.6*inch, 0.9*inch, 0.9*inch, 0.9*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066CC')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('BACKGROUND', (2, 1), (2, -1), colors.HexColor('#D4EDDA')),
            ('BACKGROUND', (3, 1), (3, -1), colors.HexColor('#FFF3CD')),
            ('BACKGROUND', (4, 1), (4, -1), colors.HexColor('#F8D7DA')),
        ]))
        story.append(metrics_table)

        # ===== 6. GLOSSARY =====
        story.append(Paragraph("6. Glossary", heading_style))

        glossary_data = [
            ['Term', 'Definition'],
            ['Escalation', 'Issue requiring elevated attention or management involvement'],
            ['Concern', 'Potential issue flagged for monitoring before becoming critical'],
            ['Lesson Learned', 'Documented insight from resolved issues to prevent recurrence'],
            ['Friction Score', 'Composite metric measuring operational drag and urgency'],
            ['Recurrence Risk', 'ML-predicted probability of same/similar issue within 30 days'],
            ['Revenue at Risk', 'Estimated revenue impact based on churn probability'],
            ['Pareto Analysis', '80/20 rule: Focus on 20% of causes creating 80% of problems'],
            ['NLP Categorization', 'AI-based text analysis for automatic ticket classification'],
        ]
        glossary_table = Table(glossary_data, colWidths=[1.5*inch, 4.2*inch])
        glossary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6C757D')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(glossary_table)

        # Footer
        story.append(Spacer(1, 0.15*inch))
        story.append(Paragraph(f"Escalation AI v2.2.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')} | Confidential",
                              ParagraphStyle('Footer', parent=body_style, alignment=TA_CENTER, fontSize=8,
                                           textColor=colors.HexColor('#888888'))))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    except ImportError:
        return None


def generate_html_report(df):
    """Generate an HTML report that can be converted to PDF."""
    total_cost = df['Financial_Impact'].sum() if 'Financial_Impact' in df.columns else len(df) * get_benchmark_costs()['avg_per_ticket']
    revenue_risk = df['Revenue_At_Risk'].sum() if 'Revenue_At_Risk' in df.columns else total_cost * 0.20
    avg_resolution = df['Predicted_Resolution_Days'].mean()
    recurrence_rate = df['AI_Recurrence_Risk'].mean() * 100
    # Check for critical/high severity using multiple column names and values
    critical_count = 0
    for sev_col in ['tickets_data_severity', 'Severity_Norm', 'Severity', 'severity']:
        if sev_col in df.columns:
            # Count Critical, High, or Major as critical issues
            critical_count = len(df[df[sev_col].astype(str).str.lower().isin(['critical', 'high', 'major'])])
            break

    category_friction = df.groupby('AI_Category')['Strategic_Friction_Score'].sum().sort_values(ascending=False)
    
    recommendations = generate_strategic_recommendations(df)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Escalation AI Executive Report</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: 'Inter', sans-serif; color: #333; line-height: 1.6; padding: 40px; max-width: 1000px; margin: 0 auto; }}
            
            .header {{ text-align: center; margin-bottom: 40px; padding: 40px; background: linear-gradient(135deg, #0066CC 0%, #004080 100%); color: white; border-radius: 12px; }}
            .header h1 {{ font-size: 2.5rem; margin-bottom: 10px; }}
            .header p {{ opacity: 0.9; }}
            
            .section {{ margin: 30px 0; }}
            .section h2 {{ color: #0066CC; border-bottom: 2px solid #0066CC; padding-bottom: 10px; margin-bottom: 20px; }}
            .section h3 {{ color: #333; margin: 20px 0 10px 0; }}
            
            .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
            .kpi-card {{ background: #f8f9fa; border-radius: 12px; padding: 24px; text-align: center; border-left: 4px solid #0066CC; }}
            .kpi-card.alert {{ border-left-color: #DC3545; }}
            .kpi-card.success {{ border-left-color: #28A745; }}
            .kpi-value {{ font-size: 2rem; font-weight: 700; color: #0066CC; }}
            .kpi-value.money {{ color: #28A745; }}
            .kpi-value.alert {{ color: #DC3545; }}
            .kpi-label {{ font-size: 0.85rem; color: #666; text-transform: uppercase; letter-spacing: 1px; margin-top: 8px; }}
            
            .rec-card {{ background: #f0f7ff; border-left: 4px solid #0066CC; padding: 20px; margin: 15px 0; border-radius: 0 8px 8px 0; }}
            .rec-card.p1 {{ border-left-color: #DC3545; background: #fff5f5; }}
            .rec-card.p2 {{ border-left-color: #FFC107; background: #fffbf0; }}
            .rec-card h4 {{ color: #333; margin-bottom: 8px; }}
            .rec-card p {{ color: #666; margin-bottom: 10px; }}
            .rec-meta {{ display: flex; gap: 20px; font-size: 0.9rem; }}
            .rec-meta span {{ color: #0066CC; font-weight: 500; }}
            
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th {{ background: #0066CC; color: white; padding: 12px; text-align: left; }}
            td {{ padding: 12px; border-bottom: 1px solid #eee; }}
            tr:hover {{ background: #f8f9fa; }}
            
            .priority-badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }}
            .priority-badge.p1 {{ background: #DC3545; color: white; }}
            .priority-badge.p2 {{ background: #FFC107; color: #333; }}
            .priority-badge.p3 {{ background: #0066CC; color: white; }}
            
            .footer {{ text-align: center; margin-top: 40px; padding: 20px; color: #888; font-size: 0.85rem; border-top: 1px solid #eee; }}
            
            @media print {{
                body {{ padding: 20px; }}
                .header {{ break-after: page; }}
                .section {{ break-inside: avoid; }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 ESCALATION AI</h1>
            <p>Executive Intelligence Report</p>
            <p>Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
            <p>Analysis Period: 90 Days | {len(df):,} Records Analyzed</p>
        </div>
        
        <div class="section">
            <h2>1. Executive Summary</h2>
            
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-value">{len(df):,}</div>
                    <div class="kpi-label">Total Records</div>
                </div>
                <div class="kpi-card alert">
                    <div class="kpi-value alert">{critical_count}</div>
                    <div class="kpi-label">Critical Issues</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value money">${total_cost:,.0f}</div>
                    <div class="kpi-label">Total Cost</div>
                </div>
                <div class="kpi-card success">
                    <div class="kpi-value money">${total_cost * 0.35:,.0f}</div>
                    <div class="kpi-label">Savings Opportunity</div>
                </div>
            </div>
            
            <table>
                <tr><th>Category</th><th>Count</th><th>Description</th></tr>
                <tr><td>📋 Escalations</td><td>{len(df[df['tickets_data_type'].astype(str).str.contains('Escalation', case=False, na=False)]) if 'tickets_data_type' in df.columns else len(df)}</td><td>Active escalation tickets</td></tr>
                <tr><td>⚠️ Concerns</td><td>{len(df[df['tickets_data_type'].astype(str).str.contains('Concern', case=False, na=False)]) if 'tickets_data_type' in df.columns else 0}</td><td>Potential issues flagged</td></tr>
                <tr><td>📚 Lessons Learned</td><td>{len(df[df['tickets_data_type'].astype(str).str.contains('Lesson', case=False, na=False)]) if 'tickets_data_type' in df.columns else 0}</td><td>Historical learnings</td></tr>
            </table>
            
            <table>
                <tr><th>Metric</th><th>Current</th><th>Benchmark</th><th>Status</th></tr>
                <tr><td>Avg Resolution Time</td><td>{avg_resolution:.1f} days</td><td>2.8 days</td><td>{'⚠️ Above' if avg_resolution > 2.8 else '✅ Below'}</td></tr>
                <tr><td>Recurrence Rate</td><td>{recurrence_rate:.1f}%</td><td>18%</td><td>{'⚠️ Above' if recurrence_rate > 18 else '✅ Below'}</td></tr>
                <tr><td>Revenue at Risk</td><td>${revenue_risk:,.0f}</td><td>—</td><td>—</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>2. Strategic Recommendations</h2>
            {''.join([f'''
            <div class="rec-card {rec['priority'].lower()}">
                <h4><span class="priority-badge {rec['priority'].lower()}">{rec['priority']}</span> {rec['title']}</h4>
                <p>{rec['description']}</p>
                <div class="rec-meta">
                    <span>Impact: {rec['impact']}</span>
                    <span>Timeline: {rec['timeline']}</span>
                    <span>Investment: {rec['investment']}</span>
                    <span>ROI: {rec['roi']}</span>
                </div>
            </div>
            ''' for rec in recommendations[:4]])}
        </div>
        
        <div class="section">
            <h2>3. Category Analysis (Pareto)</h2>
            <table>
                <tr><th>Rank</th><th>Category</th><th>Friction Score</th><th>% of Total</th></tr>
                {''.join([f"<tr><td>{i+1}</td><td>{cat}</td><td>{friction:,.0f}</td><td>{friction/category_friction.sum()*100:.1f}%</td></tr>" for i, (cat, friction) in enumerate(category_friction.items())])}
            </table>
        </div>
        
        <div class="footer">
            <p>Generated by Escalation AI v2.2.0</p>
            <p>CONFIDENTIAL - FOR EXECUTIVE REVIEW ONLY</p>
        </div>
    </body>
    </html>
    """
    
    return html


def generate_magnificent_html_report(df):
    """Generate a magnificent HTML report with interactive Plotly charts."""
    import plotly.io as pio

    # Calculate metrics
    total_cost = df['Financial_Impact'].sum() if 'Financial_Impact' in df.columns else len(df) * get_benchmark_costs()['avg_per_ticket']
    revenue_risk = df['Revenue_At_Risk'].sum() if 'Revenue_At_Risk' in df.columns else total_cost * 0.20
    avg_resolution = df['Predicted_Resolution_Days'].mean() if 'Predicted_Resolution_Days' in df.columns else 5
    recurrence_rate = df['AI_Recurrence_Risk'].mean() * 100 if 'AI_Recurrence_Risk' in df.columns else 15
    # Check for critical/high severity using multiple column names and values
    critical_count = 0
    for sev_col in ['tickets_data_severity', 'Severity_Norm', 'Severity', 'severity']:
        if sev_col in df.columns:
            critical_count = len(df[df[sev_col].astype(str).str.lower().isin(['critical', 'high', 'major'])])
            break
    total_records = len(df)

    # Generate charts as HTML
    # 1. Category Sunburst
    sunburst_fig = chart_category_sunburst(df)
    sunburst_fig.update_layout(height=500, margin=dict(t=30, b=30, l=30, r=30))
    sunburst_html = pio.to_html(sunburst_fig, full_html=False, include_plotlyjs=False)

    # 2. Severity Distribution
    if 'tickets_data_severity' in df.columns:
        sev_data = df.groupby('tickets_data_severity')['Financial_Impact'].sum().reset_index()
        sev_fig = go.Figure(data=[go.Pie(
            labels=sev_data['tickets_data_severity'],
            values=sev_data['Financial_Impact'],
            hole=0.5,
            marker=dict(colors=['#ef4444', '#f97316', '#22c55e']),
            textinfo='label+percent'
        )])
        sev_fig.update_layout(height=400, margin=dict(t=30, b=30), showlegend=True,
                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        severity_html = pio.to_html(sev_fig, full_html=False, include_plotlyjs=False)
    else:
        severity_html = "<p>Severity data not available</p>"

    # 3. Friction by Category Bar
    friction_fig = chart_friction_by_category(df)
    friction_fig.update_layout(height=400, margin=dict(t=30, b=80, l=50, r=30))
    friction_html = pio.to_html(friction_fig, full_html=False, include_plotlyjs=False)

    # 4. Timeline Trend
    trend_fig = chart_trend_timeline(df)
    trend_fig.update_layout(height=350, margin=dict(t=30, b=50, l=50, r=30))
    trend_html = pio.to_html(trend_fig, full_html=False, include_plotlyjs=False)

    # 5. Engineer Performance
    eng_fig = chart_engineer_performance(df)
    eng_fig.update_layout(height=400, margin=dict(t=30, b=80, l=50, r=30))
    engineer_html = pio.to_html(eng_fig, full_html=False, include_plotlyjs=False)

    # Get recommendations
    recommendations = generate_strategic_recommendations(df)

    # Build HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Escalation Intelligence Report</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

            * {{ margin: 0; padding: 0; box-sizing: border-box; }}

            body {{
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, #0a1628 0%, #1a365d 50%, #0a1628 100%);
                color: #e2e8f0;
                min-height: 100vh;
                padding: 40px;
            }}

            .container {{ max-width: 1400px; margin: 0 auto; }}

            .header {{
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 58, 138, 0.9) 100%);
                border-radius: 20px;
                padding: 40px;
                margin-bottom: 30px;
                border: 1px solid rgba(59, 130, 246, 0.3);
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
                text-align: center;
            }}

            .header h1 {{
                font-size: 3rem;
                font-weight: 800;
                background: linear-gradient(135deg, #ffffff 0%, #60a5fa 50%, #3b82f6 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 10px;
            }}

            .header .subtitle {{
                color: #94a3b8;
                font-size: 1.2rem;
                letter-spacing: 2px;
            }}

            .header .meta {{
                margin-top: 20px;
                color: #64748b;
                font-size: 0.9rem;
            }}

            .kpi-grid {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 20px;
                margin-bottom: 30px;
            }}

            .kpi-card {{
                background: linear-gradient(145deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.8) 100%);
                border-radius: 16px;
                padding: 25px;
                text-align: center;
                border: 1px solid rgba(59, 130, 246, 0.2);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }}

            .kpi-card.green {{ border-color: rgba(34, 197, 94, 0.4); }}
            .kpi-card.red {{ border-color: rgba(239, 68, 68, 0.4); }}
            .kpi-card.blue {{ border-color: rgba(59, 130, 246, 0.4); }}
            .kpi-card.purple {{ border-color: rgba(139, 92, 246, 0.4); }}

            .kpi-label {{
                color: #94a3b8;
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 2px;
                margin-bottom: 10px;
            }}

            .kpi-value {{
                font-size: 2.5rem;
                font-weight: 800;
                text-shadow: 0 0 30px currentColor;
            }}

            .kpi-value.green {{ color: #22c55e; }}
            .kpi-value.red {{ color: #ef4444; }}
            .kpi-value.blue {{ color: #3b82f6; }}
            .kpi-value.purple {{ color: #8b5cf6; }}

            .section {{
                background: rgba(15, 23, 42, 0.6);
                border-radius: 16px;
                padding: 25px;
                margin-bottom: 25px;
                border: 1px solid rgba(59, 130, 246, 0.2);
            }}

            .section h2 {{
                color: #e2e8f0;
                font-size: 1.3rem;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 1px solid rgba(59, 130, 246, 0.3);
            }}

            .section h2 span {{ margin-right: 10px; }}

            .chart-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 25px;
                margin-bottom: 25px;
            }}

            .chart-box {{
                background: rgba(15, 23, 42, 0.4);
                border-radius: 12px;
                padding: 20px;
                border: 1px solid rgba(59, 130, 246, 0.15);
            }}

            .chart-box h3 {{
                color: #94a3b8;
                font-size: 0.9rem;
                margin-bottom: 15px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}

            .rec-card {{
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.8) 100%);
                border-radius: 12px;
                padding: 20px;
                margin: 15px 0;
                border-left: 4px solid #3b82f6;
            }}

            .rec-card.p1 {{ border-left-color: #ef4444; }}
            .rec-card.p2 {{ border-left-color: #f97316; }}
            .rec-card.p3 {{ border-left-color: #3b82f6; }}

            .rec-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 12px;
            }}

            .rec-priority {{
                padding: 4px 12px;
                border-radius: 4px;
                font-weight: 700;
                font-size: 0.8rem;
                color: white;
            }}

            .rec-priority.p1 {{ background: #ef4444; }}
            .rec-priority.p2 {{ background: #f97316; }}
            .rec-priority.p3 {{ background: #3b82f6; }}

            .rec-confidence {{ color: #22c55e; font-size: 0.85rem; }}

            .rec-title {{ color: #e2e8f0; font-weight: 600; font-size: 1.1rem; margin-bottom: 8px; }}
            .rec-desc {{ color: #94a3b8; font-size: 0.9rem; margin-bottom: 15px; }}

            .rec-metrics {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 10px;
                font-size: 0.8rem;
            }}

            .rec-metric {{ color: #64748b; }}
            .rec-metric strong {{ color: #e2e8f0; }}

            .footer {{
                text-align: center;
                padding: 30px;
                color: #64748b;
                font-size: 0.85rem;
            }}

            .footer .confidential {{ color: #ef4444; font-weight: 600; margin-top: 10px; }}

            @media print {{
                body {{ background: white; color: #333; padding: 20px; }}
                .header {{ background: #0066CC; color: white; }}
                .header h1 {{ color: white; -webkit-text-fill-color: white; }}
                .kpi-card, .section, .chart-box {{ background: #f8f9fa; border: 1px solid #ddd; }}
                .kpi-label, .rec-desc {{ color: #666; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>ESCALATION INTELLIGENCE</h1>
                <div class="subtitle">EXECUTIVE ANALYTICS REPORT</div>
                <div class="meta">
                    Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')} |
                    Analysis Period: 90 Days | {total_records:,} Records Analyzed
                </div>
            </div>

            <div class="kpi-grid">
                <div class="kpi-card green">
                    <div class="kpi-label">💰 Total Financial Impact</div>
                    <div class="kpi-value green">${total_cost/1000:,.0f}K</div>
                </div>
                <div class="kpi-card red">
                    <div class="kpi-label">⚠️ Revenue at Risk</div>
                    <div class="kpi-value red">${revenue_risk/1000:,.0f}K</div>
                </div>
                <div class="kpi-card blue">
                    <div class="kpi-label">🔥 Critical Issues</div>
                    <div class="kpi-value blue">{critical_count}</div>
                </div>
                <div class="kpi-card purple">
                    <div class="kpi-label">⏱️ Avg Resolution</div>
                    <div class="kpi-value purple">{avg_resolution:.1f}d</div>
                </div>
            </div>

            <div class="section">
                <h2><span>🎯</span>Strategic Recommendations</h2>
                {''.join([f'''
                <div class="rec-card {rec['priority'].lower()}">
                    <div class="rec-header">
                        <span class="rec-priority {rec['priority'].lower()}">{rec['priority']}</span>
                        <span class="rec-confidence">🎯 {rec['confidence']}% confidence</span>
                    </div>
                    <div class="rec-title">{rec['title']}</div>
                    <div class="rec-desc">{rec['description']}</div>
                    <div class="rec-metrics">
                        <div class="rec-metric"><strong>Impact:</strong> {rec['impact']}</div>
                        <div class="rec-metric"><strong>Timeline:</strong> {rec['timeline']}</div>
                        <div class="rec-metric"><strong>Investment:</strong> {rec['investment']}</div>
                        <div class="rec-metric"><strong>ROI:</strong> {rec['roi']}</div>
                    </div>
                </div>
                ''' for rec in recommendations[:4]])}
            </div>

            <div class="chart-grid">
                <div class="chart-box">
                    <h3>📊 Category & Sub-Category Breakdown</h3>
                    {sunburst_html}
                </div>
                <div class="chart-box">
                    <h3>🎯 Severity Distribution</h3>
                    {severity_html}
                </div>
            </div>

            <div class="chart-grid">
                <div class="chart-box">
                    <h3>📈 Friction by Category</h3>
                    {friction_html}
                </div>
                <div class="chart-box">
                    <h3>👥 Engineer Performance</h3>
                    {engineer_html}
                </div>
            </div>

            <div class="section">
                <h2><span>📈</span>Escalation Timeline</h2>
                {trend_html}
            </div>

            <div class="footer">
                <p>Generated by Escalation AI v2.3.0 | Powered by AI-Driven Analytics</p>
                <p class="confidential">CONFIDENTIAL - FOR EXECUTIVE REVIEW ONLY</p>
            </div>
        </div>
    </body>
    </html>
    """

    return html


# ============================================================================
# EXCEL-STYLE DASHBOARD - INTERACTIVE & SPECTACULAR
# ============================================================================

def render_excel_dashboard(df):
    """
    Spectacular interactive dashboard with visual impact.
    Features large sunburst charts, animated gauges, and interactive elements.
    """

    # Calculate metrics
    total_cost = df['Financial_Impact'].sum() if 'Financial_Impact' in df.columns else 0
    total_records = len(df)
    avg_cost = total_cost / total_records if total_records > 0 else 0
    categories = df['AI_Category'].unique() if 'AI_Category' in df.columns else []
    # Check for critical/high severity using multiple column names and values
    critical_count = 0
    for sev_col in ['tickets_data_severity', 'Severity_Norm', 'Severity', 'severity']:
        if sev_col in df.columns:
            critical_count = len(df[df[sev_col].astype(str).str.lower().isin(['critical', 'high', 'major'])])
            break

    # Extract years
    if 'tickets_data_issue_datetime' in df.columns:
        df['_year'] = pd.to_datetime(df['tickets_data_issue_datetime']).dt.year
        years = sorted(df['_year'].unique())
    else:
        years = [2024, 2025]
        df['_year'] = 2024

    # ========== SPECTACULAR HEADER ==========
    from datetime import datetime
    current_time = datetime.now().strftime("%b %d, %Y at %I:%M %p")

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0a1628 0%, #1a365d 50%, #0a1628 100%);
                padding: 25px 35px; border-radius: 16px; margin-bottom: 20px;
                border: 1px solid rgba(59, 130, 246, 0.3);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255,255,255,0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="font-size: 2.5rem; font-weight: 800; margin: 0;
                           background: linear-gradient(135deg, #ffffff 0%, #60a5fa 50%, #3b82f6 100%);
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    ESCALATION <span style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent;">INTELLIGENCE</span>
                </h1>
                <p style="color: #94a3b8; font-size: 1rem; margin: 8px 0 0 0; letter-spacing: 2px;">
                    EXECUTIVE ANALYTICS DASHBOARD
                </p>
            </div>
            <div style="text-align: right;">
                <div style="color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;">Last Updated</div>
                <div style="color: #60a5fa; font-size: 1rem; font-weight: 600; margin-top: 4px;">{current_time}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ========== ROW 1: SPECTACULAR KPI CARDS ==========
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, rgba(34, 197, 94, 0.15) 0%, rgba(21, 128, 61, 0.25) 100%);
                    border-radius: 16px; padding: 25px; text-align: center;
                    border: 1px solid rgba(34, 197, 94, 0.3);
                    box-shadow: 0 4px 20px rgba(34, 197, 94, 0.2);">
            <div style="color: #86efac; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px;">
                💰 Total Financial Impact
            </div>
            <div style="color: #22c55e; font-size: 2.8rem; font-weight: 800; text-shadow: 0 0 30px rgba(34, 197, 94, 0.5);">
                ${total_cost:,.0f}
            </div>
            <div style="color: #6b7280; font-size: 0.75rem; margin-top: 8px;">
                Avg: ${avg_cost:,.0f} per record
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, rgba(59, 130, 246, 0.15) 0%, rgba(29, 78, 216, 0.25) 100%);
                    border-radius: 16px; padding: 25px; text-align: center;
                    border: 1px solid rgba(59, 130, 246, 0.3);
                    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.2);">
            <div style="color: #93c5fd; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px;">
                📊 Total Records
            </div>
            <div style="color: #3b82f6; font-size: 2.8rem; font-weight: 800; text-shadow: 0 0 30px rgba(59, 130, 246, 0.5);">
                {total_records:,}
            </div>
            <div style="color: #6b7280; font-size: 0.75rem; margin-top: 8px;">
                Across {len(categories)} categories
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.25) 100%);
                    border-radius: 16px; padding: 25px; text-align: center;
                    border: 1px solid rgba(239, 68, 68, 0.3);
                    box-shadow: 0 4px 20px rgba(239, 68, 68, 0.2);">
            <div style="color: #fca5a5; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px;">
                🔴 Critical Issues
            </div>
            <div style="color: #ef4444; font-size: 2.8rem; font-weight: 800; text-shadow: 0 0 30px rgba(239, 68, 68, 0.5);">
                {critical_count}
            </div>
            <div style="color: #6b7280; font-size: 0.75rem; margin-top: 8px;">
                {critical_count/total_records*100:.1f}% of total
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        avg_days = df['Predicted_Resolution_Days'].mean() if 'Predicted_Resolution_Days' in df.columns else 0
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, rgba(168, 85, 247, 0.15) 0%, rgba(126, 34, 206, 0.25) 100%);
                    border-radius: 16px; padding: 25px; text-align: center;
                    border: 1px solid rgba(168, 85, 247, 0.3);
                    box-shadow: 0 4px 20px rgba(168, 85, 247, 0.2);">
            <div style="color: #d8b4fe; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px;">
                ⏱️ Avg Resolution
            </div>
            <div style="color: #a855f7; font-size: 2.8rem; font-weight: 800; text-shadow: 0 0 30px rgba(168, 85, 247, 0.5);">
                {avg_days:.1f}
            </div>
            <div style="color: #6b7280; font-size: 0.75rem; margin-top: 8px;">
                Days to resolve
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

    # ========== ROW 2: STRATEGIC FRICTION + TREND ==========
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.6); border-radius: 16px; padding: 20px;
                    border: 1px solid rgba(239, 68, 68, 0.2);">
            <h3 style="color: #e2e8f0; margin: 0 0 10px 0; font-size: 1.1rem;">
                🔥 Strategic Friction by Category
            </h3>
            <p style="color: #64748b; font-size: 0.8rem; margin: 0;">Highest friction areas requiring attention</p>
        </div>
        """, unsafe_allow_html=True)

        # Strategic Friction by Category - Horizontal Bar Chart
        if 'AI_Category' in df.columns and 'Strategic_Friction_Score' in df.columns:
            friction_data = df.groupby('AI_Category').agg({
                'Strategic_Friction_Score': 'sum',
                'Financial_Impact': 'sum'
            }).sort_values('Strategic_Friction_Score', ascending=True)

            # Color gradient based on friction score
            max_friction = friction_data['Strategic_Friction_Score'].max()
            colors = [f'rgba({int(239 * (v/max_friction))}, {int(68 + 129 * (1 - v/max_friction))}, {int(68 + 178 * (1 - v/max_friction))}, 0.9)'
                     for v in friction_data['Strategic_Friction_Score']]

            fig_friction = go.Figure(data=[go.Bar(
                y=friction_data.index,
                x=friction_data['Strategic_Friction_Score'],
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(255,255,255,0.3)', width=1)
                ),
                text=[f'{v:,.0f}' for v in friction_data['Strategic_Friction_Score']],
                textposition='outside',
                textfont=dict(size=11, color='#e2e8f0'),
                customdata=friction_data['Financial_Impact'],
                hovertemplate='<b>%{y}</b><br>Friction: %{x:,.0f}<br>Cost: $%{customdata:,.0f}<extra></extra>'
            )])
            fig_friction.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=60, t=10, b=10),
                height=420,
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    tickfont=dict(size=10, color='#64748b'),
                    title=dict(text='Friction Score', font=dict(size=10, color='#64748b'))
                ),
                yaxis=dict(
                    showgrid=False,
                    tickfont=dict(size=11, color='#e2e8f0')
                ),
                showlegend=False
            )
            st.plotly_chart(fig_friction, use_container_width=True, key="main_friction")

    with col2:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.6); border-radius: 16px; padding: 20px;
                    border: 1px solid rgba(239, 68, 68, 0.2);">
            <h3 style="color: #e2e8f0; margin: 0 0 10px 0; font-size: 1.1rem;">
                📈 Financial Impact Timeline
            </h3>
            <p style="color: #64748b; font-size: 0.8rem; margin: 0;">Monthly cost trend with severity breakdown</p>
        </div>
        """, unsafe_allow_html=True)

        if 'tickets_data_issue_datetime' in df.columns:
            df_m = df.copy()
            df_m['month'] = pd.to_datetime(df_m['tickets_data_issue_datetime']).dt.to_period('M').astype(str)
            monthly = df_m.groupby(['month', 'tickets_data_severity'])['Financial_Impact'].sum().reset_index()

            fig_area = px.area(
                monthly,
                x='month',
                y='Financial_Impact',
                color='tickets_data_severity',
                color_discrete_map={'Critical': '#ef4444', 'Major': '#f97316', 'Minor': '#22c55e'},
                labels={'Financial_Impact': 'Cost', 'month': 'Month', 'tickets_data_severity': 'Severity'}
            )
            fig_area.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=40),
                height=420,
                xaxis=dict(showgrid=False, tickfont=dict(size=9, color='#64748b'), tickangle=-45),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)',
                          tickfont=dict(size=9, color='#64748b'), tickformat='$,.0f'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5,
                           font=dict(color='#94a3b8', size=10)),
                hovermode='x unified'
            )
            st.plotly_chart(fig_area, use_container_width=True, key="area_trend")

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

    # ========== ROW 3: ENGINEER TREEMAP + SEVERITY GAUGE ==========
    col1, col2 = st.columns([1.2, 0.8])

    with col1:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.6); border-radius: 16px; padding: 20px;
                    border: 1px solid rgba(34, 197, 94, 0.2);">
            <h3 style="color: #e2e8f0; margin: 0 0 10px 0; font-size: 1.1rem;">
                👥 Engineer Performance Treemap
            </h3>
            <p style="color: #64748b; font-size: 0.8rem; margin: 0;">Click to drill into engineer details</p>
        </div>
        """, unsafe_allow_html=True)

        # Try multiple possible engineer column names
        eng_col = None
        for col in ['Engineer', 'tickets_data_engineer_name', 'Engineer_Assigned', 'Assigned_To']:
            if col in df.columns:
                eng_col = col
                break

        if eng_col and 'AI_Category' in df.columns:
            eng_data = df.groupby([eng_col, 'AI_Category']).agg({
                'Financial_Impact': 'sum',
                'AI_Category': 'count'
            }).rename(columns={'AI_Category': 'Records'})
            eng_data = eng_data.reset_index()
            eng_data.columns = ['Engineer', 'Category', 'Cost', 'Records']

            if len(eng_data) > 0:
                fig_tree = px.treemap(
                    eng_data,
                    path=['Engineer', 'Category'],
                    values='Cost',
                    color='Cost',
                    color_continuous_scale='RdYlGn_r',
                    hover_data={'Records': True, 'Cost': ':$,.0f'}
                )
                fig_tree.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=350,
                    font=dict(color='white'),
                    coloraxis_colorbar=dict(
                        title=dict(text="Cost", font=dict(color='#94a3b8')),
                        tickformat="$,.0f",
                        tickfont=dict(color='#94a3b8')
                    )
                )
                fig_tree.update_traces(
                    textinfo='label+value',
                    texttemplate='%{label}<br>$%{value:,.0f}',
                    hovertemplate='<b>%{label}</b><br>Cost: $%{value:,.0f}<br>Records: %{customdata[0]}<extra></extra>'
                )
                st.plotly_chart(fig_tree, use_container_width=True, key="eng_treemap")
            else:
                st.info("No engineer data available for treemap")
        else:
            st.info("Engineer column not found in data")

    with col2:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.6); border-radius: 16px; padding: 20px;
                    border: 1px solid rgba(168, 85, 247, 0.2);">
            <h3 style="color: #e2e8f0; margin: 0 0 10px 0; font-size: 1.1rem;">
                🎯 Severity Breakdown
            </h3>
            <p style="color: #64748b; font-size: 0.8rem; margin: 0;">Distribution by severity level</p>
        </div>
        """, unsafe_allow_html=True)

        # Try multiple severity column names
        sev_col = None
        for col in ['tickets_data_severity', 'Severity_Norm', 'Severity', 'severity']:
            if col in df.columns:
                sev_col = col
                break

        if sev_col:
            sev_data = df.groupby(sev_col).agg({
                'Financial_Impact': 'sum',
                'AI_Category': 'count'
            }).rename(columns={'AI_Category': 'Count'}).reset_index()

            if len(sev_data) > 0:
                # Color map based on severity
                color_map = {'Critical': '#ef4444', 'High': '#f97316', 'Major': '#f97316',
                            'Medium': '#eab308', 'Minor': '#22c55e', 'Low': '#22c55e'}
                colors = [color_map.get(str(s), '#3b82f6') for s in sev_data[sev_col]]

                fig_sev = go.Figure(data=[go.Pie(
                    labels=sev_data[sev_col],
                    values=sev_data['Financial_Impact'],
                    hole=0.5,
                    marker=dict(
                        colors=colors,
                        line=dict(color='rgba(0,0,0,0.3)', width=2)
                    ),
                    textinfo='label+percent',
                    textfont=dict(size=12, color='white'),
                    hovertemplate='<b>%{label}</b><br>Cost: $%{value:,.0f}<br>%{percent}<extra></extra>'
                )])
                fig_sev.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=350,
                    showlegend=False,
                    annotations=[dict(
                        text=f'<b>${total_cost/1000:.0f}K</b><br>Total',
                        x=0.5, y=0.5, font_size=16, font_color='white', showarrow=False
                    )]
                )
                st.plotly_chart(fig_sev, use_container_width=True, key="sev_donut")
            else:
                st.info("No severity data available")
        else:
            st.info("Severity column not found in data")

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

    # ========== ROW 4: QUARTERLY + ORIGIN ==========
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.6); border-radius: 16px; padding: 20px;
                    border: 1px solid rgba(59, 130, 246, 0.2);">
            <h3 style="color: #e2e8f0; margin: 0 0 10px 0; font-size: 1.1rem;">
                📅 Quarterly Performance
            </h3>
            <p style="color: #64748b; font-size: 0.8rem; margin: 0;">Cost distribution by quarter</p>
        </div>
        """, unsafe_allow_html=True)

        # Try multiple date column names
        date_col = None
        for col in ['tickets_data_issue_datetime', 'Issue_Date', 'Created_Date', 'Date', 'created_at']:
            if col in df.columns:
                date_col = col
                break

        if date_col:
            try:
                df_q = df.copy()
                df_q['qtr'] = 'Q' + pd.to_datetime(df_q[date_col], errors='coerce').dt.quarter.astype(str)
                qtr_data = df_q.groupby('qtr')['Financial_Impact'].sum().reset_index()
                qtr_data.columns = ['Quarter', 'Cost']

                if len(qtr_data) > 0 and qtr_data['Cost'].sum() > 0:
                    fig_qtr = go.Figure(data=[go.Bar(
                        x=qtr_data['Quarter'],
                        y=qtr_data['Cost'],
                        marker=dict(
                            color=qtr_data['Cost'],
                            colorscale='Blues',
                            line=dict(color='rgba(59, 130, 246, 0.8)', width=2)
                        ),
                        text=[f'${v:,.0f}' for v in qtr_data['Cost']],
                        textposition='outside',
                        textfont=dict(size=12, color='white')
                    )])
                    fig_qtr.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=10, r=10, t=30, b=10),
                        height=300,
                        xaxis=dict(showgrid=False, tickfont=dict(size=12, color='#94a3b8')),
                        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)',
                                  tickfont=dict(size=10, color='#64748b'), tickformat='$,.0f'),
                        showlegend=False
                    )
                    st.plotly_chart(fig_qtr, use_container_width=True, key="qtr_bar")
                else:
                    st.info("No quarterly cost data available")
            except Exception as e:
                st.info(f"Could not parse date column: {e}")
        else:
            st.info("Date column not found in data")

    with col2:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.6); border-radius: 16px; padding: 20px;
                    border: 1px solid rgba(239, 68, 68, 0.2);">
            <h3 style="color: #e2e8f0; margin: 0 0 10px 0; font-size: 1.1rem;">
                🔄 Escalation Origin Analysis
            </h3>
            <p style="color: #64748b; font-size: 0.8rem; margin: 0;">Click an origin below to drill down</p>
        </div>
        """, unsafe_allow_html=True)

        # Try multiple origin column names
        origin_col = None
        for col in ['tickets_data_escalation_origin', 'Origin_Norm', 'Origin', 'origin', 'Escalation_Origin']:
            if col in df.columns:
                origin_col = col
                break

        if origin_col:
            # Find severity column for critical count
            sev_col_for_origin = None
            for col in ['tickets_data_severity', 'Severity_Norm', 'Severity', 'severity']:
                if col in df.columns:
                    sev_col_for_origin = col
                    break

            # Calculate comprehensive metrics by origin
            agg_dict = {
                'Financial_Impact': ['sum', 'mean', 'count'],
                'Predicted_Resolution_Days': 'mean' if 'Predicted_Resolution_Days' in df.columns else 'count'
            }
            if sev_col_for_origin:
                agg_dict[sev_col_for_origin] = lambda x: (x.astype(str).str.lower().isin(['critical', 'high', 'major'])).sum()

            origin_analysis = df.groupby(origin_col).agg(agg_dict).reset_index()

            # Flatten column names
            if sev_col_for_origin:
                origin_analysis.columns = ['Origin', 'Total_Cost', 'Avg_Cost', 'Count', 'Avg_Resolution', 'Critical_Count']
            else:
                origin_analysis.columns = ['Origin', 'Total_Cost', 'Avg_Cost', 'Count', 'Avg_Resolution']
                origin_analysis['Critical_Count'] = 0

            origin_analysis = origin_analysis.sort_values('Total_Cost', ascending=True)

            total_all = origin_analysis['Total_Cost'].sum()
            num_origins = len(origin_analysis)

            # Selector for drill-down with "Show chart only" option
            origins_list = origin_analysis.sort_values('Total_Cost', ascending=False)['Origin'].tolist()
            options_list = ["📊 Chart view (click to select)"] + origins_list
            selected_option = st.selectbox(
                "🔍 Select origin:",
                options=options_list,
                index=0,
                key="origin_drilldown",
                label_visibility="collapsed"
            )

            # Full bar chart - taller with proper spacing
            colors = ['#ef4444', '#f97316', '#22c55e', '#3b82f6', '#8b5cf6', '#06b6d4']
            fig_orig = go.Figure(data=[go.Bar(
                y=origin_analysis['Origin'],
                x=origin_analysis['Total_Cost'],
                orientation='h',
                marker=dict(
                    color=[colors[i % len(colors)] for i in range(num_origins)],
                    line=dict(color='rgba(255,255,255,0.3)', width=1)
                ),
                text=[f'${v:,.0f}' for v in origin_analysis['Total_Cost']],
                textposition='inside',
                textfont=dict(size=14, color='white', family='Arial Black'),
                hovertemplate='<b>%{y}</b><br>Total Cost: %{x:$,.0f}<extra></extra>'
            )])
            fig_orig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=20, t=10, b=10),
                height=max(200, num_origins * 60),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)',
                          tickfont=dict(size=11, color='#64748b'), tickformat='$,.0f'),
                yaxis=dict(showgrid=False, tickfont=dict(size=13, color='#e2e8f0', family='Arial')),
                showlegend=False,
                bargap=0.3
            )
            st.plotly_chart(fig_orig, use_container_width=True, key="orig_bar")

            # Show drill-down only when an origin is selected
            if selected_option != "📊 Chart view (click to select)":
                selected_origin = selected_option
                selected_data = origin_analysis[origin_analysis['Origin'] == selected_origin].iloc[0]
                selected_df = df[df[origin_col] == selected_origin]

                critical_rate = selected_data['Critical_Count'] / selected_data['Count'] * 100 if selected_data['Count'] > 0 else 0
                pct_of_total = selected_data['Total_Cost'] / total_all * 100

                # Drill-down panel
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
                            border-radius: 12px; padding: 15px; margin-top: 10px; border: 1px solid rgba(59, 130, 246, 0.3);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="color: #e2e8f0; font-weight: 700; font-size: 1rem;">📊 {selected_origin}</span>
                        <span style="color: #22c55e; font-weight: 800; font-size: 1.4rem;">${selected_data['Total_Cost']:,.0f}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                        <div style="background: rgba(0,0,0,0.2); border-radius: 8px; padding: 12px; text-align: center;">
                            <div style="color: #64748b; font-size: 0.7rem; text-transform: uppercase;">Ticket Volume</div>
                            <div style="color: #3b82f6; font-size: 1.3rem; font-weight: 700;">{selected_data['Count']:,.0f}</div>
                            <div style="color: #64748b; font-size: 0.65rem;">{pct_of_total:.1f}% of total cost</div>
                        </div>
                        <div style="background: rgba(0,0,0,0.2); border-radius: 8px; padding: 12px; text-align: center;">
                            <div style="color: #64748b; font-size: 0.7rem; text-transform: uppercase;">Avg Cost/Ticket</div>
                            <div style="color: #f97316; font-size: 1.3rem; font-weight: 700;">${selected_data['Avg_Cost']:,.0f}</div>
                        </div>
                        <div style="background: rgba(0,0,0,0.2); border-radius: 8px; padding: 12px; text-align: center;">
                            <div style="color: #64748b; font-size: 0.7rem; text-transform: uppercase;">Avg Resolution</div>
                            <div style="color: #8b5cf6; font-size: 1.3rem; font-weight: 700;">{selected_data['Avg_Resolution']:.1f}d</div>
                        </div>
                        <div style="background: rgba(0,0,0,0.2); border-radius: 8px; padding: 12px; text-align: center;">
                            <div style="color: #64748b; font-size: 0.7rem; text-transform: uppercase;">Critical Rate</div>
                            <div style="color: {'#ef4444' if critical_rate > 20 else '#22c55e'}; font-size: 1.3rem; font-weight: 700;">{critical_rate:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Top categories for this origin
                if 'AI_Category' in selected_df.columns:
                    top_cats = selected_df.groupby('AI_Category')['Financial_Impact'].sum().sort_values(ascending=False).head(3)
                    st.markdown(f"""
                    <div style="background: rgba(34, 197, 94, 0.1); border-radius: 8px; padding: 10px; margin-top: 8px; border: 1px solid rgba(34, 197, 94, 0.3);">
                        <div style="color: #86efac; font-size: 0.75rem; font-weight: 600; margin-bottom: 5px;">🏷️ Top Categories for {selected_origin}:</div>
                        <div style="color: #94a3b8; font-size: 0.7rem;">
                            {"  •  ".join([f"<b>{cat}</b> (${val:,.0f})" for cat, val in top_cats.items()])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Origin column not found in data")

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

    # ========== ROW 5: SANKEY FLOW DIAGRAM ==========
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.6); border-radius: 16px; padding: 20px;
                border: 1px solid rgba(168, 85, 247, 0.3);">
        <h3 style="color: #e2e8f0; margin: 0 0 10px 0; font-size: 1.2rem;">
            🔀 Issue Flow Analysis: Category → Severity → Resolution Speed
        </h3>
        <p style="color: #64748b; font-size: 0.85rem; margin: 0;">
            Interactive flow diagram showing how issues move through the system. Hover for details.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Create Sankey data - check for multiple severity column names
    sev_col_sankey = None
    for col in ['tickets_data_severity', 'Severity_Norm', 'Severity', 'severity']:
        if col in df.columns:
            sev_col_sankey = col
            break

    if 'AI_Category' in df.columns and sev_col_sankey:
        # Derive Resolution Speed from Predicted_Resolution_Days
        df_sankey = df.copy()
        if 'Predicted_Resolution_Days' in df_sankey.columns:
            # Create resolution speed categories
            def get_resolution_speed(days):
                if pd.isna(days):
                    return 'Standard (2-5d)'
                elif days < 2:
                    return 'Quick (<2d)'
                elif days <= 5:
                    return 'Standard (2-5d)'
                else:
                    return 'Extended (>5d)'
            df_sankey['Resolution_Speed'] = df_sankey['Predicted_Resolution_Days'].apply(get_resolution_speed)
        else:
            df_sankey['Resolution_Speed'] = 'Standard (2-5d)'

        # Get unique values for each level (convert to strings)
        categories = [str(c) for c in df_sankey['AI_Category'].unique().tolist()]
        severities = [str(s) for s in df_sankey[sev_col_sankey].unique().tolist()]
        resolutions = [str(r) for r in df_sankey['Resolution_Speed'].dropna().unique().tolist()]

        # Create node labels
        all_labels = categories + severities + resolutions

        # Create color map
        cat_colors = ['#3b82f6', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#ec4899', '#6366f1']
        sev_colors = {'Critical': '#ef4444', 'High': '#ef4444', 'Major': '#f97316', 'Medium': '#f97316', 'Minor': '#22c55e', 'Low': '#22c55e'}
        res_colors = {'Quick (<2d)': '#22c55e', 'Standard (2-5d)': '#3b82f6', 'Extended (>5d)': '#ef4444'}

        node_colors = []
        for label in all_labels:
            if label in categories:
                node_colors.append(cat_colors[categories.index(label) % len(cat_colors)])
            elif label in sev_colors:
                node_colors.append(sev_colors[label])
            elif label in res_colors:
                node_colors.append(res_colors[label])
            else:
                node_colors.append('#6b7280')

        # Create links: Category -> Severity
        links_cat_sev = df_sankey.groupby(['AI_Category', sev_col_sankey]).size().reset_index(name='count')

        # Create links: Severity -> Resolution
        links_sev_res = df_sankey.groupby([sev_col_sankey, 'Resolution_Speed']).size().reset_index(name='count')

        # Build source, target, value lists
        sources = []
        targets = []
        values = []
        link_colors = []

        # Category -> Severity links
        for _, row in links_cat_sev.iterrows():
            cat_str = str(row['AI_Category'])
            sev_str = str(row[sev_col_sankey])
            if cat_str in all_labels and sev_str in all_labels:
                src_idx = all_labels.index(cat_str)
                tgt_idx = all_labels.index(sev_str)
                sources.append(src_idx)
                targets.append(tgt_idx)
                values.append(row['count'])
                # Color based on severity (handle different naming conventions)
                sev_lower = sev_str.lower()
                if sev_lower in ['critical', 'high']:
                    link_colors.append('rgba(239, 68, 68, 0.4)')
                elif sev_lower in ['major', 'medium']:
                    link_colors.append('rgba(249, 115, 22, 0.4)')
                else:
                    link_colors.append('rgba(34, 197, 94, 0.4)')

        # Severity -> Resolution links
        for _, row in links_sev_res.iterrows():
            sev_str = str(row[sev_col_sankey])
            res_str = str(row['Resolution_Speed'])
            if pd.notna(row['Resolution_Speed']) and sev_str in all_labels and res_str in all_labels:
                src_idx = all_labels.index(sev_str)
                tgt_idx = all_labels.index(res_str)
                sources.append(src_idx)
                targets.append(tgt_idx)
                values.append(row['count'])
                # Color based on resolution speed
                if 'Quick' in res_str:
                    link_colors.append('rgba(34, 197, 94, 0.5)')
                elif 'Extended' in res_str:
                    link_colors.append('rgba(239, 68, 68, 0.5)')
                else:
                    link_colors.append('rgba(59, 130, 246, 0.5)')

        # Create Sankey diagram
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(
                pad=20,
                thickness=25,
                line=dict(color='rgba(255,255,255,0.3)', width=1),
                label=all_labels,
                color=node_colors,
                hovertemplate='<b>%{label}</b><br>Total: %{value}<extra></extra>'
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
                hovertemplate='<b>%{source.label}</b> → <b>%{target.label}</b><br>Count: %{value}<extra></extra>'
            )
        )])

        fig_sankey.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            height=450,
            font=dict(color='#e2e8f0', size=11)
        )

        st.plotly_chart(fig_sankey, use_container_width=True, key="issue_flow_sankey")

        # Legend
        st.markdown("""
        <div style="display: flex; justify-content: center; gap: 30px; margin-top: 10px; flex-wrap: wrap;">
            <div style="color: #94a3b8; font-size: 0.8rem;">
                <span style="color: #3b82f6;">■</span> Categories
            </div>
            <div style="color: #94a3b8; font-size: 0.8rem;">
                <span style="color: #ef4444;">■</span> Critical
                <span style="color: #f97316; margin-left: 10px;">■</span> Major
                <span style="color: #22c55e; margin-left: 10px;">■</span> Minor
            </div>
            <div style="color: #94a3b8; font-size: 0.8rem;">
                <span style="color: #22c55e;">■</span> Quick
                <span style="color: #3b82f6; margin-left: 10px;">■</span> Standard
                <span style="color: #ef4444; margin-left: 10px;">■</span> Extended
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Category or Severity column not found in data for Issue Flow Analysis")

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

    # ========== ROW 6: STRATEGIC RECOMMENDATIONS ==========
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.6); border-radius: 16px; padding: 20px;
                border: 1px solid rgba(34, 197, 94, 0.3);">
        <h3 style="color: #e2e8f0; margin: 0 0 10px 0; font-size: 1.2rem;">
            🎯 Strategic Recommendations
        </h3>
        <p style="color: #64748b; font-size: 0.85rem; margin: 0;">
            AI-generated insights with confidence scoring
        </p>
    </div>
    """, unsafe_allow_html=True)

    recommendations = generate_strategic_recommendations(df)

    rec_cols = st.columns(2)
    for i, rec in enumerate(recommendations[:4]):
        with rec_cols[i % 2]:
            priority_color = '#ef4444' if rec['priority'] == 'P1' else '#f97316' if rec['priority'] == 'P2' else '#3b82f6'
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.8) 100%);
                        border-radius: 12px; padding: 18px; margin: 10px 0;
                        border-left: 4px solid {priority_color};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="background: {priority_color}; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 700; font-size: 0.8rem;">
                        {rec['priority']}
                    </span>
                    <span style="color: #22c55e; font-size: 0.8rem;">🎯 {rec['confidence']}% confidence</span>
                </div>
                <div style="color: #e2e8f0; font-weight: 600; font-size: 1rem; margin-bottom: 8px;">{rec['title']}</div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 12px;">{rec['description']}</div>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; font-size: 0.75rem;">
                    <div><span style="color: #22c55e;">Impact:</span> <span style="color: #e2e8f0;">{rec['impact']}</span></div>
                    <div><span style="color: #3b82f6;">Timeline:</span> <span style="color: #e2e8f0;">{rec['timeline']}</span></div>
                    <div><span style="color: #f97316;">Investment:</span> <span style="color: #e2e8f0;">{rec['investment']}</span></div>
                    <div><span style="color: #8b5cf6;">ROI:</span> <span style="color: #e2e8f0;">{rec['roi']}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# CONSOLIDATED TAB 2: DEEP ANALYSIS
# ============================================================================

def render_deep_analysis(df):
    """Render consolidated Deep Analysis page (Analytics + Root Cause + Advanced Insights)."""
    render_spectacular_header("Deep Analysis", "Comprehensive drill-down for detailed insights", "📈")

    # Main tabs - flattened structure
    tabs = st.tabs(["🎯 Categories", "👥 Engineers", "🔬 Root Cause", "📊 Patterns & SLA", "🔗 Similarity", "📚 Lessons Learned"])

    # ===== TAB 1: CATEGORIES =====
    with tabs[0]:
        # Sub-category column detection
        sub_cat_col = 'AI_Sub_Category' if 'AI_Sub_Category' in df.columns else 'AI_SubCategory' if 'AI_SubCategory' in df.columns else None

        # ROW 1: Cross-category sub-category analysis (TOP)
        st.markdown("#### 🏆 Top & Bottom Performing Sub-Categories (All Categories)")
        if sub_cat_col and 'AI_Category' in df.columns:
            # Calculate metrics for all sub-categories
            all_sub_data = df.groupby([sub_cat_col, 'AI_Category']).agg({
                'Financial_Impact': 'sum',
                'AI_Recurrence_Risk': 'mean' if 'AI_Recurrence_Risk' in df.columns else 'count',
                'Predicted_Resolution_Days': 'mean' if 'Predicted_Resolution_Days' in df.columns else 'count'
            }).reset_index()
            all_sub_data.columns = ['SubCategory', 'Category', 'Cost', 'Recurrence', 'Resolution']
            all_sub_data['Count'] = df.groupby([sub_cat_col, 'AI_Category']).size().values

            view_mode = st.radio(
                "View by:",
                ["💰 Highest Cost", "💚 Lowest Cost", "🔴 Highest Recurrence", "⏱️ Slowest Resolution"],
                horizontal=True,
                key="subcategory_view_mode"
            )

            if view_mode == "💰 Highest Cost":
                display_data = all_sub_data.nlargest(10, 'Cost')
                metric_col = 'Cost'
                color_scale = 'Reds'
                format_str = '${:,.0f}'
            elif view_mode == "💚 Lowest Cost":
                display_data = all_sub_data[all_sub_data['Count'] >= 2].nsmallest(10, 'Cost')
                metric_col = 'Cost'
                color_scale = 'Greens'
                format_str = '${:,.0f}'
            elif view_mode == "🔴 Highest Recurrence":
                display_data = all_sub_data[all_sub_data['Count'] >= 2].nlargest(10, 'Recurrence')
                metric_col = 'Recurrence'
                color_scale = 'Reds'
                format_str = '{:.1%}'
            else:
                display_data = all_sub_data[all_sub_data['Count'] >= 2].nlargest(10, 'Resolution')
                metric_col = 'Resolution'
                color_scale = 'Oranges'
                format_str = '{:.1f}d'

            display_data = display_data.sort_values(metric_col, ascending=True)

            # Create labels with category context
            labels = [f"{row['SubCategory'][:25]}... ({row['Category'][:15]})" if len(row['SubCategory']) > 25
                     else f"{row['SubCategory']} ({row['Category'][:15]})"
                     for _, row in display_data.iterrows()]

            fig_all_sub = go.Figure(data=[go.Bar(
                y=labels,
                x=display_data[metric_col],
                orientation='h',
                marker=dict(
                    color=display_data[metric_col],
                    colorscale=color_scale,
                    line=dict(color='rgba(255,255,255,0.3)', width=1)
                ),
                text=[format_str.format(v) if metric_col != 'Recurrence' else f'{v*100:.1f}%' for v in display_data[metric_col]],
                textposition='outside',
                textfont=dict(size=11, color='#e2e8f0'),
                hovertemplate='<b>%{y}</b><br>Value: %{x}<br>Tickets: %{customdata}<extra></extra>',
                customdata=display_data['Count']
            )])
            fig_all_sub.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=max(300, len(display_data) * 40),
                margin=dict(l=10, r=80, t=10, b=10),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#94a3b8')),
                yaxis=dict(showgrid=False, tickfont=dict(size=10, color='#e2e8f0')),
                showlegend=False
            )
            st.plotly_chart(fig_all_sub, use_container_width=True)

        st.markdown("---")

        # ROW 2: Sunburst + Severity Breakdown
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🎯 Category & Sub-Category Drill-Down")
            st.plotly_chart(chart_category_sunburst(df), use_container_width=True)
        with col2:
            st.markdown("#### 📊 Severity Breakdown by Category")
            # Severity Distribution by Category - Stacked Bar
            if 'AI_Category' in df.columns and 'tickets_data_severity' in df.columns:
                sev_by_cat = df.groupby(['AI_Category', 'tickets_data_severity']).size().reset_index(name='Count')

                fig_sev_cat = go.Figure()
                severity_colors = {'Critical': '#ef4444', 'Major': '#f97316', 'Minor': '#22c55e'}

                for severity in ['Critical', 'Major', 'Minor']:
                    sev_data = sev_by_cat[sev_by_cat['tickets_data_severity'] == severity]
                    if len(sev_data) > 0:
                        fig_sev_cat.add_trace(go.Bar(
                            name=severity,
                            x=sev_data['AI_Category'],
                            y=sev_data['Count'],
                            marker_color=severity_colors.get(severity, '#3b82f6'),
                            hovertemplate=f'<b>%{{x}}</b><br>{severity}: %{{y}}<extra></extra>'
                        ))

                fig_sev_cat.update_layout(
                    barmode='stack',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    margin=dict(t=10, b=80, l=50, r=20),
                    xaxis=dict(tickangle=-45, tickfont=dict(size=9, color='#94a3b8'), showgrid=False),
                    yaxis=dict(tickfont=dict(size=10, color='#94a3b8'), gridcolor='rgba(255,255,255,0.1)', title='Count'),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, font=dict(color='#94a3b8'))
                )
                st.plotly_chart(fig_sev_cat, use_container_width=True, key="sev_by_cat_deep")

        st.markdown("---")

        # ROW 3: Category drill-down with inline selector
        st.markdown("#### 🔍 Single Category Drill-Down")
        if 'AI_Category' in df.columns:
            categories = sorted(df['AI_Category'].unique().tolist())
            selected_cat = st.selectbox("Select a category to explore:", categories, key="deep_cat_select")

            cat_df = df[df['AI_Category'] == selected_cat]
            cat_cost = cat_df['Financial_Impact'].sum() if 'Financial_Impact' in cat_df.columns else 0
            cat_count = len(cat_df)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tickets", f"{cat_count:,}")
            m2.metric("Total Cost", f"${cat_cost:,.0f}")
            m3.metric("Avg Cost", f"${cat_cost/cat_count:,.0f}" if cat_count > 0 else "$0")
            m4.metric("% of Total", f"{cat_count/len(df)*100:.1f}%")

            # Sub-category breakdown - HORIZONTAL BAR with color coding
            if sub_cat_col:
                sub_data = cat_df.groupby(sub_cat_col).agg({
                    'Financial_Impact': 'sum',
                    'AI_Category': 'count'
                }).rename(columns={'AI_Category': 'Count'}).sort_values('Financial_Impact', ascending=True)

                # Color gradient based on cost
                max_cost = sub_data['Financial_Impact'].max()
                colors = [f'rgba({int(59 + 180 * (v/max_cost))}, {int(130 - 62 * (v/max_cost))}, {int(246 - 178 * (v/max_cost))}, 0.9)'
                         for v in sub_data['Financial_Impact']]

                fig_sub = go.Figure(data=[go.Bar(
                    y=sub_data.index,
                    x=sub_data['Financial_Impact'],
                    orientation='h',
                    marker=dict(
                        color=colors,
                        line=dict(color='rgba(255,255,255,0.3)', width=1)
                    ),
                    text=[f'${v:,.0f} ({c} tickets)' for v, c in zip(sub_data['Financial_Impact'], sub_data['Count'])],
                    textposition='outside',
                    textfont=dict(size=11, color='#e2e8f0'),
                    hovertemplate='<b>%{y}</b><br>Cost: $%{x:,.0f}<extra></extra>'
                )])
                fig_sub.update_layout(
                    title=dict(text=f"Sub-Categories in {selected_cat}", font=dict(size=14, color='#e2e8f0')),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=max(250, len(sub_data) * 40),
                    margin=dict(l=10, r=120, t=40, b=10),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)',
                              tickfont=dict(size=10, color='#94a3b8'), tickformat='$,.0f'),
                    yaxis=dict(showgrid=False, tickfont=dict(size=11, color='#e2e8f0')),
                    showlegend=False
                )
                st.plotly_chart(fig_sub, use_container_width=True)

    # ===== TAB 2: ENGINEERS =====
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(chart_engineer_performance(df), use_container_width=True)
        with col2:
            # Engineer Resolution Speed vs Volume Scatter
            eng_col = 'tickets_data_engineer_name' if 'tickets_data_engineer_name' in df.columns else 'tickets_data_issue_resolved_by'
            if eng_col in df.columns and 'Predicted_Resolution_Days' in df.columns:
                eng_metrics = df.groupby(eng_col).agg({
                    'Predicted_Resolution_Days': 'mean',
                    'Financial_Impact': ['sum', 'count'],
                    'AI_Recurrence_Risk': 'mean'
                }).reset_index()
                eng_metrics.columns = ['Engineer', 'Avg_Resolution', 'Total_Cost', 'Ticket_Count', 'Recurrence_Risk']
                eng_metrics = eng_metrics[eng_metrics['Ticket_Count'] >= 3]  # Filter low volume

                # Quadrant classification
                avg_res_median = eng_metrics['Avg_Resolution'].median()
                avg_risk_median = eng_metrics['Recurrence_Risk'].median()

                def get_quadrant(row):
                    fast = row['Avg_Resolution'] <= avg_res_median
                    quality = row['Recurrence_Risk'] <= avg_risk_median
                    if fast and quality: return 'Fast & Clean'
                    elif not fast and quality: return 'Slow but Thorough'
                    elif fast and not quality: return 'Fast but Sloppy'
                    else: return 'Needs Support'

                eng_metrics['Quadrant'] = eng_metrics.apply(get_quadrant, axis=1)
                quadrant_colors = {
                    'Fast & Clean': '#22c55e',
                    'Slow but Thorough': '#3b82f6',
                    'Fast but Sloppy': '#f97316',
                    'Needs Support': '#ef4444'
                }

                fig_quad = go.Figure()
                for quadrant in quadrant_colors:
                    q_data = eng_metrics[eng_metrics['Quadrant'] == quadrant]
                    if len(q_data) > 0:
                        fig_quad.add_trace(go.Scatter(
                            x=q_data['Avg_Resolution'],
                            y=q_data['Recurrence_Risk'] * 100,
                            mode='markers+text',
                            name=quadrant,
                            text=q_data['Engineer'].str.split().str[0],  # First name only
                            textposition='top center',
                            marker=dict(size=q_data['Ticket_Count'] * 2, color=quadrant_colors[quadrant], opacity=0.7),
                            hovertemplate='<b>%{text}</b><br>Resolution: %{x:.1f}d<br>Recurrence: %{y:.1f}%<extra></extra>'
                        ))

                fig_quad.add_vline(x=avg_res_median, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                fig_quad.add_hline(y=avg_risk_median * 100, line_dash="dash", line_color="rgba(255,255,255,0.3)")

                fig_quad.update_layout(
                    title=dict(text='Engineer Performance Quadrant', font=dict(size=14, color='#e2e8f0')),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    xaxis=dict(title='Avg Resolution (days)', gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#94a3b8')),
                    yaxis=dict(title='Recurrence Risk (%)', gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#94a3b8')),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color='#94a3b8', size=10))
                )
                st.plotly_chart(fig_quad, use_container_width=True)

                st.markdown("""
                <div style="background: rgba(0,0,0,0.2); border-radius: 8px; padding: 12px; margin-top: 10px;">
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; font-size: 0.8rem;">
                        <div><span style="color: #22c55e;">●</span> <b>Fast & Clean:</b> High performers</div>
                        <div><span style="color: #3b82f6;">●</span> <b>Slow but Thorough:</b> Quality focused</div>
                        <div><span style="color: #f97316;">●</span> <b>Fast but Sloppy:</b> Speed over quality</div>
                        <div><span style="color: #ef4444;">●</span> <b>Needs Support:</b> Training required</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Engineer performance data not available")

    # ===== TAB 3: ROOT CAUSE =====
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            render_chart_with_insight('pareto_analysis', chart_pareto_analysis(df), df)
        with col2:
            st.plotly_chart(chart_driver_tree(df), use_container_width=True)

        st.markdown("#### 📊 Root Cause Impact Quantification")
        col3, col4 = st.columns(2)
        with col3:
            # Financial impact by category
            if 'AI_Category' in df.columns and 'Financial_Impact' in df.columns:
                impact_data = df.groupby('AI_Category')['Financial_Impact'].sum().sort_values(ascending=True)
                fig_impact = go.Figure(data=[go.Bar(
                    y=impact_data.index,
                    x=impact_data.values,
                    orientation='h',
                    marker_color='#ef4444',
                    text=[f'${v:,.0f}' for v in impact_data.values],
                    textposition='outside'
                )])
                fig_impact.update_layout(
                    title="Financial Impact by Root Cause",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=350,
                    margin=dict(l=10, r=80, t=40, b=10),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickformat='$,.0f'),
                    yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig_impact, use_container_width=True)
        with col4:
            render_chart_with_insight('risk_heatmap', chart_risk_heatmap(df), df)

    # ===== TAB 4: PATTERNS & SLA =====
    with tabs[3]:
        try:
            from escalation_ai.advanced_insights import (
                chart_sla_funnel, chart_aging_analysis, chart_time_heatmap,
                chart_resolution_consistency, chart_recurrence_patterns
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 📊 SLA Compliance Funnel")
                st.plotly_chart(chart_sla_funnel(df), use_container_width=True)
            with col2:
                st.markdown("##### ⏱️ Ticket Aging Analysis")
                st.plotly_chart(chart_aging_analysis(df), use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                st.markdown("##### 🕐 Peak Escalation Times")
                st.plotly_chart(chart_time_heatmap(df), use_container_width=True)
            with col4:
                st.markdown("##### 🔄 Recurrence Patterns")
                st.plotly_chart(chart_recurrence_patterns(df), use_container_width=True)

        except ImportError:
            st.plotly_chart(chart_recurrence_risk(df), use_container_width=True)
            st.plotly_chart(chart_resolution_distribution(df), use_container_width=True)

    # ===== TAB 5: SIMILARITY =====
    with tabs[4]:
        st.markdown("#### 🔗 Similarity Search Analysis")
        st.markdown("*Insights from comparing tickets to historical patterns*")

        # Check if similarity data exists
        has_similarity = any(col in df.columns for col in ['Similar_Ticket_Count', 'Best_Match_Similarity', 'Similarity_Score', 'Resolution_Consistency'])

        if not has_similarity:
            st.info("🔍 **Similarity search data not available.**\n\nRun the analysis with similarity search enabled to populate this section.")
        else:
            # Similarity sub-tabs
            sim_tabs = st.tabs(["📊 Overview", "📈 Score Analysis", "⚖️ Consistency", "🔥 Heatmap"])

            with sim_tabs[0]:
                # Overview - count distribution and key metrics
                col1, col2, col3 = st.columns(3)

                if 'Similar_Ticket_Count' in df.columns:
                    counts = df['Similar_Ticket_Count'].dropna()
                    with col1:
                        avg_similar = counts.mean()
                        st.metric("Avg Similar Tickets", f"{avg_similar:.1f}")
                    with col2:
                        zero_matches = (counts == 0).sum()
                        st.metric("No Matches Found", f"{zero_matches}", delta=f"{zero_matches/len(counts)*100:.0f}%" if len(counts) > 0 else "0%", delta_color="inverse")
                    with col3:
                        high_matches = (counts >= 5).sum()
                        st.metric("Good Coverage (5+)", f"{high_matches}", delta=f"{high_matches/len(counts)*100:.0f}%" if len(counts) > 0 else "0%")

                st.markdown("---")

                # Count distribution chart
                if 'Similar_Ticket_Count' in df.columns:
                    fig_dist = go.Figure(data=[go.Histogram(
                        x=df['Similar_Ticket_Count'].dropna(),
                        nbinsx=20,
                        marker_color='#8b5cf6'
                    )])
                    fig_dist.update_layout(
                        title="Similar Ticket Count Distribution",
                        xaxis_title="Number of Similar Tickets",
                        yaxis_title="Frequency",
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        height=350
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

            with sim_tabs[1]:
                # Score distribution
                st.markdown("##### Similarity Score Analysis")
                st.markdown("*How confident are we in the similar ticket matches?*")

                score_col = 'Best_Match_Similarity' if 'Best_Match_Similarity' in df.columns else 'Similarity_Score' if 'Similarity_Score' in df.columns else None

                if score_col:
                    scores = df[score_col].dropna()
                    scores = scores[scores > 0]

                    if len(scores) > 0:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Average Score", f"{scores.mean():.2f}")
                        with col2:
                            high_conf = (scores >= 0.7).sum()
                            st.metric("High Confidence (≥0.7)", f"{high_conf}", delta=f"{high_conf/len(scores)*100:.0f}%")
                        with col3:
                            low_conf = (scores < 0.5).sum()
                            st.metric("Low Confidence (<0.5)", f"{low_conf}", delta=f"{low_conf/len(scores)*100:.0f}%", delta_color="inverse")

                        st.markdown("---")

                        # Score distribution histogram
                        fig_score = go.Figure(data=[go.Histogram(
                            x=scores,
                            nbinsx=20,
                            marker_color='#06b6d4'
                        )])
                        fig_score.update_layout(
                            title="Similarity Score Distribution",
                            xaxis_title="Similarity Score",
                            yaxis_title="Frequency",
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            height=350
                        )
                        st.plotly_chart(fig_score, use_container_width=True)

                        # Score by category
                        if 'AI_Category' in df.columns:
                            st.markdown("##### Similarity Scores by Category")
                            cat_scores = df.groupby('AI_Category')[score_col].agg(['mean', 'std', 'count'])
                            cat_scores = cat_scores[cat_scores['count'] >= 3].sort_values('mean', ascending=True)

                            fig_cat = go.Figure(data=[go.Bar(
                                y=cat_scores.index,
                                x=cat_scores['mean'],
                                orientation='h',
                                marker_color=['#22c55e' if x >= 0.7 else '#f97316' if x >= 0.5 else '#ef4444' for x in cat_scores['mean']],
                                error_x=dict(type='data', array=cat_scores['std'], visible=True),
                                text=[f"{x:.2f}" for x in cat_scores['mean']],
                                textposition='outside'
                            )])
                            fig_cat.update_layout(
                                title="Average Similarity Score by Category",
                                xaxis_title="Average Similarity Score",
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                height=max(300, len(cat_scores) * 35),
                                margin=dict(l=10, r=80)
                            )
                            st.plotly_chart(fig_cat, use_container_width=True)
                else:
                    st.info("Similarity score data not available. Run the analysis pipeline to generate Best_Match_Similarity.")

            with sim_tabs[2]:
                # Consistency analysis
                st.markdown("##### Resolution Consistency")
                st.markdown("*Are we resolving similar issues the same way?*")

                if 'Resolution_Consistency' in df.columns:
                    consistency_counts = df['Resolution_Consistency'].value_counts()

                    col1, col2 = st.columns(2)
                    with col1:
                        # Pie chart of consistency
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=consistency_counts.index,
                            values=consistency_counts.values,
                            marker_colors=['#22c55e' if 'Consistent' in str(l) else '#ef4444' for l in consistency_counts.index],
                            hole=0.4
                        )])
                        fig_pie.update_layout(
                            title="Resolution Consistency Distribution",
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            height=350
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col2:
                        # Consistency by category
                        if 'AI_Category' in df.columns:
                            cat_cons = df.groupby('AI_Category')['Resolution_Consistency'].apply(
                                lambda x: (x.str.contains('Consistent', case=False, na=False)).mean() * 100
                            ).sort_values(ascending=True)

                            fig_bar = go.Figure(data=[go.Bar(
                                y=cat_cons.index,
                                x=cat_cons.values,
                                orientation='h',
                                marker_color=['#22c55e' if x >= 70 else '#f97316' if x >= 50 else '#ef4444' for x in cat_cons.values],
                                text=[f"{x:.0f}%" for x in cat_cons.values],
                                textposition='outside'
                            )])
                            fig_bar.update_layout(
                                title="Consistency Rate by Category",
                                xaxis_title="Consistency %",
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                height=max(300, len(cat_cons) * 35),
                                margin=dict(l=10, r=60),
                                xaxis=dict(range=[0, 110])
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)

                    # Inconsistent tickets table
                    inconsistent = df[df['Resolution_Consistency'].str.contains('Inconsistent|Different', case=False, na=False)]
                    if len(inconsistent) > 0:
                        st.markdown("##### ⚠️ Tickets with Inconsistent Resolutions")
                        st.markdown(f"*{len(inconsistent)} tickets resolved differently than similar historical cases*")

                        display_cols = ['Identity', 'AI_Category', 'Similar_Ticket_Count']
                        if 'AI_Sub_Category' in df.columns:
                            display_cols.insert(2, 'AI_Sub_Category')
                        if 'Best_Match_Similarity' in df.columns:
                            display_cols.append('Best_Match_Similarity')
                        display_cols = [c for c in display_cols if c in inconsistent.columns]

                        if display_cols:
                            st.dataframe(
                                inconsistent[display_cols].head(20),
                                use_container_width=True,
                                hide_index=True
                            )
                else:
                    st.info("Resolution consistency data not available. Run similarity analysis with consistency checking enabled.")

            with sim_tabs[3]:
                # Effectiveness heatmap
                st.markdown("##### Similarity Search Effectiveness")
                st.markdown("*Where do we have good historical coverage?*")

                if 'AI_Category' in df.columns and 'Similar_Ticket_Count' in df.columns:
                    # Try to create heatmap with origin
                    origin_col = None
                    for col in ['AI_Origin', 'Origin', 'Source']:
                        if col in df.columns:
                            origin_col = col
                            break

                    if origin_col:
                        # Create heatmap
                        heatmap_data = df.pivot_table(
                            values='Similar_Ticket_Count',
                            index='AI_Category',
                            columns=origin_col,
                            aggfunc='mean'
                        ).fillna(0)

                        fig_heat = go.Figure(data=go.Heatmap(
                            z=heatmap_data.values,
                            x=heatmap_data.columns,
                            y=heatmap_data.index,
                            colorscale='Viridis',
                            text=np.round(heatmap_data.values, 1),
                            texttemplate="%{text}",
                            hovertemplate='Category: %{y}<br>Origin: %{x}<br>Avg Matches: %{z:.1f}<extra></extra>'
                        ))
                        fig_heat.update_layout(
                            title="Average Similar Ticket Count by Category & Origin",
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            height=max(400, len(heatmap_data) * 30)
                        )
                        st.plotly_chart(fig_heat, use_container_width=True)
                    else:
                        st.info("Origin column not found for heatmap. Showing coverage table instead.")

                    # Coverage summary table
                    st.markdown("##### Coverage by Category")
                    coverage = df.groupby('AI_Category').agg({
                        'Similar_Ticket_Count': ['mean', 'sum', 'count']
                    })
                    coverage.columns = ['Avg Matches', 'Total Matches', 'Ticket Count']
                    coverage['Coverage Score'] = (coverage['Avg Matches'] * 20).clip(0, 100).round(0).astype(int)
                    coverage = coverage.sort_values('Coverage Score', ascending=False)

                    # Color code the coverage score
                    st.dataframe(
                        coverage.style.background_gradient(subset=['Coverage Score'], cmap='RdYlGn'),
                        use_container_width=True
                    )
                else:
                    st.info("Need category and similar ticket count data for effectiveness analysis.")

    # ===== TAB 6: LESSONS LEARNED =====
    with tabs[5]:
        st.markdown("#### 📚 Learning Effectiveness Analysis")
        st.markdown("*Analyzing how well lessons are being learned and applied to prevent recurrence*")

        # Calculate learning grades using comprehensive function
        grades_data = _calculate_learning_grades(df)

        if grades_data:
            # Row 1: Recurrence vs Lessons Quadrant Chart + Grade Summary
            col1, col2 = st.columns([3, 2])

            with col1:
                # Recurrence vs Lesson Completion Quadrant
                fig_recurrence = chart_recurrence_vs_lessons(df)
                if fig_recurrence:
                    st.plotly_chart(fig_recurrence, use_container_width=True)
                else:
                    st.info("Insufficient data for recurrence analysis")

            with col2:
                # Learning Grade Summary Cards
                grades_summary = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
                for cat_data in grades_data.values():
                    grades_summary[cat_data['grade']] += 1

                st.markdown("""
                <div style="background: rgba(139, 92, 246, 0.1); border-radius: 12px; padding: 20px; border: 1px solid rgba(139, 92, 246, 0.3);">
                    <div style="color: #c4b5fd; font-size: 0.85rem; font-weight: 600; margin-bottom: 15px;">📊 Learning Grade Distribution</div>
                """, unsafe_allow_html=True)

                grade_colors = {'A': '#22c55e', 'B': '#3b82f6', 'C': '#f97316', 'D': '#ef4444', 'F': '#dc2626'}
                for grade, count in grades_summary.items():
                    if count > 0:
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; margin: 8px 0;">
                            <span style="background: {grade_colors[grade]}; color: white; padding: 4px 12px; border-radius: 4px; font-weight: 700;">Grade {grade}</span>
                            <span style="color: #e2e8f0; font-size: 1.2rem; font-weight: 600;">{count} categories</span>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

                # Key insight
                avg_recurrence = sum(d['recurrence_rate'] for d in grades_data.values()) / len(grades_data)
                st.markdown(f"""
                <div style="background: rgba(239, 68, 68, 0.1); border-radius: 8px; padding: 15px; margin-top: 15px; border: 1px solid rgba(239, 68, 68, 0.3);">
                    <div style="color: #fca5a5; font-size: 0.8rem;">Average Recurrence Rate</div>
                    <div style="color: #ef4444; font-size: 2rem; font-weight: 700;">{avg_recurrence:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # Scoring Criteria Breakdown
            with st.expander("📋 **How is the Learning Effectiveness Score Calculated?**", expanded=False):
                st.markdown("##### 📊 Score Components (Total: 100 points)")

                score_cols = st.columns(2)
                with score_cols[0]:
                    st.success("🔄 **Recurrence Score** (35% weight)")
                    st.caption("Formula: 100 - Recurrence Rate")
                    st.caption("*Lower recurrence = higher score*")

                    st.warning("⚙️ **Resolution Consistency** (25% weight)")
                    st.caption("% of tickets with consistent resolution")
                    st.caption("*More consistency = higher score*")

                with score_cols[1]:
                    st.info("📝 **Lesson Completion** (30% weight)")
                    st.caption("% of lessons marked complete/done/closed")
                    st.caption("*Higher completion = higher score*")

                    st.markdown("✅ **Documentation Bonus** (+10 points)")
                    st.caption("Awarded if any lessons are documented")
                    st.caption("*for the category*")

                st.markdown("---")
                st.markdown("##### 🎓 Grade Thresholds")
                grade_cols = st.columns(5)
                with grade_cols[0]:
                    st.markdown("🟢 **A**: ≥80")
                with grade_cols[1]:
                    st.markdown("🔵 **B**: 65-79")
                with grade_cols[2]:
                    st.markdown("🟠 **C**: 50-64")
                with grade_cols[3]:
                    st.markdown("🔴 **D**: 35-49")
                with grade_cols[4]:
                    st.markdown("⛔ **F**: <35")

            # Row 2: Category Learning Scorecard
            st.markdown("#### 🎯 Category Learning Scorecard")

            # Sort by score descending
            sorted_grades = sorted(grades_data.items(), key=lambda x: x[1]['score'], reverse=True)

            # Create scorecard visualization with detailed hover
            categories = [g[0] for g in sorted_grades]
            scores = [g[1]['score'] for g in sorted_grades]
            grades = [g[1]['grade'] for g in sorted_grades]

            # Build detailed hover data
            hover_texts = []
            for cat, data in sorted_grades:
                recurrence_score = max(0, 100 - data['recurrence_rate'])
                hover_texts.append(
                    f"<b>{cat}</b><br><br>"
                    f"<b>Total Score:</b> {data['score']:.1f}/100<br>"
                    f"<b>Grade:</b> {data['grade']}<br><br>"
                    f"<b>Score Breakdown:</b><br>"
                    f"• Recurrence (35%): {recurrence_score:.1f} pts<br>"
                    f"  └ Rate: {data['recurrence_rate']:.1f}%<br>"
                    f"• Lesson Completion (30%): {data['lesson_completion']:.1f} pts<br>"
                    f"• Consistency (25%): {data['consistency']:.1f} pts<br>"
                    f"• Documentation Bonus: {'+10' if data.get('lessons_documented', 0) > 0 else '0'} pts<br><br>"
                    f"<b>Tickets:</b> {data['ticket_count']}"
                )

            fig_scorecard = go.Figure()

            # Score bars
            fig_scorecard.add_trace(go.Bar(
                y=categories,
                x=scores,
                orientation='h',
                marker_color=[grade_colors[g] for g in grades],
                text=[f"{g} ({s:.0f})" for g, s in zip(grades, scores)],
                textposition='outside',
                hovertemplate='%{customdata}<extra></extra>',
                customdata=hover_texts
            ))

            fig_scorecard.update_layout(
                title=dict(text='Learning Effectiveness Score by Category', font=dict(size=14, color='#e2e8f0')),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=max(300, len(categories) * 35),
                margin=dict(l=10, r=80, t=40, b=10),
                xaxis=dict(range=[0, 110], gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#94a3b8'), title='Score'),
                yaxis=dict(tickfont=dict(color='#e2e8f0', size=10))
            )
            st.plotly_chart(fig_scorecard, use_container_width=True)

            # Row 3: At-Risk Categories and Recommendations
            st.markdown("#### ⚠️ Categories Needing Attention")

            at_risk = [(cat, data) for cat, data in grades_data.items()
                      if data['grade'] in ['D', 'F'] or data['recurrence_rate'] > 40]

            if at_risk:
                for cat, data in sorted(at_risk, key=lambda x: x[1]['score'])[:5]:
                    risk_color = '#ef4444' if data['grade'] == 'F' else '#f97316'
                    st.markdown(f"""
                    <div style="background: rgba(239, 68, 68, 0.1); border-radius: 8px; padding: 15px; margin: 10px 0; border-left: 4px solid {risk_color};">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <span style="color: #e2e8f0; font-weight: 600; font-size: 1.1rem;">{cat}</span>
                            <span style="background: {risk_color}; color: white; padding: 4px 12px; border-radius: 4px; font-weight: 700;">Grade {data['grade']}</span>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; font-size: 0.85rem;">
                            <div>
                                <div style="color: #94a3b8;">Recurrence Rate</div>
                                <div style="color: #ef4444; font-weight: 600;">{data['recurrence_rate']:.1f}%</div>
                            </div>
                            <div>
                                <div style="color: #94a3b8;">Lesson Completion</div>
                                <div style="color: {'#22c55e' if data['lesson_completion'] > 50 else '#f97316'}; font-weight: 600;">{data['lesson_completion']:.1f}%</div>
                            </div>
                            <div>
                                <div style="color: #94a3b8;">Tickets</div>
                                <div style="color: #3b82f6; font-weight: 600;">{data['ticket_count']}</div>
                            </div>
                        </div>
                        <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1);">
                            💡 <b>Action:</b> {'Urgent - Mandate lesson documentation and review root causes' if data['recurrence_rate'] > 50 else 'Review and improve lesson application process'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ All categories are performing well on learning effectiveness!")

            # Row 4: Similarity-Based "Lessons Not Learned" Analysis
            st.markdown("#### 🔄 Lessons Not Learned - Recurrence Despite Documentation")
            st.markdown("*Identifying cases where similar issues keep appearing despite having documented lessons*")

            # Find lesson column
            lesson_col = None
            for col in ['tickets_data_lessons_learned_title', 'tickets_data_lessons_learned_preventive_actions', 'Lesson_Title']:
                if col in df.columns:
                    lesson_col = col
                    break

            if lesson_col and 'Similar_Ticket_Count' in df.columns and 'tickets_data_issue_datetime' in df.columns:
                # Analyze if lessons are preventing recurrence
                df_analysis = df.copy()
                df_analysis['Has_Lesson'] = df_analysis[lesson_col].notna() & (df_analysis[lesson_col].astype(str).str.strip() != '')
                df_analysis['Has_Similar'] = df_analysis['Similar_Ticket_Count'] > 0
                df_analysis['Issue_Date'] = pd.to_datetime(df_analysis['tickets_data_issue_datetime'], errors='coerce')

                # Calculate metrics by category
                lesson_effectiveness = []
                for cat in df_analysis['AI_Category'].dropna().unique():
                    cat_df = df_analysis[df_analysis['AI_Category'] == cat]
                    if len(cat_df) < 3:
                        continue

                    total = len(cat_df)
                    with_lessons = cat_df['Has_Lesson'].sum()
                    with_similar = cat_df['Has_Similar'].sum()

                    # Key metric: Issues that have similar tickets (recurrence) but ALSO have lessons
                    # This indicates lessons exist but similar issues still appear
                    recurring_with_lessons = ((cat_df['Has_Similar']) & (cat_df['Has_Lesson'])).sum()
                    recurring_without_lessons = ((cat_df['Has_Similar']) & (~cat_df['Has_Lesson'])).sum()

                    # Calculate lesson effectiveness: lower recurrence when lessons exist = effective
                    if with_lessons > 0:
                        recurrence_with_lesson = recurring_with_lessons / with_lessons * 100
                    else:
                        recurrence_with_lesson = 0

                    without_lessons = total - with_lessons
                    if without_lessons > 0:
                        recurrence_without_lesson = recurring_without_lessons / without_lessons * 100
                    else:
                        recurrence_without_lesson = 0

                    # Effectiveness = how much lessons reduce recurrence
                    if recurrence_without_lesson > 0:
                        effectiveness = ((recurrence_without_lesson - recurrence_with_lesson) / recurrence_without_lesson) * 100
                    else:
                        effectiveness = 0 if recurrence_with_lesson > 0 else 100

                    lesson_effectiveness.append({
                        'category': cat,
                        'total_tickets': total,
                        'with_lessons': with_lessons,
                        'recurring_with_lessons': recurring_with_lessons,
                        'recurrence_with_lesson': recurrence_with_lesson,
                        'recurrence_without_lesson': recurrence_without_lesson,
                        'effectiveness': effectiveness,
                        'lesson_coverage': (with_lessons / total * 100) if total > 0 else 0
                    })

                if lesson_effectiveness:
                    # Sort by ineffectiveness (lessons not working)
                    lesson_effectiveness.sort(key=lambda x: x['effectiveness'])

                    col1, col2 = st.columns(2)

                    with col1:
                        # Chart: Lesson Effectiveness by Category
                        eff_df = pd.DataFrame(lesson_effectiveness)
                        eff_df = eff_df.sort_values('effectiveness', ascending=True)

                        fig_eff = go.Figure()
                        fig_eff.add_trace(go.Bar(
                            y=eff_df['category'],
                            x=eff_df['effectiveness'],
                            orientation='h',
                            marker_color=[
                                '#22c55e' if e >= 50 else '#f97316' if e >= 0 else '#ef4444'
                                for e in eff_df['effectiveness']
                            ],
                            text=[f"{e:.0f}%" for e in eff_df['effectiveness']],
                            textposition='outside',
                            hovertemplate='<b>%{y}</b><br>Effectiveness: %{x:.1f}%<br>Recurrence WITH lessons: %{customdata[0]:.1f}%<br>Recurrence WITHOUT lessons: %{customdata[1]:.1f}%<extra></extra>',
                            customdata=list(zip(eff_df['recurrence_with_lesson'], eff_df['recurrence_without_lesson']))
                        ))
                        fig_eff.update_layout(
                            title='Lesson Effectiveness by Category<br><sub>Negative = lessons not preventing recurrence</sub>',
                            xaxis_title='Effectiveness %',
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            height=max(300, len(eff_df) * 35),
                            margin=dict(l=10, r=80, t=60, b=30)
                        )
                        st.plotly_chart(fig_eff, use_container_width=True)

                    with col2:
                        # Metrics summary
                        avg_effectiveness = np.mean([e['effectiveness'] for e in lesson_effectiveness])
                        total_recurring_with_lessons = sum(e['recurring_with_lessons'] for e in lesson_effectiveness)
                        total_with_lessons = sum(e['with_lessons'] for e in lesson_effectiveness)

                        st.metric("Avg Lesson Effectiveness", f"{avg_effectiveness:.1f}%",
                                 delta="Good" if avg_effectiveness > 30 else "Needs Improvement",
                                 delta_color="normal" if avg_effectiveness > 30 else "inverse")

                        st.metric("Issues Recurring Despite Lessons", f"{total_recurring_with_lessons}",
                                 delta=f"of {total_with_lessons} with lessons",
                                 delta_color="inverse")

                        # Worst offenders
                        st.markdown("##### ⚠️ Lessons Not Working")
                        for item in lesson_effectiveness[:3]:
                            if item['effectiveness'] < 30 and item['recurring_with_lessons'] > 0:
                                st.markdown(f"""
                                <div style="background: rgba(239, 68, 68, 0.1); border-radius: 6px; padding: 10px; margin: 5px 0; border-left: 3px solid #ef4444;">
                                    <div style="color: #fca5a5; font-weight: 600;">{item['category']}</div>
                                    <div style="color: #94a3b8; font-size: 0.8rem;">{item['recurring_with_lessons']} issues recurred despite having lessons</div>
                                </div>
                                """, unsafe_allow_html=True)

                    # Detailed table
                    with st.expander("📊 **Detailed Lesson Effectiveness Data**"):
                        eff_display = pd.DataFrame(lesson_effectiveness)
                        eff_display = eff_display.rename(columns={
                            'category': 'Category',
                            'total_tickets': 'Total Tickets',
                            'with_lessons': 'With Lessons',
                            'recurring_with_lessons': 'Recurring (With Lessons)',
                            'recurrence_with_lesson': 'Recurrence % (With)',
                            'recurrence_without_lesson': 'Recurrence % (Without)',
                            'effectiveness': 'Effectiveness %',
                            'lesson_coverage': 'Lesson Coverage %'
                        })
                        eff_display = eff_display.round(1)
                        st.dataframe(eff_display, use_container_width=True, hide_index=True)

            elif 'Similar_Ticket_Count' in df.columns:
                st.info("Lesson documentation column not found. Add lessons_learned data to enable effectiveness analysis.")
            else:
                st.info("Similar ticket analysis required for lessons effectiveness tracking. Run the similarity analysis pipeline.")

        else:
            # Fallback to basic lessons display
            lessons_col = 'tickets_data_lessons_learned_preventive_actions'
            if lessons_col in df.columns:
                df_lessons = df[df[lessons_col].notna() & (df[lessons_col].astype(str) != '')]
                st.markdown(f"**{len(df_lessons)}** records have documented lessons learned")

                if len(df_lessons) > 0:
                    st.markdown("#### 📚 Recent Lessons")
                    for _, row in df_lessons.head(5).iterrows():
                        lesson_text = str(row[lessons_col])[:200]
                        st.markdown(f"""
                        <div style="background: rgba(15, 23, 42, 0.6); border-radius: 8px; padding: 12px; margin: 8px 0; border-left: 3px solid #8b5cf6;">
                            <div style="color: #c4b5fd; font-weight: 600;">{row.get('AI_Category', 'Unknown')}</div>
                            <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 5px;">{lesson_text}...</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No lessons learned data available")


# ============================================================================
# CONSOLIDATED TAB 4: BENCHMARKING & MONITORING
# ============================================================================

def render_benchmarking_monitoring(df):
    """Render consolidated Benchmarking, Alerts, and Drift Detection page."""
    render_spectacular_header("Benchmarking & Monitoring", "Performance tracking against standards and thresholds", "🏆")

    tabs = st.tabs(["🏆 Industry Benchmarks", "⚠️ Alert Thresholds", "📊 Drift Detection"])

    # ===== TAB 1: BENCHMARKING =====
    with tabs[0]:
        recurrence_rate = df['AI_Recurrence_Risk'].mean() if 'AI_Recurrence_Risk' in df.columns else 0.15
        resolution_days = df['Predicted_Resolution_Days'].mean() if 'Predicted_Resolution_Days' in df.columns else 5

        current_metrics = {
            'Resolution Time': {'value': resolution_days, 'unit': 'days', 'best': 2, 'avg': 5, 'lower_better': True},
            'Recurrence Rate': {'value': recurrence_rate * 100, 'unit': '%', 'best': 5, 'avg': 15, 'lower_better': True},
            'SLA Breach': {'value': 12, 'unit': '%', 'best': 5, 'avg': 15, 'lower_better': True},
            'First Contact Resolution': {'value': 45, 'unit': '%', 'best': 70, 'avg': 50, 'lower_better': False},
            'Cost per Escalation': {'value': df['Financial_Impact'].mean() if 'Financial_Impact' in df.columns else get_benchmark_costs()['avg_per_ticket'], 'unit': '$', 'best': get_benchmark_costs()['best_in_class'], 'avg': get_benchmark_costs()['industry_avg'], 'lower_better': True},
            'Customer Satisfaction': {'value': 72, 'unit': '%', 'best': 90, 'avg': 75, 'lower_better': False}
        }

        cols = st.columns(3)
        for i, (metric_name, data) in enumerate(current_metrics.items()):
            with cols[i % 3]:
                value = data['value']
                best = data['best']
                avg = data['avg']

                if data['lower_better']:
                    position = 'Best-in-Class' if value <= best else 'Above Average' if value <= avg else 'Below Average'
                    color = '#22c55e' if value <= best else '#f97316' if value <= avg else '#ef4444'
                else:
                    position = 'Best-in-Class' if value >= best else 'Above Average' if value >= avg else 'Below Average'
                    color = '#22c55e' if value >= best else '#f97316' if value >= avg else '#ef4444'

                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.6); border-radius: 12px; padding: 20px; margin: 10px 0;
                            border: 1px solid {color}40;">
                    <div style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">{metric_name}</div>
                    <div style="color: {color}; font-size: 2rem; font-weight: 700; margin: 8px 0;">
                        {f'${value:,.0f}' if data['unit'] == '$' else f'{value:.1f}{data["unit"]}'}
                    </div>
                    <div style="color: #64748b; font-size: 0.75rem;">
                        Best: {f'${best}' if data['unit'] == '$' else f'{best}{data["unit"]}'} |
                        Avg: {f'${avg}' if data['unit'] == '$' else f'{avg}{data["unit"]}'}
                    </div>
                    <div style="color: {color}; font-size: 0.85rem; margin-top: 8px; font-weight: 600;">{position}</div>
                </div>
                """, unsafe_allow_html=True)

    # ===== TAB 2: ALERTS =====
    with tabs[1]:
        date_col = None
        for col in ['Issue_Date', 'tickets_data_issue_datetime', 'Created_Date', 'Date']:
            if col in df.columns:
                date_col = col
                break

        if date_col:
            df_alert = df.copy()
            df_alert['date'] = pd.to_datetime(df_alert[date_col]).dt.date

            daily_metrics = df_alert.groupby('date').agg({
                'Financial_Impact': 'sum',
                'Strategic_Friction_Score': 'mean' if 'Strategic_Friction_Score' in df.columns else 'count',
                'AI_Recurrence_Risk': 'mean' if 'AI_Recurrence_Risk' in df.columns else 'count'
            }).reset_index()
            daily_metrics['escalation_count'] = df_alert.groupby('date').size().values

            # Calculate thresholds
            esc_warning = daily_metrics['escalation_count'].quantile(0.75)
            esc_critical = daily_metrics['escalation_count'].quantile(0.90)
            current_esc = daily_metrics['escalation_count'].iloc[-1] if len(daily_metrics) > 0 else 0

            col1, col2, col3 = st.columns(3)

            with col1:
                status = 'Critical' if current_esc > esc_critical else 'Warning' if current_esc > esc_warning else 'Normal'
                color = '#ef4444' if status == 'Critical' else '#f97316' if status == 'Warning' else '#22c55e'
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.6); border-radius: 12px; padding: 20px; border: 2px solid {color};">
                    <div style="color: #94a3b8; font-size: 0.8rem;">DAILY ESCALATIONS</div>
                    <div style="color: {color}; font-size: 2.5rem; font-weight: 700;">{current_esc:.0f}</div>
                    <div style="color: {color}; font-size: 0.9rem;">● {status}</div>
                    <div style="color: #64748b; font-size: 0.7rem; margin-top: 8px;">
                        Warning: >{esc_warning:.0f} | Critical: >{esc_critical:.0f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                if 'Strategic_Friction_Score' in df.columns:
                    friction_warning = daily_metrics['Strategic_Friction_Score'].quantile(0.75)
                    friction_critical = daily_metrics['Strategic_Friction_Score'].quantile(0.90)
                    current_friction = daily_metrics['Strategic_Friction_Score'].iloc[-1] if len(daily_metrics) > 0 else 0
                    status = 'Critical' if current_friction > friction_critical else 'Warning' if current_friction > friction_warning else 'Normal'
                    color = '#ef4444' if status == 'Critical' else '#f97316' if status == 'Warning' else '#22c55e'
                    st.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.6); border-radius: 12px; padding: 20px; border: 2px solid {color};">
                        <div style="color: #94a3b8; font-size: 0.8rem;">DAILY FRICTION SCORE</div>
                        <div style="color: {color}; font-size: 2.5rem; font-weight: 700;">{current_friction:.1f}</div>
                        <div style="color: {color}; font-size: 0.9rem;">● {status}</div>
                    </div>
                    """, unsafe_allow_html=True)

            with col3:
                if 'AI_Recurrence_Risk' in df.columns:
                    risk_warning = daily_metrics['AI_Recurrence_Risk'].quantile(0.75) * 100
                    risk_critical = daily_metrics['AI_Recurrence_Risk'].quantile(0.90) * 100
                    current_risk = (daily_metrics['AI_Recurrence_Risk'].iloc[-1] if len(daily_metrics) > 0 else 0) * 100
                    status = 'Critical' if current_risk > risk_critical else 'Warning' if current_risk > risk_warning else 'Normal'
                    color = '#ef4444' if status == 'Critical' else '#f97316' if status == 'Warning' else '#22c55e'
                    st.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.6); border-radius: 12px; padding: 20px; border: 2px solid {color};">
                        <div style="color: #94a3b8; font-size: 0.8rem;">RECURRENCE RISK</div>
                        <div style="color: {color}; font-size: 2.5rem; font-weight: 700;">{current_risk:.1f}%</div>
                        <div style="color: {color}; font-size: 0.9rem;">● {status}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Timeline chart
            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Scatter(
                x=daily_metrics['date'], y=daily_metrics['escalation_count'],
                mode='lines+markers', name='Escalations', line=dict(color='#3b82f6')
            ))
            fig_timeline.add_hline(y=esc_warning, line_dash="dash", line_color="#f97316", annotation_text="Warning")
            fig_timeline.add_hline(y=esc_critical, line_dash="dash", line_color="#ef4444", annotation_text="Critical")
            fig_timeline.update_layout(
                title="Escalation Trend with Thresholds",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=350
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.warning("Date column not found for alert analysis")

    # ===== TAB 3: DRIFT DETECTION =====
    with tabs[2]:
        date_col = None
        for col in ['Issue_Date', 'tickets_data_issue_datetime', 'Created_Date']:
            if col in df.columns:
                date_col = col
                break

        if date_col and 'AI_Category' in df.columns:
            df_drift = df.copy()
            df_drift['date'] = pd.to_datetime(df_drift[date_col])
            df_drift = df_drift.sort_values('date')

            # Split into baseline and current
            split_idx = int(len(df_drift) * 0.6)
            baseline = df_drift.iloc[:split_idx]
            current = df_drift.iloc[split_idx:]

            baseline_dist = baseline['AI_Category'].value_counts(normalize=True)
            current_dist = current['AI_Category'].value_counts(normalize=True)

            all_cats = sorted(set(baseline_dist.index) | set(current_dist.index))

            fig_drift = go.Figure()
            fig_drift.add_trace(go.Bar(
                name='Baseline (60%)',
                x=all_cats,
                y=[baseline_dist.get(c, 0) * 100 for c in all_cats],
                marker_color='#3b82f6'
            ))
            fig_drift.add_trace(go.Bar(
                name='Current (40%)',
                x=all_cats,
                y=[current_dist.get(c, 0) * 100 for c in all_cats],
                marker_color='#22c55e'
            ))
            fig_drift.update_layout(
                title="Category Distribution: Baseline vs Current",
                barmode='group',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=400,
                xaxis_tickangle=-45,
                yaxis_title="Percentage (%)"
            )
            st.plotly_chart(fig_drift, use_container_width=True)

            # Drift metrics
            st.markdown("#### 📊 Drift Analysis")
            drift_data = []
            for cat in all_cats:
                b_pct = baseline_dist.get(cat, 0) * 100
                c_pct = current_dist.get(cat, 0) * 100
                change = c_pct - b_pct
                drift_data.append({
                    'Category': cat,
                    'Baseline %': f'{b_pct:.1f}%',
                    'Current %': f'{c_pct:.1f}%',
                    'Change': f'{change:+.1f}%',
                    'Status': '📈 Increasing' if change > 2 else '📉 Decreasing' if change < -2 else '➡️ Stable'
                })

            st.dataframe(pd.DataFrame(drift_data), use_container_width=True, hide_index=True)
        else:
            st.warning("Date and Category columns required for drift detection")


# ============================================================================
# CONSOLIDATED TAB 5: PLANNING & ACTIONS
# ============================================================================

def render_planning_actions(df):
    """Render consolidated Planning & Actions page (What-If + Action Tracker)."""
    render_spectacular_header("Planning & Actions", "Scenario modeling and initiative tracking", "🎯")

    tabs = st.tabs(["🔮 What-If Simulator", "📋 Action Tracker", "📚 Learning-Based Actions"])

    # ===== TAB 1: WHAT-IF SIMULATOR =====
    with tabs[0]:
        st.markdown("#### Adjust parameters to simulate impact on escalation metrics")

        recurrence_rate = df['AI_Recurrence_Risk'].mean() if 'AI_Recurrence_Risk' in df.columns else 0.15
        avg_resolution = df['Predicted_Resolution_Days'].mean() if 'Predicted_Resolution_Days' in df.columns else 5
        friction_sum = df['Strategic_Friction_Score'].sum() if 'Strategic_Friction_Score' in df.columns else 3000
        cost_sum = df['Financial_Impact'].sum() if 'Financial_Impact' in df.columns else 375000

        # Calculate current lesson coverage
        lesson_col = None
        for col in ['tickets_data_lessons_learned_title', 'tickets_data_lessons_learned_preventive_actions']:
            if col in df.columns:
                lesson_col = col
                break
        current_lesson_coverage = (df[lesson_col].notna().sum() / len(df) * 100) if lesson_col else 0

        # Calculate similarity-based recurrence
        similarity_recurrence = (df['Similar_Ticket_Count'] > 0).mean() * 100 if 'Similar_Ticket_Count' in df.columns else recurrence_rate * 100

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 🎛️ Scenario Parameters")
            staffing = st.slider("Staffing Changes (engineers)", -5, 10, 0, key="whatif_staff")
            training = st.slider("Training Impact (% error reduction)", 0, 50, 0, key="whatif_train")
            volume = st.slider("Volume Changes (%)", -30, 50, 0, key="whatif_vol")
            process = st.slider("Process Improvements (% efficiency)", 0, 40, 0, key="whatif_proc")

            st.markdown("##### 📚 Learning & Knowledge Parameters")
            lesson_improvement = st.slider("Lesson Documentation Rate (+%)", 0, 50, 0, key="whatif_lesson",
                                          help=f"Current: {current_lesson_coverage:.0f}% coverage")
            lesson_application = st.slider("Lesson Application Effectiveness (%)", 0, 80, 0, key="whatif_apply",
                                          help="How effectively lessons prevent recurrence")

        with col2:
            st.markdown("##### 📊 Projected Impact")

            # Calculate projections
            staff_factor = 1 - (staffing * 0.03)
            training_factor = 1 - (training / 100)
            volume_factor = 1 + (volume / 100)
            process_factor = 1 - (process / 100)

            # Lesson-based recurrence reduction
            # More lessons + better application = lower recurrence
            lesson_coverage_factor = 1 - (lesson_improvement / 100 * 0.3)  # Each % of lesson coverage reduces recurrence by 0.3%
            lesson_application_factor = 1 - (lesson_application / 100 * 0.5)  # Application effectiveness has bigger impact

            proj_resolution = avg_resolution * staff_factor * process_factor
            proj_recurrence = recurrence_rate * training_factor * lesson_coverage_factor * lesson_application_factor
            proj_similarity_recurrence = similarity_recurrence * lesson_application_factor * training_factor
            proj_friction = friction_sum * volume_factor * process_factor / len(df)
            proj_cost = cost_sum * volume_factor * staff_factor * process_factor * lesson_application_factor / len(df)

            metrics = [
                ("Resolution Time", f"{avg_resolution:.1f}d", f"{proj_resolution:.1f}d", proj_resolution < avg_resolution),
                ("Recurrence Rate", f"{recurrence_rate*100:.1f}%", f"{proj_recurrence*100:.1f}%", proj_recurrence < recurrence_rate),
                ("Similar Issue Rate", f"{similarity_recurrence:.1f}%", f"{proj_similarity_recurrence:.1f}%", proj_similarity_recurrence < similarity_recurrence),
                ("Avg Friction", f"{friction_sum/len(df):.1f}", f"{proj_friction:.1f}", proj_friction < friction_sum/len(df)),
                ("Avg Cost", f"${cost_sum/len(df):,.0f}", f"${proj_cost:,.0f}", proj_cost < cost_sum/len(df))
            ]

            for name, baseline, projected, is_better in metrics:
                color = '#22c55e' if is_better else '#ef4444'
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.6); border-radius: 8px; padding: 12px; margin: 8px 0;
                            display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #94a3b8;">{name}</span>
                    <div>
                        <span style="color: #64748b; text-decoration: line-through; margin-right: 10px;">{baseline}</span>
                        <span style="color: {color}; font-weight: 700;">{projected}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Show potential savings
            if lesson_improvement > 0 or lesson_application > 0:
                base_annual_cost = cost_sum * 4  # Assume quarterly data
                projected_annual_cost = proj_cost * len(df) * 4
                savings = base_annual_cost - projected_annual_cost
                if savings > 0:
                    st.markdown(f"""
                    <div style="background: rgba(34, 197, 94, 0.1); border-radius: 8px; padding: 15px; margin-top: 15px; border: 1px solid rgba(34, 197, 94, 0.3);">
                        <div style="color: #86efac; font-size: 0.85rem;">💰 Projected Annual Savings from Learning Improvements</div>
                        <div style="color: #22c55e; font-size: 1.8rem; font-weight: 700;">${savings:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ===== TAB 2: ACTION TRACKER =====
    with tabs[1]:
        # Top Systemic Issues
        st.markdown("#### 🎯 Top 5 Systemic Issues")
        st.markdown("*Data-driven analysis of highest-impact recurring problems*")

        if 'AI_Category' in df.columns:
            issue_analysis = df.groupby('AI_Category').agg({
                'Financial_Impact': ['sum', 'count'],
                'AI_Recurrence_Risk': 'mean' if 'AI_Recurrence_Risk' in df.columns else 'count'
            }).reset_index()
            issue_analysis.columns = ['Category', 'Total_Cost', 'Count', 'Recurrence']
            issue_analysis = issue_analysis.sort_values('Total_Cost', ascending=False).head(5)

            for idx, row in issue_analysis.iterrows():
                rec = generate_strategic_recommendations(df[df['AI_Category'] == row['Category']])
                fix = rec[0]['description'] if rec else "Implement process improvements"

                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.6); border-radius: 12px; padding: 16px; margin: 10px 0;
                            border-left: 4px solid #ef4444;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="color: #e2e8f0; font-weight: 700; font-size: 1.1rem;">{row['Category']}</span>
                            <span style="color: #64748b; margin-left: 15px;">{row['Count']} tickets</span>
                        </div>
                        <span style="color: #ef4444; font-weight: 700; font-size: 1.2rem;">${row['Total_Cost']:,.0f}</span>
                    </div>
                    <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 8px;">
                        <b>Recommended Fix:</b> {fix[:150]}...
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Initiative Management
        st.markdown("---")
        st.markdown("#### 📋 Initiative Management")

        if 'action_items' not in st.session_state:
            st.session_state.action_items = []

        # Add new initiative
        with st.expander("➕ Add New Initiative"):
            new_title = st.text_input("Initiative Title", key="new_init_title")
            new_priority = st.selectbox("Priority", ["P1", "P2", "P3"], key="new_init_priority")
            new_owner = st.text_input("Owner", key="new_init_owner")
            new_due = st.date_input("Due Date", key="new_init_due")

            if st.button("Add Initiative", key="add_init_btn"):
                if new_title:
                    st.session_state.action_items.append({
                        'title': new_title,
                        'priority': new_priority,
                        'owner': new_owner,
                        'due': str(new_due),
                        'status': 'Not Started',
                        'progress': 0
                    })
                    st.success(f"Added: {new_title}")
                    st.rerun()

        # Display initiatives
        for i, item in enumerate(st.session_state.action_items):
            priority_color = '#ef4444' if item['priority'] == 'P1' else '#f97316' if item['priority'] == 'P2' else '#3b82f6'
            status_color = '#22c55e' if item['status'] == 'Complete' else '#f97316' if item['status'] == 'In Progress' else '#64748b'

            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.6); border-radius: 8px; padding: 12px;
                            border-left: 4px solid {priority_color};">
                    <span style="background: {priority_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;">{item['priority']}</span>
                    <span style="color: #e2e8f0; font-weight: 600; margin-left: 10px;">{item['title']}</span>
                    <div style="color: #64748b; font-size: 0.8rem; margin-top: 5px;">
                        Owner: {item['owner']} | Due: {item['due']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                new_status = st.selectbox("Status", ["Not Started", "In Progress", "Complete"],
                                         index=["Not Started", "In Progress", "Complete"].index(item['status']),
                                         key=f"status_{i}", label_visibility="collapsed")
                if new_status != item['status']:
                    st.session_state.action_items[i]['status'] = new_status
                    st.rerun()
            with col3:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state.action_items.pop(i)
                    st.rerun()

    # ===== TAB 3: LEARNING-BASED ACTIONS =====
    with tabs[2]:
        st.markdown("#### 📚 AI-Generated Actions from Learning Analysis")
        st.markdown("*Prioritized actions based on similarity patterns and lesson effectiveness*")

        # Find lesson column
        lesson_col = None
        for col in ['tickets_data_lessons_learned_title', 'tickets_data_lessons_learned_preventive_actions', 'Lesson_Title']:
            if col in df.columns:
                lesson_col = col
                break

        if lesson_col and 'Similar_Ticket_Count' in df.columns and 'AI_Category' in df.columns:
            # Calculate lesson effectiveness per category
            df_analysis = df.copy()
            df_analysis['Has_Lesson'] = df_analysis[lesson_col].notna() & (df_analysis[lesson_col].astype(str).str.strip() != '')
            df_analysis['Has_Similar'] = df_analysis['Similar_Ticket_Count'] > 0

            action_items = []

            for cat in df_analysis['AI_Category'].dropna().unique():
                cat_df = df_analysis[df_analysis['AI_Category'] == cat]
                if len(cat_df) < 3:
                    continue

                total = len(cat_df)
                with_lessons = cat_df['Has_Lesson'].sum()
                recurring_with_lessons = ((cat_df['Has_Similar']) & (cat_df['Has_Lesson'])).sum()
                recurring_without_lessons = ((cat_df['Has_Similar']) & (~cat_df['Has_Lesson'])).sum()

                lesson_coverage = (with_lessons / total * 100) if total > 0 else 0
                recurrence_rate = (cat_df['Has_Similar'].sum() / total * 100) if total > 0 else 0

                # Calculate cost impact
                cat_cost = cat_df['Financial_Impact'].sum() if 'Financial_Impact' in cat_df.columns else 0

                # Determine action priority and type
                priority = None
                action_type = None
                action_desc = None
                potential_savings = 0

                # Case 1: High recurrence, low lesson coverage - Need to document lessons
                if recurrence_rate > 50 and lesson_coverage < 30:
                    priority = "P1"
                    action_type = "📝 Document Lessons"
                    action_desc = f"Only {lesson_coverage:.0f}% of tickets have lessons but {recurrence_rate:.0f}% are recurring. Mandate lesson documentation for all resolved tickets."
                    potential_savings = cat_cost * 0.3  # 30% cost reduction potential

                # Case 2: Lessons exist but not working - Need to improve lesson quality/application
                elif with_lessons > 0 and recurring_with_lessons > with_lessons * 0.4:
                    priority = "P1"
                    action_type = "🔄 Improve Lesson Application"
                    action_desc = f"{recurring_with_lessons} issues recurred despite having lessons. Review lesson quality and ensure teams are applying documented solutions."
                    potential_savings = cat_cost * 0.25

                # Case 3: Moderate recurrence, some lessons - Need better knowledge sharing
                elif recurrence_rate > 30 and lesson_coverage > 30 and lesson_coverage < 70:
                    priority = "P2"
                    action_type = "📢 Knowledge Sharing"
                    action_desc = f"Lessons exist ({lesson_coverage:.0f}% coverage) but recurrence is {recurrence_rate:.0f}%. Improve cross-team knowledge sharing and training."
                    potential_savings = cat_cost * 0.15

                # Case 4: Low lesson coverage, moderate issues
                elif lesson_coverage < 40 and total >= 10:
                    priority = "P2"
                    action_type = "📝 Document Lessons"
                    action_desc = f"Low lesson coverage ({lesson_coverage:.0f}%) for {total} tickets. Establish lesson documentation as part of resolution workflow."
                    potential_savings = cat_cost * 0.2

                # Case 5: Good lesson coverage but could improve consistency
                elif 'Resolution_Consistency' in df.columns:
                    inconsistent = cat_df['Resolution_Consistency'].str.contains('Inconsistent', na=False).sum()
                    if inconsistent > total * 0.2:
                        priority = "P3"
                        action_type = "⚙️ Standardize Resolution"
                        action_desc = f"{inconsistent} tickets ({inconsistent/total*100:.0f}%) have inconsistent resolutions. Create standard operating procedures."
                        potential_savings = cat_cost * 0.1

                if priority:
                    action_items.append({
                        'category': cat,
                        'priority': priority,
                        'action_type': action_type,
                        'description': action_desc,
                        'ticket_count': total,
                        'recurrence_rate': recurrence_rate,
                        'lesson_coverage': lesson_coverage,
                        'potential_savings': potential_savings,
                        'cost': cat_cost
                    })

            # Sort by priority then by potential savings
            priority_order = {'P1': 0, 'P2': 1, 'P3': 2}
            action_items.sort(key=lambda x: (priority_order.get(x['priority'], 3), -x['potential_savings']))

            if action_items:
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    p1_count = sum(1 for a in action_items if a['priority'] == 'P1')
                    st.metric("🔴 P1 Actions", p1_count, help="Critical - immediate action needed")
                with col2:
                    p2_count = sum(1 for a in action_items if a['priority'] == 'P2')
                    st.metric("🟠 P2 Actions", p2_count, help="Important - plan within 30 days")
                with col3:
                    total_savings = sum(a['potential_savings'] for a in action_items)
                    st.metric("💰 Potential Savings", f"${total_savings:,.0f}")

                st.markdown("---")

                # Display actions
                for item in action_items[:10]:
                    priority_color = '#ef4444' if item['priority'] == 'P1' else '#f97316' if item['priority'] == 'P2' else '#3b82f6'

                    st.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.6); border-radius: 12px; padding: 16px; margin: 12px 0;
                                border-left: 4px solid {priority_color};">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px;">
                            <div>
                                <span style="background: {priority_color}; color: white; padding: 3px 10px; border-radius: 4px; font-size: 0.75rem; font-weight: 700;">{item['priority']}</span>
                                <span style="color: #60a5fa; margin-left: 10px; font-size: 0.9rem;">{item['action_type']}</span>
                            </div>
                            <span style="color: #22c55e; font-weight: 600;">Save ${item['potential_savings']:,.0f}</span>
                        </div>
                        <div style="color: #e2e8f0; font-weight: 600; font-size: 1.1rem; margin-bottom: 8px;">{item['category']}</div>
                        <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 10px;">{item['description']}</div>
                        <div style="display: flex; gap: 20px; font-size: 0.8rem; color: #64748b;">
                            <span>📊 {item['ticket_count']} tickets</span>
                            <span>🔄 {item['recurrence_rate']:.0f}% recurrence</span>
                            <span>📝 {item['lesson_coverage']:.0f}% lesson coverage</span>
                            <span>💰 ${item['cost']:,.0f} total cost</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Export actions as table
                with st.expander("📋 Export Action Items"):
                    actions_df = pd.DataFrame(action_items)
                    actions_df = actions_df.rename(columns={
                        'category': 'Category',
                        'priority': 'Priority',
                        'action_type': 'Action Type',
                        'description': 'Description',
                        'ticket_count': 'Tickets',
                        'recurrence_rate': 'Recurrence %',
                        'lesson_coverage': 'Lesson Coverage %',
                        'potential_savings': 'Potential Savings',
                        'cost': 'Total Cost'
                    })
                    st.dataframe(actions_df, use_container_width=True, hide_index=True)

            else:
                st.success("✅ No critical learning-based actions identified. Categories are performing well!")

        else:
            st.info("📊 This analysis requires:\n- Lessons learned data column\n- Similar_Ticket_Count from similarity analysis\n- AI_Category classification\n\nRun the full analysis pipeline to enable learning-based action recommendations.")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point."""
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Escalation AI")
        st.markdown("*Executive Intelligence Platform*")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["📊 Executive Dashboard", "📈 Deep Analysis", "💰 Financial Intelligence",
             "🏆 Benchmarking & Monitoring", "🎯 Planning & Actions", "📽️ Presentation Mode"],
            label_visibility="collapsed"
        )
        
        # Load data from default source
        df, data_source = load_data()

        # Excel-style filters (shown for Executive Dashboard page)
        if page == "📊 Executive Dashboard":
            st.markdown("---")
            st.markdown("### Add filter(s)")

            # Category filter (like Regions)
            st.markdown('<div class="excel-filter-section">', unsafe_allow_html=True)
            st.markdown('<div class="excel-filter-title">Categories</div>', unsafe_allow_html=True)
            if 'AI_Category' in df.columns:
                all_categories = sorted(df['AI_Category'].unique().tolist())
                selected_categories = st.multiselect(
                    "Select Categories",
                    options=all_categories,
                    default=all_categories,
                    key="excel_cat_filter",
                    label_visibility="collapsed"
                )
                if selected_categories:
                    df = df[df['AI_Category'].isin(selected_categories)]
            st.markdown('</div>', unsafe_allow_html=True)

            # Year filter
            st.markdown('<div class="excel-filter-section">', unsafe_allow_html=True)
            st.markdown('<div class="excel-filter-title">Year</div>', unsafe_allow_html=True)
            if 'tickets_data_issue_datetime' in df.columns:
                df_temp_year = df.copy()
                df_temp_year['year'] = pd.to_datetime(df_temp_year['tickets_data_issue_datetime']).dt.year
                all_years = sorted(df_temp_year['year'].unique().tolist())
                selected_years = st.multiselect(
                    "Select Years",
                    options=all_years,
                    default=all_years,
                    key="excel_year_filter",
                    label_visibility="collapsed"
                )
                if selected_years:
                    df['year'] = pd.to_datetime(df['tickets_data_issue_datetime']).dt.year
                    df = df[df['year'].isin(selected_years)]
            st.markdown('</div>', unsafe_allow_html=True)

            # Severity filter (like Stores)
            st.markdown('<div class="excel-filter-section">', unsafe_allow_html=True)
            st.markdown('<div class="excel-filter-title">Severity</div>', unsafe_allow_html=True)
            if 'tickets_data_severity' in df.columns:
                all_severities = sorted(df['tickets_data_severity'].unique().tolist())
                selected_severities = st.multiselect(
                    "Select Severities",
                    options=all_severities,
                    default=all_severities,
                    key="excel_sev_filter",
                    label_visibility="collapsed"
                )
                if selected_severities:
                    df = df[df['tickets_data_severity'].isin(selected_severities)]
            st.markdown('</div>', unsafe_allow_html=True)

            # Clear Filters button
            if st.button("🗑️ Clear Filter(s)", key="clear_filters", type="secondary"):
                st.rerun()

        st.markdown("---")
        st.markdown(f"**📁 Data Source:**")
        st.caption(data_source)
        st.markdown(f"**Records:** {len(df):,}")

        # Quick stats
        if 'Financial_Impact' in df.columns:
            total_cost = df['Financial_Impact'].sum()
            st.markdown(f"**Total Cost:** ${total_cost:,.0f}")

        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        # Date filter
        if 'tickets_data_issue_datetime' in df.columns:
            min_date = pd.to_datetime(df['tickets_data_issue_datetime']).min().date()
            max_date = pd.to_datetime(df['tickets_data_issue_datetime']).max().date()
            
            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                df = df[
                    (pd.to_datetime(df['tickets_data_issue_datetime']).dt.date >= date_range[0]) &
                    (pd.to_datetime(df['tickets_data_issue_datetime']).dt.date <= date_range[1])
                ]
        
        st.markdown("---")
        st.markdown("### 📤 Export Reports")

        export_format = st.selectbox(
            "Select Report Type",
            ["📦 All Reports (ZIP)", "📊 Strategic Report (Excel)", "📄 Executive Report (PDF)", "🌐 Interactive Report (HTML)", "📁 Raw Data (CSV)"],
            key="export_format_select"
        )

        if export_format == "📦 All Reports (ZIP)":
            st.markdown("""
            <div style="background: rgba(251, 191, 36, 0.1); border-radius: 8px; padding: 12px; margin: 10px 0; border: 1px solid rgba(251, 191, 36, 0.3);">
                <div style="color: #fcd34d; font-weight: 600;">📦 Complete Report Package</div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 5px;">
                    Download all reports in a single ZIP file: Strategic Report (Excel),
                    Executive Report (PDF), Interactive Report (HTML), and Raw Data (CSV).
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("📥 Generate All Reports", key="gen_all_reports"):
                with st.spinner("Generating all reports... This may take a moment."):
                    # Create ZIP file in memory
                    zip_buffer = io.BytesIO()
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                    reports_generated = []
                    reports_failed = []

                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        # 1. Strategic Report (Excel)
                        project_root = Path(__file__).parent.parent.parent
                        strategic_paths = [
                            project_root / "Strategic_Report.xlsx",
                            Path.cwd() / "Strategic_Report.xlsx",
                            Path("Strategic_Report.xlsx"),
                        ]
                        strategic_report_path = None
                        for path in strategic_paths:
                            if path.exists():
                                strategic_report_path = path
                                break

                        if strategic_report_path:
                            try:
                                with open(strategic_report_path, "rb") as f:
                                    zip_file.writestr(f"Strategic_Report_{timestamp}.xlsx", f.read())
                                reports_generated.append("Strategic Report (Excel)")
                            except Exception as e:
                                reports_failed.append(f"Strategic Report: {e}")
                        else:
                            reports_failed.append("Strategic Report: File not found")

                        # 2. Executive Report (PDF)
                        try:
                            pdf_data = generate_executive_pdf_report(df)
                            if pdf_data:
                                zip_file.writestr(f"Executive_Report_{timestamp}.pdf", pdf_data)
                                reports_generated.append("Executive Report (PDF)")
                            else:
                                reports_failed.append("Executive Report: PDF generation not available")
                        except Exception as e:
                            reports_failed.append(f"Executive Report: {e}")

                        # 3. Interactive Report (HTML)
                        try:
                            html_data = generate_magnificent_html_report(df)
                            zip_file.writestr(f"Interactive_Report_{timestamp}.html", html_data)
                            reports_generated.append("Interactive Report (HTML)")
                        except Exception as e:
                            reports_failed.append(f"Interactive Report: {e}")

                        # 4. Raw Data (CSV)
                        try:
                            csv_data = df.to_csv(index=False)
                            zip_file.writestr(f"Escalation_Data_{timestamp}.csv", csv_data)
                            reports_generated.append("Raw Data (CSV)")
                        except Exception as e:
                            reports_failed.append(f"Raw Data: {e}")

                    # Show status
                    if reports_generated:
                        st.success(f"✅ Generated: {', '.join(reports_generated)}")
                    if reports_failed:
                        st.warning(f"⚠️ Failed: {', '.join(reports_failed)}")

                    # Download button
                    zip_buffer.seek(0)
                    st.download_button(
                        label=f"⬇️ Download All Reports ({len(reports_generated)} files)",
                        data=zip_buffer.getvalue(),
                        file_name=f"Escalation_Reports_{timestamp}.zip",
                        mime="application/zip",
                        key="download_all_zip"
                    )

        elif export_format == "📊 Strategic Report (Excel)":
            st.markdown("""
            <div style="background: rgba(34, 197, 94, 0.1); border-radius: 8px; padding: 12px; margin: 10px 0; border: 1px solid rgba(34, 197, 94, 0.3);">
                <div style="color: #86efac; font-weight: 600;">📊 Strategic Report</div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 5px;">
                    Comprehensive Excel report with multiple sheets: Executive Summary, Financial Analysis,
                    Category Breakdown, Engineer Performance, Recommendations, and Charts.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Load the Strategic Report file - search multiple locations
            project_root = Path(__file__).parent.parent.parent
            strategic_paths = [
                project_root / "Strategic_Report.xlsx",
                Path.cwd() / "Strategic_Report.xlsx",
                Path("Strategic_Report.xlsx"),
            ]

            strategic_report_path = None
            for path in strategic_paths:
                if path.exists():
                    strategic_report_path = path
                    break

            if strategic_report_path:
                try:
                    with open(strategic_report_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Strategic Report",
                            data=f.read(),
                            file_name=f"Strategic_Report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_strategic_report"
                        )
                except Exception as e:
                    st.warning(f"Could not read Strategic Report: {e}")
            else:
                st.warning("Strategic Report not found. Please run the report generation pipeline first.")

        elif export_format == "📄 Executive Report (PDF)":
            st.markdown("""
            <div style="background: rgba(239, 68, 68, 0.1); border-radius: 8px; padding: 12px; margin: 10px 0; border: 1px solid rgba(239, 68, 68, 0.3);">
                <div style="color: #fca5a5; font-weight: 600;">📄 Executive Report</div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 5px;">
                    Professional PDF report with KPIs, strategic recommendations,
                    and key insights for executive presentation.
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("📥 Generate PDF Report", key="gen_pdf"):
                with st.spinner("Generating executive report..."):
                    pdf_data = generate_executive_pdf_report(df)
                    if pdf_data:
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=pdf_data,
                            file_name=f"Executive_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            key="download_pdf"
                        )
                    else:
                        st.warning("PDF generation requires reportlab. Try the HTML report instead.")

        elif export_format == "🌐 Interactive Report (HTML)":
            st.markdown("""
            <div style="background: rgba(59, 130, 246, 0.1); border-radius: 8px; padding: 12px; margin: 10px 0; border: 1px solid rgba(59, 130, 246, 0.3);">
                <div style="color: #93c5fd; font-weight: 600;">🌐 Interactive Report</div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 5px;">
                    Beautiful HTML report with interactive charts. Open in any browser,
                    hover over charts for details. Can be printed to PDF.
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("📥 Generate HTML Report", key="gen_html"):
                with st.spinner("Generating interactive report..."):
                    html_data = generate_magnificent_html_report(df)
                    st.download_button(
                        label="⬇️ Download HTML",
                        data=html_data,
                        file_name=f"Interactive_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                        mime="text/html",
                        key="download_html"
                    )

        elif export_format == "📁 Raw Data (CSV)":
            st.markdown("""
            <div style="background: rgba(168, 85, 247, 0.1); border-radius: 8px; padding: 12px; margin: 10px 0; border: 1px solid rgba(168, 85, 247, 0.3);">
                <div style="color: #c4b5fd; font-weight: 600;">📁 Raw Data Export</div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 5px;">
                    Export all escalation data as CSV for further analysis in Excel,
                    Power BI, or other tools.
                </div>
            </div>
            """, unsafe_allow_html=True)

            csv_data = df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_data,
                file_name=f"Escalation_Data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key="download_csv"
            )
    
    # Auto-scroll to top when switching pages
    if 'current_page' not in st.session_state:
        st.session_state.current_page = page

    if st.session_state.current_page != page:
        st.session_state.current_page = page
        streamlit_js_eval(js_expressions="parent.document.querySelector('section.main').scrollTo(0, 0)")

    # Main content - Route to appropriate page (6 consolidated tabs)
    if page == "📊 Executive Dashboard":
        render_excel_dashboard(df)
    elif page == "📈 Deep Analysis":
        render_deep_analysis(df)
    elif page == "💰 Financial Intelligence":
        render_financial_analysis(df)
    elif page == "🏆 Benchmarking & Monitoring":
        render_benchmarking_monitoring(df)
    elif page == "🎯 Planning & Actions":
        render_planning_actions(df)
    elif page == "📽️ Presentation Mode":
        render_presentation_mode(df)


# ============================================================================
# ADVANCED INSIGHTS PAGE
# ============================================================================

def render_advanced_insights(df):
    """Render the Advanced Insights page with high-value visualizations."""
    render_spectacular_header("Advanced Insights", "Strategic visualizations for executive decision-making", "🚀")
    
    # Create tabs for different insight categories
    tabs = st.tabs(["📊 SLA & Aging", "👥 Engineer Efficiency", "💰 Cost Analysis", "🔄 Patterns"])
    
    with tabs[0]:
        st.markdown("### SLA Compliance & Ticket Aging Analysis")
        st.markdown("*Track resolution performance against service level agreements*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(chart_sla_funnel(df), use_container_width=True)
        
        with col2:
            st.plotly_chart(chart_aging_analysis(df), use_container_width=True)
        
        st.markdown("---")
        st.markdown("### Time Pattern Analysis")
        st.markdown("*Identify peak escalation times and shift handoff issues*")
        st.plotly_chart(chart_time_heatmap(df), use_container_width=True)
    
    with tabs[1]:
        st.markdown("### Engineer Efficiency Quadrant")
        st.markdown("*Speed vs Quality: Identify top performers and those needing support*")
        
        st.plotly_chart(chart_engineer_quadrant(df), use_container_width=True)
        
        # Quadrant legend explanation
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style="background: rgba(40,167,69,0.2); padding: 15px; border-radius: 8px; border-left: 4px solid #28A745;">
                <strong>⭐ Fast & Clean</strong><br>
                <small>Low resolution time, low recurrence. Top performers.</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: rgba(23,162,184,0.2); padding: 15px; border-radius: 8px; border-left: 4px solid #17A2B8;">
                <strong>🐢 Slow but Thorough</strong><br>
                <small>Higher resolution time, but quality work.</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: rgba(255,193,7,0.2); padding: 15px; border-radius: 8px; border-left: 4px solid #FFC107;">
                <strong>⚡ Fast but Sloppy</strong><br>
                <small>Quick fixes that may recur. Need coaching.</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="background: rgba(220,53,69,0.2); padding: 15px; border-radius: 8px; border-left: 4px solid #DC3545;">
                <strong>🆘 Needs Support</strong><br>
                <small>Slow and issues recur. Priority for training.</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Resolution Consistency")
        st.markdown("*Categories with high variability may indicate process gaps or training needs*")
        st.plotly_chart(chart_resolution_consistency(df), use_container_width=True)
    
    with tabs[2]:
        st.markdown("### Cost Avoidance Waterfall")
        st.markdown("*Path from current costs to achievable target through strategic interventions*")
        
        st.plotly_chart(chart_cost_waterfall(df), use_container_width=True)
        
        # Cost insights
        st.markdown("---")
        st.markdown("### 💡 Cost Reduction Opportunities")
        
        total_cost = df['Financial_Impact'].sum() if 'Financial_Impact' in df.columns else len(df) * get_benchmark_costs()['avg_per_ticket']
        recurrence_rate = df['AI_Recurrence_Probability'].mean() if 'AI_Recurrence_Probability' in df.columns else 0.2
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            recurrence_savings = total_cost * recurrence_rate * 0.5
            st.markdown(f"""
            <div class="kpi-container success">
                <h3 style="color: #28A745;">${recurrence_savings/1000:.0f}K</h3>
                <p>Recurrence Prevention</p>
                <small>50% of recurring issue costs</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            resolution_savings = total_cost * 0.15
            st.markdown(f"""
            <div class="kpi-container success">
                <h3 style="color: #28A745;">${resolution_savings/1000:.0f}K</h3>
                <p>Faster Resolution</p>
                <small>15% from reduced handling time</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            process_savings = total_cost * 0.05
            st.markdown(f"""
            <div class="kpi-container success">
                <h3 style="color: #28A745;">${process_savings/1000:.0f}K</h3>
                <p>Process Improvement</p>
                <small>5% from automation & efficiency</small>
            </div>
            """, unsafe_allow_html=True)
    
    with tabs[3]:
        st.markdown("### Operational Health Score")
        st.markdown("*Composite score based on recurrence, resolution time, and critical issues*")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.plotly_chart(chart_health_gauge(df), use_container_width=True)
        
        with col2:
            # Health score breakdown
            recurrence_rate = df['AI_Recurrence_Probability'].mean() * 100 if 'AI_Recurrence_Probability' in df.columns else 15
            resolution_days = df['Predicted_Resolution_Days'].mean() if 'Predicted_Resolution_Days' in df.columns else 5
            critical_pct = (df['tickets_data_severity'] == 'Critical').mean() * 100 if 'tickets_data_severity' in df.columns else 10
            
            st.markdown("#### Score Components")
            st.markdown(f"""
            | Component | Value | Impact |
            |-----------|-------|--------|
            | Recurrence Rate | {recurrence_rate:.1f}% | -{ recurrence_rate * 1.5:.0f} pts |
            | Resolution Time | {resolution_days:.1f} days | -{resolution_days * 5:.0f} pts |
            | Critical Issues | {critical_pct:.1f}% | -{critical_pct * 0.5:.0f} pts |
            """)
            
            st.info("💡 **Tip:** Focus on reducing recurrence rate for the biggest health score improvement.")
        
        st.markdown("---")
        st.markdown("### Category to Recurrence Flow")
        st.markdown("*Which categories are driving high-risk outcomes?*")
        st.plotly_chart(chart_recurrence_patterns(df), use_container_width=True)


def render_dashboard(df):
    """Render the main dashboard page."""
    render_spectacular_header("Dashboard", "Real-time escalation intelligence at a glance", "📊")
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    
    # Count by ticket type
    type_col = 'tickets_data_type'
    total = len(df)
    escalations_count = len(df[df[type_col].astype(str).str.contains('Escalation', case=False, na=False)]) if type_col in df.columns else total
    concerns_count = len(df[df[type_col].astype(str).str.contains('Concern', case=False, na=False)]) if type_col in df.columns else 0
    lessons_count = len(df[df[type_col].astype(str).str.contains('Lesson', case=False, na=False)]) if type_col in df.columns else 0
    
    with col1:
        st.markdown(f"""
        <div class="kpi-container">
            <p class="kpi-value">{total:,}</p>
            <p class="kpi-label">Total Records</p>
            <p class="kpi-delta" style="font-size: 0.7rem; color: #888;">📋 {escalations_count} Escalations | ⚠️ {concerns_count} Concerns | 📚 {lessons_count} Lessons</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        critical = len(df[df['tickets_data_severity'] == 'Critical'])
        critical_pct = (critical / total * 100) if total > 0 else 0
        st.markdown(f"""
        <div class="kpi-container critical">
            <p class="kpi-value">{critical}</p>
            <p class="kpi-label">Critical Issues</p>
            <p class="kpi-delta delta-up">{critical_pct:.1f}% of total</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_friction = df['Strategic_Friction_Score'].mean()
        st.markdown(f"""
        <div class="kpi-container warning">
            <p class="kpi-value">{avg_friction:.0f}</p>
            <p class="kpi-label">Avg Friction Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_days = df['Predicted_Resolution_Days'].mean()
        st.markdown(f"""
        <div class="kpi-container success">
            <p class="kpi-value">{avg_days:.1f}</p>
            <p class="kpi-label">Avg Resolution (Days)</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

    # Charts row 1
    col1, col2 = st.columns([2, 1])

    with col1:
        render_chart_with_insight('trend_timeline', chart_trend_timeline(df), df)

    with col2:
        render_chart_with_insight('severity_distribution', chart_severity_distribution(df), df)

    # Charts row 2
    col1, col2 = st.columns(2)

    with col1:
        render_chart_with_insight('friction_by_category', chart_friction_by_category(df), df)

    with col2:
        render_chart_with_insight('recurrence_risk', chart_recurrence_risk(df), df)


def render_analytics(df):
    """Render the analytics page with detailed charts."""
    render_spectacular_header("Analytics", "Deep dive into escalation patterns and performance", "📈")

    tabs = st.tabs(["🎯 Categories", "👥 Engineers", "📊 Distributions", "💰 Financial", "🔗 Similarity", "📚 Lessons Learned"])

    with tabs[0]:
        # Sub-tabs for different category views
        cat_tabs = st.tabs(["📊 Overview", "🔍 Drill-Down", "📈 Treemap", "📋 Details"])

        with cat_tabs[0]:
            # Overview - Sunburst and friction chart with insights
            render_chart_with_insight('category_sunburst', chart_category_sunburst(df), df)
            render_chart_with_insight('friction_by_category', chart_friction_by_category(df), df)

        with cat_tabs[1]:
            # Drill-down - Category selector with sub-category breakdown
            st.markdown("### Sub-Category Drill-Down")
            st.markdown("Select a category to view detailed sub-category breakdown")

            # Category selector
            categories = ['All Categories'] + sorted(df['AI_Category'].unique().tolist())
            selected_cat = st.selectbox("Select Category", categories, key="cat_drilldown_select")

            if selected_cat == 'All Categories':
                # Show overall sub-category breakdown
                st.plotly_chart(chart_subcategory_breakdown(df, None), use_container_width=True)
            else:
                # Show breakdown for selected category
                st.plotly_chart(chart_subcategory_breakdown(df, selected_cat), use_container_width=True)

            # Sub-category comparison table
            st.markdown("### Sub-Category Comparison")
            comparison_df = chart_subcategory_comparison_table(df)
            if not comparison_df.empty:
                if selected_cat != 'All Categories':
                    comparison_df = comparison_df[comparison_df['Category'] == selected_cat]
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        with cat_tabs[2]:
            # Treemap view
            st.markdown("### Category Treemap")
            st.markdown("*Click on categories to drill down into sub-categories*")
            st.plotly_chart(chart_category_treemap(df), use_container_width=True)

        with cat_tabs[3]:
            # Detailed category statistics
            st.markdown("### Category Statistics")

            # Summary table
            if 'AI_Sub_Category' in df.columns:
                # Build aggregation dict dynamically based on available columns
                agg_dict = {'AI_Sub_Category': 'count'}
                col_names = ['Ticket Count']

                if 'AI_Confidence' in df.columns:
                    agg_dict['AI_Confidence'] = 'mean'
                    col_names.append('Avg Confidence')

                if 'Strategic_Friction_Score' in df.columns:
                    agg_dict['Strategic_Friction_Score'] = 'sum'
                    col_names.append('Total Friction')

                cat_stats = df.groupby('AI_Category').agg(agg_dict).round(2)
                cat_stats.columns = col_names
                cat_stats = cat_stats.sort_values('Ticket Count', ascending=False)

                # Add financial if available
                if 'Financial_Impact' in df.columns:
                    fin_stats = df.groupby('AI_Category')['Financial_Impact'].sum()
                    cat_stats['Total Impact'] = fin_stats
                    cat_stats['Total Impact'] = cat_stats['Total Impact'].apply(lambda x: f"${x:,.0f}")

                st.dataframe(cat_stats, use_container_width=True)

                # Sub-category distribution within each category
                st.markdown("### Sub-Category Distribution")
                for cat in df['AI_Category'].unique():
                    with st.expander(f"📁 {cat}"):
                        cat_df = df[df['AI_Category'] == cat]
                        sub_counts = cat_df['AI_Sub_Category'].value_counts()

                        col1, col2 = st.columns([2, 1])
                        with col1:
                            fig = go.Figure(go.Bar(
                                x=sub_counts.values,
                                y=sub_counts.index,
                                orientation='h',
                                marker_color='#0066CC'
                            ))
                            fig.update_layout(
                                **{
                                    **create_plotly_theme(),
                                    'margin': dict(l=10, r=10, t=10, b=10),
                                },
                                height=200,
                                yaxis_title='',
                                xaxis_title='Count'
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            st.metric("Total Tickets", len(cat_df))
                            if 'Financial_Impact' in df.columns:
                                st.metric("Total Impact", f"${cat_df['Financial_Impact'].sum():,.0f}")
            else:
                st.info("Sub-category data not available. Run classification with the updated system.")
    
    with tabs[1]:
        fig = chart_engineer_performance(df)
        if fig:
            render_chart_with_insight('engineer_performance', fig, df)
        else:
            st.info("Engineer data not available.")

    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            render_chart_with_insight('resolution_distribution', chart_resolution_distribution(df), df)
        with col2:
            render_chart_with_insight('recurrence_risk', chart_recurrence_risk(df), df)
    
    with tabs[3]:
        if 'Financial_Impact' in df.columns:
            # Sub-tabs for financial drill-down
            fin_tabs = st.tabs(["📊 By Category", "🔍 Sub-Category Drill-Down", "📋 Summary Table"])

            with fin_tabs[0]:
                # Financial by category
                fin_by_cat = df.groupby('AI_Category')['Financial_Impact'].sum().sort_values(ascending=False)

                fig = go.Figure(go.Bar(
                    x=fin_by_cat.index,
                    y=fin_by_cat.values,
                    marker=dict(
                        color=fin_by_cat.values,
                        colorscale='Reds'
                    ),
                    text=[f"${v/1000:.0f}K" for v in fin_by_cat.values],
                    textposition='outside'
                ))

                # Get theme without margin
                theme = create_plotly_theme()
                theme.pop('margin', None)

                fig.update_layout(
                    **theme,
                    title='Financial Impact by Category',
                    xaxis_tickangle=-45,
                    height=420,
                    margin=dict(l=40, r=60, t=80, b=100)  # Room for labels above bars
                )

                st.plotly_chart(fig, use_container_width=True)

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Impact", f"${df['Financial_Impact'].sum():,.0f}")
                with col2:
                    st.metric("Average per Ticket", f"${df['Financial_Impact'].mean():,.0f}")
                with col3:
                    st.metric("Max Single Ticket", f"${df['Financial_Impact'].max():,.0f}")

            with fin_tabs[1]:
                # Sub-category financial drill-down
                st.markdown("### Financial Impact by Sub-Category")
                st.markdown("*Click on categories in the chart to drill down*")

                # Financial drill-down chart
                st.plotly_chart(chart_category_financial_drilldown(df), use_container_width=True)

                # Sub-category breakdown if available
                if 'AI_Sub_Category' in df.columns:
                    st.markdown("### Sub-Category Cost Breakdown")

                    # Category selector for detailed view
                    categories = ['All Categories'] + sorted(df['AI_Category'].unique().tolist())
                    selected_cat = st.selectbox("Select Category for Details", categories, key="fin_cat_select")

                    if selected_cat == 'All Categories':
                        subcat_fin = df.groupby('AI_Sub_Category')['Financial_Impact'].agg(['sum', 'mean', 'count'])
                    else:
                        subcat_fin = df[df['AI_Category'] == selected_cat].groupby('AI_Sub_Category')['Financial_Impact'].agg(['sum', 'mean', 'count'])

                    subcat_fin.columns = ['Total', 'Average', 'Count']
                    subcat_fin = subcat_fin.sort_values('Total', ascending=False)

                    # Bar chart
                    fig = go.Figure(go.Bar(
                        x=subcat_fin['Total'],
                        y=subcat_fin.index,
                        orientation='h',
                        marker=dict(
                            color=subcat_fin['Total'],
                            colorscale='Reds'
                        ),
                        text=[f"${v:,.0f}" for v in subcat_fin['Total']],
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Total: $%{x:,.0f}<extra></extra>'
                    ))

                    theme = create_plotly_theme()
                    theme.pop('margin', None)

                    fig.update_layout(
                        **theme,
                        title=f'Financial Impact: {selected_cat}',
                        height=max(300, len(subcat_fin) * 35),
                        margin=dict(l=200, r=80, t=60, b=40)
                    )

                    st.plotly_chart(fig, use_container_width=True)

            with fin_tabs[2]:
                # Summary table
                st.markdown("### Financial Summary Table")

                if 'AI_Sub_Category' in df.columns:
                    summary = df.groupby(['AI_Category', 'AI_Sub_Category']).agg({
                        'Financial_Impact': ['sum', 'mean', 'count']
                    }).round(2)
                    summary.columns = ['Total Impact', 'Avg Impact', 'Ticket Count']
                    summary = summary.reset_index()
                    summary['Total Impact'] = summary['Total Impact'].apply(lambda x: f"${x:,.0f}")
                    summary['Avg Impact'] = summary['Avg Impact'].apply(lambda x: f"${x:,.0f}")
                    summary = summary.sort_values(['AI_Category', 'Ticket Count'], ascending=[True, False])
                else:
                    summary = df.groupby('AI_Category').agg({
                        'Financial_Impact': ['sum', 'mean', 'count']
                    }).round(2)
                    summary.columns = ['Total Impact', 'Avg Impact', 'Ticket Count']
                    summary = summary.reset_index()
                    summary['Total Impact'] = summary['Total Impact'].apply(lambda x: f"${x:,.0f}")
                    summary['Avg Impact'] = summary['Avg Impact'].apply(lambda x: f"${x:,.0f}")

                st.dataframe(summary, use_container_width=True, hide_index=True)
        else:
            st.info("Financial impact data not available.")

    with tabs[4]:
        # Similarity Search Analysis Tab
        st.markdown("### 🔗 Similarity Search Analysis")
        st.markdown("*Insights from comparing tickets to historical patterns*")

        # Check if similarity data exists
        has_similarity = any(col in df.columns for col in ['Similar_Ticket_Count', 'Best_Match_Similarity', 'Resolution_Consistency'])

        if not has_similarity:
            st.info("🔍 **Similarity search data not available.**\n\nRun the analysis with similarity search enabled to populate this section.")
        else:
            # Similarity sub-tabs
            sim_tabs = st.tabs(["📊 Overview", "📈 Score Analysis", "⚖️ Consistency", "🔥 Heatmap"])

            with sim_tabs[0]:
                # Overview - count distribution and key metrics
                col1, col2, col3 = st.columns(3)

                if 'Similar_Ticket_Count' in df.columns:
                    counts = df['Similar_Ticket_Count'].dropna()
                    with col1:
                        avg_similar = counts.mean()
                        st.metric("Avg Similar Tickets", f"{avg_similar:.1f}")
                    with col2:
                        zero_matches = (counts == 0).sum()
                        st.metric("No Matches Found", f"{zero_matches}", delta=f"{zero_matches/len(counts)*100:.0f}%", delta_color="inverse")
                    with col3:
                        high_matches = (counts >= 5).sum()
                        st.metric("Good Coverage (5+)", f"{high_matches}", delta=f"{high_matches/len(counts)*100:.0f}%")

                st.markdown("---")

                # Count distribution chart
                fig = chart_similarity_count_distribution(df)
                if fig:
                    render_chart_with_insight('similarity_count', fig, df)
                else:
                    st.info("Similar ticket count data not available.")

            with sim_tabs[1]:
                # Score distribution
                st.markdown("### Similarity Score Analysis")
                st.markdown("*How confident are we in the similar ticket matches?*")

                if 'Best_Match_Similarity' in df.columns:
                    scores = df['Best_Match_Similarity'].dropna()
                    scores = scores[scores > 0]

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Score", f"{scores.mean():.2f}")
                    with col2:
                        high_conf = (scores >= 0.7).sum()
                        st.metric("High Confidence", f"{high_conf}", delta=f"{high_conf/len(scores)*100:.0f}%")
                    with col3:
                        low_conf = (scores < 0.5).sum()
                        st.metric("Low Confidence", f"{low_conf}", delta=f"{low_conf/len(scores)*100:.0f}%", delta_color="inverse")

                    st.markdown("---")
                    fig = chart_similarity_score_distribution(df)
                    if fig:
                        render_chart_with_insight('similarity_score', fig, df)

                # Resolution comparison
                fig = chart_expected_vs_predicted_resolution(df)
                if fig:
                    st.markdown("### Expected vs AI Predicted Resolution")
                    st.plotly_chart(fig, use_container_width=True)

            with sim_tabs[2]:
                # Consistency analysis
                st.markdown("### Resolution Consistency")
                st.markdown("*Are we resolving similar issues the same way?*")

                col1, col2 = st.columns(2)

                with col1:
                    fig = chart_resolution_consistency(df)
                    if fig:
                        render_chart_with_insight('resolution_consistency', fig, df)
                    else:
                        st.info("Resolution consistency data not available.")

                with col2:
                    fig = chart_inconsistent_by_category(df)
                    if fig:
                        render_chart_with_insight('inconsistent_resolution', fig, df)
                    else:
                        st.info("Inconsistent resolution data not available.")

                # Detailed inconsistency table
                if 'Inconsistent_Resolution' in df.columns:
                    inconsistent = df[df['Inconsistent_Resolution'] == True]
                    if len(inconsistent) > 0:
                        st.markdown("### Tickets with Inconsistent Resolutions")
                        st.markdown(f"*{len(inconsistent)} tickets resolved differently than similar historical cases*")

                        display_cols = ['Identity', 'AI_Category', 'AI_Sub_Category', 'Similar_Ticket_Count', 'Best_Match_Similarity']
                        display_cols = [c for c in display_cols if c in inconsistent.columns]

                        if display_cols:
                            st.dataframe(
                                inconsistent[display_cols].head(20),
                                use_container_width=True,
                                hide_index=True
                            )

            with sim_tabs[3]:
                # Effectiveness heatmap
                st.markdown("### Similarity Search Effectiveness")
                st.markdown("*Where do we have good historical coverage?*")

                fig = chart_similarity_effectiveness_heatmap(df)
                if fig:
                    render_chart_with_insight('similarity_effectiveness', fig, df)
                else:
                    st.info("Need both category and origin data for effectiveness heatmap.")

                # Coverage summary by category
                if 'Similar_Ticket_Count' in df.columns and 'AI_Category' in df.columns:
                    st.markdown("### Coverage by Category")
                    coverage = df.groupby('AI_Category').agg({
                        'Similar_Ticket_Count': ['mean', 'sum'],
                        'AI_Category': 'count'
                    })
                    coverage.columns = ['Avg Matches', 'Total Matches', 'Ticket Count']
                    coverage['Coverage Score'] = (coverage['Avg Matches'] * 20).clip(0, 100).round(0).astype(int)
                    coverage = coverage.sort_values('Coverage Score', ascending=False)

                    st.dataframe(coverage, use_container_width=True)

    with tabs[5]:
        # Lessons Learned Effectiveness Tab - Comprehensive 6-Pillar Scorecard
        st.markdown("### 📚 Lessons Learned Effectiveness")
        st.markdown("*Comprehensive 6-pillar assessment of organizational learning*")

        # Get comprehensive scorecard
        scorecard = get_comprehensive_scorecard(df)

        if not scorecard or not scorecard.category_scorecards:
            # Fall back to simple grading
            grades_data = _calculate_learning_grades(df)
            if not grades_data:
                st.info("🔍 **Lessons learned data not available.**\n\nThis analysis requires:\n- `AI_Category` column\n- `tickets_data_lessons_learned_title` or similar column\n- Ideally `AI_Recurrence_Probability` from similarity analysis")
            else:
                # Use simple grades (backwards compatibility)
                total_cats = len(grades_data)
                avg_score = np.mean([d['score'] for d in grades_data.values()])
                st.metric("Average Score (Simple)", f"{avg_score:.0f}/100")
                fig = chart_learning_grades(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="learning_grades_simple")
        else:
            # Full comprehensive scorecard view
            summary_df = scorecard.get_summary_df()
            at_risk = scorecard.get_at_risk_categories()

            # Summary metrics
            total_cats = len(scorecard.category_scorecards)
            avg_score = summary_df['Overall Score'].mean()

            # Count grades by letter (ignore +/-)
            grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
            for grade in summary_df['Grade']:
                base_grade = grade[0] if grade else 'F'
                if base_grade in grade_counts:
                    grade_counts[base_grade] += 1

            at_risk_count = grade_counts['D'] + grade_counts['F']

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Score", f"{avg_score:.0f}/100")
            with col2:
                st.metric("Categories Analyzed", total_cats)
            with col3:
                st.metric("At-Risk (D/F)", at_risk_count,
                         delta=f"{at_risk_count/total_cats*100:.0f}%" if total_cats > 0 else "0%",
                         delta_color="inverse")
            with col4:
                excellent = grade_counts['A'] + grade_counts['B']
                st.metric("Excellent (A/B)", excellent,
                         delta=f"{excellent/total_cats*100:.0f}%" if total_cats > 0 else "0%")

            st.markdown("---")

            # Comprehensive sub-tabs
            lesson_tabs = st.tabs([
                "🎯 Scorecard Overview",
                "📊 Category Rankings",
                "🔬 Pillar Deep-Dive",
                "📈 Trends & Patterns",
                "⚠️ At-Risk Categories",
                "💡 AI Recommendations"
            ])

            with lesson_tabs[0]:
                # Scorecard Overview with Radar
                st.markdown("### Organization Learning Effectiveness Scorecard")
                st.markdown("""
                *The 6-pillar scorecard evaluates learning effectiveness across:*
                - **Learning Velocity**: Improvement trends over time
                - **Impact Management**: Handling of high-severity recurring issues
                - **Knowledge Quality**: Quality of documented lessons (AI-assessed)
                - **Process Maturity**: Consistency and documentation completeness
                - **Knowledge Transfer**: Cross-team learning and knowledge sharing
                - **Outcome Effectiveness**: Actual results and improvements
                """)

                col1, col2 = st.columns([1, 1])

                with col1:
                    # Organization-wide radar
                    fig = chart_scorecard_radar(df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="radar_org_overview")

                with col2:
                    # Pillar averages table
                    st.markdown("### Pillar Averages")
                    pillar_avgs = []
                    for pillar_key in ['learning_velocity', 'impact_management', 'knowledge_quality',
                                      'process_maturity', 'knowledge_transfer', 'outcome_effectiveness']:
                        scores = [sc.pillars[pillar_key].score for sc in scorecard.category_scorecards.values()]
                        avg = np.mean(scores)
                        trend = 'improving' if avg > 60 else ('needs work' if avg < 40 else 'stable')
                        pillar_avgs.append({
                            'Pillar': pillar_key.replace('_', ' ').title(),
                            'Avg Score': f"{avg:.0f}",
                            'Status': '✅' if avg >= 70 else ('⚠️' if avg >= 50 else '🔴'),
                            'Trend': trend
                        })
                    st.dataframe(pd.DataFrame(pillar_avgs), use_container_width=True, hide_index=True)

            with lesson_tabs[1]:
                # Category Rankings
                st.markdown("### Category Rankings")
                st.markdown("*All categories ranked by overall learning effectiveness score*")

                fig = chart_learning_grades(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="learning_grades_ranked")

                # Detailed table
                st.markdown("### Detailed Scores")
                display_df = summary_df.copy()
                display_df['Overall Score'] = display_df['Overall Score'].round(1)
                for col in display_df.columns:
                    if col not in ['Category', 'Rank', 'Grade', 'Overall Score']:
                        display_df[col] = display_df[col].round(1)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

            with lesson_tabs[2]:
                # Pillar Deep-Dive
                st.markdown("### Pillar Deep-Dive Analysis")
                st.markdown("*Select a category to see detailed pillar breakdown*")

                selected_cat = st.selectbox(
                    "Select Category",
                    options=['Organization Average'] + list(scorecard.category_scorecards.keys()),
                    key="pillar_drilldown_cat"
                )

                col1, col2 = st.columns([1, 1])

                with col1:
                    if selected_cat == 'Organization Average':
                        fig = chart_scorecard_radar(df)
                    else:
                        fig = chart_scorecard_radar(df, selected_cat)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=f"radar_pillar_{selected_cat}")

                with col2:
                    if selected_cat != 'Organization Average' and selected_cat in scorecard.category_scorecards:
                        cat_sc = scorecard.category_scorecards[selected_cat]

                        st.markdown(f"### {selected_cat}")
                        st.markdown(f"**Grade: {cat_sc.overall_grade}** | Score: {cat_sc.overall_score:.0f}/100 | Rank: #{cat_sc.rank}")

                        if cat_sc.strengths:
                            st.markdown("**Strengths:** " + ", ".join(cat_sc.strengths))
                        if cat_sc.weaknesses:
                            st.markdown("**Weaknesses:** " + ", ".join(cat_sc.weaknesses))

                        st.markdown("#### Pillar Details")
                        for name, pillar in cat_sc.pillars.items():
                            status = '✅' if pillar.score >= 70 else ('⚠️' if pillar.score >= 50 else '🔴')
                            with st.expander(f"{status} {name.replace('_', ' ').title()}: {pillar.score:.0f}"):
                                for sub_name, sub_score in pillar.sub_scores.items():
                                    st.write(f"  • {sub_name.replace('_', ' ').title()}: {sub_score:.0f}")
                                if pillar.insights:
                                    st.markdown("**Insights:**")
                                    for insight in pillar.insights:
                                        st.write(f"  {insight}")

                # Comparison view
                st.markdown("---")
                st.markdown("### Compare Categories")
                compare_cats = st.multiselect(
                    "Select categories to compare (max 5)",
                    options=list(scorecard.category_scorecards.keys()),
                    default=list(scorecard.category_scorecards.keys())[:3],
                    max_selections=5,
                    key="compare_cats"
                )
                if compare_cats:
                    fig = chart_scorecard_comparison(df, compare_cats)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="scorecard_comparison")

            with lesson_tabs[3]:
                # Trends & Patterns
                st.markdown("### Trends & Patterns")

                fig = chart_recurrence_vs_lessons(df)
                if fig:
                    st.markdown("#### Recurrence vs Lesson Completion")
                    render_chart_with_insight('recurrence_lessons', fig, df)

                fig = chart_learning_heatmap(df)
                if fig:
                    st.markdown("#### Learning Effectiveness by Category & LOB")
                    st.plotly_chart(fig, use_container_width=True, key="learning_heatmap")

            with lesson_tabs[4]:
                # At-Risk Categories
                st.markdown("### ⚠️ At-Risk Categories")
                st.markdown("*Categories scoring below C- require immediate attention*")

                if at_risk:
                    for cat_sc in at_risk[:10]:
                        with st.expander(f"🔴 {cat_sc.category} - Grade {cat_sc.overall_grade} ({cat_sc.overall_score:.0f})"):
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                fig = chart_scorecard_radar(df, cat_sc.category)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True, key=f"radar_atrisk_{cat_sc.category}")
                            with col2:
                                st.markdown("**Weaknesses:**")
                                for w in cat_sc.weaknesses:
                                    st.write(f"  • {w}")

                                st.markdown("**Key Recommendations:**")
                                for rec in cat_sc.recommendations[:5]:
                                    st.write(f"  • {rec}")

                                # Show worst pillars
                                worst_pillars = sorted(
                                    cat_sc.pillars.items(),
                                    key=lambda x: x[1].score
                                )[:2]
                                st.markdown("**Focus Areas:**")
                                for name, pillar in worst_pillars:
                                    st.write(f"  🎯 {name.replace('_', ' ').title()}: {pillar.score:.0f}")
                else:
                    st.success("✅ No categories with grades below C-! Organization is learning effectively.")

            with lesson_tabs[5]:
                # AI Recommendations
                st.markdown("### 💡 AI-Powered Recommendations")

                top_recs = scorecard.get_top_recommendations(10)

                if st.button("🤖 Generate AI Executive Summary", key="gen_ai_summary"):
                    with st.spinner("Generating AI analysis..."):
                        summary = scorecard.generate_ai_summary(use_ollama=True)
                        st.markdown("#### Executive Summary")
                        st.markdown(summary)
                else:
                    st.markdown("*Click button above for AI-generated executive summary*")

                st.markdown("---")
                st.markdown("### Priority Recommendations")

                if top_recs:
                    for rec in top_recs:
                        priority = "🔴 HIGH" if rec['grade'] in ['F', 'D-', 'D'] else "🟡 MEDIUM"
                        st.markdown(f"**{priority}** | {rec['category']} ({rec['grade']})")
                        st.write(f"  → {rec['recommendation']}")
                        st.markdown("")
                else:
                    st.success("✅ All categories performing well!")


if __name__ == "__main__":
    main()
