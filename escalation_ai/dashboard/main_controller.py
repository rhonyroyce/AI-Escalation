"""
Main application controller for the Escalation AI dashboard.

Extracted from streamlit_app.py. Provides main() which orchestrates
sidebar navigation, data loading, filters, export, and page routing.
"""

import streamlit as st
import pandas as pd
import io
import zipfile
from pathlib import Path
from datetime import datetime
from streamlit_js_eval import streamlit_js_eval


def _setup_sidebar_navigation():
    """Set up sidebar branding and navigation radio.

    Returns:
        str: Selected page name.
    """
    st.markdown("## 🎯 Escalation AI")
    st.markdown("*Executive Intelligence Platform*")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊 Executive Dashboard", "📈 Deep Analysis", "💰 Financial Intelligence",
         "🏆 Benchmarking & Monitoring", "🎯 Planning & Actions", "📽️ Presentation Mode"],
        label_visibility="collapsed"
    )
    return page


def _setup_sidebar_filters(df, page):
    """Apply Excel-style filters when on Executive Dashboard page.

    Args:
        df: The loaded DataFrame.
        page: Currently selected page.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if page != "📊 Executive Dashboard":
        return df

    st.markdown("---")
    st.markdown("### Add filter(s)")

    # Category filter
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

    # Severity filter
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

    return df


def _setup_sidebar_data_info(df, data_source):
    """Display data source info and quick stats in sidebar."""
    st.markdown("---")
    st.markdown(f"**📁 Data Source:**")
    st.caption(data_source)
    st.markdown(f"**Records:** {len(df):,}")

    if 'Financial_Impact' in df.columns:
        total_cost = df['Financial_Impact'].sum()
        st.markdown(f"**Total Cost:** ${total_cost:,.0f}")

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()


def _setup_sidebar_date_filter(df):
    """Apply date range filter from Settings section.

    Returns:
        pd.DataFrame: Date-filtered DataFrame.
    """
    st.markdown("---")
    st.markdown("### ⚙️ Settings")

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

    return df


def _setup_sidebar_exports(df):
    """Render export section in sidebar with multiple report format options."""
    from escalation_ai.dashboard.streamlit_app import (
        generate_executive_pdf_report,
    )
    from escalation_ai.dashboard.report_generator_view import (
        generate_magnificent_html_report,
    )

    st.markdown("---")
    st.markdown("### 📤 Export Reports")

    export_format = st.selectbox(
        "Select Report Type",
        ["📦 All Reports (ZIP)", "📊 Strategic Report (Excel)", "📄 Executive Report (PDF)", "🌐 Interactive Report (HTML)", "📁 Raw Data (CSV)"],
        key="export_format_select"
    )

    if export_format == "📦 All Reports (ZIP)":
        _export_all_reports_zip(df, generate_executive_pdf_report, generate_magnificent_html_report)
    elif export_format == "📊 Strategic Report (Excel)":
        _export_strategic_report()
    elif export_format == "📄 Executive Report (PDF)":
        _export_pdf_report(df, generate_executive_pdf_report)
    elif export_format == "🌐 Interactive Report (HTML)":
        _export_html_report(df, generate_magnificent_html_report)
    elif export_format == "📁 Raw Data (CSV)":
        _export_csv_data(df)


def _export_all_reports_zip(df, generate_executive_pdf_report, generate_magnificent_html_report):
    """Generate and download all reports as a ZIP bundle."""
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
            zip_buffer = io.BytesIO()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            reports_generated = []
            reports_failed = []

            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # 1. Strategic Report (Excel)
                strategic_report_path = _find_strategic_report()
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

            if reports_generated:
                st.success(f"✅ Generated: {', '.join(reports_generated)}")
            if reports_failed:
                st.warning(f"⚠️ Failed: {', '.join(reports_failed)}")

            zip_buffer.seek(0)
            st.download_button(
                label=f"⬇️ Download All Reports ({len(reports_generated)} files)",
                data=zip_buffer.getvalue(),
                file_name=f"Escalation_Reports_{timestamp}.zip",
                mime="application/zip",
                key="download_all_zip"
            )


def _find_strategic_report():
    """Search for Strategic_Report.xlsx in common locations."""
    project_root = Path(__file__).parent.parent.parent
    strategic_paths = [
        project_root / "Strategic_Report.xlsx",
        Path.cwd() / "Strategic_Report.xlsx",
        Path("Strategic_Report.xlsx"),
    ]
    for path in strategic_paths:
        if path.exists():
            return path
    return None


def _export_strategic_report():
    """Export Strategic Report (Excel) download."""
    st.markdown("""
    <div style="background: rgba(34, 197, 94, 0.1); border-radius: 8px; padding: 12px; margin: 10px 0; border: 1px solid rgba(34, 197, 94, 0.3);">
        <div style="color: #86efac; font-weight: 600;">📊 Strategic Report</div>
        <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 5px;">
            Comprehensive Excel report with multiple sheets: Executive Summary, Financial Analysis,
            Category Breakdown, Engineer Performance, Recommendations, and Charts.
        </div>
    </div>
    """, unsafe_allow_html=True)

    strategic_report_path = _find_strategic_report()

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


def _export_pdf_report(df, generate_executive_pdf_report):
    """Export Executive Report (PDF) download."""
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


def _export_html_report(df, generate_magnificent_html_report):
    """Export Interactive Report (HTML) download."""
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


def _export_csv_data(df):
    """Export Raw Data (CSV) download."""
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


def _route_to_page(page, df):
    """Route to the appropriate render function based on page selection."""
    from escalation_ai.dashboard.streamlit_app import (
        render_excel_dashboard, render_deep_analysis,
        render_financial_analysis, render_benchmarking_monitoring,
        render_presentation_mode,
    )
    from escalation_ai.dashboard.planning_view import render_planning_actions

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


def main():
    """Main application entry point for standalone dashboard mode.

    Orchestrates the entire dashboard experience:
    1. Sidebar: Navigation radio (6 pages), data loading via load_data(),
       Excel-style filters (category, year, severity, date range) when on
       Executive Dashboard page
    2. Data loading: Reads Strategic_Report.xlsx, applies process_dataframe()
       which recalculates Financial_Impact from price_catalog.xlsx
    3. Export section: ZIP bundle, Excel, PDF, interactive HTML, and CSV
       download buttons
    4. Page routing: Dispatches to the appropriate render_* function based
       on sidebar radio selection

    Pages:
    - Executive Dashboard -> render_excel_dashboard()
    - Deep Analysis -> render_deep_analysis()
    - Financial Intelligence -> render_financial_analysis()
    - Benchmarking & Monitoring -> render_benchmarking_monitoring()
    - Planning & Actions -> render_planning_actions()
    - Presentation Mode -> render_presentation_mode()
    """
    from escalation_ai.dashboard.streamlit_app import load_data

    # Sidebar
    with st.sidebar:
        page = _setup_sidebar_navigation()

        # Load data from default source
        df, data_source = load_data()

        # Excel-style filters (shown for Executive Dashboard page)
        df = _setup_sidebar_filters(df, page)

        _setup_sidebar_data_info(df, data_source)

        df = _setup_sidebar_date_filter(df)

        _setup_sidebar_exports(df)

    # Auto-scroll to top when switching pages
    if 'current_page' not in st.session_state:
        st.session_state.current_page = page

    if st.session_state.current_page != page:
        st.session_state.current_page = page
        streamlit_js_eval(js_expressions="parent.document.querySelector('section.main').scrollTo(0, 0)")

    # Main content - Route to appropriate page
    _route_to_page(page, df)
