"""
Interactive HTML report generator for the Escalation AI dashboard.

Extracted from streamlit_app.py. Provides generate_magnificent_html_report()
which creates a dark-themed executive HTML report with embedded Plotly charts.
"""

import plotly.graph_objects as go
from datetime import datetime


def _build_report_css():
    """Return the CSS stylesheet for the HTML report."""
    return """
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

            * { margin: 0; padding: 0; box-sizing: border-box; }

            body {
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, #0a1628 0%, #1a365d 50%, #0a1628 100%);
                color: #e2e8f0;
                min-height: 100vh;
                padding: 40px;
            }

            .container { max-width: 1400px; margin: 0 auto; }

            .header {
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 58, 138, 0.9) 100%);
                border-radius: 20px;
                padding: 40px;
                margin-bottom: 30px;
                border: 1px solid rgba(59, 130, 246, 0.3);
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
                text-align: center;
            }

            .header h1 {
                font-size: 3rem;
                font-weight: 800;
                background: linear-gradient(135deg, #ffffff 0%, #60a5fa 50%, #3b82f6 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 10px;
            }

            .header .subtitle {
                color: #94a3b8;
                font-size: 1.2rem;
                letter-spacing: 2px;
            }

            .header .meta {
                margin-top: 20px;
                color: #64748b;
                font-size: 0.9rem;
            }

            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 20px;
                margin-bottom: 30px;
            }

            .kpi-card {
                background: linear-gradient(145deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.8) 100%);
                border-radius: 16px;
                padding: 25px;
                text-align: center;
                border: 1px solid rgba(59, 130, 246, 0.2);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }

            .kpi-card.green { border-color: rgba(34, 197, 94, 0.4); }
            .kpi-card.red { border-color: rgba(239, 68, 68, 0.4); }
            .kpi-card.blue { border-color: rgba(59, 130, 246, 0.4); }
            .kpi-card.purple { border-color: rgba(139, 92, 246, 0.4); }

            .kpi-label {
                color: #94a3b8;
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 2px;
                margin-bottom: 10px;
            }

            .kpi-value {
                font-size: 2.5rem;
                font-weight: 800;
                text-shadow: 0 0 30px currentColor;
            }

            .kpi-value.green { color: #22c55e; }
            .kpi-value.red { color: #ef4444; }
            .kpi-value.blue { color: #3b82f6; }
            .kpi-value.purple { color: #8b5cf6; }

            .section {
                background: rgba(15, 23, 42, 0.6);
                border-radius: 16px;
                padding: 25px;
                margin-bottom: 25px;
                border: 1px solid rgba(59, 130, 246, 0.2);
            }

            .section h2 {
                color: #e2e8f0;
                font-size: 1.3rem;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 1px solid rgba(59, 130, 246, 0.3);
            }

            .section h2 span { margin-right: 10px; }

            .chart-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 25px;
                margin-bottom: 25px;
            }

            .chart-box {
                background: rgba(15, 23, 42, 0.4);
                border-radius: 12px;
                padding: 20px;
                border: 1px solid rgba(59, 130, 246, 0.15);
            }

            .chart-box h3 {
                color: #94a3b8;
                font-size: 0.9rem;
                margin-bottom: 15px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }

            .rec-card {
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.8) 100%);
                border-radius: 12px;
                padding: 20px;
                margin: 15px 0;
                border-left: 4px solid #3b82f6;
            }

            .rec-card.p1 { border-left-color: #ef4444; }
            .rec-card.p2 { border-left-color: #f97316; }
            .rec-card.p3 { border-left-color: #3b82f6; }

            .rec-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 12px;
            }

            .rec-priority {
                padding: 4px 12px;
                border-radius: 4px;
                font-weight: 700;
                font-size: 0.8rem;
                color: white;
            }

            .rec-priority.p1 { background: #ef4444; }
            .rec-priority.p2 { background: #f97316; }
            .rec-priority.p3 { background: #3b82f6; }

            .rec-confidence { color: #22c55e; font-size: 0.85rem; }

            .rec-title { color: #e2e8f0; font-weight: 600; font-size: 1.1rem; margin-bottom: 8px; }
            .rec-desc { color: #94a3b8; font-size: 0.9rem; margin-bottom: 15px; }

            .rec-metrics {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 10px;
                font-size: 0.8rem;
            }

            .rec-metric { color: #64748b; }
            .rec-metric strong { color: #e2e8f0; }

            .footer {
                text-align: center;
                padding: 30px;
                color: #64748b;
                font-size: 0.85rem;
            }

            .footer .confidential { color: #ef4444; font-weight: 600; margin-top: 10px; }

            @media print {
                body { background: white; color: #333; padding: 20px; }
                .header { background: #0066CC; color: white; }
                .header h1 { color: white; -webkit-text-fill-color: white; }
                .kpi-card, .section, .chart-box { background: #f8f9fa; border: 1px solid #ddd; }
                .kpi-label, .rec-desc { color: #666; }
            }
    """


def _build_report_charts(df):
    """Generate all chart HTML fragments for the report.

    Returns:
        dict with keys: sunburst_html, severity_html, friction_html, trend_html, engineer_html
    """
    import plotly.io as pio
    from escalation_ai.dashboard.streamlit_app import (
        chart_category_sunburst, chart_friction_by_category,
        chart_trend_timeline, chart_engineer_performance,
    )

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

    return {
        'sunburst_html': sunburst_html,
        'severity_html': severity_html,
        'friction_html': friction_html,
        'trend_html': trend_html,
        'engineer_html': engineer_html,
    }


def _build_report_metrics(df):
    """Calculate KPI metrics for the report header.

    Returns:
        dict with keys: total_cost, revenue_risk, avg_resolution, critical_count, total_records
    """
    from escalation_ai.dashboard.streamlit_app import get_benchmark_costs

    total_cost = df['Financial_Impact'].sum() if 'Financial_Impact' in df.columns else len(df) * get_benchmark_costs()['avg_per_ticket']
    revenue_risk = df['Revenue_At_Risk'].sum() if 'Revenue_At_Risk' in df.columns else total_cost * 0.20
    avg_resolution = df['Predicted_Resolution_Days'].mean() if 'Predicted_Resolution_Days' in df.columns else 5

    critical_count = 0
    for sev_col in ['tickets_data_severity', 'Severity_Norm', 'Severity', 'severity']:
        if sev_col in df.columns:
            critical_count = len(df[df[sev_col].astype(str).str.lower().isin(['critical', 'high', 'major'])])
            break

    return {
        'total_cost': total_cost,
        'revenue_risk': revenue_risk,
        'avg_resolution': avg_resolution,
        'critical_count': critical_count,
        'total_records': len(df),
    }


def _build_recommendations_html(recommendations):
    """Build HTML for strategic recommendation cards."""
    return ''.join([f'''
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
    ''' for rec in recommendations[:4]])


def generate_magnificent_html_report(df):
    """Generate an interactive HTML report with embedded Plotly charts.

    Dark-themed executive report with:
    - KPI cards (Financial Impact, Revenue at Risk, Critical Issues, Avg Resolution)
    - Strategic Recommendations (P1/P2/P3 cards with confidence scores)
    - Embedded interactive charts: Category Sunburst, Severity Distribution,
      Friction by Category, Engineer Performance, Escalation Timeline
    - Plotly.js loaded from CDN for interactivity
    - Print-friendly CSS overrides

    Args:
        df: Processed DataFrame with standard columns.

    Returns:
        str: Complete HTML document with embedded Plotly charts.
    """
    from escalation_ai.dashboard.streamlit_app import generate_strategic_recommendations

    metrics = _build_report_metrics(df)
    charts = _build_report_charts(df)
    recommendations = generate_strategic_recommendations(df)
    css = _build_report_css()
    recs_html = _build_recommendations_html(recommendations)

    total_cost = metrics['total_cost']
    revenue_risk = metrics['revenue_risk']
    avg_resolution = metrics['avg_resolution']
    critical_count = metrics['critical_count']
    total_records = metrics['total_records']

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Escalation Intelligence Report</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
{css}
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
                {recs_html}
            </div>

            <div class="chart-grid">
                <div class="chart-box">
                    <h3>📊 Category & Sub-Category Breakdown</h3>
                    {charts['sunburst_html']}
                </div>
                <div class="chart-box">
                    <h3>🎯 Severity Distribution</h3>
                    {charts['severity_html']}
                </div>
            </div>

            <div class="chart-grid">
                <div class="chart-box">
                    <h3>📈 Friction by Category</h3>
                    {charts['friction_html']}
                </div>
                <div class="chart-box">
                    <h3>👥 Engineer Performance</h3>
                    {charts['engineer_html']}
                </div>
            </div>

            <div class="section">
                <h2><span>📈</span>Escalation Timeline</h2>
                {charts['trend_html']}
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
