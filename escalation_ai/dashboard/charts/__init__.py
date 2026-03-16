"""Deep Analysis chart modules.

Each module corresponds to one tab in render_deep_analysis() and exposes
a single ``render_tab(df)`` entry-point that handles Streamlit layout and
chart rendering for that tab.  Pure figure-building helpers return
``go.Figure`` (or ``None`` when required columns are missing).
"""

from escalation_ai.dashboard.charts.category_charts import render_tab as render_categories_tab
from escalation_ai.dashboard.charts.engineer_charts import render_tab as render_engineers_tab
from escalation_ai.dashboard.charts.root_cause_charts import render_tab as render_root_cause_tab
from escalation_ai.dashboard.charts.pattern_charts import render_tab as render_patterns_tab
from escalation_ai.dashboard.charts.similarity_charts import render_tab as render_similarity_tab
from escalation_ai.dashboard.charts.lessons_charts import render_tab as render_lessons_tab

__all__ = [
    "render_categories_tab",
    "render_engineers_tab",
    "render_root_cause_tab",
    "render_patterns_tab",
    "render_similarity_tab",
    "render_lessons_tab",
]
