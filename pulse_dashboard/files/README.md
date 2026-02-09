# CSE Unit — Project Pulse Dashboard

Streamlit recreation of the Power BI Project Pulse dashboard with full interactivity and drill-down capabilities.

## Features

### 1. Pulse Ranking (Flow Chart)
- Flowing bands showing each scoring dimension's relative ranking week-over-week
- Smooth spline interpolation between data points
- **Click any band** to drill down into that dimension

### 2. Project Pulse – Region | Area Table
- Hierarchical view: Region → Area breakdown
- Sparkline trends for each row
- Conditional color formatting on all score cells
- Expand/collapse regions

### 3. Weekly Trend Heatmap
- Regions as rows, weeks as columns
- Color-coded cells (Red < 14, Yellow 14-16, Green 17+)
- Total row aggregation

### 4. Ratings Legend
- LOB, CSAT, PM Perf, Potential rating definitions
- Project Pulse color scale reference

### 5. Drill-Down Panel
When you click a dimension in the Pulse Ranking chart:
- Bar chart: Avg score by region
- Distribution: Score 0/1/2/3 breakdown
- Project table: All projects sorted by that score
- Trend lines: Weekly trend by region

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Place your Pulse_Tracker.xlsx in the same folder

# Run the dashboard
streamlit run pulse_dashboard_full.py
```

---

## Leveraging Claude Code

Open Claude Code in VS Code and try prompts like:

### Add More Drill-Downs
```
Add click handlers to the Region/Area table so when I click a region, 
it drills down to show all projects in that region with their scores.
```

### Customize Styling
```
Change the color theme to match our corporate brand: 
primary blue #0047AB, accent orange #FF6B35
```

### Add Export
```
Add a button to export the current filtered view to Excel with formatting
```

### Add Filters
```
Add a PM Name filter dropdown that filters all charts and tables
```

### Enhance the Heatmap
```
Make the weekly heatmap cells clickable - clicking a cell should 
show all projects for that region+week combination
```

---

## File Structure

```
├── pulse_dashboard_full.py    # Main Streamlit app
├── Pulse_Tracker.xlsx         # Data source (your file)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Data Requirements

The Excel file must have a sheet named "Project Pulse" with columns:
- Wk (week number)
- Region
- Area
- Project
- PM Name
- Design, IX, PAG, RF Opt, Field, CSAT, PM Performance, Potential (scores 0-3)
- Total Score (sum of above, 0-24)

---

## Customization Tips

### Change Week Range Display
Edit line ~320 to show more/fewer weeks in the heatmap:
```python
display_weeks = sorted(fdf['Wk'].unique())[-12:]  # Show last 12 weeks
```

### Add New Scoring Dimensions
Add to the SCORE_COLS list and COLORS dict at the top of the file.

### Change Pulse Thresholds
Edit the `pulse_class()` function around line 180.
