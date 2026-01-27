# AI Escalation

🚀 **AI-powered telecom escalation analysis with McKinsey-style executive reporting**

## Overview

AI Escalation is a comprehensive Python application that analyzes telecom escalation tickets using AI/ML to provide:

- **AI Classification**: Semantic embedding-based categorization of issues
- **Strategic Friction Scoring**: Weighted risk assessment using McKinsey framework
- **Recidivism Detection**: Pattern analysis to identify repeat offenses
- **Similar Ticket Finder**: Find historical similar tickets with resolution comparison
- **Resolution Time Prediction**: ML-based prediction of expected resolution times
- **Human-in-the-Loop Feedback**: Continuous learning from human corrections
- **Executive Reporting**: Professional charts and Excel reports

## Features

### 🧠 AI-Powered Analysis

- Uses Ollama with local LLM models (qwen3-embedding, gemma3)
- Semantic similarity matching for ticket classification
- 11-category telecom-specific classification system

### 📊 McKinsey-Style Reporting

- 15 executive-quality visualization charts
- Strategic friction scoring
- Risk stratification analysis
- Automated executive summary generation

### 🔄 Continuous Learning

- Human feedback integration for classification improvement
- Similar ticket validation with dropdown feedback
- Resolution time expectation calibration

### 🎯 Predictive Analytics

- Recurrence risk prediction (ML-based)
- Resolution time prediction with accuracy tracking
- Similar ticket pattern analysis

## Prerequisites

### 1. Ollama Installation

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull qwen3-embedding:8b
ollama pull gemma3:27b

# Start Ollama server
ollama serve
```

### 2. Python Dependencies

```bash
pip install -r requirements.txt
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AI-Escalation.git
cd AI-Escalation

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Analysis

```bash
python main.py
```

This will:

1. Open a file dialog to select your Excel data file
2. Process the tickets through the AI pipeline
3. Generate charts in the `output/plots/` directory
4. Create an Excel report with analysis results

### Input Data Format

Your Excel file should contain columns like:

- `tickets_data_issue_summary` - Issue description
- `tickets_data_severity` - Severity level (Critical/Major/Minor)
- `tickets_data_type_1` - Ticket type
- `tickets_data_escalation_origin` - External/Internal
- `tickets_data_issue_datetime` - Issue date
- `tickets_data_close_datetime` - Resolution date

### Output Files

- `output/[timestamp]_analysis.xlsx` - Complete Excel report with multiple sheets
- `output/plots/*.png` - 15 visualization charts
- `classification_feedback.xlsx` - Feedback file for human corrections
- `similarity_feedback.xlsx` - Similar ticket validation feedback

## Project Structure

```text
AI-Escalation/
├── main.py                     # Main entry point
├── EscalationAI0126.py        # Core analysis engine (5700+ lines)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .gitignore                 # Git ignore rules
├── escalation_ai/             # Modular package (in progress)
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration constants
│   │   ├── ai_engine.py       # Ollama AI wrapper
│   │   └── utils.py           # Utility functions
│   ├── models/
│   │   └── __init__.py
│   └── visualization/
│       └── __init__.py
└── output/                    # Generated reports (gitignored)
    └── plots/
```

## Configuration

Key settings in `escalation_ai/core/config.py`:

```python
# AI Models
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "qwen3-embedding:8b"
GEN_MODEL = "gemma3:27b"

# Thresholds
SIMILARITY_THRESHOLD_HIGH = 0.60
MIN_CLASSIFICATION_CONFIDENCE = 0.25

# Weights (McKinsey Framework)
WEIGHTS = {
    'BASE_SEVERITY': {'Critical': 100, 'Major': 50, 'Minor': 10},
    'ORIGIN_MULTIPLIER': {'External': 2.5, 'Internal': 1.0},
    ...
}
```

## Charts Generated

1. **Friction Pareto** - Top strategic friction sources
2. **Risk Origin** - External vs Internal breakdown
3. **Severity Waterfall** - Severity distribution
4. **Root Cause Pie** - Root cause categories
5. **Timeline Trend** - Issues over time
6. **Top Offenders** - Highest friction tickets
7. **Recidivism Matrix** - Repeat offense analysis
8. **Heatmap** - Category × Severity distribution
9. **Engineer Performance** - Engineer patterns
10. **LOB Risk** - Line of Business analysis
11. **Root Cause Deep Dive** - Root cause analysis
12. **PM Prediction Accuracy** - PM forecast accuracy
13. **AI Recurrence Prediction** - ML risk predictions
14. **Resolution Consistency** - Resolution pattern analysis
15. **Resolution Time Comparison** - Expected vs Actual vs Predicted

## Human Feedback Workflow

### Classification Feedback

1. Open `classification_feedback.xlsx`
2. Review AI classifications
3. Correct wrong categories in dropdown
4. Save and re-run analysis

### Similarity Feedback

1. Open `similarity_feedback.xlsx`
2. Review ticket pairs marked by AI
3. Select "Similar" or "Not Similar" in dropdown
4. Enter expected resolution times
5. Save and re-run for improved predictions

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## Support

For issues and feature requests, please use the GitHub Issues tracker.
