"""
Escalation AI - Main Entry Point

AI-powered telecom escalation analysis with McKinsey-style executive reporting.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main pipeline from the monolithic file (for now)
# TODO: Complete refactoring into modular structure
from EscalationAI0126 import main_pipeline

if __name__ == "__main__":
    main_pipeline()
