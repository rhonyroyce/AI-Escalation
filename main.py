"""
Escalation AI - Main Entry Point (LEGACY)

WARNING: This entry point is DEPRECATED. Use run.py instead:
    python run.py                  # Full pipeline + dashboard
    python run.py --no-gui         # Pipeline only
    python run.py --dashboard-only # Dashboard only

This file is maintained only for backward compatibility and will be
removed in a future release.
"""

import sys
import os
import warnings

# Emit deprecation warning
warnings.warn(
    "main.py is deprecated. Use 'python run.py' instead. "
    "See run.py --help for all options.",
    DeprecationWarning,
    stacklevel=2,
)

# Route through the modular package instead of the monolith
from escalation_ai.pipeline.orchestrator import EscalationPipeline

def main_pipeline():
    """Legacy entry point — delegates to modular EscalationPipeline."""
    pipeline = EscalationPipeline()
    pipeline.run()

if __name__ == "__main__":
    main_pipeline()
