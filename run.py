#!/usr/bin/env python3
"""
Escalation AI - Main Entry Point

Run this script to start the Escalation AI analysis pipeline.

Usage:
    python run.py
    
Or with the package installed:
    python -m escalation_ai
"""

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

def main():
    """Main entry point."""
    print("=" * 60)
    print("ESCALATION AI - Strategic Friction Analysis")
    print("=" * 60)
    print()
    
    try:
        from escalation_ai import main_pipeline
        main_pipeline()
    except ImportError as e:
        print(f"Error importing escalation_ai package: {e}")
        print("\nMake sure you're running from the project directory.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nAnalysis cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        logging.exception("Unexpected error")
        sys.exit(1)


if __name__ == "__main__":
    main()
