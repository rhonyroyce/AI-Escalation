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
import os
import subprocess

# ==========================================
# CONDA ENVIRONMENT CHECK
# ==========================================
REQUIRED_ENV = "ml-gpu"

def ensure_conda_env():
    """Ensure we're running in the correct conda environment."""
    current_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    
    if current_env != REQUIRED_ENV:
        print(f"⚠️  Wrong conda environment: '{current_env or 'none'}'")
        print(f"📦 Required environment: '{REQUIRED_ENV}'")
        print()
        print(f"Please activate the correct environment:")
        print(f"    conda activate {REQUIRED_ENV}")
        print()
        
        # Ask user if they want to continue anyway
        try:
            response = input("Continue anyway? (y/N): ").strip().lower()
            if response != 'y':
                print("Exiting. Please activate ml-gpu environment.")
                sys.exit(1)
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(1)
    else:
        print(f"✅ Conda environment: {REQUIRED_ENV}")

# Check environment before anything else
ensure_conda_env()

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
