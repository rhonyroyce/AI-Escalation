#!/usr/bin/env python3
"""
Escalation AI - Main Entry Point

Run this script to start the Escalation AI analysis pipeline
and launch the interactive Streamlit dashboard.

Usage:
    python run.py                  # Full pipeline + launch dashboard
    python run.py --no-gui         # Just run pipeline (no dashboard)
    python run.py --dashboard-only # Skip pipeline, just launch dashboard
    
Or with the package installed:
    python -m escalation_ai
"""

import logging
import sys
import os
import subprocess
import argparse
import webbrowser
from pathlib import Path
import time

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

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Escalation AI - Strategic Friction Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                  Run full pipeline + launch dashboard
  python run.py --no-gui         Run pipeline only (no dashboard)
  python run.py --dashboard-only Launch dashboard without running pipeline
  python run.py --port 8502      Use custom port for dashboard
        """
    )
    
    parser.add_argument(
        '--no-gui', 
        action='store_true',
        help='Run pipeline only, do not launch dashboard'
    )
    
    parser.add_argument(
        '--dashboard-only', 
        action='store_true',
        help='Launch dashboard without running pipeline'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port for Streamlit dashboard (default: 8501)'
    )
    
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not automatically open browser'
    )
    
    return parser.parse_args()


def run_pipeline():
    """Run the analysis pipeline."""
    print()
    print("=" * 60)
    print("🚀 ESCALATION AI - Strategic Friction Analysis")
    print("=" * 60)
    print()
    
    try:
        from escalation_ai import main_pipeline
        main_pipeline()
        return True
    except ImportError as e:
        print(f"Error importing escalation_ai package: {e}")
        print("\nMake sure you're running from the project directory.")
        return False
    except Exception as e:
        print(f"\nPipeline error: {e}")
        logger.exception("Pipeline error")
        return False


def launch_dashboard(port: int = 8501, open_browser: bool = True):
    """Launch the Streamlit dashboard."""
    print()
    print("=" * 60)
    print("🌐 Launching Interactive Dashboard")
    print("=" * 60)
    print()
    
    # Get the streamlit app path
    dashboard_path = Path(__file__).parent / "escalation_ai" / "dashboard" / "streamlit_app.py"
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard not found at {dashboard_path}")
        return False
    
    print(f"📊 Starting Streamlit server on port {port}...")
    print(f"🔗 URL: http://localhost:{port}")
    print()
    print("Press Ctrl+C to stop the dashboard")
    print("-" * 60)
    
    # Build streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(port),
        "--server.headless", "true",  # Always headless, we handle browser opening ourselves
        "--browser.gatherUsageStats", "false",
        "--theme.base", "dark",
        "--theme.primaryColor", "#0066CC",
        "--theme.backgroundColor", "#0a1929",
        "--theme.secondaryBackgroundColor", "#001e3c",
        "--theme.textColor", "#E0E0E0"
    ]
    
    try:
        # Open browser after a short delay (only once)
        if open_browser:
            def open_browser_delayed():
                time.sleep(2)
                webbrowser.open(f"http://localhost:{port}")
            
            import threading
            browser_thread = threading.Thread(target=open_browser_delayed)
            browser_thread.daemon = True
            browser_thread.start()
        
        # Run streamlit
        subprocess.run(cmd)
        return True
        
    except KeyboardInterrupt:
        print("\n\n✅ Dashboard stopped.")
        return True
    except FileNotFoundError:
        print("\n❌ Streamlit not found. Install with: pip install streamlit plotly")
        return False
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")
        return False


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        if args.dashboard_only:
            # Just launch dashboard
            launch_dashboard(port=args.port, open_browser=not args.no_browser)
            
        elif args.no_gui:
            # Just run pipeline
            success = run_pipeline()
            if not success:
                sys.exit(1)
            print("\n✅ Pipeline complete. Run with --dashboard-only to view results.")
            
        else:
            # Full flow: pipeline + dashboard
            print("🔄 Running full analysis pipeline...")
            success = run_pipeline()
            
            if success:
                print("\n✅ Pipeline complete! Launching dashboard...")
                launch_dashboard(port=args.port, open_browser=not args.no_browser)
            else:
                print("\n⚠️ Pipeline had errors. Launching dashboard anyway...")
                launch_dashboard(port=args.port, open_browser=not args.no_browser)
                
    except KeyboardInterrupt:
        print("\n\n👋 Cancelled by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
