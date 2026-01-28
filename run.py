#!/usr/bin/env python3
"""
Escalation AI - Main Entry Point

Run this script to start the Escalation AI analysis pipeline
and launch the interactive Streamlit dashboard.

Usage:
    python run.py                  # Full pipeline + launch dashboard
    python run.py --no-gui         # Just run pipeline (no dashboard)
    python run.py --dashboard-only # Skip pipeline, just launch dashboard
    python run.py --file input.xlsx --output report.xlsx  # Non-interactive mode
    
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
import warnings

# Suppress cuML n_bins warning (it auto-adjusts, just informational)
warnings.filterwarnings('ignore', message='.*n_bins.*greater than.*number of samples.*')

# Force unbuffered output for real-time progress display
os.environ['PYTHONUNBUFFERED'] = '1'

# ==========================================
# ENVIRONMENT CHECK - Check for GPU availability
# ==========================================
REQUIRED_ENV = "ml-gpu"

def check_gpu_available():
    """Check if GPU is available via nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            return True, result.stdout.strip().split('\n')[0]
    except:
        pass
    return False, None

def ensure_environment():
    """Check environment - support both conda and venv with GPU."""
    # First check if GPU is actually available
    gpu_available, gpu_name = check_gpu_available()
    
    # Check for conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    
    # Check for venv (VIRTUAL_ENV is set by venv/virtualenv activation)
    venv_path = os.environ.get('VIRTUAL_ENV', '')
    
    if conda_env == REQUIRED_ENV:
        print(f"✅ Conda environment: {REQUIRED_ENV}")
        if gpu_available:
            print(f"✅ GPU detected: {gpu_name}")
        return True
    elif venv_path and REQUIRED_ENV in venv_path:
        print(f"✅ Virtual environment: {venv_path}")
        if gpu_available:
            print(f"✅ GPU detected: {gpu_name}")
        return True
    elif gpu_available:
        # No specific env but GPU is available - proceed
        if venv_path:
            print(f"✅ Virtual environment: {venv_path}")
        print(f"✅ GPU detected: {gpu_name}")
        return True
    elif conda_env:
        # Wrong conda env - this will cause issues with GPU libs
        print(f"❌ Wrong conda environment: '{conda_env}'")
        print(f"   Required: {REQUIRED_ENV}")
        print()
        print(f"   Please run: conda activate {REQUIRED_ENV}")
        print(f"   Or use: conda run -n {REQUIRED_ENV} python run.py")
        print()
        
        # Ask to continue anyway
        try:
            response = input("   Continue without GPU acceleration? (y/N): ").strip().lower()
            if response != 'y':
                sys.exit(1)
            print("   ⚠️  Continuing with limited functionality...")
            return False
        except (KeyboardInterrupt, EOFError):
            print("\n   Exiting.")
            sys.exit(1)
    else:
        # No conda/venv and no GPU detected
        print(f"⚠️  No GPU detected and no {REQUIRED_ENV} environment")
        print(f"   For GPU acceleration, ensure NVIDIA drivers are installed")
        return False

# Check environment before anything else
_gpu_env = ensure_environment()

# ==========================================
# GPU / VRAM DETECTION DISPLAY
# ==========================================
def display_gpu_info():
    """Display GPU and model configuration."""
    try:
        from escalation_ai.core.config import EMBED_MODEL, GEN_MODEL, DETECTED_VRAM_GB
        print(f"🎮 GPU VRAM Detected: {DETECTED_VRAM_GB:.1f} GB")
        print(f"🧠 Embedding Model: {EMBED_MODEL}")
        print(f"🤖 Generation Model: {GEN_MODEL}")
    except ImportError:
        print("⚠️  Could not load GPU config")

display_gpu_info()
print()
sys.stdout.flush()

# Configure logging - less verbose to let progress bars shine
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
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
  python run.py --file data.xlsx Run with specific input file
  python run.py --file data.xlsx --output report.xlsx  Fully non-interactive
  python run.py --port 8502      Use custom port for dashboard
        """
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Input Excel/CSV file (skips file dialog)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output report file path (skips save dialog)'
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
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed logging output'
    )
    
    return parser.parse_args()


def run_pipeline(input_file=None, output_file=None):
    """Run the analysis pipeline."""
    print()
    print("=" * 60)
    print("  🚀 ESCALATION AI - Strategic Friction Analysis")
    print("=" * 60)
    print()
    sys.stdout.flush()
    
    try:
        from escalation_ai.pipeline.orchestrator import EscalationPipeline
        from escalation_ai.reports import generate_report
        import tkinter as tk
        from tkinter import filedialog, messagebox
        
        # Create pipeline
        pipeline = EscalationPipeline()
        
        # Initialize
        if not pipeline.initialize():
            return False
        
        # Load data
        if input_file:
            if not pipeline.load_data(input_file):
                return False
        else:
            if not pipeline.load_data():
                return False
        
        # Run all phases
        df = pipeline.run_all_phases()
        
        # Generate executive summary
        exec_summary = pipeline.generate_executive_summary()
        
        # Determine output path
        if output_file:
            save_path = output_file
        else:
            root = tk.Tk()
            root.withdraw()
            save_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                initialfile="Strategic_Report.xlsx"
            )
            root.destroy()
            if not save_path:
                print("❌ No output file selected. Aborting.")
                return False
        
        # Generate report
        print()
        print("-" * 60)
        print("  SAVING REPORT")
        print("-" * 60)
        print(f"  📊 Generating Excel report with charts...")
        sys.stdout.flush()
        
        generate_report(df, save_path, exec_summary, pipeline.df_raw)
        
        print(f"  ✅ Saved to: {save_path}")
        print()
        print("=" * 60)
        print("  🎉 ANALYSIS COMPLETE!")
        print("=" * 60)
        print()
        sys.stdout.flush()
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importing escalation_ai package: {e}")
        print("\nMake sure you're running from the project directory.")
        return False
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


def launch_dashboard(port: int = 8501, open_browser: bool = True):
    """Launch the Streamlit dashboard."""
    print()
    print("=" * 60)
    print("  🌐 Launching Interactive Dashboard")
    print("=" * 60)
    print()
    
    # Get the streamlit app path
    dashboard_path = Path(__file__).parent / "escalation_ai" / "dashboard" / "streamlit_app.py"
    
    if not dashboard_path.exists():
        print(f"❌ Error: Dashboard not found at {dashboard_path}")
        return False
    
    print(f"  📊 Starting Streamlit server on port {port}...")
    print(f"  🔗 URL: http://localhost:{port}")
    print()
    print("  Press Ctrl+C to stop the dashboard")
    print("-" * 60)
    sys.stdout.flush()
    
    # Build streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(port),
        "--server.headless", "true",
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
    
    # Enable verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    try:
        if args.dashboard_only:
            # Just launch dashboard
            launch_dashboard(port=args.port, open_browser=not args.no_browser)
            
        elif args.no_gui:
            # Just run pipeline
            success = run_pipeline(input_file=args.file, output_file=args.output)
            if not success:
                sys.exit(1)
            print("✅ Pipeline complete. Run with --dashboard-only to view results.")
            
        else:
            # Full flow: pipeline + dashboard
            print("🔄 Running full analysis pipeline...")
            sys.stdout.flush()
            success = run_pipeline(input_file=args.file, output_file=args.output)
            
            if success:
                print("✅ Launching dashboard to view results...")
                launch_dashboard(port=args.port, open_browser=not args.no_browser)
            else:
                print("\n⚠️ Pipeline had errors. Launching dashboard anyway...")
                launch_dashboard(port=args.port, open_browser=not args.no_browser)
                
    except KeyboardInterrupt:
        print("\n\n👋 Cancelled by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
