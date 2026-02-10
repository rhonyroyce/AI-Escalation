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
import signal
import atexit
import socket
from tqdm import tqdm

# Suppress cuML n_bins warning (it auto-adjusts, just informational)
warnings.filterwarnings('ignore', message='.*n_bins.*greater than.*number of samples.*')

# ==========================================
# PATH VALIDATION & SECURITY
# ==========================================
def validate_file_path(path: str, must_exist: bool = False) -> Path:
    """
    Validate and sanitize file paths to prevent path traversal attacks.

    Args:
        path: File path to validate
        must_exist: If True, raise error if file doesn't exist

    Returns:
        Resolved Path object

    Raises:
        ValueError: If path is invalid or fails validation
    """
    try:
        resolved = Path(path).resolve()

        # Check if file exists when required
        if must_exist and not resolved.exists():
            raise ValueError(f"File not found: {path}")

        # Prevent path traversal outside project root
        # Allow files in home directory and project directory
        project_root = Path(__file__).parent.resolve()
        home_dir = Path.home().resolve()

        # Check if path is within allowed directories
        resolved_str = str(resolved)
        allowed = (
            resolved_str.startswith(str(project_root)) or
            resolved_str.startswith(str(home_dir))
        )

        if not allowed:
            raise ValueError(f"Path outside allowed directories: {path}")

        return resolved

    except Exception as e:
        raise ValueError(f"Invalid file path '{path}': {e}")


def sanitize_filename(filename: str) -> str:
    """
    Remove potentially dangerous characters from filename.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for filesystem operations
    """
    import re

    # Remove any path separators
    filename = os.path.basename(filename)

    # Remove dangerous characters, keep alphanumeric, spaces, hyphens, underscores, dots
    filename = re.sub(r'[^\w\s\-\.]', '', filename)

    # Limit length to avoid filesystem issues
    return filename[:255]


# ==========================================
# CUDA Environment Setup for Blackwell GPUs (RTX 50xx series, sm_120)
# ==========================================
def setup_cuda_environment():
    """
    Auto-detect and configure CUDA environment for GPU acceleration.
    Handles Blackwell GPUs (sm_120) which require CUDA 12.8+ and NVRTC 12.8+.

    Sets up:
    - CUDA_HOME: Path to CUDA toolkit (for nvcc, libs)
    - CUDA_PATH: Also used by CuPy to find CUDA headers
    - CPATH: Include path for cuda_fp16.h and other CUDA headers
    - LD_LIBRARY_PATH: Path to NVRTC 12.8+ library (from nvidia-cuda-nvrtc-cu12)

    Returns True if environment was modified and process needs restart.
    """
    needs_restart = False
    cuda_install_path = None

    try:
        # Find the best available CUDA installation
        cuda_paths = [
            '/usr/local/cuda-12.9',
            '/usr/local/cuda-12.8',
            '/usr/local/cuda-12',
            '/usr/local/cuda',
        ]

        for cuda_path in cuda_paths:
            try:
                if os.path.exists(cuda_path):
                    cuda_install_path = cuda_path
                    break
            except (OSError, PermissionError):
                continue

        # Set CUDA_HOME, CUDA_PATH, and CPATH for CuPy NVRTC compilation
        if cuda_install_path:
            cuda_include = os.path.join(cuda_install_path, 'include')

            # CUDA_HOME and CUDA_PATH - both used by different tools
            if os.environ.get('CUDA_HOME') != cuda_install_path:
                os.environ['CUDA_HOME'] = cuda_install_path
                needs_restart = True
            if os.environ.get('CUDA_PATH') != cuda_install_path:
                os.environ['CUDA_PATH'] = cuda_install_path
                needs_restart = True

            # CPATH - C include path for NVRTC to find cuda_fp16.h, etc.
            cpath = os.environ.get('CPATH', '')
            if cuda_include not in cpath:
                os.environ['CPATH'] = f"{cuda_include}:{cpath}" if cpath else cuda_include
                needs_restart = True

        # Find NVRTC library in the current Python environment's nvidia packages
        # This is needed for Blackwell GPUs (sm_120) which require NVRTC 12.8+
        try:
            import nvidia.cuda_nvrtc.lib as nvrtc_lib
            nvrtc_lib_path = Path(nvrtc_lib.__path__[0]) if hasattr(nvrtc_lib, '__path__') else None
            if nvrtc_lib_path is None:
                # Try alternate method
                import nvidia.cuda_nvrtc
                pkg_path = getattr(nvidia.cuda_nvrtc, '__path__', None)
                if pkg_path:
                    nvrtc_lib_path = Path(pkg_path[0]) / 'lib'

            if nvrtc_lib_path and nvrtc_lib_path.exists():
                ld_path = os.environ.get('LD_LIBRARY_PATH', '')
                nvrtc_str = str(nvrtc_lib_path)
                if nvrtc_str not in ld_path:
                    os.environ['LD_LIBRARY_PATH'] = f"{nvrtc_str}:{ld_path}"
                    needs_restart = True
        except ImportError:
            pass  # nvidia-cuda-nvrtc not installed, use system NVRTC
        except Exception as e:
            # Log warning but don't fail - system NVRTC might still work
            print(f"⚠️ Warning: Could not configure NVRTC library: {e}", file=sys.stderr)

        return needs_restart

    except Exception as e:
        # Critical error in CUDA setup - log and continue without CUDA
        print(f"⚠️ Warning: Error setting up CUDA environment: {e}", file=sys.stderr)
        return False

# Check if we need to restart with updated LD_LIBRARY_PATH
# This is required because LD_LIBRARY_PATH must be set before Python loads CUDA libraries
MAX_CUDA_RESTARTS = 1

if '_CUDA_ENV_SET' not in os.environ:
    restart_count = int(os.environ.get('_CUDA_RESTART_COUNT', '0'))

    if restart_count >= MAX_CUDA_RESTARTS:
        print(f"⚠️ Maximum CUDA environment restarts reached ({MAX_CUDA_RESTARTS})")
        print("   Continuing without CUDA restart...")
        os.environ['_CUDA_ENV_SET'] = '1'
    else:
        needs_restart = setup_cuda_environment()
        if needs_restart:
            os.environ['_CUDA_ENV_SET'] = '1'
            os.environ['_CUDA_RESTART_COUNT'] = str(restart_count + 1)
            # Restart Python with the updated environment
            try:
                os.execv(sys.executable, [sys.executable] + sys.argv)
            except Exception as e:
                print(f"⚠️ Failed to restart with updated environment: {e}")
                print("   Continuing with current environment...")
        os.environ['_CUDA_ENV_SET'] = '1'


# Force unbuffered output for real-time progress display
os.environ['PYTHONUNBUFFERED'] = '1'

# ==========================================
# ENVIRONMENT CHECK - Check for GPU availability
# ==========================================
REQUIRED_ENV = "ml-gpu"

def check_gpu_available():
    """
    Check if GPU is available via nvidia-smi.

    Returns:
        Tuple of (is_available: bool, gpu_name: str or None)
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=2  # Reduced from 5 seconds for faster startup
        )
        if result.returncode == 0 and result.stdout.strip():
            gpus = result.stdout.strip().split('\n')
            return True, gpus[0] if gpus else None
    except subprocess.TimeoutExpired:
        print("⚠️ GPU check timed out", file=sys.stderr)
    except FileNotFoundError:
        # nvidia-smi not found - no NVIDIA driver installed
        pass
    except Exception as e:
        print(f"⚠️ Unexpected error during GPU check: {e}", file=sys.stderr)

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

# ==========================================
# LOGGING CONFIGURATION
# ==========================================
def setup_logging(verbose: bool = False):
    """
    Configure logging with both file and console handlers.

    Args:
        verbose: If True, set console output to INFO level, otherwise WARNING

    Returns:
        Path: Path to the log file
    """
    # Create logs directory
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Create log file with timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"escalation_ai_{timestamp}.log"

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # File handler (always DEBUG level for complete logs)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Console handler (WARNING or INFO based on verbose flag)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add our handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    if verbose:
        print(f"📝 Verbose logging enabled. Log file: {log_file}")
    else:
        print(f"📝 Logging to: {log_file}")

    return log_file


# Initialize logging with default settings (will be reconfigured in main() if --verbose)
_default_log_file = setup_logging(verbose=False)

logger = logging.getLogger(__name__)


# ==========================================
# HEALTH CHECK UTILITIES
# ==========================================
def check_ollama_server():
    """
    Check if Ollama server is running and accessible.

    Returns:
        bool: True if Ollama is accessible, False otherwise
    """
    try:
        import requests
        response = requests.get('http://localhost:11434/api/version', timeout=2)
        return response.status_code == 200
    except ImportError:
        print("⚠️ requests library not available for Ollama check")
        return False
    except Exception:
        return False


def check_required_packages():
    """
    Check if all required packages are installed.

    Returns:
        Tuple of (all_installed: bool, missing_packages: list)
    """
    required = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'streamlit': 'streamlit',
        'sklearn': 'scikit-learn',
        'openpyxl': 'openpyxl',
        'matplotlib': 'matplotlib',
        'requests': 'requests',
    }

    missing = []
    for import_name, package_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)

    return len(missing) == 0, missing


def health_check():
    """
    Run comprehensive health check of the environment.

    Returns:
        bool: True if all checks pass, False otherwise
    """
    print()
    print("=" * 60)
    print("  🏥 ESCALATION AI - HEALTH CHECK")
    print("=" * 60)
    print()

    # Python version check
    python_version = sys.version_info
    python_ok = python_version >= (3, 9)
    status = "✅" if python_ok else "❌"
    print(f"{status} Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if not python_ok:
        print(f"   Required: Python 3.9+")

    # GPU check
    gpu_available, gpu_name = check_gpu_available()
    status = "✅" if gpu_available else "⚠️ "
    gpu_display = gpu_name if gpu_name else "None"
    print(f"{status} GPU Available: {gpu_display}")

    # CUDA environment check
    cuda_home = os.environ.get('CUDA_HOME')
    cuda_ok = cuda_home is not None
    status = "✅" if cuda_ok else "⚠️ "
    cuda_display = cuda_home if cuda_home else "Not configured"
    print(f"{status} CUDA Environment: {cuda_display}")

    # Ollama server check
    ollama_ok = check_ollama_server()
    status = "✅" if ollama_ok else "❌"
    print(f"{status} Ollama Server: {'Running' if ollama_ok else 'Not accessible'}")
    if not ollama_ok:
        print(f"   Start with: ollama serve")

    # Required packages check
    packages_ok, missing = check_required_packages()
    status = "✅" if packages_ok else "❌"
    print(f"{status} Required Packages: {'All installed' if packages_ok else f'{len(missing)} missing'}")
    if missing:
        print(f"   Missing: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")

    # Project structure check
    project_root = Path(__file__).parent
    required_dirs = ['escalation_ai', 'escalation_ai/pipeline', 'escalation_ai/dashboard']
    structure_ok = all((project_root / d).exists() for d in required_dirs)
    status = "✅" if structure_ok else "❌"
    print(f"{status} Project Structure: {'Valid' if structure_ok else 'Missing directories'}")

    print()
    print("=" * 60)

    # Summary
    all_critical = python_ok and ollama_ok and packages_ok and structure_ok
    if all_critical:
        print("  ✅ All critical checks passed!")
    else:
        print("  ❌ Some critical checks failed. Please fix the issues above.")

    if gpu_available:
        print("  ℹ️  GPU acceleration available")
    else:
        print("  ℹ️  Running in CPU mode (slower)")

    print("=" * 60)
    print()

    return all_critical


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

    parser.add_argument(
        '--health-check',
        action='store_true',
        help='Run system health check and exit'
    )

    return parser.parse_args()


def run_pipeline(input_file=None, output_file=None):
    """
    Run the analysis pipeline with improved error handling.

    Args:
        input_file: Path to input Excel/CSV file (optional)
        output_file: Path to save output report (optional)

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print()
    print("=" * 60)
    print("  🚀 ESCALATION AI - Strategic Friction Analysis")
    print("=" * 60)
    print()
    sys.stdout.flush()

    try:
        # Lazy import to speed up startup
        print("📦 Loading analysis modules...")
        from escalation_ai.pipeline.orchestrator import EscalationPipeline
        from escalation_ai.reports import generate_report

        # Only import GUI libraries if needed
        if not input_file or not output_file:
            import tkinter as tk
            from tkinter import filedialog, messagebox

        # Validate input file if provided
        if input_file:
            try:
                input_file = str(validate_file_path(input_file, must_exist=True))
            except ValueError as e:
                print(f"❌ Invalid input file: {e}")
                return False

        # Validate output file if provided
        if output_file:
            try:
                output_file = str(validate_file_path(output_file, must_exist=False))
            except ValueError as e:
                print(f"❌ Invalid output file: {e}")
                return False

        # Create pipeline
        pipeline = EscalationPipeline()

        # Define pipeline phases for progress tracking
        phases = [
            {"name": "Initialization", "desc": "🔧 Initializing AI models and services"},
            {"name": "Data Loading", "desc": "📂 Loading and validating input data"},
            {"name": "Analysis", "desc": "⚙️  Running 7-phase analysis pipeline"},
            {"name": "Summary", "desc": "📝 Generating executive summary"},
            {"name": "Report", "desc": "💾 Creating Excel report with charts"},
        ]

        # Progress bar for overall pipeline
        with tqdm(total=len(phases), desc="Pipeline Progress", unit="phase",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:

            # Phase 1: Initialize with retry logic
            pbar.set_description(phases[0]["desc"])
            max_retries = 3
            initialized = False

            for attempt in range(max_retries):
                try:
                    if pipeline.initialize():
                        initialized = True
                        break
                    if attempt < max_retries - 1:
                        tqdm.write(f"⚠️  Initialization failed, retrying ({attempt + 1}/{max_retries})...")
                        time.sleep(2)
                except Exception as e:
                    if attempt == max_retries - 1:
                        tqdm.write(f"❌ Failed to initialize after {max_retries} attempts: {e}")
                        logger.error("Pipeline initialization failed", exc_info=True)
                        return False
                    tqdm.write(f"⚠️  Attempt {attempt + 1} failed: {e}")
                    time.sleep(2)

            if not initialized:
                tqdm.write("❌ Pipeline initialization failed")
                return False

            pbar.update(1)

            # Phase 2: Load data
            pbar.set_description(phases[1]["desc"])
            if input_file:
                if not pipeline.load_data(input_file):
                    tqdm.write("❌ Failed to load data file")
                    return False
            else:
                # Temporarily hide progress bar for file dialog
                pbar.clear()
                if not pipeline.load_data():
                    tqdm.write("❌ Data loading cancelled or failed")
                    return False
                pbar.refresh()

            pbar.update(1)

            # Phase 3: Run all analysis phases
            pbar.set_description(phases[2]["desc"])
            df = pipeline.run_all_phases()

            if df is None or df.empty:
                tqdm.write("❌ Pipeline produced no results")
                return False

            pbar.update(1)

            # Phase 4: Generate executive summary
            pbar.set_description(phases[3]["desc"])
            exec_summary = pipeline.generate_executive_summary()
            pbar.update(1)

            # Phase 5: Determine output path and generate report
            pbar.set_description(phases[4]["desc"])

            if output_file:
                save_path = output_file
            else:
                # Temporarily hide progress bar for file dialog
                pbar.clear()
                tqdm.write("💾 Select save location...")
                root = tk.Tk()
                root.withdraw()
                save_path = filedialog.asksaveasfilename(
                    defaultextension=".xlsx",
                    initialfile="Strategic_Report.xlsx",
                    filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
                )
                root.destroy()

                if not save_path:
                    tqdm.write("❌ No output file selected. Aborting.")
                    return False
                pbar.refresh()

            # Generate report
            generate_report(df, save_path, exec_summary, pipeline.df_raw)
            pbar.update(1)

        # Pipeline complete
        print()
        print("=" * 60)
        print("  🎉 ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"  ✅ Report saved to: {save_path}")
        print("=" * 60)
        print()
        sys.stdout.flush()

        return True

    except ImportError as e:
        print(f"❌ Error importing escalation_ai package: {e}")
        print("\nMake sure you're running from the project directory.")
        print("Try: pip install -e .")
        logger.error("Import error", exc_info=True)
        return False
    except KeyboardInterrupt:
        print("\n\n👋 Pipeline interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        logger.error("Pipeline execution failed", exc_info=True)
        print("\nFor detailed logs, run with --verbose flag")
        return False


def launch_dashboard(port: int = 8501, open_browser: bool = True):
    """
    Launch the Streamlit dashboard with proper resource management.

    Args:
        port: Port number for Streamlit server (default: 8501)
        open_browser: Whether to automatically open browser (default: True)

    Returns:
        bool: True if dashboard ran successfully, False on error
    """
    print()
    print("=" * 60)
    print("  🌐 Launching Interactive Dashboard")
    print("=" * 60)
    print()

    # Get the unified app path (serves both Pulse + Escalation AI)
    dashboard_path = Path(__file__).parent / "unified_app.py"

    if not dashboard_path.exists():
        print(f"❌ Error: Dashboard not found at {dashboard_path}")
        return False

    # Check if port is already in use
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port_in_use = sock.connect_ex(('localhost', port)) == 0
        sock.close()

        if port_in_use:
            print(f"⚠️  Port {port} is already in use")
            try:
                response = input("   Try to stop existing process? (y/N): ").strip().lower()
                if response == 'y':
                    print("   Attempting to stop existing Streamlit process...")
                    try:
                        subprocess.run(['pkill', '-f', f'streamlit.*{port}'], timeout=5)
                        time.sleep(2)
                        print("   ✅ Process stopped")
                    except subprocess.TimeoutExpired:
                        print("   ⚠️  Timeout stopping process")
                    except Exception as e:
                        print(f"   ⚠️  Could not stop process: {e}")
                else:
                    print("   Please use a different port with --port flag")
                    return False
            except (KeyboardInterrupt, EOFError):
                print("\n   Cancelled")
                return False
    except Exception as e:
        logger.warning(f"Could not check port status: {e}")

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

    streamlit_process = None

    def cleanup():
        """Cleanup streamlit process on exit."""
        nonlocal streamlit_process
        if streamlit_process and streamlit_process.poll() is None:
            print("\n🧹 Cleaning up dashboard process...")
            streamlit_process.terminate()
            try:
                streamlit_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("   Force killing process...")
                streamlit_process.kill()

    # Register cleanup handler
    atexit.register(cleanup)

    try:
        # Open browser after a short delay (only once)
        if open_browser:
            def open_browser_delayed():
                time.sleep(3)  # Give server time to start
                try:
                    webbrowser.open(f"http://localhost:{port}")
                except Exception as e:
                    logger.warning(f"Failed to open browser: {e}")

            import threading
            browser_thread = threading.Thread(target=open_browser_delayed, daemon=True)
            browser_thread.start()

        # Run streamlit as subprocess with proper management
        streamlit_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait for process to complete
        streamlit_process.wait()
        return True

    except KeyboardInterrupt:
        print("\n\n✅ Dashboard stopped by user.")
        cleanup()
        return True
    except FileNotFoundError:
        print("\n❌ Streamlit not found. Install with: pip install streamlit plotly")
        return False
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")
        logger.error("Dashboard launch failed", exc_info=True)
        cleanup()
        return False


def main():
    """Main entry point."""
    args = parse_args()

    # Reconfigure logging if verbose requested
    if args.verbose:
        setup_logging(verbose=True)

    # Run health check if requested
    if args.health_check:
        success = health_check()
        sys.exit(0 if success else 1)

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
