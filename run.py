#!/usr/bin/env python3
"""
Escalation AI - Main CLI Entry Point
=====================================

This script is the single command-line entry point that orchestrates the entire
Escalation AI system. It performs three major stages in sequence:

STAGE 1 -- ML Analysis Pipeline  (run_pipeline)
    Runs a 7-phase machine-learning pipeline via the EscalationPipeline orchestrator:
      Phase 1: Data Loading & Validation   -- ingest raw Excel/CSV escalation data
      Phase 2: AI Classification           -- Ollama LLM categorises each ticket
      Phase 3: Strategic Friction Scoring  -- composite risk score per escalation
      Phase 4: Recidivism / Learning       -- detects repeat offenders & patterns
      Phase 5: Similar Ticket Analysis     -- GPU-accelerated embedding search
      Phase 6: Resolution Time Prediction  -- ML model predicts time-to-resolve
      Phase 7: Report Generation           -- produces a multi-sheet Excel report
    The pipeline outputs Strategic_Report.xlsx with all enriched data and charts.

STAGE 2 -- AI Insight Pre-Generation  (pre_generate_ai_cache)
    Before launching the interactive dashboard, this stage calls the local
    Ollama LLM server (qwen3:14b for generation, qwen2:1.5b for embeddings)
    to pre-compute expensive AI artefacts:
      - Executive Summary   -- LLM-written summary of top pain points
      - Issue Classification -- each pain point classified into 11 categories
      - Embeddings Index    -- vector index for semantic search in the dashboard
    Results are serialised to .cache/ai_insights.pkl so the Streamlit dashboard
    can load them instantly at startup instead of re-generating on every page view.

STAGE 3 -- Unified Streamlit Dashboard  (launch_dashboard)
    Spawns a Streamlit subprocess running unified_app.py, which serves both the
    Project Pulse dashboard and the Escalation AI dashboard as a single app.
    The dashboard reads Strategic_Report.xlsx and .cache/ai_insights.pkl.

Usage:
    python run.py                  # Full pipeline + launch dashboard
    python run.py --no-gui         # Just run pipeline (no dashboard)
    python run.py --dashboard-only # Skip pipeline, just launch dashboard
    python run.py --file input.xlsx --output report.xlsx  # Non-interactive mode

Or with the package installed:
    python -m escalation_ai

Environment Requirements:
    - Python 3.9+
    - NVIDIA GPU with CUDA 12.8+ (for Blackwell / sm_120) -- optional, falls back to CPU
    - Ollama server running locally on port 11434 (for AI classification & insights)
    - Conda environment "ml-gpu" or equivalent venv with GPU packages installed
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

# ---------------------------------------------------------------------------
# Suppress a benign cuML warning that fires when the number of histogram bins
# exceeds the sample count.  cuML auto-adjusts internally, so the warning is
# purely informational and clutters the output.
# ---------------------------------------------------------------------------
warnings.filterwarnings('ignore', message='.*n_bins.*greater than.*number of samples.*')

# ==========================================
# PATH VALIDATION & SECURITY
# ==========================================
# These utilities guard against path-traversal attacks when accepting file
# paths from CLI arguments.  Every user-supplied path is resolved to an
# absolute path and checked against an allowlist of directories (the project
# root and the user's home directory) before being used.

def validate_file_path(path: str, must_exist: bool = False) -> Path:
    """
    Validate and sanitize a file path to prevent path traversal attacks.

    Security rationale: user-supplied paths from --file / --output flags could
    reference files outside the project (e.g. "../../etc/passwd").  This
    function resolves symlinks, then checks the result lives under either
    the project root or the user's home directory.

    Args:
        path: Raw file path string from user input or CLI argument.
        must_exist: When True, raise ValueError if the resolved path does not
                    exist on disk (used for input files that must already exist).

    Returns:
        A fully-resolved Path object guaranteed to be within allowed directories.

    Raises:
        ValueError: If the path is malformed, outside allowed directories,
                    or does not exist when must_exist is True.
    """
    try:
        # Resolve symlinks and ".." components to get the canonical absolute path
        resolved = Path(path).resolve()

        # Check if file exists when required (input files must be present)
        if must_exist and not resolved.exists():
            raise ValueError(f"File not found: {path}")

        # Prevent path traversal outside project root.
        # We allow the project directory (for report output) and the user's
        # home directory (for input files stored elsewhere on the machine).
        project_root = Path(__file__).parent.resolve()
        home_dir = Path.home().resolve()

        # Convert to string for prefix comparison
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
    Strip dangerous characters from a filename to make it filesystem-safe.

    This is a defence-in-depth measure: even after validate_file_path confirms
    the directory is allowed, we strip characters that could cause issues on
    certain filesystems or be used for injection.

    Args:
        filename: Original filename string (may contain path separators or
                  special characters).

    Returns:
        A cleaned filename containing only alphanumeric characters, spaces,
        hyphens, underscores, and dots -- truncated to 255 chars to avoid
        filesystem length limits.
    """
    import re

    # os.path.basename strips directory components, so "../foo" becomes "foo"
    filename = os.path.basename(filename)

    # Keep only safe characters: word chars (\w = [a-zA-Z0-9_]), spaces,
    # hyphens, and dots.  Everything else (semicolons, pipes, backticks, etc.)
    # is removed.
    filename = re.sub(r'[^\w\s\-\.]', '', filename)

    # Truncate to 255 characters -- the max filename length on ext4 / NTFS
    return filename[:255]


# ==========================================
# CUDA Environment Setup for Blackwell GPUs (RTX 50xx series, sm_120)
# ==========================================
# Blackwell-architecture GPUs (compute capability sm_120) require CUDA 12.8+
# and a matching NVRTC (runtime compiler) library.  Most pip-installed packages
# ship NVRTC as a Python wheel (nvidia-cuda-nvrtc-cu12), but the dynamic linker
# won't find it unless LD_LIBRARY_PATH is set BEFORE Python loads CUDA.
#
# Because LD_LIBRARY_PATH is read at process start, we detect the need and
# re-exec the Python process with the updated environment.  A sentinel env var
# (_CUDA_ENV_SET) prevents infinite restart loops.

def setup_cuda_environment():
    """
    Auto-detect and configure CUDA environment variables for GPU acceleration.

    This function solves a bootstrapping problem: CuPy and cuML need CUDA
    headers and the NVRTC shared library at import time, but the paths differ
    between system CUDA installations (/usr/local/cuda-*) and pip-installed
    nvidia-* wheels.  We probe both locations and set the appropriate env vars.

    Environment variables configured:
        CUDA_HOME    -- toolkit root, used by nvcc and CuPy
        CUDA_PATH    -- alias used by some build tools (CMake, CuPy fallback)
        CPATH        -- C include path so NVRTC can find cuda_fp16.h at JIT time
        LD_LIBRARY_PATH -- path to the NVRTC .so from the nvidia-cuda-nvrtc wheel

    Returns:
        True if any environment variable was changed, meaning the process must
        be re-executed for the changes to take effect (because the dynamic
        linker caches LD_LIBRARY_PATH at startup).
    """
    needs_restart = False
    cuda_install_path = None

    try:
        # Probe for the best (newest) system CUDA installation.
        # We prefer 12.9 > 12.8 > 12 > generic symlink, because Blackwell
        # requires at least 12.8.
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
                # Path might be on an inaccessible mount -- skip it
                continue

        # Set CUDA_HOME, CUDA_PATH, and CPATH so CuPy's NVRTC can compile
        # device kernels at import time (e.g. for cuML's KNN).
        if cuda_install_path:
            cuda_include = os.path.join(cuda_install_path, 'include')

            # CUDA_HOME -- the "official" way to point tools at the CUDA toolkit
            if os.environ.get('CUDA_HOME') != cuda_install_path:
                os.environ['CUDA_HOME'] = cuda_install_path
                needs_restart = True
            # CUDA_PATH -- secondary variable checked by CuPy and CMake
            if os.environ.get('CUDA_PATH') != cuda_install_path:
                os.environ['CUDA_PATH'] = cuda_install_path
                needs_restart = True

            # CPATH -- tells the C preprocessor where to find cuda_fp16.h and
            # similar headers that NVRTC needs for JIT-compiling GPU kernels.
            cpath = os.environ.get('CPATH', '')
            if cuda_include not in cpath:
                os.environ['CPATH'] = f"{cuda_include}:{cpath}" if cpath else cuda_include
                needs_restart = True

        # Locate the NVRTC shared library inside the pip-installed nvidia wheel.
        # This is necessary because Blackwell requires NVRTC 12.8+, and the
        # system NVRTC (if present) might be older.  The pip wheel ships the
        # correct version but installs it in a non-standard location that the
        # dynamic linker won't search by default.
        try:
            import nvidia.cuda_nvrtc.lib as nvrtc_lib
            nvrtc_lib_path = Path(nvrtc_lib.__path__[0]) if hasattr(nvrtc_lib, '__path__') else None
            if nvrtc_lib_path is None:
                # Alternate discovery method: navigate from the package root
                import nvidia.cuda_nvrtc
                pkg_path = getattr(nvidia.cuda_nvrtc, '__path__', None)
                if pkg_path:
                    nvrtc_lib_path = Path(pkg_path[0]) / 'lib'

            # Prepend the wheel's lib directory to LD_LIBRARY_PATH so the
            # dynamic linker finds libnvrtc.so before the (possibly older)
            # system version.
            if nvrtc_lib_path and nvrtc_lib_path.exists():
                ld_path = os.environ.get('LD_LIBRARY_PATH', '')
                nvrtc_str = str(nvrtc_lib_path)
                if nvrtc_str not in ld_path:
                    os.environ['LD_LIBRARY_PATH'] = f"{nvrtc_str}:{ld_path}"
                    needs_restart = True
        except ImportError:
            pass  # nvidia-cuda-nvrtc pip package not installed; fall back to system NVRTC
        except Exception as e:
            # Non-fatal: log and continue.  If the system NVRTC is new enough
            # (>= 12.8), everything will still work.
            print(f"\u26a0\ufe0f Warning: Could not configure NVRTC library: {e}", file=sys.stderr)

        return needs_restart

    except Exception as e:
        # If CUDA setup itself crashes, we still want the pipeline to run on CPU
        print(f"\u26a0\ufe0f Warning: Error setting up CUDA environment: {e}", file=sys.stderr)
        return False

# ---------------------------------------------------------------------------
# One-time CUDA environment bootstrap (runs at import time)
#
# Because LD_LIBRARY_PATH is baked into the dynamic linker at process start,
# simply calling os.environ[...] = ... is not enough -- we must re-exec the
# Python process.  The sentinel variable _CUDA_ENV_SET prevents an infinite
# restart loop, and _CUDA_RESTART_COUNT caps restarts at MAX_CUDA_RESTARTS
# in case of pathological configurations.
# ---------------------------------------------------------------------------
MAX_CUDA_RESTARTS = 1  # At most one restart is needed (set env + re-exec once)

if '_CUDA_ENV_SET' not in os.environ:
    restart_count = int(os.environ.get('_CUDA_RESTART_COUNT', '0'))

    if restart_count >= MAX_CUDA_RESTARTS:
        # Safety valve: if we already restarted once and still land here,
        # something unexpected happened.  Give up on CUDA env setup and
        # proceed with whatever environment we have.
        print(f"\u26a0\ufe0f Maximum CUDA environment restarts reached ({MAX_CUDA_RESTARTS})")
        print("   Continuing without CUDA restart...")
        os.environ['_CUDA_ENV_SET'] = '1'
    else:
        needs_restart = setup_cuda_environment()
        if needs_restart:
            # Mark that we've configured the environment, bump the counter,
            # and replace the current process with a fresh Python invocation
            # that will inherit the updated LD_LIBRARY_PATH.
            os.environ['_CUDA_ENV_SET'] = '1'
            os.environ['_CUDA_RESTART_COUNT'] = str(restart_count + 1)
            # os.execv replaces the process image -- it never returns on success.
            try:
                os.execv(sys.executable, [sys.executable] + sys.argv)
            except Exception as e:
                # If execv fails (e.g. permission error), fall through gracefully
                print(f"\u26a0\ufe0f Failed to restart with updated environment: {e}")
                print("   Continuing with current environment...")
        # Either no restart was needed, or execv failed -- either way, mark done
        os.environ['_CUDA_ENV_SET'] = '1'


# ---------------------------------------------------------------------------
# Force unbuffered stdout so tqdm progress bars and print() output appear in
# real time, even when stdout is redirected to a pipe or log file.
# ---------------------------------------------------------------------------
os.environ['PYTHONUNBUFFERED'] = '1'

# ==========================================
# ENVIRONMENT CHECK - Check for GPU availability
# ==========================================
# The pipeline benefits enormously from GPU acceleration (cuML for KNN,
# CuPy for matrix ops).  We check for the expected conda/venv environment
# and for a working NVIDIA GPU via nvidia-smi.  If neither is found, the
# user is warned and given the option to continue on CPU.

REQUIRED_ENV = "ml-gpu"  # Expected conda environment name for GPU support

def check_gpu_available():
    """
    Probe for an NVIDIA GPU by running nvidia-smi.

    We query nvidia-smi rather than importing a CUDA library because nvidia-smi
    is always available when NVIDIA drivers are installed, even if no Python GPU
    packages are present.  The 2-second timeout keeps startup fast on machines
    without a GPU (where nvidia-smi may hang or be absent).

    Returns:
        Tuple of (is_available: bool, gpu_name: str or None).
        gpu_name is the human-readable name of the first detected GPU,
        e.g. "NVIDIA GeForce RTX 5090".
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=2  # Short timeout: nvidia-smi responds instantly if drivers are loaded
        )
        if result.returncode == 0 and result.stdout.strip():
            gpus = result.stdout.strip().split('\n')
            return True, gpus[0] if gpus else None
    except subprocess.TimeoutExpired:
        print("\u26a0\ufe0f GPU check timed out", file=sys.stderr)
    except FileNotFoundError:
        # nvidia-smi binary not found -- NVIDIA drivers are not installed
        pass
    except Exception as e:
        print(f"\u26a0\ufe0f Unexpected error during GPU check: {e}", file=sys.stderr)

    return False, None

def ensure_environment():
    """
    Verify that the runtime environment is suitable for the pipeline.

    Checks for the correct conda/venv environment and GPU availability.
    If the wrong conda env is active (e.g. "base" instead of "ml-gpu"),
    the user is warned because GPU libraries like cuML won't be importable.
    The function offers an interactive prompt to continue on CPU if desired.

    Returns:
        bool: True if GPU acceleration is available, False if running CPU-only.
    """
    # First check if GPU is actually available (driver-level check)
    gpu_available, gpu_name = check_gpu_available()

    # Check for conda environment (set by `conda activate`)
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')

    # Check for venv (VIRTUAL_ENV is set by `source .venv/bin/activate`)
    venv_path = os.environ.get('VIRTUAL_ENV', '')

    # Priority 1: correct conda env -- best case, all GPU libs are available
    if conda_env == REQUIRED_ENV:
        print(f"\u2705 Conda environment: {REQUIRED_ENV}")
        if gpu_available:
            print(f"\u2705 GPU detected: {gpu_name}")
        return True
    # Priority 2: venv whose path contains the expected env name
    elif venv_path and REQUIRED_ENV in venv_path:
        print(f"\u2705 Virtual environment: {venv_path}")
        if gpu_available:
            print(f"\u2705 GPU detected: {gpu_name}")
        return True
    # Priority 3: GPU is present even without the expected env name
    elif gpu_available:
        # No specific env but GPU is available -- optimistically proceed
        if venv_path:
            print(f"\u2705 Virtual environment: {venv_path}")
        print(f"\u2705 GPU detected: {gpu_name}")
        return True
    elif conda_env:
        # Wrong conda env AND no GPU -- GPU libs will be missing
        print(f"\u274c Wrong conda environment: '{conda_env}'")
        print(f"   Required: {REQUIRED_ENV}")
        print()
        print(f"   Please run: conda activate {REQUIRED_ENV}")
        print(f"   Or use: conda run -n {REQUIRED_ENV} python run.py")
        print()

        # Interactive fallback: let the user decide whether to continue on CPU
        try:
            response = input("   Continue without GPU acceleration? (y/N): ").strip().lower()
            if response != 'y':
                sys.exit(1)
            print("   \u26a0\ufe0f  Continuing with limited functionality...")
            return False
        except (KeyboardInterrupt, EOFError):
            print("\n   Exiting.")
            sys.exit(1)
    else:
        # No conda/venv and no GPU detected -- lowest-priority path
        print(f"\u26a0\ufe0f  No GPU detected and no {REQUIRED_ENV} environment")
        print(f"   For GPU acceleration, ensure NVIDIA drivers are installed")
        return False

# ---------------------------------------------------------------------------
# Run the environment check immediately at import time so the user sees
# feedback before the slow imports of pandas/numpy/sklearn that follow.
# The result (_gpu_env) is not used further -- the check is purely for
# user-facing status output and the optional early-exit prompt.
# ---------------------------------------------------------------------------
_gpu_env = ensure_environment()

# ==========================================
# GPU / VRAM DETECTION DISPLAY
# ==========================================
# After confirming the environment, display which AI models were auto-selected
# based on available GPU VRAM.  The config module (escalation_ai.core.config)
# probes VRAM and chooses heavier or lighter Ollama models accordingly
# (e.g. qwen3:14b on 16 GB VRAM vs qwen3:8b on 8 GB).

def display_gpu_info():
    """Display the auto-selected GPU configuration to the user.

    Imports EMBED_MODEL, GEN_MODEL, and DETECTED_VRAM_GB from the core config
    module and prints them.  This import also triggers VRAM detection, so it
    doubles as a warm-up for the config subsystem.
    """
    try:
        from escalation_ai.core.config import EMBED_MODEL, GEN_MODEL, DETECTED_VRAM_GB
        print(f"\U0001f3ae GPU VRAM Detected: {DETECTED_VRAM_GB:.1f} GB")
        print(f"\U0001f9e0 Embedding Model: {EMBED_MODEL}")
        print(f"\U0001f916 Generation Model: {GEN_MODEL}")
    except ImportError:
        print("\u26a0\ufe0f  Could not load GPU config")

# Print GPU/model info immediately so the user sees it during the slow
# import phase that follows.
display_gpu_info()
print()
sys.stdout.flush()  # Flush before heavy imports may block stdout

# ==========================================
# LOGGING CONFIGURATION
# ==========================================
# We configure dual-output logging: a verbose DEBUG-level log file (for
# post-mortem debugging) and a quieter console handler (WARNING by default,
# INFO with --verbose).  The log file is timestamped so successive runs
# don't overwrite each other.

def setup_logging(verbose: bool = False):
    """
    Configure the root logger with file and console handlers.

    Every run produces a dedicated log file under logs/ with a timestamp in
    the filename.  The file handler always captures DEBUG-level messages for
    full traceability, while the console handler shows only warnings (or info
    in verbose mode) to keep terminal output clean.

    Args:
        verbose: When True, lower the console handler to INFO level so the
                 user sees detailed progress messages in the terminal.

    Returns:
        Path: Absolute path to the newly created log file.
    """
    # Create the logs directory next to this script if it doesn't exist
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Timestamp-based filename prevents overwriting previous runs' logs
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"escalation_ai_{timestamp}.log"

    # File formatter includes the logger name for tracing across submodules
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Console formatter is shorter because the user is watching in real time
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # File handler: always DEBUG so nothing is lost
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Console handler: WARNING normally, INFO with --verbose
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    console_handler.setFormatter(console_formatter)

    # Configure the root logger (all library loggers inherit from this)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove any pre-existing handlers to avoid duplicate log lines
    # (this function may be called twice: once at import, once in main()
    # if the user passes --verbose).
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Attach our handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    if verbose:
        print(f"\U0001f4dd Verbose logging enabled. Log file: {log_file}")
    else:
        print(f"\U0001f4dd Logging to: {log_file}")

    return log_file


# ---------------------------------------------------------------------------
# Initialize logging immediately with default (non-verbose) settings.
# If the user passes --verbose, main() will call setup_logging again to
# reconfigure the console handler to INFO level.
# ---------------------------------------------------------------------------
_default_log_file = setup_logging(verbose=False)

# Module-level logger for this file
logger = logging.getLogger(__name__)


# ==========================================
# HEALTH CHECK UTILITIES
# ==========================================
# These functions verify that all external dependencies (Ollama, Python
# packages, project structure) are operational.  They are used by the
# --health-check CLI flag to give the user a quick diagnostic report
# before attempting a full pipeline run.

def check_ollama_server():
    """
    Check if the Ollama LLM server is running and responding.

    Ollama must be running locally on port 11434 for AI classification and
    insight generation.  We hit the /api/version endpoint because it is the
    lightest endpoint and doesn't load a model.

    Returns:
        bool: True if Ollama responded with HTTP 200, False otherwise.
    """
    try:
        import requests
        response = requests.get('http://localhost:11434/api/version', timeout=2)
        return response.status_code == 200
    except ImportError:
        print("\u26a0\ufe0f requests library not available for Ollama check")
        return False
    except Exception:
        return False


def check_required_packages():
    """
    Verify that core Python packages are importable.

    This is a quick smoke test: if any of these imports fail, the pipeline
    will crash later with a less helpful error.  We map import names to pip
    package names so the error message tells the user exactly what to install.

    Returns:
        Tuple of (all_installed: bool, missing_packages: list[str]).
        missing_packages contains pip install names, not import names.
    """
    # Mapping: Python import name -> pip install name
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
    Run a comprehensive diagnostic check and print a human-readable report.

    Checks performed:
        1. Python version (>= 3.9 required)
        2. GPU availability via nvidia-smi
        3. CUDA_HOME environment variable
        4. Ollama server connectivity
        5. Required Python packages
        6. Project directory structure (escalation_ai/, pipeline/, dashboard/)

    Returns:
        bool: True if all CRITICAL checks passed (Python, Ollama, packages,
              structure).  GPU/CUDA are informational only -- the pipeline
              can run on CPU, just slower.
    """
    print()
    print("=" * 60)
    print("  \U0001f3e5 ESCALATION AI - HEALTH CHECK")
    print("=" * 60)
    print()

    # --- Python version ---
    python_version = sys.version_info
    python_ok = python_version >= (3, 9)
    status = "\u2705" if python_ok else "\u274c"
    print(f"{status} Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if not python_ok:
        print(f"   Required: Python 3.9+")

    # --- GPU availability (informational, not critical) ---
    gpu_available, gpu_name = check_gpu_available()
    status = "\u2705" if gpu_available else "\u26a0\ufe0f "
    gpu_display = gpu_name if gpu_name else "None"
    print(f"{status} GPU Available: {gpu_display}")

    # --- CUDA environment (informational, not critical) ---
    cuda_home = os.environ.get('CUDA_HOME')
    cuda_ok = cuda_home is not None
    status = "\u2705" if cuda_ok else "\u26a0\ufe0f "
    cuda_display = cuda_home if cuda_home else "Not configured"
    print(f"{status} CUDA Environment: {cuda_display}")

    # --- Ollama LLM server (critical for AI classification) ---
    ollama_ok = check_ollama_server()
    status = "\u2705" if ollama_ok else "\u274c"
    print(f"{status} Ollama Server: {'Running' if ollama_ok else 'Not accessible'}")
    if not ollama_ok:
        print(f"   Start with: ollama serve")

    # --- Python packages (critical for pipeline execution) ---
    packages_ok, missing = check_required_packages()
    status = "\u2705" if packages_ok else "\u274c"
    print(f"{status} Required Packages: {'All installed' if packages_ok else f'{len(missing)} missing'}")
    if missing:
        print(f"   Missing: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")

    # --- Project directory structure ---
    project_root = Path(__file__).parent
    required_dirs = ['escalation_ai', 'escalation_ai/pipeline', 'escalation_ai/dashboard']
    structure_ok = all((project_root / d).exists() for d in required_dirs)
    status = "\u2705" if structure_ok else "\u274c"
    print(f"{status} Project Structure: {'Valid' if structure_ok else 'Missing directories'}")

    print()
    print("=" * 60)

    # --- Summary ---
    all_critical = python_ok and ollama_ok and packages_ok and structure_ok
    if all_critical:
        print("  \u2705 All critical checks passed!")
    else:
        print("  \u274c Some critical checks failed. Please fix the issues above.")

    if gpu_available:
        print("  \u2139\ufe0f  GPU acceleration available")
    else:
        print("  \u2139\ufe0f  Running in CPU mode (slower)")

    print("=" * 60)
    print()

    return all_critical


def parse_args():
    """
    Parse and return command-line arguments.

    Supported modes:
        (default)        -- run full pipeline, pre-generate AI cache, launch dashboard
        --no-gui         -- run pipeline only, skip dashboard (for CI or headless servers)
        --dashboard-only -- skip pipeline, just launch Streamlit (assumes report exists)
        --health-check   -- run diagnostics and exit
        --file / --output-- bypass tkinter file dialogs for fully non-interactive runs

    Returns:
        argparse.Namespace with the parsed flags.
    """
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

    # Input/output file overrides -- bypass tkinter file dialogs
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

    # Execution mode flags (mutually exclusive in practice)
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

    # Dashboard configuration
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

    # Debugging / diagnostics
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
    Execute the 7-phase ML analysis pipeline (Stage 1 of the system).

    This is the computational core of Escalation AI.  It delegates to the
    EscalationPipeline orchestrator which runs these phases in order:
        1. Data Loading & Validation
        2. AI Classification (Ollama LLM assigns categories per ticket)
        3. Strategic Friction Scoring (composite risk score)
        4. Recidivism / Learning Analysis (repeat-offender detection)
        5. Similar Ticket Analysis (embedding-based nearest-neighbor search)
        6. Resolution Time Prediction (ML regression model)
        7. Report Generation (multi-sheet Excel with embedded charts)

    Initialization has retry logic (3 attempts with 2-second backoff) because
    Ollama model loading can be slow on first run and may time out.

    If --file and --output are not provided, tkinter file dialogs are shown
    for interactive file selection.  When both are provided, the pipeline
    runs fully non-interactively (suitable for CI/cron).

    Args:
        input_file: Path to input Excel/CSV file.  If None, a tkinter file
                    dialog is shown.
        output_file: Path to save the output report.  If None, a tkinter
                     save dialog is shown.

    Returns:
        bool: True if the pipeline completed and saved the report successfully,
              False on any error (import failure, data loading error, empty
              results, user cancellation, etc.).
    """
    print()
    print("=" * 60)
    print("  \U0001f680 ESCALATION AI - Strategic Friction Analysis")
    print("=" * 60)
    print()
    sys.stdout.flush()

    try:
        # Lazy-import heavy modules here (not at file top) so that startup
        # remains fast for --dashboard-only and --health-check modes.
        print("\U0001f4e6 Loading analysis modules...")
        from escalation_ai.pipeline.orchestrator import EscalationPipeline
        from escalation_ai.reports import generate_report

        # Only import tkinter GUI libraries if we'll need file dialogs
        # (avoids errors on headless servers when both paths are provided)
        if not input_file or not output_file:
            import tkinter as tk
            from tkinter import filedialog, messagebox

        # Validate user-supplied paths through the security layer
        if input_file:
            try:
                input_file = str(validate_file_path(input_file, must_exist=True))
            except ValueError as e:
                print(f"\u274c Invalid input file: {e}")
                return False

        if output_file:
            try:
                output_file = str(validate_file_path(output_file, must_exist=False))
            except ValueError as e:
                print(f"\u274c Invalid output file: {e}")
                return False

        # Instantiate the pipeline orchestrator (does not load data yet)
        pipeline = EscalationPipeline()

        # Define the 5 top-level progress phases shown to the user.
        # Phase 3 ("Analysis") internally contains the 7 sub-phases listed
        # in the orchestrator, but we present it as a single progress step
        # to keep the CLI output readable.
        phases = [
            {"name": "Initialization", "desc": "\U0001f527 Initializing AI models and services"},
            {"name": "Data Loading", "desc": "\U0001f4c2 Loading and validating input data"},
            {"name": "Analysis", "desc": "\u2699\ufe0f  Running 7-phase analysis pipeline"},
            {"name": "Summary", "desc": "\U0001f4dd Generating executive summary"},
            {"name": "Report", "desc": "\U0001f4be Creating Excel report with charts"},
        ]

        # Wrap the pipeline in a tqdm progress bar for visual feedback
        with tqdm(total=len(phases), desc="Pipeline Progress", unit="phase",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:

            # ---- Phase 1: Initialize AI models (Ollama connection, embedding model) ----
            # Retry up to 3 times because the Ollama server may need time to
            # load the model into GPU memory on the first request.
            pbar.set_description(phases[0]["desc"])
            max_retries = 3
            initialized = False

            for attempt in range(max_retries):
                try:
                    if pipeline.initialize():
                        initialized = True
                        break
                    # initialize() returned False (non-exception failure)
                    if attempt < max_retries - 1:
                        tqdm.write(f"\u26a0\ufe0f  Initialization failed, retrying ({attempt + 1}/{max_retries})...")
                        time.sleep(2)
                except Exception as e:
                    if attempt == max_retries - 1:
                        tqdm.write(f"\u274c Failed to initialize after {max_retries} attempts: {e}")
                        logger.error("Pipeline initialization failed", exc_info=True)
                        return False
                    tqdm.write(f"\u26a0\ufe0f  Attempt {attempt + 1} failed: {e}")
                    time.sleep(2)

            if not initialized:
                tqdm.write("\u274c Pipeline initialization failed")
                return False

            pbar.update(1)

            # ---- Phase 2: Load and validate input data ----
            pbar.set_description(phases[1]["desc"])
            if input_file:
                # Non-interactive: load the file directly
                if not pipeline.load_data(input_file):
                    tqdm.write("\u274c Failed to load data file")
                    return False
            else:
                # Interactive: show a tkinter file-open dialog.
                # We temporarily clear the tqdm bar so the dialog isn't hidden.
                pbar.clear()
                if not pipeline.load_data():
                    tqdm.write("\u274c Data loading cancelled or failed")
                    return False
                pbar.refresh()

            pbar.update(1)

            # ---- Phase 3: Run all 7 analysis sub-phases ----
            # This is the heavy computation: classification, scoring, embeddings,
            # similarity search, recurrence prediction, financial metrics, etc.
            pbar.set_description(phases[2]["desc"])
            df = pipeline.run_all_phases()

            if df is None or df.empty:
                tqdm.write("\u274c Pipeline produced no results")
                return False

            pbar.update(1)

            # ---- Phase 4: Generate executive summary via LLM ----
            pbar.set_description(phases[3]["desc"])
            exec_summary = pipeline.generate_executive_summary()
            pbar.update(1)

            # ---- Phase 5: Save the multi-sheet Excel report ----
            pbar.set_description(phases[4]["desc"])

            if output_file:
                save_path = output_file
            else:
                # Interactive: show a tkinter save dialog
                pbar.clear()
                tqdm.write("\U0001f4be Select save location...")
                root = tk.Tk()
                root.withdraw()  # Hide the empty root window
                save_path = filedialog.asksaveasfilename(
                    defaultextension=".xlsx",
                    initialfile="Strategic_Report.xlsx",
                    filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
                )
                root.destroy()

                if not save_path:
                    tqdm.write("\u274c No output file selected. Aborting.")
                    return False
                pbar.refresh()

            # generate_report writes the enriched DataFrame, charts, and
            # executive summary into a multi-sheet .xlsx workbook.
            generate_report(df, save_path, exec_summary, pipeline.df_raw)
            pbar.update(1)

        # Pipeline complete -- print the final success banner
        print()
        print("=" * 60)
        print("  \U0001f389 ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"  \u2705 Report saved to: {save_path}")
        print("=" * 60)
        print()
        sys.stdout.flush()

        return True

    except ImportError as e:
        # This typically means the user is not in the right virtualenv or
        # hasn't run `pip install -e .` to install the escalation_ai package.
        print(f"\u274c Error importing escalation_ai package: {e}")
        print("\nMake sure you're running from the project directory.")
        print("Try: pip install -e .")
        logger.error("Import error", exc_info=True)
        return False
    except KeyboardInterrupt:
        print("\n\n\U0001f44b Pipeline interrupted by user")
        return False
    except Exception as e:
        # Catch-all for unexpected errors; full traceback goes to the log file
        print(f"\n\u274c Pipeline error: {e}")
        logger.error("Pipeline execution failed", exc_info=True)
        print("\nFor detailed logs, run with --verbose flag")
        return False


def pre_generate_ai_cache():
    """
    Pre-generate AI insights and persist them to disk (Stage 2 of the system).

    Why this exists:
        The Streamlit dashboard needs AI-generated content (executive summary,
        issue classifications, embedding vectors) to populate its pages.
        Generating these on-the-fly inside Streamlit would cause multi-minute
        delays every time a page loads or the app re-runs.  By pre-computing
        them here and saving to .cache/ai_insights.pkl, the dashboard can
        deserialise the results in milliseconds.

    The cache file (.cache/ai_insights.pkl) is a Python pickle dictionary with
    these keys:
        'ollama_available'     -- bool, True (confirms Ollama was reachable)
        'ai_exec_summary'      -- str, LLM-generated executive summary text
        'ai_issue_categories'  -- list[str], category label per pain point
        'ai_issue_texts'       -- list[str], the raw pain point texts (parallel)
        'embeddings_index'     -- dict with 'texts' and 'embeddings' arrays for
                                  semantic search in the dashboard

    Data source: ProjectPulse.xlsx (sheet "Project Pulse") -- this is a separate
    dataset from the escalation tickets; it contains project health assessments
    with free-text "Pain Points", "Comments", and "Resolution Plan" columns.

    Graceful degradation: if Ollama is not running, or ProjectPulse.xlsx is
    missing, this function prints a warning and returns without creating a cache.
    The dashboard will then fall back to non-AI mode.
    """
    import pickle
    project_root = Path(__file__).parent
    cache_dir = project_root / '.cache'
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / 'ai_insights.pkl'

    # Add the pulse_dashboard directory to sys.path so we can import its
    # utility module.  This is done at runtime (not at the top of the file)
    # because pulse_dashboard is an optional component.
    sys.path.insert(0, str(project_root / 'pulse_dashboard'))
    try:
        from utils.pulse_insights import (
            check_ollama, ollama_generate, build_embeddings_index,
        )
    except ImportError as e:
        print(f"  \u26a0\ufe0f  Could not import pulse_insights: {e}")
        return

    # Bail out early if Ollama is not reachable -- all three generation steps
    # require it, so there's nothing useful we can do without it.
    if not check_ollama():
        print("  \u26a0\ufe0f  Ollama not running \u2014 skipping AI pre-generation")
        return

    print()
    print("=" * 60)
    print("  \U0001f9e0 Pre-generating AI Insights")
    print("=" * 60)
    print()

    # Load the Project Pulse dataset directly with pandas (not via Streamlit's
    # caching, since we're running outside Streamlit here).
    pulse_file = project_root / 'ProjectPulse.xlsx'
    if not pulse_file.exists():
        print("  \u26a0\ufe0f  ProjectPulse.xlsx not found \u2014 skipping AI pre-generation")
        return

    import pandas as pd
    import numpy as np
    try:
        df = pd.read_excel(pulse_file, sheet_name='Project Pulse')
        # Clean text columns: replace non-breaking spaces (\xa0) and normalise
        # empty-like values to NaN so downstream LLM prompts don't contain
        # literal "nan" strings.
        for col in ['Comments', 'Pain Points', 'Resolution Plan']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('\xa0', ' ', regex=False).str.strip()
                df[col] = df[col].replace({'nan': np.nan, '': np.nan, 'None': np.nan})
        # Ensure numeric score column is actually numeric (Excel sometimes
        # stores numbers as strings when cells have mixed formatting).
        for col in ['Total Score']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    except Exception as e:
        print(f"  \u26a0\ufe0f  Failed to load Pulse data: {e}")
        return

    # Initialize the cache dictionary.  The dashboard checks 'ollama_available'
    # to decide whether to show AI-powered widgets or fall back to static mode.
    cache = {'ollama_available': True}

    # ---- Step 1: Executive Summary ----
    # Feed the top 20 pain points (truncated to 300 chars each) to the LLM
    # and ask for a structured summary with themes and urgency assessment.
    print("  \U0001f4dd Generating Executive Summary...")
    sys.stdout.flush()
    pain = df['Pain Points'].dropna()
    if not pain.empty:
        texts = pain.head(20).tolist()
        combined = "\n".join([f"- {t[:300]}" for t in texts])
        prompt = f"""Analyze these project pain points from a telecom portfolio and provide:
1. A 2-3 sentence executive summary
2. Top 5 recurring themes
3. Most urgent issue requiring immediate attention

Pain Points:
{combined}

Format your response exactly as:
SUMMARY: <summary>
THEMES:
1. <theme>: <description>
2. <theme>: <description>
3. <theme>: <description>
4. <theme>: <description>
5. <theme>: <description>
URGENT: <urgent issue>"""
        result = ollama_generate(prompt)
        if result:
            cache['ai_exec_summary'] = result
            print("     \u2705 Executive Summary generated")
        else:
            print("     \u274c Executive Summary failed")
    sys.stdout.flush()

    # ---- Step 2: Issue Classification ----
    # Classify each pain point into one of 11 predefined categories using the
    # LLM with low temperature (0.1) for deterministic, consistent labelling.
    # We process up to 30 issues and print progress every 10 items.
    print("  \U0001f3f7\ufe0f  Classifying issues...")
    sys.stdout.flush()
    CATEGORIES = [
        "Resource/Staffing", "Timeline/Delays", "Technical/Engineering",
        "Vendor/Partner", "Communication", "Process/Workflow",
        "Customer Satisfaction", "Budget/Commercial", "Equipment/Tools",
        "Scope Change", "Other"
    ]
    if not pain.empty:
        texts = pain.head(30).tolist()
        categories_result = []
        for i, text in enumerate(texts):
            # Prompt the LLM to return ONLY the category name -- this makes
            # parsing trivial and keeps token usage low.
            p = f"Categorize this telecom project issue into ONE of: {', '.join(CATEGORIES)}\nIssue: {text[:300]}\nRespond with ONLY the category name, nothing else."
            r = ollama_generate(p, temperature=0.1, timeout=30)
            if r:
                # Fuzzy-match the LLM response to a known category.
                # If the LLM returns "timeline/delays issues", we still match
                # "Timeline/Delays" via case-insensitive substring search.
                matched = next((c for c in CATEGORIES if c.lower() in r.strip().lower()), 'Other')
                categories_result.append(matched)
            else:
                categories_result.append('Other')
            # Progress feedback every 10 issues
            if (i + 1) % 10 == 0:
                print(f"     ... classified {i + 1}/{len(texts)}")
                sys.stdout.flush()
        # Store both the category labels and the original texts so the
        # dashboard can display them side by side.
        cache['ai_issue_categories'] = categories_result
        cache['ai_issue_texts'] = texts
        print(f"     \u2705 {len(categories_result)} issues classified")
    sys.stdout.flush()

    # ---- Step 3: Embeddings Index ----
    # Build a vector index over the text columns (Comments, Pain Points,
    # Resolution Plan) using the embedding model (qwen2:1.5b via Ollama).
    # This index powers the semantic search feature in the dashboard -- users
    # can type a natural-language query and find the most relevant issues.
    print("  \U0001f50d Building embeddings index...")
    sys.stdout.flush()
    try:
        # build_embeddings_index concatenates text from the specified columns,
        # generates embeddings via Ollama, and returns a dict with 'texts'
        # (list[str]) and 'embeddings' (numpy array of shape [N, dim]).
        index = build_embeddings_index(df, columns=['Comments', 'Pain Points', 'Resolution Plan'])
        if index and len(index.get('texts', [])) > 0:
            cache['embeddings_index'] = index
            print(f"     \u2705 Index built: {len(index['texts'])} documents embedded")
        else:
            print("     \u26a0\ufe0f  No documents to embed")
    except Exception as e:
        print(f"     \u274c Embeddings failed: {e}")
    sys.stdout.flush()

    # ---- Persist the cache to disk ----
    # Using pickle for serialisation because the cache contains numpy arrays
    # (embeddings) which pickle handles natively and efficiently.
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        print(f"\n  \U0001f4be Cache saved to {cache_file}")
    except Exception as e:
        print(f"  \u274c Failed to save cache: {e}")

    print()


def launch_dashboard(port: int = 8501, open_browser: bool = True):
    """
    Launch the unified Streamlit dashboard as a managed subprocess (Stage 3).

    This function:
        1. Checks that the unified_app.py entry point exists.
        2. Detects if the requested port is already in use (e.g. from a
           previous run) and offers to kill the existing process.
        3. Starts Streamlit with a dark theme and usage-stats disabled.
        4. Optionally opens the user's default browser after a 3-second delay
           (in a daemon thread so it doesn't block the main process).
        5. Registers an atexit cleanup handler that terminates the Streamlit
           subprocess on exit (Ctrl+C, normal exit, or crash).

    The dashboard is run in headless mode (--server.headless true) because we
    handle browser-opening ourselves, and because Streamlit's built-in browser
    opener doesn't work reliably inside WSL or SSH sessions.

    Args:
        port: TCP port for the Streamlit HTTP server (default: 8501).
        open_browser: If True, auto-open http://localhost:{port} in the
                      default browser after a short delay.

    Returns:
        bool: True if the dashboard ran and exited cleanly (including Ctrl+C
              shutdown), False on errors (missing files, port conflicts, etc.).
    """
    print()
    print("=" * 60)
    print("  \U0001f310 Launching Interactive Dashboard")
    print("=" * 60)
    print()

    # The unified app serves both Project Pulse and Escalation AI dashboards
    # from a single Streamlit process with tabbed navigation.
    dashboard_path = Path(__file__).parent / "unified_app.py"

    if not dashboard_path.exists():
        print(f"\u274c Error: Dashboard not found at {dashboard_path}")
        return False

    # --- Port conflict detection ---
    # Try to connect to the port; if it succeeds, something is already listening.
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port_in_use = sock.connect_ex(('localhost', port)) == 0
        sock.close()

        if port_in_use:
            print(f"\u26a0\ufe0f  Port {port} is already in use")
            try:
                response = input("   Try to stop existing process? (y/N): ").strip().lower()
                if response == 'y':
                    print("   Attempting to stop existing Streamlit process...")
                    try:
                        # pkill sends SIGTERM to processes matching the pattern
                        subprocess.run(['pkill', '-f', f'streamlit.*{port}'], timeout=5)
                        time.sleep(2)  # Give the process time to release the port
                        print("   \u2705 Process stopped")
                    except subprocess.TimeoutExpired:
                        print("   \u26a0\ufe0f  Timeout stopping process")
                    except Exception as e:
                        print(f"   \u26a0\ufe0f  Could not stop process: {e}")
                else:
                    print("   Please use a different port with --port flag")
                    return False
            except (KeyboardInterrupt, EOFError):
                print("\n   Cancelled")
                return False
    except Exception as e:
        logger.warning(f"Could not check port status: {e}")

    print(f"  \U0001f4ca Starting Streamlit server on port {port}...")
    print(f"  \U0001f517 URL: http://localhost:{port}")
    print()
    print("  Press Ctrl+C to stop the dashboard")
    print("-" * 60)
    sys.stdout.flush()

    # Build the Streamlit command with dark-theme configuration.
    # --server.headless true: prevents Streamlit from trying to open a browser
    #   itself (we handle that separately with better timing).
    # --browser.gatherUsageStats false: opt out of Streamlit telemetry.
    # --theme.*: apply a dark blue colour scheme matching the CSE branding.
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

    # Track the subprocess so we can clean it up on exit
    streamlit_process = None

    def cleanup():
        """Terminate the Streamlit subprocess on exit.

        Called by atexit or explicitly on Ctrl+C.  Sends SIGTERM first (polite
        shutdown), waits 5 seconds, then SIGKILL if it's still running.
        This prevents zombie Streamlit processes from holding the port.
        """
        nonlocal streamlit_process
        if streamlit_process and streamlit_process.poll() is None:
            print("\n\U0001f9f9 Cleaning up dashboard process...")
            streamlit_process.terminate()
            try:
                streamlit_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("   Force killing process...")
                streamlit_process.kill()

    # Register the cleanup function so it runs even if the script crashes
    # or is killed by a signal other than SIGINT.
    atexit.register(cleanup)

    try:
        # --- Browser auto-open ---
        # We launch a daemon thread that waits 3 seconds (giving the Streamlit
        # server time to bind the port) then opens the URL.  Using a daemon
        # thread means it won't prevent the process from exiting.
        if open_browser:
            def open_browser_delayed():
                time.sleep(3)  # Wait for server to start accepting connections
                try:
                    webbrowser.open(f"http://localhost:{port}")
                except Exception as e:
                    logger.warning(f"Failed to open browser: {e}")

            import threading
            browser_thread = threading.Thread(target=open_browser_delayed, daemon=True)
            browser_thread.start()

        # Start Streamlit as a subprocess.  stdout/stderr are piped so they
        # don't interleave with our own output.  The main thread blocks on
        # .wait() until the user presses Ctrl+C or the process exits.
        streamlit_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Block until the Streamlit process exits (normally via Ctrl+C)
        streamlit_process.wait()
        return True

    except KeyboardInterrupt:
        # User pressed Ctrl+C -- this is the expected way to stop the dashboard
        print("\n\n\u2705 Dashboard stopped by user.")
        cleanup()
        return True
    except FileNotFoundError:
        # sys.executable or streamlit not found in PATH
        print("\n\u274c Streamlit not found. Install with: pip install streamlit plotly")
        return False
    except Exception as e:
        print(f"\n\u274c Error launching dashboard: {e}")
        logger.error("Dashboard launch failed", exc_info=True)
        cleanup()
        return False


def main():
    """
    Top-level entry point: parse CLI args and dispatch to the appropriate mode.

    Execution modes:
        --health-check   -> run diagnostics, print report, exit
        --dashboard-only -> pre-generate AI cache + launch Streamlit (no pipeline)
        --no-gui         -> run ML pipeline only, print results, exit (no Streamlit)
        (default)        -> full flow: pipeline -> AI cache -> dashboard

    In the default (full) mode, the dashboard is launched regardless of whether
    the pipeline succeeded.  This is intentional: even if some phases failed,
    the dashboard can still display partial results and let the user investigate.
    """
    args = parse_args()

    # Reconfigure logging if the user requested verbose output.
    # This re-creates the console handler with INFO level (instead of WARNING).
    if args.verbose:
        setup_logging(verbose=True)

    # Health check mode: run diagnostics and exit immediately
    if args.health_check:
        success = health_check()
        sys.exit(0 if success else 1)

    try:
        if args.dashboard_only:
            # Skip the ML pipeline; just pre-generate AI insights (if Ollama is
            # available) and launch the dashboard.  Useful when the pipeline was
            # already run separately via --no-gui.
            pre_generate_ai_cache()
            launch_dashboard(port=args.port, open_browser=not args.no_browser)

        elif args.no_gui:
            # Run the ML pipeline without launching a dashboard.  Useful for
            # CI/CD, cron jobs, or when the user only wants the Excel report.
            success = run_pipeline(input_file=args.file, output_file=args.output)
            if not success:
                sys.exit(1)
            print("\u2705 Pipeline complete. Run with --dashboard-only to view results.")

        else:
            # Default full flow: pipeline -> AI cache pre-generation -> dashboard.
            # The three stages run sequentially because each depends on the output
            # of the previous one (pipeline produces the report that the dashboard
            # reads, and AI cache pre-generation needs to happen before the
            # dashboard loads the cache).
            print("\U0001f504 Running full analysis pipeline...")
            sys.stdout.flush()
            success = run_pipeline(input_file=args.file, output_file=args.output)

            # Pre-generate AI insights regardless of pipeline success, because
            # the Pulse insights come from ProjectPulse.xlsx (a separate file).
            pre_generate_ai_cache()

            if success:
                print("\u2705 Launching dashboard to view results...")
                launch_dashboard(port=args.port, open_browser=not args.no_browser)
            else:
                # Launch dashboard even after pipeline errors so the user can
                # inspect partial results and diagnose what went wrong.
                print("\n\u26a0\ufe0f Pipeline had errors. Launching dashboard anyway...")
                launch_dashboard(port=args.port, open_browser=not args.no_browser)

    except KeyboardInterrupt:
        print("\n\n\U0001f44b Cancelled by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
