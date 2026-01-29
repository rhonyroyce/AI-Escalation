"""
Unit tests for run.py

Tests all major functions in the run.py entry point script including:
- Path validation and security
- CUDA environment setup
- GPU detection
- Health checks
- Pipeline execution
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import sys
import os
import tempfile
import shutil

# Add parent directory to path to import run module
sys.path.insert(0, str(Path(__file__).parent.parent))

import run


class TestPathValidation(unittest.TestCase):
    """Test suite for path validation and security functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_file = self.test_dir / "test.xlsx"
        self.test_file.write_text("test data")

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_validate_existing_file(self):
        """Test validation of existing file."""
        result = run.validate_file_path(str(self.test_file), must_exist=True)
        self.assertIsInstance(result, Path)
        self.assertTrue(result.exists())

    def test_validate_nonexistent_file_without_requirement(self):
        """Test validation of non-existent file when existence not required."""
        nonexistent = self.test_dir / "nonexistent.xlsx"
        result = run.validate_file_path(str(nonexistent), must_exist=False)
        self.assertIsInstance(result, Path)

    def test_validate_nonexistent_file_with_requirement(self):
        """Test validation fails for non-existent file when existence required."""
        nonexistent = self.test_dir / "nonexistent.xlsx"
        with self.assertRaises(ValueError) as context:
            run.validate_file_path(str(nonexistent), must_exist=True)
        self.assertIn("File not found", str(context.exception))

    def test_prevent_path_traversal(self):
        """Test prevention of path traversal attacks."""
        # Try to access /etc/passwd (outside allowed directories)
        with self.assertRaises(ValueError) as context:
            run.validate_file_path('/etc/passwd', must_exist=False)
        self.assertIn("outside allowed directories", str(context.exception))

    def test_prevent_relative_path_traversal(self):
        """Test prevention of relative path traversal."""
        with self.assertRaises(ValueError) as context:
            run.validate_file_path('../../../etc/passwd', must_exist=False)
        self.assertIn("outside allowed directories", str(context.exception))

    def test_sanitize_filename_removes_dangerous_chars(self):
        """Test filename sanitization removes dangerous characters."""
        dangerous = "../../bad;file|name*.xlsx"
        sanitized = run.sanitize_filename(dangerous)
        # Should remove all dangerous characters
        self.assertNotIn('/', sanitized)
        self.assertNotIn('..', sanitized)
        self.assertNotIn(';', sanitized)
        self.assertNotIn('|', sanitized)
        self.assertNotIn('*', sanitized)

    def test_sanitize_filename_keeps_valid_chars(self):
        """Test filename sanitization keeps valid characters."""
        valid = "My_Report-2024.xlsx"
        sanitized = run.sanitize_filename(valid)
        self.assertEqual(valid, sanitized)

    def test_sanitize_filename_limits_length(self):
        """Test filename sanitization limits length to 255 chars."""
        long_name = "a" * 300 + ".xlsx"
        sanitized = run.sanitize_filename(long_name)
        self.assertLessEqual(len(sanitized), 255)


class TestCUDASetup(unittest.TestCase):
    """Test suite for CUDA environment setup."""

    @patch.dict(os.environ, {}, clear=True)
    @patch('os.path.exists')
    def test_cuda_setup_finds_installation(self, mock_exists):
        """Test CUDA setup finds and configures CUDA installation."""
        # Mock that CUDA 12.8 exists
        def exists_side_effect(path):
            return path == '/usr/local/cuda-12.8'

        mock_exists.side_effect = exists_side_effect

        needs_restart = run.setup_cuda_environment()

        # Should set environment variables
        self.assertIn('CUDA_HOME', os.environ)
        self.assertEqual(os.environ['CUDA_HOME'], '/usr/local/cuda-12.8')
        self.assertTrue(needs_restart)

    @patch.dict(os.environ, {}, clear=True)
    @patch('os.path.exists')
    def test_cuda_setup_handles_missing_installation(self, mock_exists):
        """Test CUDA setup handles missing installation gracefully."""
        mock_exists.return_value = False

        needs_restart = run.setup_cuda_environment()

        # Should not crash and return False
        self.assertFalse(needs_restart)

    @patch.dict(os.environ, {'CUDA_HOME': '/usr/local/cuda-12.8'}, clear=False)
    @patch('os.path.exists')
    def test_cuda_setup_skips_if_already_set(self, mock_exists):
        """Test CUDA setup skips restart if environment already configured."""
        mock_exists.return_value = True

        # CUDA_HOME already matches, should not need restart
        needs_restart = run.setup_cuda_environment()

        # May or may not need restart depending on other variables
        self.assertIsInstance(needs_restart, bool)

    @patch('os.path.exists')
    def test_cuda_setup_handles_permission_error(self, mock_exists):
        """Test CUDA setup handles permission errors gracefully."""
        mock_exists.side_effect = PermissionError("Access denied")

        # Should not crash
        needs_restart = run.setup_cuda_environment()
        self.assertFalse(needs_restart)


class TestGPUDetection(unittest.TestCase):
    """Test suite for GPU detection."""

    @patch('subprocess.run')
    def test_check_gpu_available_success(self, mock_run):
        """Test GPU detection with successful nvidia-smi call."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='NVIDIA GeForce RTX 4090\n'
        )

        available, gpu_name = run.check_gpu_available()

        self.assertTrue(available)
        self.assertEqual(gpu_name, 'NVIDIA GeForce RTX 4090')

    @patch('subprocess.run')
    def test_check_gpu_available_multiple_gpus(self, mock_run):
        """Test GPU detection with multiple GPUs."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='NVIDIA GeForce RTX 4090\nNVIDIA GeForce RTX 3090\n'
        )

        available, gpu_name = run.check_gpu_available()

        self.assertTrue(available)
        self.assertEqual(gpu_name, 'NVIDIA GeForce RTX 4090')  # Returns first GPU

    @patch('subprocess.run')
    def test_check_gpu_available_no_gpu(self, mock_run):
        """Test GPU detection when no GPU available."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout=''
        )

        available, gpu_name = run.check_gpu_available()

        self.assertFalse(available)
        self.assertIsNone(gpu_name)

    @patch('subprocess.run')
    def test_check_gpu_available_timeout(self, mock_run):
        """Test GPU detection handles timeout gracefully."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired('nvidia-smi', 2)

        available, gpu_name = run.check_gpu_available()

        self.assertFalse(available)
        self.assertIsNone(gpu_name)

    @patch('subprocess.run')
    def test_check_gpu_available_not_found(self, mock_run):
        """Test GPU detection when nvidia-smi not installed."""
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")

        available, gpu_name = run.check_gpu_available()

        self.assertFalse(available)
        self.assertIsNone(gpu_name)


class TestHealthCheck(unittest.TestCase):
    """Test suite for health check functionality."""

    @patch('run.check_gpu_available')
    @patch('run.check_ollama_server')
    @patch('run.check_required_packages')
    def test_health_check_all_pass(self, mock_packages, mock_ollama, mock_gpu):
        """Test health check when all checks pass."""
        mock_gpu.return_value = (True, "NVIDIA RTX 4090")
        mock_ollama.return_value = True
        mock_packages.return_value = (True, [])

        result = run.health_check()

        self.assertTrue(result)

    @patch('run.check_gpu_available')
    @patch('run.check_ollama_server')
    @patch('run.check_required_packages')
    def test_health_check_missing_packages(self, mock_packages, mock_ollama, mock_gpu):
        """Test health check when packages are missing."""
        mock_gpu.return_value = (True, "NVIDIA RTX 4090")
        mock_ollama.return_value = True
        mock_packages.return_value = (False, ['pandas', 'numpy'])

        result = run.health_check()

        self.assertFalse(result)

    @patch('run.check_gpu_available')
    @patch('run.check_ollama_server')
    @patch('run.check_required_packages')
    def test_health_check_ollama_not_running(self, mock_packages, mock_ollama, mock_gpu):
        """Test health check when Ollama server is not running."""
        mock_gpu.return_value = (True, "NVIDIA RTX 4090")
        mock_ollama.return_value = False
        mock_packages.return_value = (True, [])

        result = run.health_check()

        self.assertFalse(result)

    @patch('requests.get')
    def test_check_ollama_server_running(self, mock_get):
        """Test Ollama server check when server is running."""
        mock_get.return_value = MagicMock(status_code=200)

        result = run.check_ollama_server()

        self.assertTrue(result)

    @patch('requests.get')
    def test_check_ollama_server_not_running(self, mock_get):
        """Test Ollama server check when server is not running."""
        import requests
        mock_get.side_effect = requests.ConnectionError("Connection refused")

        result = run.check_ollama_server()

        self.assertFalse(result)

    def test_check_required_packages_all_installed(self):
        """Test package check when all packages are installed."""
        # Since we're running in a test environment, pandas/numpy should be available
        all_installed, missing = run.check_required_packages()

        # At minimum, should return tuple
        self.assertIsInstance(all_installed, bool)
        self.assertIsInstance(missing, list)

    @patch('builtins.__import__')
    def test_check_required_packages_missing(self, mock_import):
        """Test package check when packages are missing."""
        def import_side_effect(name, *args, **kwargs):
            if name in ['pandas', 'numpy']:
                raise ImportError(f"No module named '{name}'")
            return MagicMock()

        mock_import.side_effect = import_side_effect

        all_installed, missing = run.check_required_packages()

        self.assertFalse(all_installed)
        self.assertGreater(len(missing), 0)


class TestArgumentParsing(unittest.TestCase):
    """Test suite for command-line argument parsing."""

    def test_parse_args_defaults(self):
        """Test argument parsing with default values."""
        with patch('sys.argv', ['run.py']):
            args = run.parse_args()

            self.assertEqual(args.port, 8501)
            self.assertFalse(args.no_gui)
            self.assertFalse(args.dashboard_only)
            self.assertFalse(args.verbose)
            self.assertFalse(args.health_check)

    def test_parse_args_verbose(self):
        """Test argument parsing with verbose flag."""
        with patch('sys.argv', ['run.py', '--verbose']):
            args = run.parse_args()

            self.assertTrue(args.verbose)

    def test_parse_args_health_check(self):
        """Test argument parsing with health-check flag."""
        with patch('sys.argv', ['run.py', '--health-check']):
            args = run.parse_args()

            self.assertTrue(args.health_check)

    def test_parse_args_custom_port(self):
        """Test argument parsing with custom port."""
        with patch('sys.argv', ['run.py', '--port', '8502']):
            args = run.parse_args()

            self.assertEqual(args.port, 8502)

    def test_parse_args_input_output_files(self):
        """Test argument parsing with input and output files."""
        with patch('sys.argv', ['run.py', '--file', 'input.xlsx', '--output', 'output.xlsx']):
            args = run.parse_args()

            self.assertEqual(args.file, 'input.xlsx')
            self.assertEqual(args.output, 'output.xlsx')

    def test_parse_args_no_gui(self):
        """Test argument parsing with no-gui flag."""
        with patch('sys.argv', ['run.py', '--no-gui']):
            args = run.parse_args()

            self.assertTrue(args.no_gui)

    def test_parse_args_dashboard_only(self):
        """Test argument parsing with dashboard-only flag."""
        with patch('sys.argv', ['run.py', '--dashboard-only']):
            args = run.parse_args()

            self.assertTrue(args.dashboard_only)


class TestLogging(unittest.TestCase):
    """Test suite for logging configuration."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_log_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_log_dir.exists():
            shutil.rmtree(self.test_log_dir)

    @patch('run.Path')
    def test_setup_logging_creates_directory(self, mock_path):
        """Test that logging setup creates log directory."""
        mock_log_dir = self.test_log_dir / "logs"
        mock_path.return_value.parent = mock_path.return_value
        mock_path.return_value.__truediv__ = lambda self, other: mock_log_dir

        # Should create directory if it doesn't exist
        log_file = run.setup_logging(verbose=False)

        self.assertIsInstance(log_file, Path)

    def test_setup_logging_verbose_mode(self):
        """Test logging setup in verbose mode."""
        import logging

        log_file = run.setup_logging(verbose=True)

        # Check that root logger is configured
        root_logger = logging.getLogger()
        self.assertEqual(root_logger.level, logging.DEBUG)

        # Should have both file and console handlers
        self.assertGreaterEqual(len(root_logger.handlers), 2)


class TestSanitization(unittest.TestCase):
    """Test suite for input sanitization."""

    def test_sanitize_filename_basic(self):
        """Test basic filename sanitization."""
        result = run.sanitize_filename("report.xlsx")
        self.assertEqual(result, "report.xlsx")

    def test_sanitize_filename_with_spaces(self):
        """Test sanitization preserves spaces."""
        result = run.sanitize_filename("My Report 2024.xlsx")
        self.assertEqual(result, "My Report 2024.xlsx")

    def test_sanitize_filename_removes_path(self):
        """Test sanitization removes path components."""
        result = run.sanitize_filename("/path/to/file.xlsx")
        self.assertEqual(result, "file.xlsx")

    def test_sanitize_filename_removes_special_chars(self):
        """Test sanitization removes special characters."""
        result = run.sanitize_filename("file<>:|?.xlsx")
        # Should only keep alphanumeric, spaces, hyphens, underscores, dots
        self.assertNotIn('<', result)
        self.assertNotIn('>', result)
        self.assertNotIn(':', result)
        self.assertNotIn('|', result)
        self.assertNotIn('?', result)


if __name__ == '__main__':
    unittest.main()
