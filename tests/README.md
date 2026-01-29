# Escalation AI Test Suite

Comprehensive unit tests for the Escalation AI application.

## Overview

This test suite provides coverage for:
- Path validation and security
- CUDA environment setup
- GPU detection
- Health check functionality
- Command-line argument parsing
- Logging configuration
- Input sanitization

## Installation

Install test dependencies:

```bash
pip install pytest pytest-cov pytest-mock
```

Or install all requirements including dev dependencies:

```bash
pip install -r requirements.txt
```

## Running Tests

### Run All Tests

```bash
# From project root
pytest tests/

# With verbose output
pytest tests/ -v

# With coverage report
pytest tests/ --cov=run --cov=escalation_ai --cov-report=html
```

### Run Specific Test Files

```bash
# Run only run.py tests
pytest tests/test_run.py

# Run with extra verbosity
pytest tests/test_run.py -vv
```

### Run Specific Test Classes

```bash
# Test only path validation
pytest tests/test_run.py::TestPathValidation -v

# Test only GPU detection
pytest tests/test_run.py::TestGPUDetection -v

# Test only health checks
pytest tests/test_run.py::TestHealthCheck -v
```

### Run Specific Test Methods

```bash
# Test a single function
pytest tests/test_run.py::TestPathValidation::test_prevent_path_traversal -v

# Test GPU availability detection
pytest tests/test_run.py::TestGPUDetection::test_check_gpu_available_success -v
```

## Test Organization

### Test Files

- **test_run.py**: Tests for run.py entry point
  - Path validation and security
  - CUDA environment setup
  - GPU detection
  - Health checks
  - Argument parsing
  - Logging configuration

### Test Classes

Each test file is organized into test classes by functionality:

```
TestPathValidation
├── test_validate_existing_file
├── test_prevent_path_traversal
├── test_sanitize_filename_removes_dangerous_chars
└── ...

TestCUDASetup
├── test_cuda_setup_finds_installation
├── test_cuda_setup_handles_missing_installation
└── ...

TestGPUDetection
├── test_check_gpu_available_success
├── test_check_gpu_available_no_gpu
└── ...

TestHealthCheck
├── test_health_check_all_pass
├── test_check_ollama_server_running
└── ...
```

### Fixtures

Sample data and test fixtures are located in `tests/fixtures/`:

- **sample_data.py**: Functions to generate sample escalation data
  - `create_sample_escalation_data()`: DataFrame with sample tickets
  - `create_sample_excel_file()`: Creates temporary Excel file
  - `get_sample_ticket_text()`: Sample ticket descriptions
  - `get_sample_categories()`: Expected category mappings

## Coverage Reports

After running tests with coverage:

```bash
pytest tests/ --cov=run --cov=escalation_ai --cov-report=html
```

Open the HTML coverage report:

```bash
# Linux/WSL
xdg-open htmlcov/index.html

# macOS
open htmlcov/index.html

# Windows
start htmlcov/index.html
```

## Test Markers

Tests can be marked for selective execution:

```python
@pytest.mark.unit
def test_something():
    pass

@pytest.mark.integration
def test_integration():
    pass

@pytest.mark.slow
def test_long_running():
    pass

@pytest.mark.gpu
def test_gpu_required():
    pass
```

Run only specific markers:

```bash
# Run only unit tests
pytest tests/ -m unit

# Run only GPU tests
pytest tests/ -m gpu

# Exclude slow tests
pytest tests/ -m "not slow"
```

## Writing New Tests

### Test Structure

Follow this structure for new tests:

```python
import unittest
from unittest.mock import patch, MagicMock

class TestMyFeature(unittest.TestCase):
    """Test suite for my feature."""

    def setUp(self):
        """Set up test fixtures before each test."""
        pass

    def tearDown(self):
        """Clean up after each test."""
        pass

    def test_basic_functionality(self):
        """Test basic functionality works."""
        # Arrange
        input_data = "test"

        # Act
        result = my_function(input_data)

        # Assert
        self.assertEqual(result, "expected")

    @patch('module.external_dependency')
    def test_with_mock(self, mock_dep):
        """Test with mocked dependencies."""
        mock_dep.return_value = "mocked"

        result = my_function()

        self.assertEqual(result, "mocked")
        mock_dep.assert_called_once()
```

### Best Practices

1. **One assertion per test** (when possible)
2. **Clear test names** describing what is being tested
3. **Use fixtures** for common setup/teardown
4. **Mock external dependencies** (network, filesystem, GPU)
5. **Test edge cases** and error conditions
6. **Keep tests isolated** - no dependencies between tests

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest tests/ --cov=run --cov=escalation_ai
```

## Troubleshooting

### Import Errors

If you get import errors when running tests:

```bash
# Make sure you're in the project root
cd /path/to/AI-Escalation

# Install package in editable mode
pip install -e .

# Run tests
pytest tests/
```

### Mock Failures

If mocks aren't working as expected:

```python
# Use patch with full module path
@patch('run.subprocess.run')  # NOT just 'subprocess.run'
def test_something(self, mock_run):
    pass
```

### GPU Tests

GPU-related tests are mocked by default. To test with real GPU:

```python
@pytest.mark.gpu
def test_real_gpu():
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    # Test with real GPU
```

## Test Coverage Goals

Target coverage levels:
- **Critical paths**: 100% coverage
- **Core functionality**: >90% coverage
- **Utility functions**: >80% coverage
- **Overall project**: >75% coverage

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure all tests pass before committing
3. Maintain or improve coverage percentages
4. Document any new test utilities or fixtures

## Questions or Issues?

For questions about testing:
- Check this README
- Review existing tests for examples
- Open an issue on GitHub
