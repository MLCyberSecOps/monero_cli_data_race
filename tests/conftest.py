"""Pytest configuration and fixtures for ThreadGuard tests."""
"""Pytest configuration and fixtures for ThreadGuard tests."""
import os
import sys
import tempfile
from pathlib import Path
from typing import Generator, List, Optional

import pytest

# Add the parent directory to the path so we can import threadguard_new
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_cpp_file():
    """Create a temporary C++ file for testing."""

    def _create_file(content, suffix=".cpp"):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            return f.name

    return _create_file


@pytest.fixture
def temp_py_file():
    """Create a temporary Python file for testing."""

    def _create_file(content, suffix=".py"):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            return f.name

    return _create_file


@pytest.fixture
def cleanup_temp_files(request):
    """Clean up temporary files after tests."""
    files = []

    def _add_file(filepath):
        if filepath:
            files.append(filepath)
        return filepath

    yield _add_file

    for filepath in files:
        try:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)
        except (OSError, PermissionError):
            pass


@pytest.fixture
def analyzer():
    """Create a new ThreadGuardAnalyzer instance for testing."""
    from threadguard_new import ThreadGuardAnalyzer

    return ThreadGuardAnalyzer()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark")
