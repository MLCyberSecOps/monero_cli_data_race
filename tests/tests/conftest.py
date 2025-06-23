import os
import tempfile
from pathlib import Path

import pytest


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
