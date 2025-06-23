"""Unit tests for ThreadGuard analyzer functionality."""

import os

# Add parent directory to path so we can import threadguard_new
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_utils import TestData, cleanup_temp_file, create_temp_cpp_file
from threadguard_new import ThreadGuardAnalyzer


class TestThreadGuardAnalyzer(unittest.TestCase):
    """Test cases for ThreadGuardAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ThreadGuardAnalyzer()
        self.test_files = []

    def tearDown(self):
        """Clean up test fixtures."""
        for filepath in self.test_files:
            cleanup_temp_file(filepath)

    def _create_test_file(self, content: str, suffix: str = ".cpp") -> str:
        """Create a test file and track it for cleanup."""
        filepath = create_temp_cpp_file(content, suffix)
        self.test_files.append(filepath)
        return filepath

    def test_detect_simple_data_race(self):
        """Test detection of a simple data race."""
        code = TestData.get_simple_data_race()
        filepath = self._create_test_file(code)

        # Analyze the file
        result = self.analyzer.analyze_file(Path(filepath))

        # Verify we found the data race
        self.assertGreaterEqual(
            len(result.races),
            1,
            "Should detect at least one data race in the simple example",
        )
        race_var = result.races[0].variable
        self.assertIn("counter", race_var, "Should detect race on 'counter' variable")

    def test_detect_monero_stdin_reader_pattern(self):
        """Test detection of Monero's async_stdin_reader race pattern."""
        code = TestData.get_monero_stdin_reader_snippet()
        filepath = self._create_test_file(code, ".h")

        # Analyze the file
        result = self.analyzer.analyze_file(Path(filepath))

        # Verify we found the race condition on m_read_status
        race_found = any("m_read_status" in race.variable for race in result.races)
        self.assertTrue(
            race_found,
            "Should detect race condition on m_read_status in async_stdin_reader",
        )

    @patch("threadguard_new.ThreadGuardAnalyzer._extract_shared_variables")
    def test_analyze_file_error_handling(self, mock_extract):
        """Test error handling in analyze_file method."""
        # Setup mock to raise an exception
        mock_extract.side_effect = Exception("Test error")

        # Create a test file
        code = "int x = 0;"
        filepath = self._create_test_file(code)

        # Should not raise an exception
        result = self.analyzer.analyze_file(Path(filepath))

        # Should have recorded the error
        self.assertTrue(
            any("ERROR" in issue for issue in result.inconsistent_locking),
            "Should record analysis errors in the result",
        )


if __name__ == "__main__":
    unittest.main()
