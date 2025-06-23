"""Integration tests for ThreadGuard analyzer with real C++ files."""

import os
import unittest
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path so we can import threadguard_new
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from threadguard_new import ThreadGuardAnalyzer
from tests.test_utils import TestData, create_temp_cpp_file, cleanup_temp_file


class TestThreadGuardIntegration(unittest.TestCase):
    """Integration tests for ThreadGuard analyzer."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for the test case."""
        cls.test_dir = tempfile.mkdtemp(prefix='threadguard_test_')
        cls.analyzer = ThreadGuardAnalyzer()
        
        # Create test files
        cls.simple_race_file = os.path.join(cls.test_dir, 'simple_race.cpp')
        with open(cls.simple_race_file, 'w', encoding='utf-8') as f:
            f.write(TestData.get_simple_data_race())
            
        cls.monero_file = os.path.join(cls.test_dir, 'monero_reader.h')
        with open(cls.monero_file, 'w', encoding='utf-8') as f:
            f.write(TestData.get_monero_stdin_reader_snippet())
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)
    
    def test_analyze_simple_race_file(self):
        """Test analyzing a simple C++ file with a data race."""
        result = self.analyzer.analyze_file(Path(self.simple_race_file))
        
        # Should find at least one data race
        self.assertGreater(len(result.races), 0, 
                         "Should find data races in simple_race.cpp")
        
        # Verify the race is on the counter variable
        race_vars = [r.variable for r in result.races]
        self.assertTrue(any('counter' in var for var in race_vars),
                      "Should detect race on 'counter' variable")
    
    def test_analyze_monero_pattern_file(self):
        """Test analyzing a file with Monero's async_stdin_reader pattern."""
        result = self.analyzer.analyze_file(Path(self.monero_file))
        
        # Should find race conditions on m_read_status
        race_vars = [r.variable for r in result.races]
        self.assertTrue(
            any('m_read_status' in var for var in race_vars),
            "Should detect race condition on m_read_status in async_stdin_reader"
        )
    
    def test_analyze_nonexistent_file(self):
        """Test analyzing a non-existent file."""
        non_existent = Path(self.test_dir) / "does_not_exist.cpp"
        result = self.analyzer.analyze_file(non_existent)
        
        # Should have recorded an error with the correct message
        self.assertTrue(
            any(f"File does not exist: {non_existent}" in str(issue) for issue in result.inconsistent_locking),
            "Should handle non-existent files with a clear error message"
        )


if __name__ == '__main__':
    unittest.main()
