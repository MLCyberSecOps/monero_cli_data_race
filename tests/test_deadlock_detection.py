"""
Test file: test_deadlock_detection.py
Purpose: Tests for detecting potential deadlocks in code.
         Verifies identification of lock ordering issues and potential deadlocks.
"""
from pathlib import Path

import pytest

from threadguard_new import ThreadGuardAnalyzer


class TestDeadlockDetection:
    """Tests for deadlock detection."""

    @pytest.fixture
    def analyzer(self):
        return ThreadGuardAnalyzer()

    def test_detect_potential_deadlock(
        self, analyzer, temp_cpp_file, cleanup_temp_files
    ):
        """Test detection of potential deadlocks."""
        content = """
        #include <mutex>
        #include <thread>

        std::mutex m1, m2;

        void thread1() {
            std::lock_guard<std::mutex> lock1(m1);  // Acquires m1 first
            std::lock_guard<std::mutex> lock2(m2);  // Then acquires m2
        }

        void thread2() {
            std::lock_guard<std::mutex> lock2(m2);  // Acquires m2 first
            std::lock_guard<std::mutex> lock1(m1);  // Then acquires m1 - potential deadlock with thread1
        }

        int main() {
            std::thread t1(thread1);
            std::thread t2(thread2);
            t1.join();
            t2.join();
            return 0;
        }
        """
        filepath = temp_cpp_file(content)
        cleanup_temp_files(filepath)

        result = analyzer.analyze_file(Path(filepath))
        assert len(result.deadlock_risks) > 0, "Should detect potential deadlock"

    def test_properly_ordered_locks(self, analyzer, temp_cpp_file, cleanup_temp_files):
        """Test that properly ordered locks don't trigger deadlock detection."""
        content = """
        #include <mutex>
        #include <thread>

        std::mutex m1, m2;

        void thread1() {
            std::lock_guard<std::mutex> lock1(m1);
            std::lock_guard<std::mutex> lock2(m2);
        }

        void thread2() {
            std::lock_guard<std::mutex> lock1(m1);  // Same order as thread1
            std::lock_guard<std::mutex> lock2(m2);  // No potential deadlock
        }

        int main() {
            std::thread t1(thread1);
            std::thread t2(thread2);
            t1.join();
            t2.join();
            return 0;
        }
        """
        filepath = temp_cpp_file(content)
        cleanup_temp_files(filepath)

        result = analyzer.analyze_file(Path(filepath))
        assert (
            len(result.deadlock_risks) == 0
        ), "Should not detect deadlocks in properly ordered locks"
