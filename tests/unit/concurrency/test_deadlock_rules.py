"""Unit tests for deadlock detection rules."""
from pathlib import Path

import pytest

from threadguard_new import ThreadGuardAnalyzer


class TestDeadlockRules:
    """Test suite for deadlock detection rules."""

    @pytest.fixture
    def analyzer(self):
        """Provide a fresh analyzer instance for each test."""
        return ThreadGuardAnalyzer()

    def test_detect_double_lock(self, analyzer, tmp_path):
        """Test detection of recursive locking on the same mutex."""
        code = """
        #include <mutex>
        std::mutex mtx;

        void func() {
            mtx.lock();
            mtx.lock();  // Double lock - should be detected
            // ...
            mtx.unlock();
            mtx.unlock();
        }
        """
        test_file = tmp_path / "double_lock.cpp"
        test_file.write_text(code)

        result = analyzer.analyze_file(test_file)
        assert any(
            "recursive lock" in str(issue).lower() for issue in result.deadlock_risks
        )

    def test_detect_missing_unlock(self, analyzer, tmp_path):
        """Test detection of missing unlock operations."""
        code = """
        #include <mutex>
        std::mutex mtx;

        void func() {
            mtx.lock();
            if (some_condition) {
                return;  // Missing unlock
            }
            mtx.unlock();
        }
        """
        test_file = tmp_path / "missing_unlock.cpp"
        test_file.write_text(code)

        result = analyzer.analyze_file(test_file)
        assert any(
            "not released" in str(issue).lower() for issue in result.deadlock_risks
        )

    def test_detect_lock_order_violation(self, analyzer, tmp_path):
        """Test detection of inconsistent lock ordering."""
        code = """
        #include <mutex>
        std::mutex mtx1, mtx2;

        void func1() {
            std::lock_guard<std::mutex> lk1(mtx1);
            std::lock_guard<std::mutex> lk2(mtx2);  // Order: mtx1 -> mtx2
        }

        void func2() {
            std::lock_guard<std::mutex> lk2(mtx2);
            std::lock_guard<std::mutex> lk1(mtx1);  // Reverse order: mtx2 -> mtx1
        }
        """
        test_file = tmp_path / "lock_order.cpp"
        test_file.write_text(code)

        result = analyzer.analyze_file(test_file)
        assert any(
            "inconsistent lock ordering" in str(issue).lower()
            for issue in result.deadlock_risks
        )

    def test_negative_case(self, analyzer, tmp_path):
        """Test that correct locking patterns don't trigger warnings."""
        code = """
        #include <mutex>
        #include <vector>

        class ThreadSafeVector {
            std::vector<int> data;
            mutable std::mutex mtx;

        public:
            void push_back(int value) {
                std::lock_guard<std::mutex> lock(mtx);
                data.push_back(value);
            }

            size_t size() const {
                std::lock_guard<std::mutex> lock(mtx);
                return data.size();
            }
        };
        """
        test_file = tmp_path / "correct_usage.cpp"
        test_file.write_text(code)

        result = analyzer.analyze_file(test_file)
        assert not result.races, "No data races should be detected"
        assert not result.deadlock_risks, "No deadlock risks should be detected"
