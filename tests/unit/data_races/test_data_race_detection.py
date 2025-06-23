"""Unit tests for data race detection."""
from pathlib import Path

import pytest


class TestDataRaceDetection:
    """Test suite for data race detection."""

    def test_shared_counter_without_lock(self, analyzer, tmp_path):
        """Test detection of unsynchronized counter increment."""
        code = """
        #include <thread>
        #include <vector>

        int counter = 0;

        void increment() {
            counter++;  // Unsynchronized access
        }

        void test_race() {
            std::vector<std::thread> threads;
            for (int i = 0; i < 10; ++i) {
                threads.emplace_back(increment);
            }

            for (auto& t : threads) {
                t.join();
            }
        }
        """
        test_file = tmp_path / "race_condition.cpp"
        test_file.write_text(code)

        result = analyzer.analyze_file(test_file)
        assert any("race condition" in str(issue).lower() for issue in result.races)

    def test_shared_vector_without_lock(self, analyzer, tmp_path):
        """Test detection of unsynchronized vector access."""
        code = """
        #include <vector>
        #include <thread>

        std::vector<int> shared_data;

        void add_data(int value) {
            shared_data.push_back(value);  // Unsynchronized access
        }

        void test_race() {
            std::thread t1(add_data, 1);
            std::thread t2(add_data, 2);

            t1.join();
            t2.join();
        }
        """
        test_file = tmp_path / "vector_race.cpp"
        test_file.write_text(code)

        result = analyzer.analyze_file(test_file)
        assert any("race condition" in str(issue).lower() for issue in result.races)

    def test_atomic_counter_no_race(self, analyzer, tmp_path):
        """Test that atomic operations don't trigger race detection."""
        code = """
        #include <atomic>
        #include <thread>
        #include <vector>

        std::atomic<int> counter{0};

        void increment() {
            counter.fetch_add(1, std::memory_order_relaxed);
        }

        void test_no_race() {
            std::vector<std::thread> threads;
            for (int i = 0; i < 10; ++i) {
                threads.emplace_back(increment);
            }

            for (auto& t : threads) {
                t.join();
            }
        }
        """
        test_file = tmp_path / "atomic_counter.cpp"
        test_file.write_text(code)

        result = analyzer.analyze_file(test_file)
        assert (
            not result.races
        ), "No data races should be detected with atomic operations"

    @pytest.mark.slow
    def test_false_positive_analysis(self, analyzer, tmp_path):
        """Test for false positives in race detection."""
        code = """
        #include <mutex>
        #include <vector>

        class SafeContainer {
            std::vector<int> data;
            mutable std::mutex mtx;

        public:
            void add(int value) {
                std::lock_guard<std::mutex> lock(mtx);
                data.push_back(value);
            }

            bool contains(int value) const {
                std::lock_guard<std::mutex> lock(mtx);
                return std::find(data.begin(), data.end(), value) != data.end();
            }

            size_t size() const {
                std::lock_guard<std::mutex> lock(mtx);
                return data.size();
            }
        };
        """
        test_file = tmp_path / "false_positive.cpp"
        test_file.write_text(code)

        result = analyzer.analyze_file(test_file)
        assert (
            not result.races
        ), "No false positives should be detected in properly synchronized code"
