"""
Test file: test_condition_variable.py
Purpose: Tests for analyzing condition variable usage patterns.
         Ensures proper wait/notify patterns and predicate usage.
"""
from pathlib import Path

import pytest

from threadguard_new import ThreadGuardAnalyzer


class TestConditionVariable:
    """Tests for condition variable pattern analysis."""

    @pytest.fixture
    def analyzer(self):
        return ThreadGuardAnalyzer()

    def test_condition_variable_pattern(
        self, analyzer, temp_cpp_file, cleanup_temp_files
    ):
        """Test analysis of condition variable usage."""
        content = """
        #include <mutex>
        #include <condition_variable>
        #include <queue>

        template<typename T>
        class BoundedBuffer {
            std::queue<T> m_buffer;
            const size_t m_max_size;
            std::mutex m_mutex;
            std::condition_variable m_not_full;
            std::condition_variable m_not_empty;

        public:
            BoundedBuffer(size_t max_size) : m_max_size(max_size) {}

            void put(T item) {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_not_full.wait(lock, [this] { return m_buffer.size() < m_max_size; });
                m_buffer.push(std::move(item));
                m_not_empty.notify_one();
            }

            T get() {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_not_empty.wait(lock, [this] { return !m_buffer.empty(); });
                T item = std::move(m_buffer.front());
                m_buffer.pop();
                m_not_full.notify_one();
                return item;
            }
        };
        """
        filepath = temp_cpp_file(content, suffix=".h")
        cleanup_temp_files(filepath)

        result = analyzer.analyze_file(Path(filepath))
        # Should not find any thread safety issues
        assert len(result.races) == 0
        assert len(result.deadlock_risks) == 0
