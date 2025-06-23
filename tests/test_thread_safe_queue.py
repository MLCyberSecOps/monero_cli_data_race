"""
Test file: test_thread_safe_queue.py
Purpose: Tests for analyzing thread-safe queue implementation patterns.
         Verifies proper synchronization in producer-consumer scenarios.
"""
from pathlib import Path

import pytest

from threadguard_new import ThreadGuardAnalyzer


class TestThreadSafeQueue:
    """Tests for thread-safe queue pattern analysis."""

    @pytest.fixture
    def analyzer(self):
        return ThreadGuardAnalyzer()

    def test_thread_safe_queue(self, analyzer, temp_cpp_file, cleanup_temp_files):
        """Test analysis of thread-safe queue implementation."""
        content = """
        #include <queue>
        #include <mutex>
        #include <condition_variable>
        #include <optional>

        template<typename T>
        class ThreadSafeQueue {
            std::queue<T> m_queue;
            mutable std::mutex m_mutex;
            std::condition_variable m_cv;
            bool m_shutdown = false;

        public:
            void push(T item) {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (m_shutdown) return;
                m_queue.push(std::move(item));
                m_cv.notify_one();
            }

            std::optional<T> pop() {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cv.wait(lock, [this] { return !m_queue.empty() || m_shutdown; });
                if (m_queue.empty()) return std::nullopt;
                T item = std::move(m_queue.front());
                m_queue.pop();
                return item;
            }

            void shutdown() {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_shutdown = true;
                m_cv.notify_all();
            }

            bool empty() const {
                std::lock_guard<std::mutex> lock(m_mutex);
                return m_queue.empty();
            }
        };
        """
        filepath = temp_cpp_file(content, suffix=".h")
        cleanup_temp_files(filepath)

        result = analyzer.analyze_file(Path(filepath))
        # Should not find any thread safety issues
        assert len(result.races) == 0
        assert len(result.deadlock_risks) == 0
