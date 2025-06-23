"""Test file: test_singleton.py
Purpose: Tests for analyzing thread-safe singleton patterns.
         Ensures proper lazy initialization and double-checked locking.
"""
from pathlib import Path

import pytest

from threadguard_new import ThreadGuardAnalyzer


class TestSingleton:
    """Tests for singleton pattern analysis."""

    @pytest.fixture
    def analyzer(self):
        return ThreadGuardAnalyzer()

    def test_thread_safe_singleton(self, analyzer, temp_cpp_file, cleanup_temp_files):
        """Test analysis of thread-safe singleton implementation."""
        content = """
        #include <mutex>
        #include <memory>

        class Singleton {
            static std::unique_ptr<Singleton> m_instance;
            static std::once_flag m_once_flag;

            Singleton() = default;
            ~Singleton() = default;

        public:
            Singleton(const Singleton&) = delete;
            Singleton& operator=(const Singleton&) = delete;

            static Singleton& get_instance() {
                std::call_once(m_once_flag, [] {
                    m_instance.reset(new Singleton);
                });
                return *m_instance;
            }

            void do_something() {
                // Implementation
            }
        };

        std::unique_ptr<Singleton> Singleton::m_instance;
        std::once_flag Singleton::m_once_flag;
        """
        filepath = temp_cpp_file(content, suffix=".h")
        cleanup_temp_files(filepath)

        result = analyzer.analyze_file(Path(filepath))
        # Should not find any thread safety issues
        assert len(result.races) == 0
        assert len(result.deadlock_risks) == 0
