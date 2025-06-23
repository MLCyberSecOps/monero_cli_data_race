"""Test file: test_command_handler.py
Purpose: Tests for analyzing thread-safe command handler patterns.
         Ensures proper synchronization in command dispatch systems.
"""
from pathlib import Path

import pytest

from threadguard_new import ThreadGuardAnalyzer


class TestCommandHandler:
    """Tests for command handler pattern analysis."""

    @pytest.fixture
    def analyzer(self):
        return ThreadGuardAnalyzer()

    def test_command_handler_pattern(self, analyzer, temp_cpp_file, cleanup_temp_files):
        """Test analysis of command handler implementation."""
        content = """
        #include <map>
        #include <functional>
        #include <string>
        #include <vector>
        #include <mutex>

        class CommandHandler {
            using CommandFunc = std::function<bool(const std::vector<std::string>&)>;
            std::map<std::string, CommandFunc> m_handlers;
            mutable std::mutex m_mutex;

        public:
            void register_command(const std::string& cmd, CommandFunc handler) {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_handlers[cmd] = std::move(handler);
            }

            bool execute(const std::string& cmd) {
                CommandFunc handler;
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    auto it = m_handlers.find(cmd);
                    if (it == m_handlers.end()) return false;
                    handler = it->second;
                }
                return handler({cmd});
            }
        };
        """
        filepath = temp_cpp_file(content, suffix=".h")
        cleanup_temp_files(filepath)

        result = analyzer.analyze_file(Path(filepath))
        # Should not find any thread safety issues
        assert len(result.races) == 0
        assert len(result.deadlock_risks) == 0
