import pytest
from pathlib import Path
from threadguard_new import ThreadGuardAnalyzer

class TestAsyncReader:
    """Tests for async reader pattern analysis."""
    
    @pytest.fixture
    def analyzer(self):
        return ThreadGuardAnalyzer()
    
    def test_async_reader_pattern(self, analyzer, temp_cpp_file, cleanup_temp_files):
        """Test analysis of async reader pattern."""
        content = """
        #include <atomic>
        #include <condition_variable>
        #include <mutex>
        #include <string>
        #include <thread>

        class AsyncReader {
            std::atomic<bool> m_run{true};
            std::string m_line;
            std::mutex m_mutex;
            std::condition_variable m_cv;
            bool m_ready = false;

        public:
            void run() {
                std::thread([this]() {
                    while (m_run) {
                        std::string input;
                        std::getline(std::cin, input);
                        
                        {
                            std::lock_guard<std::mutex> lock(m_mutex);
                            m_line = std::move(input);
                            m_ready = true;
                        }
                        m_cv.notify_one();
                    }
                }).detach();
            }

            bool get_line(std::string& line) {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cv.wait(lock, [this] { return m_ready || !m_run; });
                if (m_ready) {
                    line = std::move(m_line);
                    m_ready = false;
                    return true;
                }
                return false;
            }

            void stop() {
                m_run = false;
                m_cv.notify_one();
            }
        };
        """
        filepath = temp_cpp_file(content, suffix='.h')
        cleanup_temp_files(filepath)
        
        result = analyzer.analyze_file(Path(filepath))
        # Should not find any thread safety issues
        assert len(result.races) == 0
        assert len(result.deadlock_risks) == 0

