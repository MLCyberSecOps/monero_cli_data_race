"""
Test file: test_thread_pool.py
Purpose: Tests for analyzing thread pool implementations.
         Verifies proper task queuing and worker thread synchronization.
"""
import pytest
from pathlib import Path
from threadguard_new import ThreadGuardAnalyzer

class TestThreadPool:
    """Tests for thread pool pattern analysis."""
    
    @pytest.fixture
    def analyzer(self):
        return ThreadGuardAnalyzer()
    
    def test_thread_pool_pattern(self, analyzer, temp_cpp_file, cleanup_temp_files):
        """Test analysis of thread pool implementation."""
        content = """
        #include <vector>
        #include <thread>
        #include <queue>
        #include <functional>
        #include <mutex>
        #include <condition_variable>
        #include <atomic>
        #include <future>
        #include <stdexcept>
        
        class ThreadPool {
            using Task = std::function<void()>;
            
            std::vector<std::thread> m_threads;
            std::queue<Task> m_tasks;
            std::mutex m_mutex;
            std::condition_variable m_cv;
            std::atomic<bool> m_stop{false};
            
            void worker() {
                while (true) {
                    Task task;
                    {
                        std::unique_lock<std::mutex> lock(m_mutex);
                        m_cv.wait(lock, [this] { 
                            return m_stop || !m_tasks.empty(); 
                        });
                        
                        if (m_stop && m_tasks.empty()) return;
                        
                        task = std::move(m_tasks.front());
                        m_tasks.pop();
                    }
                    task();
                }
            }
            
        public:
            explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) {
                for (size_t i = 0; i < num_threads; ++i) {
                    m_threads.emplace_back(&ThreadPool::worker, this);
                }
            }
            
            ~ThreadPool() {
                m_stop = true;
                m_cv.notify_all();
                for (auto& t : m_threads) {
                    if (t.joinable()) t.join();
                }
            }
            
            template<class F, class... Args>
            auto enqueue(F&& f, Args&&... args) {
                using return_type = std::invoke_result_t<F, Args...>;
                
                auto task = std::make_shared<std::packaged_task<return_type()>>(
                    std::bind(std::forward<F>(f), std::forward<Args>(args)...)
                );
                
                std::future<return_type> res = task->get_future();
                {
                    std::unique_lock<std::mutex> lock(m_mutex);
                    if (m_stop) {
                        throw std::runtime_error("enqueue on stopped ThreadPool");
                    }
                    m_tasks.emplace([task](){ (*task)(); });
                }
                m_cv.notify_one();
                return res;
            }
        };
        """
        filepath = temp_cpp_file(content, suffix='.h')
        cleanup_temp_files(filepath)
        
        result = analyzer.analyze_file(Path(filepath))
        # Should not find any thread safety issues
        assert len(result.races) == 0
        assert len(result.deadlock_risks) == 0

