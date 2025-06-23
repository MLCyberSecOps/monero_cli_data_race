"""Unit tests for atomic operation analysis."""
import pytest
from pathlib import Path

class TestAtomicOperations:
    """Test suite for atomic operation analysis."""
    
    def test_atomic_counter(self, analyzer, tmp_path):
        """Test correct usage of atomic counter."""
        code = """
        #include <atomic>
        #include <thread>
        #include <vector>
        
        std::atomic<int> counter{0};
        
        void increment() {
            counter.fetch_add(1, std::memory_order_relaxed);
        }
        
        void test_atomic() {
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
        assert not result.races, "No data races should be detected with atomic operations"
    
    def test_memory_ordering(self, analyzer, tmp_path):
        """Test different memory orderings with atomics."""
        code = """
        #include <atomic>
        #include <thread>
        
        std::atomic<bool> flag{false};
        int data = 0;
        
        void producer() {
            data = 42;  // This write happens before the flag is set
            flag.store(true, std::memory_order_release);
        }
        
        void consumer() {
            while (!flag.load(std::memory_order_acquire)) {
                // Wait for the flag
            }
            int local = data;  // Should see data = 42
        }
        
        void test_memory_order() {
            std::thread t1(producer);
            std::thread t2(consumer);
            
            t1.join();
            t2.join();
        }
        """
        test_file = tmp_path / "memory_ordering.cpp"
        test_file.write_text(code)
        
        result = analyzer.analyze_file(test_file)
        assert not result.races, "Properly synchronized with memory orderings"
    
    def test_atomic_flag(self, analyzer, tmp_path):
        """Test usage of std::atomic_flag."""
        code = """
        #include <atomic>
        #include <thread>
        #include <vector>
        
        std::atomic_flag lock = ATOMIC_FLAG_INIT;
        int shared_data = 0;
        
        void increment() {
            while (lock.test_and_set(std::memory_order_acquire)) {
                // Spin until we get the lock
            }
            shared_data++;  // Critical section
            lock.clear(std::memory_order_release);
        }
        
        void test_spinlock() {
            std::vector<std::thread> threads;
            for (int i = 0; i < 10; ++i) {
                threads.emplace_back(increment);
            }
            
            for (auto& t : threads) {
                t.join();
            }
        }
        """
        test_file = tmp_path / "atomic_flag.cpp"
        test_file.write_text(code)
        
        result = analyzer.analyze_file(test_file)
        assert not result.races, "No data races should be detected with atomic_flag"
    
    def test_atomic_compare_exchange(self, analyzer, tmp_path):
        """Test compare-and-exchange operations."""
        code = """
        #include <atomic>
        #include <thread>
        #include <vector>
        
        std::atomic<int> max_value{0};
        
        void update_max(int value) {
            int current = max_value.load(std::memory_order_relaxed);
            while (current < value && 
                   !max_value.compare_exchange_weak(
                       current, value,
                       std::memory_order_release,
                       std::memory_order_relaxed)) {
                // Try again with the new current value
            }
        }
        
        void test_atomic_update() {
            std::vector<std::thread> threads;
            for (int i = 1; i <= 100; ++i) {
                threads.emplace_back(update_max, i);
            }
            
            for (auto& t : threads) {
                t.join();
            }
        }
        """
        test_file = tmp_path / "atomic_compare_exchange.cpp"
        test_file.write_text(code)
        
        result = analyzer.analyze_file(test_file)
        assert not result.races, "No data races should be detected with compare-exchange"
