"""
Test file: test_atomic_operations.py
Purpose: Tests for analyzing atomic operations usage.
         Verifies correct memory ordering and atomic access patterns.
"""
import pytest
from pathlib import Path
from threadguard_new import ThreadGuardAnalyzer

class TestAtomicOperations:
    """Tests for atomic operations analysis."""
    
    @pytest.fixture
    def analyzer(self):
        return ThreadGuardAnalyzer()
    
    def test_atomic_counter(self, analyzer, temp_cpp_file, cleanup_temp_files):
        """Test analysis of atomic counter implementation."""
        content = """
        #include <atomic>
        #include <thread>
        
        class AtomicCounter {
            std::atomic<int> m_count{0};
            
        public:
            void increment() {
                m_count.fetch_add(1, std::memory_order_relaxed);
            }
            
            void decrement() {
                m_count.fetch_sub(1, std::memory_order_relaxed);
            }
            
            int get() const {
                return m_count.load(std::memory_order_relaxed);
            }
            
            // Test different memory orderings
            void increment_acquire_release() {
                m_count.fetch_add(1, std::memory_order_acq_rel);
            }
            
            int get_acquire() const {
                return m_count.load(std::memory_order_acquire);
            }
            
            void set_release(int value) {
                m_count.store(value, std::memory_order_release);
            }
        };
        """
        filepath = temp_cpp_file(content, suffix='.h')
        cleanup_temp_files(filepath)
        
        result = analyzer.analyze_file(Path(filepath))
        # Should not find any thread safety issues
        assert len(result.races) == 0
        assert len(result.deadlock_risks) == 0

