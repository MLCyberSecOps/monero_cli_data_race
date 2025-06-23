"""Unit tests for deadlock detection."""
import pytest
from pathlib import Path

class TestDeadlockDetection:
    """Test suite for deadlock detection."""
    
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
        assert any("recursive lock" in str(issue).lower() for issue in result.deadlock_risks)
    
    def test_detect_missing_unlock(self, analyzer, tmp_path):
        """Test detection of missing unlock operations."""
        code = """
        #include <mutex>
        std::mutex mtx;
        
        void func(bool condition) {
            mtx.lock();
            if (condition) {
                return;  // Missing unlock
            }
            mtx.unlock();
        }
        """
        test_file = tmp_path / "missing_unlock.cpp"
        test_file.write_text(code)
        
        result = analyzer.analyze_file(test_file)
        assert any("not released" in str(issue).lower() for issue in result.deadlock_risks)
    
    def test_lock_guard_usage(self, analyzer, tmp_path):
        """Test correct usage of std::lock_guard."""
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
        test_file = tmp_path / "lock_guard_usage.cpp"
        test_file.write_text(code)
        
        result = analyzer.analyze_file(test_file)
        assert not result.races, "No data races should be detected"
        assert not result.deadlock_risks, "No deadlock risks should be detected"
    
    def test_detect_deadlock_cycle(self, analyzer, tmp_path):
        """Test detection of potential deadlock cycles."""
        code = """
        #include <mutex>
        
        std::mutex mtx1, mtx2;
        
        void thread1() {
            std::lock_guard<std::mutex> lock1(mtx1);
            // ...
            std::lock_guard<std::mutex> lock2(mtx2);  // Potential deadlock
        }
        
        void thread2() {
            std::lock_guard<std::mutex> lock2(mtx2);
            // ...
            std::lock_guard<std::mutex> lock1(mtx1);  // Reverse order -> deadlock
        }
        """
        test_file = tmp_path / "deadlock_cycle.cpp"
        test_file.write_text(code)
        
        result = analyzer.analyze_file(test_file)
        assert any("potential deadlock" in str(issue).lower() for issue in result.deadlock_risks)
    
    @pytest.mark.slow
    def test_complex_deadlock_scenario(self, analyzer, tmp_path):
        """Test detection in a more complex deadlock scenario."""
        code = """
        #include <mutex>
        #include <vector>
        #include <map>
        
        class ResourceManager {
            std::map<int, std::vector<int>> resources;
            mutable std::mutex mtx;
            
        public:
            void add_resource(int id, int value) {
                std::lock_guard<std::mutex> lock(mtx);
                resources[id].push_back(value);
            }
            
            bool transfer(int from, int to, int value) {
                std::lock_guard<std::mutex> lock(mtx);
                
                auto& source = resources[from];
                auto it = std::find(source.begin(), source.end(), value);
                if (it == source.end()) {
                    return false;
                }
                
                // Simulate a potential deadlock by calling another method that locks
                if (!validate_transfer(from, to, value)) {
                    return false;
                }
                
                resources[to].push_back(value);
                source.erase(it);
                return true;
            }
            
            bool validate_transfer(int from, int to, int value) {
                std::lock_guard<std::mutex> lock(mtx);  // Potential recursive lock
                // Complex validation logic here
                return true;
            }
        };
        """
        test_file = tmp_path / "complex_deadlock.cpp"
        test_file.write_text(code)
        
        result = analyzer.analyze_file(test_file)
        assert any("potential deadlock" in str(issue).lower() for issue in result.deadlock_risks)
