"""Test utilities for ThreadGuard test suite."""

import os
import tempfile
from pathlib import Path
from typing import Optional


def create_temp_cpp_file(content: str, suffix: str = '.cpp') -> str:
    """Create a temporary C++ file with the given content.
    
    Args:
        content: The C++ code to write to the file
        suffix: File extension (default: .cpp)
        
    Returns:
        Path to the created temporary file
    """
    with tempfile.NamedTemporaryFile(
        mode='w', 
        suffix=suffix, 
        delete=False,
        encoding='utf-8'
    ) as f:
        f.write(content)
        return f.name


def cleanup_temp_file(filepath: str) -> None:
    """Remove a temporary file if it exists.
    
    Args:
        filepath: Path to the file to remove
    """
    try:
        if filepath and os.path.exists(filepath):
            os.unlink(filepath)
    except (OSError, PermissionError):
        pass  # Ignore cleanup errors in tests


class TestData:
    """Test data for ThreadGuard test cases."""
    
    @staticmethod
    def get_simple_data_race() -> str:
        """Get a simple C++ code snippet with a data race."""
        return """
        #include <thread>
        
        int counter = 0;
        
        void increment() {
            for (int i = 0; i < 1000; ++i) {
                counter++;  // Data race here
            }
        }
        
        int main() {
            std::thread t1(increment);
            std::thread t2(increment);
            t1.join();
            t2.join();
            return 0;
        }
        """
    
    @staticmethod
    def get_monero_stdin_reader_snippet() -> str:
        """Get a simplified version of Monero's async_stdin_reader pattern."""
        return """
        class async_stdin_reader {
        public:
            void run() {
                while (m_read_status != state_exit) {  // Race condition
                    // ...
                }
            }
            
            void stop() {
                m_read_status = state_exit;  // Another race condition
            }
            
        private:
            enum states { state_ready, state_exit };
            states m_read_status = state_ready;
        };
        """
