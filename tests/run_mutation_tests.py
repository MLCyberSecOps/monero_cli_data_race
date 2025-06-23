#!/usr/bin/env python3
"""Mutation testing for ThreadGuard concurrency analysis."""
import sys
import pytest
from pathlib import Path
from typing import Callable, Dict, List
from mutators import (
    CodeMutator, 
    RemoveLockMutator, 
    SwapLockOrderMutator,
    AddDataRaceMutator,
    run_mutation_test,
    print_mutation_test_results
)
from threadguard_new import ThreadGuardAnalyzer

# Add parent directory to path to import test utilities
sys.path.append(str(Path(__file__).parent.parent))

# Sample test code to mutate
SAMPLE_CODE = """
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

def test_thread_safety(code: str) -> bool:
    """Test if the code has thread safety issues."""
    analyzer = ThreadGuardAnalyzer()
    with open("temp_test.cpp", "w") as f:
        f.write(code)
    
    result = analyzer.analyze_file(Path("temp_test.cpp"))
    
    # Test passes if no races or deadlocks are detected
    return not (result.races or result.deadlock_risks)

def main():
    """Run mutation testing on the sample code."""
    # Define mutators
    mutators: List[CodeMutator] = [
        RemoveLockMutator(),
        SwapLockOrderMutator(),
        AddDataRaceMutator()
    ]
    
    # Run mutation tests
    print("Running mutation tests...")
    results = run_mutation_test(
        original_code=SAMPLE_CODE,
        test_func=test_thread_safety,
        mutators=mutators,
        num_iterations=20  # Number of mutations to try per mutator
    )
    
    # Print results
    print_mutation_test_results(results)
    
    # Return non-zero exit code if any tests failed
    if any(any(status != "PASSED" for status in status_list) 
           for status_list in results.values()):
        sys.exit(1)

if __name__ == "__main__":
    main()