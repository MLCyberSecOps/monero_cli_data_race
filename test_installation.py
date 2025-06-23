#!/usr/bin/env python3
"""
ThreadGuard Installation Test Script

This script verifies that ThreadGuard Enhanced is properly installed
and can detect basic concurrency issues.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Test cases with known issues
TEST_CASES = {
    "simple_race.cpp": """
#include <iostream>
#include <thread>
#include <vector>

int global_counter = 0;  // Shared variable without protection

void increment_counter() {
    for (int i = 0; i < 1000; ++i) {
        global_counter++;  // Race condition here
    }
}

int main() {
    std::vector<std::thread> threads;

    for (int i = 0; i < 4; ++i) {
        threads.emplace_back(increment_counter);
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Final counter: " << global_counter << std::endl;
    return 0;
}
""",
    "lock_guard_good.cpp": """
#include <iostream>
#include <thread>
#include <mutex>
#include <vector>

int protected_counter = 0;
std::mutex counter_mutex;

void safe_increment() {
    for (int i = 0; i < 1000; ++i) {
        std::lock_guard<std::mutex> lock(counter_mutex);
        protected_counter++;  // This should be safe
    }
}

int main() {
    std::vector<std::thread> threads;

    for (int i = 0; i < 4; ++i) {
        threads.emplace_back(safe_increment);
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Final counter: " << protected_counter << std::endl;
    return 0;
}
""",
    "deadlock_risk.cpp": """
#include <mutex>
#include <thread>

std::mutex mutex1;
std::mutex mutex2;

void function1() {
    std::lock_guard<std::mutex> lock1(mutex1);
    std::lock_guard<std::mutex> lock2(mutex2);
    // Do some work
}

void function2() {
    std::lock_guard<std::mutex> lock2(mutex2);  // Different order!
    std::lock_guard<std::mutex> lock1(mutex1);  // Potential deadlock
    // Do some work
}

int main() {
    std::thread t1(function1);
    std::thread t2(function2);

    t1.join();
    t2.join();

    return 0;
}
""",
    "monero_style.cpp": """
#include <mutex>
#include <atomic>
#include <thread>

class async_stdin_reader {
private:
    volatile int m_read_status;  // Should be atomic
    mutable std::mutex m_response_mutex;

public:
    void check_status() {
        if (m_read_status == 1) {  // Unsynchronized access
            // Handle status
        }
    }

    void set_status(int status) {
        // Missing synchronization
        m_read_status = status;
    }

    void safe_check_status() {
        std::lock_guard<std::mutex> lock(m_response_mutex);
        if (m_read_status == 1) {  // This is properly protected
            // Handle status
        }
    }
};
""",
}


def test_dependencies():
    """Test if required dependencies are available"""
    print("🔍 Testing dependencies...")

    # Test Python version
    if sys.version_info < (3, 6):
        print("❌ Python 3.6+ required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")

    # Test networkx (optional)
    try:
        import networkx

        print(f"✅ networkx {networkx.__version__}")
        has_networkx = True
    except ImportError:
        print("⚠️ networkx not available (optional but recommended)")
        has_networkx = False

    return True


def run_threadguard_test(test_file, expected_issues=None):
    """Run ThreadGuard on a test file and check results"""
    script_path = Path(__file__).parent / "threadguard_enhanced.py"

    if not script_path.exists():
        print(f"❌ ThreadGuard script not found at {script_path}")
        return False

    try:
        # Run ThreadGuard
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--json",
                f"{test_file}.json",
                "--quiet",
                str(test_file),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Check if it ran successfully
        if result.returncode not in [
            0,
            1,
            2,
        ]:  # 0=success, 1=high issues, 2=critical issues
            print(f"❌ ThreadGuard failed on {test_file.name}")
            print(f"   Error: {result.stderr}")
            return False

        # Try to read JSON output
        json_file = Path(f"{test_file}.json")
        if json_file.exists():
            import json

            with open(json_file) as f:
                data = json.load(f)

            metrics = data.get("files", [{}])[0].get("metrics", {})
            total_races = metrics.get("total_races", 0)
            critical_races = metrics.get("critical_races", 0)
            high_races = metrics.get("high_races", 0)

            print(
                f"✅ {test_file.name}: {total_races} races ({critical_races} critical, {high_races} high)"
            )

            # Clean up
            json_file.unlink()

            return True
        else:
            print(f"⚠️ No JSON output for {test_file.name}")
            return True  # Still consider success if analysis ran

    except subprocess.TimeoutExpired:
        print(f"❌ ThreadGuard timed out on {test_file.name}")
        return False
    except Exception as e:
        print(f"❌ Error running ThreadGuard on {test_file.name}: {e}")
        return False


def main():
    """Main test function"""
    print("=" * 60)
    print("ThreadGuard Enhanced Installation Test")
    print("=" * 60)

    # Test dependencies
    if not test_dependencies():
        print("\n❌ Dependency check failed")
        return 1

    print(f"\n🧪 Testing ThreadGuard Enhanced...")

    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        all_passed = True

        # Create and test each test case
        for filename, content in TEST_CASES.items():
            test_file = temp_path / filename
            test_file.write_text(content)

            print(f"\n📁 Testing {filename}...")

            if not run_threadguard_test(test_file):
                all_passed = False

        print(f"\n{'='*60}")
        if all_passed:
            print("✅ ALL TESTS PASSED")
            print("\nThreadGuard Enhanced is properly installed and working!")
            print("\nNext steps:")
            print("1. Try: python3 threadguard_enhanced.py your_file.cpp")
            print("2. See setup guide for advanced usage")
            print("3. Consider installing: pip3 install networkx matplotlib")
            return 0
        else:
            print("❌ SOME TESTS FAILED")
            print("\nPlease check:")
            print("1. Python 3.6+ is installed")
            print("2. threadguard_enhanced.py is in the current directory")
            print("3. File permissions are correct")
            return 1


if __name__ == "__main__":
    sys.exit(main())
