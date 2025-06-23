# ThreadGuard: A Static Analysis Tool for Concurrency Bug Detection

[![Version](https://img.shields.io/badge/version-1.1.0-blue)](https://github.com/MLCyberSecOps/monero_cli_data_race)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

ThreadGuard was originally developed to assist in the static analysis of a confirmed data race vulnerability in Monero's `async_stdin_reader` class (console_handler.h). The tool is capable of identifying thread safety violations, data races, and synchronization bugs in multithreaded C++ codebases. While its initial focus was the Monero CLI wallet, ThreadGuard's pattern-based engine is extensible to broader C++ concurrency analysis.

## Features

### 1. Data Race Detection
- Identifies unsynchronized access to shared variables across multiple threads
- Detects potential race conditions in critical sections
- Flags non-atomic operations on shared memory

### 2. Locking Pattern Analysis
- Verifies proper mutex locking/unlocking patterns
- Detects potential deadlocks and lock leaks
- Identifies inconsistent locking strategies

### 3. Thread Safety Analysis
- Analyzes thread entry points and execution contexts
- Identifies thread-unsafe operations
- Flags potential issues with thread-local storage

### 4. Monero-Specific Analysis
- Specialized detection for `async_stdin_reader` patterns
- Identifies common concurrency issues in Monero's codebase
- Provides targeted fix recommendations

## Installation

### Prerequisites
- Python 3.6 or higher
- NetworkX library

### Setup
```bash
# Clone the repository
https://github.com/MLCyberSecOps/monero_cli_data_race
cd monero_cli_data_race

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  
# This script has not been tested on Windows. On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
# Analyze a single file
python threadguard.py path/to/your/file.cpp

# Analyze all header files in a directory
find /path/to/source -name "*.h" -exec python threadguard.py {} \;
```

### Command Line Options
```
--output FILE     Save analysis report to FILE
--json FILE       Output results in JSON format
--ci-mode         Run in CI mode with non-zero exit on issues
--max-critical N  Maximum allowed critical issues (CI mode)
--max-high N      Maximum allowed high severity issues (CI mode)
```

### Example Output
```
================================================================================
ThreadGuard Analysis Report - console_handler.h
================================================================================

🔍 SUMMARY
----------------------------------------
Critical Races: 3
High Severity Races: 2
Locking Issues: 5
Deadlock Risks: 1

🚨 CRITICAL ISSUES
----------------------------------------
• Race condition on m_read_status in thread()
  File: console_handler.h:86
  Fix: Protect access to m_read_status with a mutex lock in thread()
```

## Integration with Build Systems

### CMake Integration
```cmake
add_custom_target(threadguard_analysis
    COMMAND python ${CMAKE_SOURCE_DIR}/tools/threadguard.py
                   --output ${CMAKE_BINARY_DIR}/threadguard_report.txt
                   ${SOURCES}
    COMMENT "Running ThreadGuard static analysis"
    VERBATIM
)
```

### CI/CD Pipeline Example (GitHub Actions)
```yaml
name: ThreadGuard Analysis

on: [push, pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install networkx
    - name: Run ThreadGuard
      run: |
        python tools/threadguard.py --ci-mode --max-critical 0 src/
```

## Development

### Adding New Detectors
1. Create a new method in the `ThreadGuardAnalyzer` class
2. Register it in the `analyze_file` method
3. Add test cases in the `tests/` directory

### Running Tests
```bash
python -m pytest tests/
```

## License
MIT License - See [LICENSE](LICENSE) for details.

## Contributing
Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Authors
- Pradeep Kumar

## Acknowledgments
- Inspired by various static analysis tools and research in concurrent programming
- Special thanks to the Monero community for their feedback and testing
