# ThreadGuard: Advanced Static Analysis for Concurrency Bug Detection

[![Version](https://img.shields.io/badge/version-2.0.0-blue)](https://github.com/MLCyberSecOps/monero_cli_data_race)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)

ThreadGuard is an advanced static analysis tool designed to detect concurrency issues in C++ codebases. Initially developed to analyze the Monero CLI wallet, it has evolved into a comprehensive solution for identifying thread safety violations, data races, and synchronization bugs in multithreaded applications.

## 🔄 Version 2.0.0 Changelog (2024-06-23)

### 🚀 New Features
- **Enhanced Deadlock Detection**: Improved algorithms for detecting potential deadlocks and lock ordering issues
- **Mutex Checker Tool**: Added `simple_mutex_checker.py` for quick analysis of mutex usage patterns
- **Monero Analysis Scripts**: Specialized scripts for analyzing Monero's codebase
- **Performance Optimizations**: Faster analysis through optimized pattern matching

### 🐛 Bug Fixes
- Fixed false positives in lock acquisition order analysis
- Improved handling of RAII lock guards
- Better detection of recursive locking patterns

### 📦 Dependencies
- Updated to support Python 3.7+
- Added new dependencies for enhanced analysis

---

## ✨ Key Features

- **Data Race Detection**: Identify unsynchronized access to shared variables across threads
- **Thread Safety Analysis**: Detect potential thread safety violations in critical sections
- **Advanced Locking Analysis**: 
  - Verify proper mutex locking/unlocking patterns
  - Detect recursive locking
  - Identify potential deadlocks and lock ordering issues
- **Monero-Specific Analysis**: Specialized detection for common concurrency patterns in Monero's codebase
- **Comprehensive Reporting**: Generate detailed reports in multiple formats (JSON, console)

## Research Purpose

This tool is developed for education and research purposes to analyze potential concurrency issues. It does not contain any exploits. The analysis is not intended for publication.

## Author

**Pradeep Kumar**  
[![LinkedIn](https://img.shields.io/badge/Connect-LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kumar07/)

> *I have used various coding co-pilots to fine tune this tool*

## Features

- **Data Race Detection**: Identifies unsynchronized access to shared variables across multiple threads
- **Locking Pattern Analysis**: Verifies proper mutex locking/unlocking patterns
- **Thread Safety Analysis**: Detects potential thread safety violations in critical sections.
- **Monero-Specific Patterns**: Specialized detection for common concurrency patterns in Monero's codebase.

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`
- For development: Additional dependencies in `requirements-dev.txt`

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- GCC/Clang for C++ code analysis
- Git (for cloning the repository)

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/MLCyberSecOps/monero_cli_data_race.git
cd monero_cli_data_race

# Install dependencies
pip install -r requirements.txt

# For development (optional)
pip install -r requirements-dev.txt
cd monero_cli_data_race

# Install development dependencies
pip install -r requirements-dev.txt
```

## Testing

Run the test suite with:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage report
pytest --cov=threadguard_new tests/


# Run a specific test file
python -m pytest tests/test_analyzer.py -v
```

### Test Structure

- `tests/test_analyzer.py`: Unit tests for core analyzer functionality
- `tests/test_integration.py`: Integration tests with real C++ files
- `tests/test_utils.py`: Test utilities and test data

## Usage

```bash
python threadguard_new.py [options] <source_file.cpp>
```

For detailed documentation, see [THREADGUARD.md](THREADGUARD.md).

## Research Purpose

This tool is developed for education and research purposes to analyze potential concurrency issues. It does not contain any exploits. The analysis is not intended for publication.

## License

MIT License - see [LICENSE](LICENSE) for details.
