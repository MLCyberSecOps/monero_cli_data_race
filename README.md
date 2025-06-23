# ThreadGuard: Static Analysis for Concurrency Bug Detection

[![Version](https://img.shields.io/badge/version-1.1.0-blue)](https://github.com/MLCyberSecOps/monero_cli_data_race)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ThreadGuard was originally developed to assist in the static analysis of a confirmed data race vulnerability in Monero's `async_stdin_reader` class (console_handler.h). The tool is capable of identifying thread safety violations, data races, and synchronization bugs in multithreaded C++ codebases. While its initial focus was the Monero CLI wallet, ThreadGuard's pattern-based engine is extensible to broader C++ concurrency analysis.

## Key Features

- **Data Race Detection**: Identify unsynchronized access to shared variables
- **Thread Safety Analysis**: Detect potential thread safety violations in critical sections
- **Locking Pattern Analysis**: Verify proper mutex locking/unlocking patterns
- **Deadlock Detection**: Spot potential deadlocks and lock ordering issues
- **Monero-Specific Patterns**: Specialized detection for common concurrency patterns in Monero's codebase

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

## Installation

### Basic Installation
```bash
pip install -r requirements.txt
```

### Development Setup
```bash
# Clone the repository
git clone https://github.com/MLCyberSecOps/monero_cli_data_race.git
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
