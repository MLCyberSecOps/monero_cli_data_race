# ThreadGuard: Static Analysis for Concurrency Bug Detection

ThreadGuard is a static analysis tool designed to detect concurrency-related issues in C++ code, with special focus on thread safety, data races, and locking patterns. It was specifically developed to analyze Monero's `async_stdin_reader` class but can be adapted for general C++ codebase analysis.

## Author

[![LinkedIn](https://img.shields.io/badge/Connect-LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kumar07/)

## Features

- **Data Race Detection**: Identifies unsynchronized access to shared variables across multiple threads
- **Locking Pattern Analysis**: Verifies proper mutex locking/unlocking patterns
- **Thread Safety Analysis**: Detects potential thread safety violations in critical sections.
- **Monero-Specific Patterns**: Specialized detection for common concurrency patterns in Monero's codebase.

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python threadguard_new.py [options] <source_file.cpp>
```

For detailed documentation, see [THREADGUARD.md](THREADGUARD.md).

## Research Purpose

This tool is developed for education and research purposes to analyze potential concurrency issues. It does not contain any exploits. The analysis is not intended for publication.

## License

MIT License - see [LICENSE](LICENSE) for details.
