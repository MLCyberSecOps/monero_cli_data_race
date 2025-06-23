# ThreadGuard: Static Analysis for Concurrency Bug Detection and Stratification

[![Version](https://img.shields.io/badge/version-1.1.0-blue)](https://github.com/MLCyberSecOps/monero_cli_data_race)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Originally developed to analyze concurrency issues in Monero's `async_stdin_reader` class, ThreadGuard is a powerful static analysis tool that can audit any multithreaded C++ project for:

- **Data Races**: Detect unsynchronized access to shared variables across threads
- **Thread Safety**: Identify potential thread safety violations in critical sections
- **Synchronization Flaws**: Analyze locking patterns and mutex usage
- **Deadlock Risks**: Spot potential deadlocks and lock ordering issues
- **Atomic Operation Validation**: Ensure proper use of atomic operations

The tool's modular architecture and configurable rules make it adaptable to various codebases beyond its original Monero use case.

## Author

**Pradeep Kumar**  
[![LinkedIn](https://img.shields.io/badge/Connect-LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kumar07/)

> *Assisted by various coding co-pilots*

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
