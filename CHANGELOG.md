# Changelog

All notable changes to the ThreadGuard project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and documentation
- Basic static analysis capabilities

## [2.0.0] - 2024-06-23

### Added
- Enhanced deadlock detection algorithms
- New `simple_mutex_checker.py` tool for quick mutex analysis
- Specialized Monero analysis scripts
- Comprehensive test suite with concurrency patterns
- Detailed documentation and examples

### Changed
- Improved pattern matching for C++ concurrency primitives
- Better handling of RAII lock guards
- Enhanced reporting format with more context
- Optimized analysis performance

### Fixed
- False positives in lock acquisition order analysis
- Issues with recursive locking detection
- Path handling in analysis scripts

## [1.1.0] - 2024-06-20

### Added
- Initial version of ThreadGuard analyzer
- Basic data race detection
- Simple lock analysis
- Monero-specific pattern detection

### Changed
- Improved error handling and reporting
- Better documentation

## [1.0.0] - 2024-06-15

### Added
- Initial research and development version
- Basic static analysis capabilities
- Experimental features

[Unreleased]: https://github.com/MLCyberSecOps/monero_cli_data_race/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/MLCyberSecOps/monero_cli_data_race/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/MLCyberSecOps/monero_cli_data_race/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/MLCyberSecOps/monero_cli_data_race/releases/tag/v1.0.0
