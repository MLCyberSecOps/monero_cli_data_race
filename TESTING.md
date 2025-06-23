# ThreadGuard Testing Documentation

This document provides an overview of the testing strategy, test organization, and instructions for running tests for the ThreadGuard static analysis tool.

## Test Organization

The test suite is organized into several test files, each focusing on different aspects of the codebase:

### Core Functionality Tests
- `test_async_reader.py`: Tests for async reader pattern detection
- `test_thread_safe_queue.py`: Tests for thread-safe queue implementation
- `test_command_handler.py`: Tests for command handler pattern
- `test_thread_pool.py`: Tests for thread pool implementation
- `test_singleton.py`: Tests for thread-safe singleton pattern

### Concurrency Issue Detection
- `test_deadlock_detection.py`: Tests for deadlock detection
- `test_condition_variable.py`: Tests for condition variable usage
- `test_atomic_operations.py`: Tests for atomic operation handling
- `test_data_race.py`: Tests for data race detection

### Integration Tests
- `test_integration.py`: End-to-end tests with real C++ code examples

### Test Data
- `test_data/`: Directory containing sample C++ files used in tests

## Running Tests

### Prerequisites
- Python 3.8+
- pytest
- pytest-cov (for coverage reporting)
- networkx (for deadlock detection)

### Basic Test Execution

Run all tests:
```bash
pytest -v
```

Run a specific test file:
```bash
pytest tests/test_async_reader.py -v
```

Run a specific test function:
```bash
pytest tests/test_async_reader.py::TestAsyncReader::test_async_reader_pattern -v
```

### Test Coverage

Generate coverage report:
```bash
pytest --cov=threadguard tests/
```

Generate HTML coverage report:
```bash
pytest --cov=threadguard --cov-report=html tests/
open htmlcov/index.html  # View the report
```

## Test Patterns

### Fixtures

The test suite uses pytest fixtures for common test resources:

- `temp_cpp_file`: Creates a temporary C++ file for testing
- `analyzer`: Provides a fresh instance of ThreadGuardAnalyzer for each test

Example:
```python
def test_example(temp_cpp_file, analyzer):
    content = """
    # C++ code here
    """
    temp_cpp_file.write_text(content)
    result = analyzer.analyze_file(temp_cpp_file)
    assert not result.races, "No races should be detected"
```

### Assertions

Common assertion patterns:

1. Check for data races:
```python
assert not result.races, f"Found unexpected races: {result.races}"
```

2. Check for deadlock risks:
```python
assert not result.deadlock_risks, f"Found deadlock risks: {result.deadlock_risks}"
```

3. Verify specific issues are detected:
```python
assert any("race condition" in str(issue) for issue in result.races), \
    "Expected race condition not detected"
```

## Adding New Tests

1. **Unit Tests**: Add tests for new functionality in the appropriate test file
2. **Integration Tests**: Add end-to-end tests for complete scenarios
3. **Test Data**: Add sample C++ files to `test_data/` if needed
4. **Documentation**: Update this document if test organization changes

## Continuous Integration

The test suite runs automatically on pull requests and merges to the main branch. The CI pipeline includes:

1. Unit tests
2. Integration tests
3. Code coverage reporting
4. Static type checking
5. Linting

## Debugging Tests

To debug a failing test:

1. Run the specific test with `-s` to see print output:
   ```bash
   pytest tests/test_example.py -v -s
   ```

2. Use `pdb` for interactive debugging:
   ```python
   import pdb; pdb.set_trace()  # Add this line where you want to break
   ```

3. Check the test logs in the CI/CD pipeline for environment-specific issues

## Performance Considerations

- Keep test files focused and fast
- Use appropriate test fixtures to avoid code duplication
- Mock external dependencies when possible
- Use `pytest-xdist` for parallel test execution:
  ```bash
  pytest -n auto  # Run tests in parallel using all available CPUs
  ```

## Code Coverage

Current test coverage: [Coverage Percentage]%

To improve coverage:
1. Identify untested code paths using coverage reports
2. Add test cases for edge conditions
3. Test error handling paths
4. Verify all public API endpoints are tested

## Known Issues

- [ ] Some tests may be flaky due to timing issues
- [ ] Deadlock detection has some false positives in complex scenarios
- [ ] Test coverage could be improved for edge cases

## Contributing

When contributing tests:
1. Follow existing test patterns
2. Add clear docstrings
3. Include assertions that clearly express the expected behavior
4. Update this documentation if you add new test categories or patterns

## License

This test suite is part of the ThreadGuard project and is covered under the same license as the main project.
