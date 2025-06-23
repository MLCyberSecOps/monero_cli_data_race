# Contributing to ThreadGuard

Thank you for your interest in contributing to ThreadGuard! We welcome contributions from the community to help improve this project.

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project, you agree to abide by its terms.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
3. **Create a branch** for your changes
4. **Commit** your changes and **push** to your fork
5. Open a **pull request** with a clear description of your changes

## Development Setup

1. Install Python 3.7 or higher
2. Clone the repository:
   ```bash
   git clone https://github.com/MLCyberSecOps/monero_cli_data_race.git
   cd monero_cli_data_race
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   make install
   make install-hooks
   ```

## Reporting Monero Issues

If you've used ThreadGuard to identify potential concurrency issues in the Monero codebase, please follow these steps:

1. **Verify the Issue**
   - Run the analysis multiple times to confirm consistency
   - Check if the issue exists in the latest Monero master branch
   - Review the code to understand the context

2. **Report to Monero**
   - Create an issue in the [Monero GitHub repository](https://github.com/monero-project/monero/issues/new/choose)
   - Use a clear, descriptive title (e.g., "Potential data race in [file]:[line]")
   - Include the ThreadGuard version and analysis output
   - Tag with appropriate labels (e.g., `bug`, `security`)

3. **Reference in This Repo (Optional)**
   - Open an issue using the [Monero Issue template](.github/ISSUE_TEMPLATE/monero-issue.md)
   - Link to the Monero issue
   - This helps us track how the tool is being used to improve Monero

## Development Workflow

1. Run all checks before committing:
   ```bash
   make check
   ```
2. The pre-commit hooks will automatically format and check your code
3. Write tests for new features or bug fixes
4. Ensure all tests pass:
   ```bash
   make test
   ```
5. Update documentation if needed
6. Commit your changes with a descriptive message
7. Push to your fork and open a pull request

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use type hints for all function signatures
- Write docstrings following [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Keep lines under 88 characters (Black's default)
- Use `snake_case` for variables and functions, `CamelCase` for classes

## Testing

- Write tests for new features and bug fixes
- Run tests with `make test`
- Aim for good test coverage (90%+)
- Use descriptive test names that explain what's being tested

## Documentation

- Update README.md with any changes to setup or usage
- Add docstrings to all public functions and classes
- Document any new command-line arguments or configuration options

## Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build
2. Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations and container parameters
3. Increase the version numbers in any examples files and the README.md to the new version that this Pull Request would represent. The versioning scheme we use is [SemVer](http://semver.org/)
4. You may merge the Pull Request once you have the sign-off of two other developers, or if you do not have permission to do that, you may request the second reviewer to merge it for you

## Reporting Issues

When reporting issues, please include:

- A clear description of the problem
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc.)
- Any relevant error messages or logs

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
